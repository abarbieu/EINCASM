import torch
import taichi as ti
import torch.nn as nn

from coralai.substrate.nn_lib import ch_norm


def activate_outputs(substrate):
    inds = substrate.ti_indices[None]
    substrate.mem[:, inds.com] = torch.sigmoid(ch_norm(substrate.mem[:, inds.com]))
    substrate.mem[:, [inds.acts_invest, inds.acts_liquidate]] = torch.softmax(substrate.mem[0, [inds.acts_invest, inds.acts_liquidate]], dim=0)

    substrate.mem[:, inds.acts_explore] = nn.ReLU()(ch_norm(substrate.mem[:, inds.acts_explore]))
    # substrate.mem[0, inds.acts_explore[0]] = mean_activation
    # substrate.mem[0, inds.acts_explore] = torch.softmax(substrate.mem[0, inds.acts_explore], dim=0)
    substrate.mem[0, inds.acts_explore] /= torch.mean(substrate.mem[0, inds.acts_explore], dim=0)

    substrate.mem[0, inds.acts] = torch.where((substrate.mem[0, inds.genome] < 0) |
                                              (substrate.mem[0, inds.infra] < 0.1),
                                              0, substrate.mem[0, inds.acts])

@ti.kernel
def apply_weights_and_biases(mem: ti.types.ndarray(), out_mem: ti.types.ndarray(),
                             sense_chinds: ti.types.ndarray(),
                             combined_weights: ti.types.ndarray(), combined_biases: ti.types.ndarray(),
                             dir_kernel: ti.types.ndarray(), dir_order: ti.types.ndarray(),
                             ti_inds: ti.template()):
    inds = ti_inds[None]
    for i, j, act_k in ti.ndrange(mem.shape[2], mem.shape[3], out_mem.shape[0]):
        val = 0.0
        rot = mem[0, inds.rot, i, j]
        genome_key = int(mem[0, inds.genome, i, j])
        for sense_ch_n in ti.ndrange(sense_chinds.shape[0]):
            # base case [0,0]
            start_weight_ind = sense_ch_n * (dir_kernel.shape[0]+1)
            val += (mem[0, sense_chinds[sense_ch_n], i, j] *
                    combined_weights[genome_key, 0, act_k, start_weight_ind])
            for offset_m in ti.ndrange(dir_kernel.shape[0]):
                ind = int((rot + dir_order[offset_m]) % dir_kernel.shape[0])
                neigh_x = (i + dir_kernel[ind, 0]) % mem.shape[2]
                neigh_y = (j + dir_kernel[ind, 1]) % mem.shape[3]
                weight_ind = start_weight_ind + offset_m
                val += mem[0, sense_chinds[sense_ch_n], neigh_x, neigh_y] * combined_weights[genome_key, 0, act_k, weight_ind]
        out_mem[act_k, i, j] = val + combined_biases[genome_key, 0, act_k, 0]


@ti.kernel
def explore(mem: ti.types.ndarray(), max_act_i: ti.types.ndarray(),
            infra_delta: ti.types.ndarray(), energy_delta: ti.types.ndarray(),
            winning_genomes: ti.types.ndarray(), winning_rots: ti.types.ndarray(),
            dir_kernel: ti.types.ndarray(), dir_order: ti.types.ndarray(), ti_inds: ti.template()):
    inds = ti_inds[None]
    first_explore_act = int(inds.acts_explore[0])
    for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
        winning_genome = mem[0, inds.genome, i, j]
        max_bid = mem[0, inds.infra, i, j] * 0.5
        winning_rot = mem[0, inds.rot, i, j]

        for offset_n in ti.ndrange(dir_kernel.shape[0]): # this order doesn't matter
            neigh_x = (i + dir_kernel[offset_n, 0]) % mem.shape[2]
            neigh_y = (j + dir_kernel[offset_n, 1]) % mem.shape[3]
            if mem[0, inds.genome, neigh_x, neigh_y] < 0:
                continue
            neigh_max_act_i = max_act_i[neigh_x, neigh_y] # Could be [0,0], so could overflow dir_kernel
            if neigh_max_act_i == 0:
                continue
            neigh_max_act_i -= 1 # aligns with dir_kernel now
            neigh_rot = mem[0, inds.rot, neigh_x, neigh_y] # represents the dir the cell is pointing
            neigh_dir_ind = int((neigh_rot+dir_order[neigh_max_act_i]) % dir_kernel.shape[0])
            neigh_dir_x = dir_kernel[neigh_dir_ind, 0]
            neigh_dir_y = dir_kernel[neigh_dir_ind, 1]
            bid = 0.0
            # If neigh's explore dir points towards this center
            if ((neigh_dir_x + dir_kernel[offset_n, 0]) == 0 and (neigh_dir_y + dir_kernel[offset_n, 1]) == 0):
                neigh_act = mem[0, first_explore_act + neigh_max_act_i, neigh_x, neigh_y]
                neigh_infra = mem[0, inds.infra, neigh_x, neigh_y]
                bid = neigh_infra * neigh_act * 0.2
                energy_delta[neigh_x, neigh_y] -= bid# * neigh_act # bids are always taken as investment
                infra_delta[i, j] += bid * 0.8
                if bid > max_bid:
                    max_bid = bid
                    winning_genome = mem[0, inds.genome, neigh_x, neigh_y]
                    winning_rot = (neigh_rot+dir_order[neigh_max_act_i]) % dir_kernel.shape[0] # aligns with the dir the winning neighbor explored from
        winning_genomes[i, j] = winning_genome
        winning_rots[i, j] = winning_rot


def explore_physics(substrate, dir_kernel, dir_order):
    inds = substrate.ti_indices[None]

    max_act_i = torch.argmax(substrate.mem[0, inds.acts_explore], dim=0) # be warned, this is the index of the actuator not the index in memory, so 0-6 not
    infra_delta = torch.zeros_like(substrate.mem[0, inds.infra])
    energy_delta = torch.zeros_like(infra_delta)
    winning_genome = torch.zeros_like(substrate.mem[0, inds.genome])
    winning_rots = torch.zeros_like(substrate.mem[0, inds.rot])
    explore(substrate.mem, max_act_i,
            infra_delta, energy_delta,
            winning_genome, winning_rots,
            dir_kernel, dir_order, substrate.ti_indices)
    # handle_investment(substrate, infra_delta)
    substrate.mem[0, inds.infra] += infra_delta
    substrate.mem[0, inds.energy] += energy_delta
    substrate.mem[0, inds.genome] = winning_genome
    substrate.mem[0, inds.rot] = winning_rots

@ti.kernel
def flow_energy_down(mem: ti.types.ndarray(), out_energy_mem: ti.types.ndarray(),
                     max_energy: ti.f32, kernel: ti.types.ndarray(), ti_inds: ti.template()):
    inds = ti_inds[None]
    for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
        central_energy = mem[0, inds.energy, i, j]
        if central_energy > max_energy:
            energy_sum_inverse = 0.0
            # Calculate the sum of the inverse of energy levels for neighboring cells
            for off_n in ti.ndrange(kernel.shape[0]):
                neigh_x = (i + kernel[off_n, 0]) % mem.shape[2]
                neigh_y = (j + kernel[off_n, 1]) % mem.shape[3]
                energy_level = mem[0, inds.energy, neigh_x, neigh_y]
                # Avoid division by zero by ensuring a minimum energy level
                energy_level = max(energy_level, 0.0001)  # Assuming 0.0001 as a minimum to avoid division by zero
                energy_sum_inverse += 1.0 / energy_level
            # Distribute energy based on the inverse proportion of energy levels
            for off_n in ti.ndrange(kernel.shape[0]):
                neigh_x = (i + kernel[off_n, 0]) % mem.shape[2]
                neigh_y = (j + kernel[off_n, 1]) % mem.shape[3]
                neigh_energy = mem[0, inds.energy, neigh_x, neigh_y]
                neigh_energy = max(neigh_energy, 0.0001)  # Again, ensuring a minimum energy level
                # Calculate the share of energy based on the inverse of energy level
                energy_share = central_energy * ((1.0 / neigh_energy) / energy_sum_inverse)
                out_energy_mem[neigh_x, neigh_y] += energy_share
        else:
            out_energy_mem[i, j] += mem[0, inds.energy, i, j]

@ti.kernel
def distribute_energy(mem: ti.types.ndarray(), out_energy: ti.types.ndarray(), max_energy: ti.f32, kernel: ti.types.ndarray(), ti_inds: ti.template()):
    inds = ti_inds[None]
    for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
        # if mem[0, inds.energy, i, j] > max_energy:
        for off_n in ti.ndrange(kernel.shape[0]):
            neigh_x = (i + kernel[off_n, 0]) % mem.shape[2]
            neigh_y = (j + kernel[off_n, 1]) % mem.shape[3]
            out_energy[neigh_x, neigh_y] += (mem[0, inds.energy, i, j] / kernel.shape[0])
        # else:
        #     out_energy[i, j] += mem[0, inds.energy, i, j]

@ti.kernel
def flow_energy_up(mem: ti.types.ndarray(), out_energy_mem: ti.types.ndarray(),
                      kernel: ti.types.ndarray(), ti_inds: ti.template()):
    inds = ti_inds[None]
    for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
        central_energy = mem[0, inds.energy, i, j]
        infra_sum = 0.00001
        for off_n in ti.ndrange(kernel.shape[0]):
            neigh_x = (i + kernel[off_n, 0]) % mem.shape[2]
            neigh_y = (j + kernel[off_n, 1]) % mem.shape[3]
            infra_sum += mem[0, inds.infra, neigh_x, neigh_y]
        for off_n in ti.ndrange(kernel.shape[0]):
            neigh_x = (i + kernel[off_n, 0]) % mem.shape[2]
            neigh_y = (j + kernel[off_n, 1]) % mem.shape[3]
            neigh_infra = mem[0, inds.infra, neigh_x, neigh_y]
            out_energy_mem[neigh_x, neigh_y] += (
                central_energy * ((neigh_infra/infra_sum)))
            
@ti.kernel
def distribute_infra(mem: ti.types.ndarray(), out_infra: ti.types.ndarray(), out_energy: ti.types.ndarray(), 
                     max_infra: ti.f32, kernel: ti.types.ndarray(), ti_inds: ti.template()):
    inds = ti_inds[None]
    for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
        if mem[0, inds.infra, i, j] > max_infra:
            for off_n in ti.ndrange(kernel.shape[0]):
                neigh_x = (i + kernel[off_n, 0]) % mem.shape[2]
                neigh_y = (j + kernel[off_n, 1]) % mem.shape[3]
                out_infra[neigh_x, neigh_y] += (mem[0, inds.infra, i, j] / kernel.shape[0])
            # above_max = out_infra[neigh_x, neigh_y] - max_infra
            # out_infra[neigh_x, neigh_y] -= above_max
            # out_energy[neigh_x, neigh_y] += above_max
        else:
            out_infra[i, j] += mem[0, inds.infra, i, j]
    

def energy_physics(substrate, kernel, max_infra, max_energy):
    # TODO: Implement infra->energy conversion, apply before energy flow
    inds = substrate.ti_indices[None]
    # substrate.mem[0, inds.infra] = torch.clamp(substrate.mem[0, inds.infra], 0.0001, 100)

    energy_out_mem = torch.zeros_like(substrate.mem[0, inds.energy])
    flow_energy_up(substrate.mem, energy_out_mem, kernel, substrate.ti_indices)
    print(f"Energy Out Mem Sum Difference: {energy_out_mem.sum().item() - substrate.mem[0, inds.energy].sum().item():.4f}")
    substrate.mem[0, inds.energy] = energy_out_mem
    
    energy_out_mem = torch.zeros_like(substrate.mem[0, inds.energy])
    distribute_energy(substrate.mem, energy_out_mem, max_energy, kernel, substrate.ti_indices)
    substrate.mem[0, inds.energy] = energy_out_mem

    infra_out_mem = torch.zeros_like(substrate.mem[0, inds.infra])
    energy_out_mem = torch.zeros_like(substrate.mem[0, inds.energy])
    distribute_infra(substrate.mem, infra_out_mem, energy_out_mem, max_infra, kernel, substrate.ti_indices)
    substrate.mem[0, inds.infra] = infra_out_mem
    substrate.mem[0, inds.energy] = energy_out_mem


def invest_liquidate(substrate):
    inds = substrate.ti_indices[None]
    investments = substrate.mem[0, inds.acts_invest] * substrate.mem[0, inds.energy]
    liquidations = substrate.mem[0, inds.acts_liquidate] * substrate.mem[0, inds.infra]
    substrate.mem[0, inds.energy] += liquidations - investments
    substrate.mem[0, inds.infra] += investments - liquidations


