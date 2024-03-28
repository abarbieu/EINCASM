import copy
from datetime import datetime
import os
import random
import numpy as np
from neat.reporting import ReporterSet
from neat.reporting import BaseReporter
import pickle

import torch
import neat
from neat.six_util import iteritems, itervalues

import taichi as ti
import torch.nn as nn


from coralai.instances.coral.coral_physics import invest_liquidate, explore_physics, energy_physics, activate_outputs, apply_weights_and_biases

# from pytorch_neat.cppn import create_cppn
from pytorch_neat.activations import identity_activation
from pytorch_neat.linear_net import LinearNet
from ..substrate.nn_lib import ch_norm

@ti.data_oriented
class SpaceEvolver():
    def __init__(self, config_path, substrate, kernel, dir_order, sense_chs, act_chs):
        torch_device = substrate.torch_device
        self.torch_device = torch_device
        self.substrate = substrate
        self.reporters = ReporterSet()
        self.substrate = substrate
        self.kernel = torch.tensor(kernel, device=torch_device)
        self.dir_kernel = self.kernel[1:] # for directional operations
        # Represents the order in which the kernel should be applied
        # in this case, in alternating closest vectors from a direction
        self.dir_order = torch.tensor(dir_order, device=torch_device)
        
        self.sense_chs = sense_chs
        self.sense_chinds = substrate.windex[sense_chs]
        self.n_senses = len(self.sense_chinds)

        self.act_chs = act_chs
        self.act_chinds = substrate.windex[act_chs]
        self.n_acts = len(self.act_chinds)


        self.neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           config_path)
        
        self.timestep = 0
        self.out_mem = None
        self.energy_offset = 0.0

        self.genomes = []
        self.ages = []
        self.combined_weights = []
        self.combined_biases = []
        self.init_population()
        self.init_substrate(self.genomes)
        self.time_last_cull = 0


        self.add_reporter(neat.StdOutReporter(True))
        self.add_reporter(neat.StatisticsReporter())

        current_datetime = datetime.now().strftime("%y%m%d-%H%M_%S")
        self.checkpoint_dir = f'history/space_evolver_run_{current_datetime}'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_prefix_full = os.path.join(self.checkpoint_dir, f"checkpoint")
        self.add_reporter(neat.Checkpointer(generation_interval=5, filename_prefix=checkpoint_prefix_full))
        # Copy config from config_path to checkpoint_dir
        with open(config_path, "r") as f:
            config_str = f.read()
        with open(os.path.join(self.checkpoint_dir, "neat_config"), "w") as f:
            f.write(config_str)
        self.substrate.save_metadata_to_json(filepath = os.path.join(self.checkpoint_dir, "sub_metadata.json"))

    def init_population(self):
        genomes = []
        for i in range(self.neat_config.pop_size):
            genome = neat.DefaultGenome(str(i))
            genome.configure_new(self.neat_config.genome_config)
            self.add_organism_get_key(genome)

        return genomes


    def init_substrate(self, genomes):
        inds = self.substrate.ti_indices[None]
        self.substrate.mem[0, inds.genome] = torch.where(
            torch.rand_like(self.substrate.mem[0, inds.genome]) > 0.2 ,
            torch.randint_like(self.substrate.mem[0, inds.genome], 0, len(genomes)),
            -1
        )
        self.substrate.mem[0, inds.genome, self.substrate.w//8:-self.substrate.w//8, self.substrate.h//8:-self.substrate.h//8] = -1
        self.substrate.mem[0, inds.energy, ...] = 0.5
        self.substrate.mem[0, inds.infra, ...] = 0.5
        self.substrate.mem[0, inds.rot] = torch.randint_like(self.substrate.mem[0, inds.rot], 0, self.dir_kernel.shape[0])

    def run(self, n_timesteps, vis, n_rad_spots, radiate_interval, cull_max_pop, cull_interval=100):
        timestep = 0
        while timestep < n_timesteps and vis.window.running:
            combined_weights = torch.stack(self.combined_weights, dim=0)
            combined_biases = torch.stack(self.combined_biases, dim=0)
            self.step_sim(combined_weights, combined_biases)
            self.report_if_necessary(timestep)
            vis.update()
            if timestep % radiate_interval == 0:
                self.apply_radiation_mutation(n_rad_spots)
                print("RADIATING")
            if vis.next_generation:
                vis.next_generation = False
                break
            if len(self.genomes) > cull_max_pop and (self.timestep - self.time_last_cull) > cull_interval:
                # self.cull_genomes(cull_cell_thresh, cull_age_thresh)
                # if len(self.genomes) > cull_max_pop:
                self.reduce_population_to_threshold(cull_max_pop)
            timestep += 1
            self.timestep = timestep

    
    def step_sim(self, combined_weights, combined_biases):
        inds = self.substrate.ti_indices[None]
        self.forward(combined_weights, combined_biases)
        self.energy_offset = self.get_energy_offset(self.timestep)
        self.ages = [age + 1 for age in self.ages]
        self.substrate.mem[0, inds.energy] += (torch.randn_like(self.substrate.mem[0, inds.energy]) + (self.energy_offset-2)) * 0.2
        self.substrate.mem[0, inds.energy] += (torch.randn_like(self.substrate.mem[0, inds.energy]) + (self.energy_offset-2)) * 0.2
        self.substrate.mem[0, inds.energy] = torch.clamp(self.substrate.mem[0, inds.energy], 0.000001, 10000)
        self.substrate.mem[0, inds.infra] = torch.clamp(self.substrate.mem[0, inds.infra], 0.000001, 10000)
        self.substrate.mem[0, inds.rot] += torch.clamp((torch.randn_like(self.substrate.mem[0, inds.rot])*0.1)//1, -1.0, 1.0)
        self.substrate.mem[0, inds.infra, self.substrate.w//3, self.substrate.h//3] += 1.0
        self.substrate.mem[0, inds.energy, self.substrate.w//3, self.substrate.h//3] += 1.0
        self.substrate.mem[0, inds.infra, 2*self.substrate.w//3, 2*self.substrate.h//3] += 1.0
        self.substrate.mem[0, inds.energy, 2*self.substrate.w//3, 2*self.substrate.h//3] += 1.0
        self.substrate.mem[0, inds.infra, self.substrate.w//3, 2*self.substrate.h//3] += 1.0
        self.substrate.mem[0, inds.energy, self.substrate.w//3, 2*self.substrate.h//3] += 1.0
        self.substrate.mem[0, inds.infra, 2*self.substrate.w//3, self.substrate.h//3] += 1.0
        self.substrate.mem[0, inds.energy, 2*self.substrate.w//3, self.substrate.h//3] += 1.0

        if self.timestep % 20 == 0:
            inds_chunk = torch.tensor(self.substrate.windex[['genome', 'infra', 'energy']], device=self.torch_device)
            kill_vals = torch.tensor([-1, 0.01, 0.01], device=self.torch_device)
            grow_vals = torch.tensor([-1, 0.01, 3.0], device=self.torch_device)
            x_kill = np.random.randint(0, self.substrate.w)
            y_kill = np.random.randint(0, self.substrate.h)
            out_mem = self.substrate.mem.clone()
            self.set_chunk(out_mem, x_kill, y_kill, 4, inds_chunk, kill_vals)
            # x_grow = np.random.randint(0, self.substrate.w)
            # y_grow = np.random.randint(0, self.substrate.h)
            # self.set_chunk(out_mem, x_grow, y_grow, 2, inds_chunk, grow_vals)
            self.substrate.mem = out_mem

    @ti.kernel
    def set_chunk(self, out_mem: ti.types.ndarray(), cx: ti.i32, cy: ti.i32, radius: ti.i32,
                  inds: ti.types.ndarray(), vals: ti.types.ndarray()):
        for off_x, off_y, k in ti.ndrange((-radius, +radius), (-radius, +radius), inds.shape[0]):
            x = (cx + off_x) % out_mem.shape[2]
            y = (cy + off_y) % out_mem.shape[3]
            out_mem[0, inds[k], x, y] *= 0.0
            out_mem[0, inds[k], x, y] += vals[k]

    def forward(self, weights, biases):
        inds = self.substrate.ti_indices[None]
        out_mem = torch.zeros_like(self.substrate.mem[0, self.act_chinds])
        apply_weights_and_biases(
            self.substrate.mem, out_mem,
            self.sense_chinds,
            weights, biases,
            self.dir_kernel, self.dir_order,
            self.substrate.ti_indices)
        self.substrate.mem[0, self.act_chinds] = out_mem
        self.apply_physics()
    

    def apply_physics(self):
        inds = self.substrate.ti_indices[None]
        # self.substrate.mem[0, inds.energy, self.substrate.w//2, self.substrate.h//2] += 10
        # sum_before = self.substrate.mem[0, inds.energy].sum() + self.substrate.mem[0, inds.infra].sum()

        activate_outputs(self.substrate)
        # sum_after = self.substrate.mem[0, inds.energy].sum() + self.substrate.mem[0, inds.infra].sum()
        # assert abs(sum_before - sum_after) < 1e-5, f"Sum before: {sum_before:.4f}, Sum after: {sum_after:.4f}"

        invest_liquidate(self.substrate)
        # sum_after = self.substrate.mem[0, inds.energy].sum() + self.substrate.mem[0, inds.infra].sum()
        # assert abs(sum_before - sum_after) < 1e-5, f"Sum before: {sum_before:.4f}, Sum after: {sum_after:.4f}"

        explore_physics(self.substrate, self.kernel, self.dir_order)
        # sum_after = self.substrate.mem[0, inds.energy].sum() + self.substrate.mem[0, inds.infra].sum()
        # assert abs(sum_before - sum_after) < 1e-5, f"Sum before: {sum_before:.4f}, Sum after: {sum_after:.4f}"

        energy_physics(self.substrate, self.kernel, max_infra=5, max_energy=4)
        # sum_after = self.substrate.mem[0, inds.energy].sum() + self.substrate.mem[0, inds.infra].sum()
        # assert abs(sum_before - sum_after) < 1e-5, f"Sum before: {sum_before:.4f}, Sum after: {sum_after:.4f}"

        self.substrate.mem[0, inds.genome] = torch.where(
            (self.substrate.mem[0, inds.infra] + self.substrate.mem[0, inds.energy]) > 0.1,
            self.substrate.mem[0, inds.genome],
            -1
        )

    def apply_radiation_mutation(self, n_spots, spot_live_radius=2, spot_dead_radius=3):
        inds = self.substrate.ti_indices[None]
        xs = torch.randint(0, self.substrate.w, (n_spots,))
        ys = torch.randint(0, self.substrate.h, (n_spots,))
        
        out_mem = self.substrate.mem.clone()
        for i in range(n_spots):
            genome_key = int(self.substrate.mem[0, inds.genome, xs[i], ys[i]].item())
            inds_genome = torch.tensor([inds.genome], device=self.torch_device)
            vals = torch.tensor([0.0], device=self.torch_device)
            x = xs[i].item()
            y = ys[i].item()
            vals[0] = -1
            self.set_chunk(out_mem, x, y, spot_dead_radius, inds_genome, vals)

            rand_genome_key = torch.randint(0, len(self.genomes), (1,))
            if genome_key < 0:
                vals[0] = rand_genome_key
                self.set_chunk(out_mem, x, y, 1, inds_genome, vals)
            else:
                if random.random() < 0.5:
                    new_genome = copy.deepcopy(self.genomes[genome_key])
                    new_genome.mutate(self.neat_config.genome_config)
                    new_genome_key = self.add_organism_get_key(new_genome)
                    vals[0] = new_genome_key
                    self.set_chunk(out_mem, x, y, spot_live_radius, inds_genome, vals)
                else: 
                    new_genome = neat.DefaultGenome(str(len(self.genomes)))
                    self.genomes[genome_key].fitness = 0.0
                    self.genomes[rand_genome_key].fitness = 0.0
                    new_genome.configure_crossover(self.genomes[genome_key], self.genomes[rand_genome_key], self.neat_config)
                    new_genome_key = self.add_organism_get_key(new_genome)
                    vals[0] = new_genome_key
                    self.set_chunk(out_mem, x, y, spot_live_radius, inds_genome, vals)
        inds_2 = torch.tensor(self.substrate.windex[['infra', 'energy']], device=self.torch_device)
        vals = torch.tensor([1.0, 1.0], device=self.torch_device)
        self.set_chunk(out_mem, x, y, spot_live_radius, inds_2, vals)
        self.substrate.mem = out_mem

    @ti.kernel
    def replace_genomes(self, mem: ti.types.ndarray(), out_mem: ti.types.ndarray(),
                        genome_transitions: ti.types.ndarray(), ti_indices: ti.template()):
        inds = ti_indices[None]
        for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
            if mem[0, inds.genome, i, j] < 0:
                out_mem[i, j] = mem[0, inds.genome, i, j]
            else:
                out_mem[i, j] = genome_transitions[int(mem[0, inds.genome, i, j])]

    def reduce_population_to_threshold(self, max_population):
        print(f"REDUCING pop to max of {max_population} from current size: {len(self.genomes)}")
        if len(self.genomes) <= max_population:
            print("Population within threshold. No reduction needed.")
            return

        inds = self.substrate.ti_indices[None]
        # Combine genome indices with their cell counts and ages
        genome_info = [(i, self.substrate.mem[0, inds.genome].eq(i).sum().item(), self.ages[i]) for i in range(len(self.genomes))]
        # Sort genomes first by cell count (descending) then by age (ascending) to prioritize younger genomes
        sorted_genomes_by_info = sorted(genome_info, key=lambda x: (-x[1], -x[2]))
        new_genomes = []
        new_ages = []
        new_combined_weights = []
        new_combined_biases = []
        genome_transitions = [None] * len(self.genomes)
        for i, (index_of_genome, _, _) in enumerate(sorted_genomes_by_info):
            if i >= max_population:
                print(f"KILLING {index_of_genome}")
                genome_transitions[index_of_genome] = -1
            else:
                new_genomes.append(self.genomes[index_of_genome])
                new_ages.append(self.ages[index_of_genome])
                new_combined_weights.append(self.combined_weights[index_of_genome])
                new_combined_biases.append(self.combined_biases[index_of_genome])
                genome_transitions[index_of_genome] = len(new_genomes) - 1
        genome_transitions = torch.tensor(genome_transitions, device=self.torch_device)
        out_mem = torch.zeros_like(self.substrate.mem[0, inds.genome])
        self.replace_genomes(self.substrate.mem, out_mem, genome_transitions, self.substrate.ti_indices)
        self.substrate.mem[0, inds.genome] = out_mem

        self.genomes = new_genomes
        self.ages = new_ages
        self.combined_weights = new_combined_weights
        self.combined_biases = new_combined_biases
        self.time_last_cull = self.timestep
        print(f"\tPop size after reduction: {len(self.genomes)}")
        if len(self.genomes) == 0:
            print("NO GENOMES LEFT. REINITIALIZING")
            self.genomes = self.init_population()
            self.init_substrate(self.genomes)


    def load_checkpoint(self, folderpath):
        config_path = os.path.join(folderpath, "..", "neat_config")
        self.neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                        config_path)
        self.load_genomes(os.path.join(folderpath, "genomes"))
        with open(os.path.join(folderpath, "ages"), 'rb') as f:
            self.ages = pickle.load(f)

        self.combined_biases = []
        self.combined_weights = []
        for genome in self.genomes:
            genome.fitness = 0.0
            net = self.create_torch_net(genome)
            self.combined_biases.append(net.biases)
            self.combined_weights.append(net.weights) 
        loaded_mem = torch.load(os.path.join(folderpath, "sub_mem"))
        self.substrate.mem = loaded_mem


    def save_genomes(self, genomes, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.genomes, f)
    
    def load_genomes(self, filename):
        with open(filename, 'rb') as f:
            genomes = pickle.load(f)
        return genomes
    
    def report_if_necessary(self, timestep):
        if timestep % 300 == 0:
            timestep_dir = os.path.join(self.checkpoint_dir, f"step_{timestep}")
            os.makedirs(timestep_dir, exist_ok=True)
            self.substrate.save_mem_to_pt(filepath = os.path.join(timestep_dir, "sub_mem"))
            self.save_genomes(self.genomes, os.path.join(timestep_dir, "genomes"))
            with open(os.path.join(timestep_dir, "ages"), 'wb') as f:
                pickle.dump(self.ages, f)

        # for i in range(len(self.genomes)):
        #     # org['genome'].fitness += self.substrate.mem[0, inds.genome].eq(i).sum().item()
        #     self.genomes[i].fitness = fitness_function(self.genomes[i], i)
        # self.reporters.start_generation(self.generation)

        # # Evaluate all genomes using the user-provided function.
        # fitness_function(list(iteritems(self.population)), self.config)

        # # Gather and report statistics.
        # best = None
        # for g in itervalues(self.population):
        #     if best is None or g.fitness > best.fitness:
        #         best = g

        # self.reporters.post_evaluate(self.config, self.population, self.species, best)

        # # # Create the next generation from the current generation.
        # # self.population = self.reproduction.reproduce(self.config, self.species,
        # #                                               self.config.pop_size, self.generation)

        # # Check for complete extinction.
        # if not self.species.species:
        #     self.reporters.complete_extinction()
        #     # TODO: re-seed env

        # # Divide the new population into species.
        # self.species.speciate(self.config, self.population, self.generation)

        # self.reporters.end_generation(self.config, self.population, self.species)

        # self.generation += 1

        # if self.config.no_fitness_termination:
        #     self.reporters.found_solution(self.config, self.generation, self.best_genome)

        # return self.best_genome
    
    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)    


    def add_organism_get_key(self, genome):
        # TODO: implement culling and memory consolidation
        self.genomes.append(genome)
        net = self.create_torch_net(genome)
        self.combined_weights.append(net.weights)
        self.combined_biases.append(net.biases)
        self.ages.append(0)
        return len(self.combined_biases) - 1


    def get_energy_offset(self, timestep, repeat_steps=100, amplitude=1, positive_scale=1, negative_scale=1):
        frequency = (2 * np.pi) / repeat_steps
        value = amplitude * np.sin(frequency * timestep)
        if value > 0:
            return value * positive_scale
        else:
            return value * negative_scale


    def create_torch_net(self, genome):
        input_coords = []
        # TODO: adjust for direcitonal kernel
        for ch in range(self.n_senses):
             input_coords.append([0, 0, self.sense_chinds[ch]])
             for offset_i in range(self.dir_order.shape[0]):
                offset_x = self.dir_kernel[self.dir_order[offset_i], 0]
                offset_y = self.dir_kernel[self.dir_order[offset_i], 1]
                input_coords.append([offset_x, offset_y, self.sense_chinds[ch]])

        output_coords = []
        for ch in range(self.n_acts):
            output_coords.append([0, 0, self.act_chinds[ch]])

        net = LinearNet.create(
            genome,
            self.neat_config,
            input_coords=input_coords,
            output_coords=output_coords,
            weight_threshold=0.0,
            weight_max=3.0,
            activation=identity_activation,
            cppn_activation=identity_activation,
            device=self.torch_device,
        )
        return net
    

    def get_genome_infra_sum(self, genome_key):
        inds = self.substrate.ti_indices[None]
        infra_sum = torch.where(self.substrate.mem[0, inds.genome] == genome_key, self.substrate.mem[0, inds.infra], 0).sum()
        return infra_sum

    # def cull_genomes(self, n_cells_thresh, age_thresh):
    #     print(f"CULLING pop of size: {len(self.genomes)}")
    #     inds = self.substrate.ti_indices[None]
    #     new_genomes = []
    #     new_ages = []
    #     new_combined_weights = []
    #     new_combined_biases = []
    #     for i in range(len(self.genomes)):
    #         where_i = self.substrate.mem[0, inds.genome].eq(i)
    #         n_cells = where_i.sum().item()
    #         print(f"\tGenome {i} is {self.ages[i]} steps old and has {n_cells} cells")
    #         if n_cells == 0 or (self.ages[i] > age_thresh and n_cells < n_cells_thresh):
    #             print(f"\t\tCulling genome {i}")
    #             self.substrate.mem[0, inds.genome] = torch.where(where_i, -1, self.substrate.mem[0, inds.genome])
    #         else:
    #             self.substrate.mem[0, inds.genome]= torch.where(where_i, len(new_genomes), self.substrate.mem[0, inds.genome])
    #             new_genomes.append(self.genomes[i])
    #             new_ages.append(self.ages[i])
    #             new_combined_weights.append(self.combined_weights[i])
    #             new_combined_biases.append(self.combined_biases[i])
    #     self.genomes = new_genomes
    #     self.ages = new_ages
    #     self.combined_weights = new_combined_weights
    #     self.combined_biases = new_combined_biases
    #     print(f"\tPop size after culling: {len(self.genomes)}")
    #     if len(self.genomes) == 0:
    #         print("Population extinct")
    #         self.init_population()
    #         self.init_substrate(self.genomes)
    #         self.timestep = 0import json
import warnings
import torch
import os
import taichi as ti
import numpy as np
from ..utils.ti_struct_factory import TaichiStructFactory
from .channel import Channel
from .substrate_index import SubstrateIndex


@ti.data_oriented
class Substrate:
    # TODO: Support multi-level indexing beyond 2 levels
    # TODO: Support mixed taichi and torch tensors - which will be transferred more?
    def __init__(self, shape, torch_dtype, torch_device, channels: dict = None):
        self.w = shape[0]
        self.h = shape[1]
        self.shape = (*shape, 0) # changed in malloc
        self.mem = None
        self.windex = None
        self.torch_dtype = torch_dtype
        self.torch_device = torch_device
        self.channels = {}
        if channels is not None:
            self.add_channels(channels)
        self.ti_ind_builder = TaichiStructFactory()
        self.ti_lims_builder = TaichiStructFactory()
        self.ti_indices = -1
        self.ti_lims = -1

    def save_metadata_to_json(self, filepath):
        config = {
            "shape": self.shape,
            "windex": self.windex.index_tree
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)

    def save_mem_to_pt(self, filepath):
        # Saves channels, channel metadata, dims, dtypes, etc
        torch.save(self.mem, filepath)

    def index_to_chname(self, index):
        return self.windex.index_to_chname(index)


    def add_channel(self, chid: str, ti_dtype=ti.f32, **kwargs):
        if self.mem is not None:
            raise ValueError(
                f"World: When adding channel {chid}: Cannot add channel after world memory is allocated (yet)."
            )
        self.channels[chid] = Channel(chid, self, ti_dtype=ti_dtype, **kwargs)


    def add_channels(self, channels: dict):
        if self.mem is not None:
            raise ValueError(
                f"World: When adding channels {channels}: Cannot add channels after world memory is allocated (yet)."
            )
        for chid in channels.keys():
            ch = channels[chid]
            if isinstance(ch, dict):
                self.add_channel(chid, **ch)
            else:
                self.add_channel(chid, ch)


    def check_ch_shape(self, shape):
        lshape = len(shape)
        if lshape > 3 or lshape < 2:
            raise ValueError(
                f"World: Channel shape must be 2 or 3 dimensional. Got shape: {shape}"
            )
        if shape[:2] != self.shape[:2]:
            print(shape[:2], self.shape[:2])
            raise ValueError(
                f"World: Channel shape must be (w, h, ...) where w and h are the world dimensions: {self.shape}. Got shape: {shape}"
            )
        if lshape == 2:
            return 1
        else:
            return shape[2]


    def stat(self, key):
        # Prints useful metrics about the channel(s) and contents
        minval = self[key].min()
        maxval = self[key].max()
        meanval = self[key].mean()
        stdval = self[key].std()
        shape = self[key].shape
        print(
            f"{key} stats:\n\tShape: {shape}\n\tMin: {minval}\n\tMax: {maxval}\n\tMean: {meanval}\n\tStd: {stdval}"
        )


    def _transfer_to_mem(self, mem, tensor_dict, index_tree, channel_dict):
        for chid, chindices in index_tree.items():
            if "subchannels" in chindices:
                for subchid, subchtree in chindices["subchannels"].items():
                    if tensor_dict[chid][subchid].dtype != self.torch_dtype:
                        warnings.warn(
                            f"\033[93mWorld: Casting {chid} of dtype: {tensor_dict[chid].dtype} to world dtype: {self.torch_dtype}\033[0m",
                            stacklevel=3,
                        )
                    if len(tensor_dict[chid][subchid].shape) == 2:
                        tensor_dict[chid][subchid] = tensor_dict[chid][
                            subchid
                        ].unsqueeze(2)
                    mem[:, :, subchtree["indices"]] = tensor_dict[chid][subchid].type(
                        self.torch_dtype
                    )
                    channel_dict[chid].add_subchannel(
                        subchid, ti_dtype=channel_dict[chid].ti_dtype
                    )
                    channel_dict[chid][subchid].link_to_mem(subchtree["indices"], mem)
                channel_dict[chid].link_to_mem(chindices["indices"], mem)
            else:
                if tensor_dict[chid].dtype != self.torch_dtype:
                    warnings.warn(
                        f"\033[93mWorld: Casting {chid} of dtype: {tensor_dict[chid].dtype} to world dtype: {self.torch_dtype}\033[0m",
                        stacklevel=3,
                    )
                if len(tensor_dict[chid].shape) == 2:
                    tensor_dict[chid] = tensor_dict[chid].unsqueeze(2)
                mem[:, :, chindices["indices"]] = tensor_dict[chid].type(
                    self.torch_dtype
                )
                channel_dict[chid].link_to_mem(chindices["indices"], mem)
        return mem, channel_dict


    def add_ti_inds(self, key, inds):
        if len(inds) == 1:
            self.ti_ind_builder.add_i(key, inds[0])
        else:
            self.ti_ind_builder.add_nparr_int(key, np.array(inds))


    def _index_subchannels(self, subchdict, start_index, parent_chid):
        end_index = start_index
        subch_tree = {}
        for subchid, subch in subchdict.items():
            if not isinstance(subch, torch.Tensor):
                raise ValueError(
                    f"World: Channel grouping only supported up to a depth of 2. Subchannel {subchid} of channel {parent_chid} must be a torch.Tensor. Got type: {type(subch)}"
                )
            subch_depth = self.check_ch_shape(subch.shape)
            indices = [i for i in range(end_index, end_index + subch_depth)]
            self.add_ti_inds(parent_chid + "_" + subchid, indices)
            self.ti_lims_builder.add_nparr_float(
                parent_chid + "_" + subchid, self.channels[parent_chid].lims
            )
            subch_tree[subchid] = {
                "indices": indices,
            }
            end_index += subch_depth
        return subch_tree, end_index - start_index


    def malloc(self):
        if self.mem is not None:
            raise ValueError("World: Cannot allocate world memory twice.")
        celltype = ti.types.struct(
            **{chid: self.channels[chid].ti_dtype for chid in self.channels.keys()}
        )
        tensor_dict = celltype.field(shape=self.shape[:2]).to_torch(
            device=self.torch_device
        )

        index_tree = {}
        endlayer_pointer = self.shape[2]
        for chid, chdata in tensor_dict.items():
            if isinstance(chdata, torch.Tensor):
                ch_depth = self.check_ch_shape(chdata.shape)
                indices = [
                    i for i in range(endlayer_pointer, endlayer_pointer + ch_depth)
                ]
                self.add_ti_inds(chid, indices)
                self.ti_lims_builder.add_nparr_float(chid, self.channels[chid].lims)
                index_tree[chid] = {"indices": indices}
                endlayer_pointer += ch_depth
            elif isinstance(chdata, dict):
                subch_tree, total_depth = self._index_subchannels(
                    chdata, endlayer_pointer, chid
                )
                indices = [
                    i for i in range(endlayer_pointer, endlayer_pointer + total_depth)
                ]
                self.add_ti_inds(chid, indices)
                self.ti_lims_builder.add_nparr_float(chid, self.channels[chid].lims)
                index_tree[chid] = {
                    "subchannels": subch_tree,
                    "indices": indices,
                }
                endlayer_pointer += total_depth

        self.shape = (*self.shape[:2], endlayer_pointer)
        mem = torch.zeros(self.shape, dtype=self.torch_dtype, device=self.torch_device)
        self.mem, self.channels = self._transfer_to_mem(
            mem, tensor_dict, index_tree, self.channels
        )
        self.windex = SubstrateIndex(index_tree)
        self.ti_indices = self.ti_ind_builder.build()
        self.ti_lims = self.ti_lims_builder.build()
        self.mem = self.mem.permute(2, 0, 1).unsqueeze(0).contiguous()
        self.shape = self.mem.shape


    def __getitem__(self, key):
        if self.mem is None:
            raise ValueError(f"World: World memory not allocated yet, cannot get {key}")
        val = self.mem[:, self.windex[key], :, :]
        return val
    
    def __setitem__(self, key, value):
        if self.mem is None:
            raise ValueError(f"World: World memory not allocated yet, cannot set {key}")
        raise NotImplementedError("World: Setting world values not implemented yet. (Just manipulate memory directly)")


    def get_inds_tivec(self, key):
        indices = self.windex[key]
        itype = ti.types.vector(n=len(indices), dtype=ti.i32)
        return itype(indices)


    def get_lims_timat(self, key):
        lims = []
        if isinstance(key, str):
            key = [key]
        if isinstance(key, tuple):
            key = [key[0]]
        for k in key:
            if isinstance(k, tuple):
                lims.append(self.channels[k[0]].lims)
            else:
                lims.append(self.channels[k].lims)
        if len(lims) == 1:
            lims = lims[0]
        lims = np.array(lims, dtype=np.float32)
        ltype = ti.types.matrix(lims.shape[0], lims.shape[1], dtype=ti.f32)
        return ltype(lims)


import time
import torch
import taichi as ti
from .substrate import Substrate


@ti.data_oriented
class Visualization:
    def __init__(self,
                 substrate: Substrate,
                 chids: list = None,
                 chinds: list = None,
                 name: str = None,
                 scale: int = None,):
        self.substrate = substrate
        self.w = substrate.w
        self.h = substrate.h
        self.chids = chids
        self.scale = 1 if scale is None else scale
        chinds = substrate.get_inds_tivec(chids)
        self.chinds = torch.tensor(list(chinds), device = substrate.torch_device)
        # self.name = f"Vis: {[self.substrate.index_to_chname(chindices[i]) for i in range(len(chindices))]}" if name is None else name
        self.name = "Vis"

        if scale is None:
            max_dim = max(self.substrate.w, self.substrate.h)
            desired_max_dim = 800
            scale = desired_max_dim // max_dim
            
        self.scale = scale
        self.img_w = self.substrate.w * scale
        self.img_h = self.substrate.h * scale
        self.n_channels = len(chinds)
        self.image = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.img_w, self.img_h))

        self.window = ti.ui.Window(
            f"{self.name}", (self.img_w, self.img_h), fps_limit=200, vsync=True
        )
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()
        self.paused = False
        self.brush_radius = 4
        self.mutating = False
        self.perturbation_strength = 0.1
        self.drawing = False
        self.prev_time = time.time()
        self.prev_pos = self.window.get_cursor_pos()
        self.channel_to_paint = 0
        self.val_to_paint = 0.1

    def set_channels(self, chindices):
        self.chinds = chindices

    @ti.kernel
    def add_val_to_loc(self,
            val: ti.f32,
            pos_x: ti.f32,
            pos_y: ti.f32,
            radius: ti.i32,
            channel_to_paint: ti.i32,
            mem: ti.types.ndarray()
        ):
        ind_x = int(pos_x * self.w)
        ind_y = int(pos_y * self.h)
        offset = int(pos_x) * 3
        for i, j in ti.ndrange((-radius, radius), (-radius, radius)):
            if (i**2) + j**2 < radius**2:
                mem[0, channel_to_paint, (i + ind_x) % self.w, (j + ind_y) % self.h] += val


    @ti.kernel
    def write_to_renderer(self, mem: ti.types.ndarray(), max_vals: ti.types.ndarray(), chinds: ti.types.ndarray()):
        for i, j in self.image:
            xind = (i//self.scale) % self.w
            yind = (j//self.scale) % self.h
            for k in ti.static(range(3)):
                chid = chinds[k]
                self.image[i, j][k] = mem[0, chid, xind, yind] / max_vals[k]

    def opt_window(self, sub_w):
        self.channel_to_paint = sub_w.slider_int("Paint channel: " +
                                                 f"{self.substrate.index_to_chname(self.channel_to_paint)}",
                                                 self.channel_to_paint, 0, 10)
        self.val_to_paint = sub_w.slider_float("Value to Paint", self.val_to_paint, -1.0, 1.0)
        self.val_to_paint = round(self.val_to_paint * 10) / 10
        self.brush_radius = sub_w.slider_int("Brush Radius", self.brush_radius, 1, 200)
        self.paused = sub_w.checkbox("Pause", self.paused)
        self.mutating = sub_w.checkbox("Perturb Weights", self.mutating)
        self.perturbation_strength = sub_w.slider_float("Perturbation Strength", self.perturbation_strength, 0.0, 5.0)

    def render_opt_window(self):
        self.canvas.set_background_color((1, 1, 1))
        opt_w = min(480 / self.img_w, self.img_w)
        opt_h = min(240 / self.img_h, self.img_h * 2)
        with self.gui.sub_window("Options", 0.05, 0.05, opt_w, opt_h) as sub_w:
            self.opt_window(sub_w)


    def check_events(self):
        for e in self.window.get_events(ti.ui.PRESS):
            if e.key in  [ti.ui.ESCAPE]:
                exit()
            if e.key == ti.ui.LMB and self.window.is_pressed(ti.ui.SHIFT):
                self.drawing = True
            elif e.key == ti.ui.SPACE:
                self.substrate.mem *= 0.0
        for e in self.window.get_events(ti.ui.RELEASE):
            if e.key == ti.ui.LMB:
                self.drawing = False


    def update(self):
        current_time = time.time()
        current_pos = self.window.get_cursor_pos()
        if not self.paused:
            self.check_events()
            if self.drawing and ((current_time - self.prev_time) > 0.1): # or (current_pos != self.prev_pos)):
                self.add_val_to_loc(self.val_to_paint, current_pos[0], current_pos[1], self.brush_radius, self.channel_to_paint, self.substrate.mem)
                self.prev_time = current_time  # Update the time of the last action
                self.prev_pos = current_pos

            max_vals = torch.tensor([self.substrate.mem[0, ch].max() for ch in self.chinds])
            self.write_to_renderer(self.substrate.mem, max_vals, self.chinds)
        self.render_opt_window()
        self.canvas.set_image(self.image)
        self.window.show()


import torch
import taichi as ti
import torch.nn as nn

from ...substrate.nn_lib import ch_norm


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
                ind = int((rot+dir_order[offset_m]) % dir_kernel.shape[0])
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


import copy
from datetime import datetime
import os
import random
import numpy as np
from neat.reporting import ReporterSet
from neat.reporting import BaseReporter
import pickle

import torch
import neat
from neat.six_util import iteritems, itervalues

import taichi as ti
import torch.nn as nn


from coralai.instances.coral.coral_physics import invest_liquidate, explore_physics, energy_physics, activate_outputs, apply_weights_and_biases

# from pytorch_neat.cppn import create_cppn
from pytorch_neat.activations import identity_activation
from pytorch_neat.linear_net import LinearNet
from ..substrate.nn_lib import ch_norm

@ti.data_oriented
class SpaceEvolver():
    def __init__(self, config_path, substrate, kernel, dir_order, sense_chs, act_chs):
        torch_device = substrate.torch_device
        self.torch_device = torch_device
        self.substrate = substrate
        self.reporters = ReporterSet()
        self.substrate = substrate
        self.kernel = torch.tensor(kernel, device=torch_device)
        self.dir_kernel = self.kernel[1:] # for directional operations
        # Represents the order in which the kernel should be applied
        # in this case, in alternating closest vectors from a direction
        self.dir_order = torch.tensor(dir_order, device=torch_device)
        
        self.sense_chs = sense_chs
        self.sense_chinds = substrate.windex[sense_chs]
        self.n_senses = len(self.sense_chinds)

        self.act_chs = act_chs
        self.act_chinds = substrate.windex[act_chs]
        self.n_acts = len(self.act_chinds)


        self.neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           config_path)
        
        self.timestep = 0
        self.out_mem = None
        self.energy_offset = 0.0

        self.genomes = []
        self.ages = []
        self.combined_weights = []
        self.combined_biases = []
        self.init_population()
        self.init_substrate(self.genomes)
        self.time_last_cull = 0


        self.add_reporter(neat.StdOutReporter(True))
        self.add_reporter(neat.StatisticsReporter())

        current_datetime = datetime.now().strftime("%y%m%d-%H%M_%S")
        self.checkpoint_dir = f'history/space_evolver_run_{current_datetime}'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_prefix_full = os.path.join(self.checkpoint_dir, f"checkpoint")
        self.add_reporter(neat.Checkpointer(generation_interval=5, filename_prefix=checkpoint_prefix_full))
        # Copy config from config_path to checkpoint_dir
        with open(config_path, "r") as f:
            config_str = f.read()
        with open(os.path.join(self.checkpoint_dir, "neat_config"), "w") as f:
            f.write(config_str)
        self.substrate.save_metadata_to_json(filepath = os.path.join(self.checkpoint_dir, "sub_metadata.json"))

    def init_population(self):
        genomes = []
        for i in range(self.neat_config.pop_size):
            genome = neat.DefaultGenome(str(i))
            genome.configure_new(self.neat_config.genome_config)
            self.add_organism_get_key(genome)

        return genomes


    def init_substrate(self, genomes):
        inds = self.substrate.ti_indices[None]
        self.substrate.mem[0, inds.genome] = torch.where(
            torch.rand_like(self.substrate.mem[0, inds.genome]) > 0.2 ,
            torch.randint_like(self.substrate.mem[0, inds.genome], 0, len(genomes)),
            -1
        )
        self.substrate.mem[0, inds.genome, self.substrate.w//8:-self.substrate.w//8, self.substrate.h//8:-self.substrate.h//8] = -1
        self.substrate.mem[0, inds.energy, ...] = 0.5
        self.substrate.mem[0, inds.infra, ...] = 0.5
        self.substrate.mem[0, inds.rot] = torch.randint_like(self.substrate.mem[0, inds.rot], 0, self.dir_kernel.shape[0])

    def run(self, n_timesteps, vis, n_rad_spots, radiate_interval, cull_max_pop, cull_interval=100):
        timestep = 0
        while timestep < n_timesteps and vis.window.running:
            combined_weights = torch.stack(self.combined_weights, dim=0)
            combined_biases = torch.stack(self.combined_biases, dim=0)
            self.step_sim(combined_weights, combined_biases)
            self.report_if_necessary(timestep)
            vis.update()
            if timestep % radiate_interval == 0:
                self.apply_radiation_mutation(n_rad_spots)
                print("RADIATING")
            if vis.next_generation:
                vis.next_generation = False
                break
            if len(self.genomes) > cull_max_pop and (self.timestep - self.time_last_cull) > cull_interval:
                # self.cull_genomes(cull_cell_thresh, cull_age_thresh)
                # if len(self.genomes) > cull_max_pop:
                self.reduce_population_to_threshold(cull_max_pop)
            timestep += 1
            self.timestep = timestep

    
    def step_sim(self, combined_weights, combined_biases):
        inds = self.substrate.ti_indices[None]
        self.forward(combined_weights, combined_biases)
        self.energy_offset = self.get_energy_offset(self.timestep)
        self.ages = [age + 1 for age in self.ages]
        self.substrate.mem[0, inds.energy] += (torch.randn_like(self.substrate.mem[0, inds.energy]) + (self.energy_offset-2)) * 0.2
        self.substrate.mem[0, inds.energy] += (torch.randn_like(self.substrate.mem[0, inds.energy]) + (self.energy_offset-2)) * 0.2
        self.substrate.mem[0, inds.energy] = torch.clamp(self.substrate.mem[0, inds.energy], 0.000001, 10000)
        self.substrate.mem[0, inds.infra] = torch.clamp(self.substrate.mem[0, inds.infra], 0.000001, 10000)
        self.substrate.mem[0, inds.rot] += torch.clamp((torch.randn_like(self.substrate.mem[0, inds.rot])*0.1)//1, -1.0, 1.0)
        self.substrate.mem[0, inds.infra, self.substrate.w//3, self.substrate.h//3] += 1.0
        self.substrate.mem[0, inds.energy, self.substrate.w//3, self.substrate.h//3] += 1.0
        self.substrate.mem[0, inds.infra, 2*self.substrate.w//3, 2*self.substrate.h//3] += 1.0
        self.substrate.mem[0, inds.energy, 2*self.substrate.w//3, 2*self.substrate.h//3] += 1.0
        self.substrate.mem[0, inds.infra, self.substrate.w//3, 2*self.substrate.h//3] += 1.0
        self.substrate.mem[0, inds.energy, self.substrate.w//3, 2*self.substrate.h//3] += 1.0
        self.substrate.mem[0, inds.infra, 2*self.substrate.w//3, self.substrate.h//3] += 1.0
        self.substrate.mem[0, inds.energy, 2*self.substrate.w//3, self.substrate.h//3] += 1.0

        if self.timestep % 20 == 0:
            inds_chunk = torch.tensor(self.substrate.windex[['genome', 'infra', 'energy']], device=self.torch_device)
            kill_vals = torch.tensor([-1, 0.01, 0.01], device=self.torch_device)
            grow_vals = torch.tensor([-1, 0.01, 3.0], device=self.torch_device)
            x_kill = np.random.randint(0, self.substrate.w)
            y_kill = np.random.randint(0, self.substrate.h)
            out_mem = self.substrate.mem.clone()
            self.set_chunk(out_mem, x_kill, y_kill, 4, inds_chunk, kill_vals)
            # x_grow = np.random.randint(0, self.substrate.w)
            # y_grow = np.random.randint(0, self.substrate.h)
            # self.set_chunk(out_mem, x_grow, y_grow, 2, inds_chunk, grow_vals)
            self.substrate.mem = out_mem

    @ti.kernel
    def set_chunk(self, out_mem: ti.types.ndarray(), cx: ti.i32, cy: ti.i32, radius: ti.i32,
                  inds: ti.types.ndarray(), vals: ti.types.ndarray()):
        for off_x, off_y, k in ti.ndrange((-radius, +radius), (-radius, +radius), inds.shape[0]):
            x = (cx + off_x) % out_mem.shape[2]
            y = (cy + off_y) % out_mem.shape[3]
            out_mem[0, inds[k], x, y] *= 0.0
            out_mem[0, inds[k], x, y] += vals[k]

    def forward(self, weights, biases):
        inds = self.substrate.ti_indices[None]
        out_mem = torch.zeros_like(self.substrate.mem[0, self.act_chinds])
        apply_weights_and_biases(
            self.substrate.mem, out_mem,
            self.sense_chinds,
            weights, biases,
            self.dir_kernel, self.dir_order,
            self.substrate.ti_indices)
        self.substrate.mem[0, self.act_chinds] = out_mem
        self.apply_physics()
    

    def apply_physics(self):
        inds = self.substrate.ti_indices[None]
        # self.substrate.mem[0, inds.energy, self.substrate.w//2, self.substrate.h//2] += 10
        # sum_before = self.substrate.mem[0, inds.energy].sum() + self.substrate.mem[0, inds.infra].sum()

        activate_outputs(self.substrate)
        # sum_after = self.substrate.mem[0, inds.energy].sum() + self.substrate.mem[0, inds.infra].sum()
        # assert abs(sum_before - sum_after) < 1e-5, f"Sum before: {sum_before:.4f}, Sum after: {sum_after:.4f}"

        invest_liquidate(self.substrate)
        # sum_after = self.substrate.mem[0, inds.energy].sum() + self.substrate.mem[0, inds.infra].sum()
        # assert abs(sum_before - sum_after) < 1e-5, f"Sum before: {sum_before:.4f}, Sum after: {sum_after:.4f}"

        explore_physics(self.substrate, self.kernel, self.dir_order)
        # sum_after = self.substrate.mem[0, inds.energy].sum() + self.substrate.mem[0, inds.infra].sum()
        # assert abs(sum_before - sum_after) < 1e-5, f"Sum before: {sum_before:.4f}, Sum after: {sum_after:.4f}"

        energy_physics(self.substrate, self.kernel, max_infra=5, max_energy=4)
        # sum_after = self.substrate.mem[0, inds.energy].sum() + self.substrate.mem[0, inds.infra].sum()
        # assert abs(sum_before - sum_after) < 1e-5, f"Sum before: {sum_before:.4f}, Sum after: {sum_after:.4f}"

        self.substrate.mem[0, inds.genome] = torch.where(
            (self.substrate.mem[0, inds.infra] + self.substrate.mem[0, inds.energy]) > 0.1,
            self.substrate.mem[0, inds.genome],
            -1
        )

    def apply_radiation_mutation(self, n_spots, spot_live_radius=2, spot_dead_radius=3):
        inds = self.substrate.ti_indices[None]
        xs = torch.randint(0, self.substrate.w, (n_spots,))
        ys = torch.randint(0, self.substrate.h, (n_spots,))
        
        out_mem = self.substrate.mem.clone()
        for i in range(n_spots):
            genome_key = int(self.substrate.mem[0, inds.genome, xs[i], ys[i]].item())
            inds_genome = torch.tensor([inds.genome], device=self.torch_device)
            vals = torch.tensor([0.0], device=self.torch_device)
            x = xs[i].item()
            y = ys[i].item()
            vals[0] = -1
            self.set_chunk(out_mem, x, y, spot_dead_radius, inds_genome, vals)

            rand_genome_key = torch.randint(0, len(self.genomes), (1,))
            if genome_key < 0:
                vals[0] = rand_genome_key
                self.set_chunk(out_mem, x, y, 1, inds_genome, vals)
            else:
                if random.random() < 0.5:
                    new_genome = copy.deepcopy(self.genomes[genome_key])
                    new_genome.mutate(self.neat_config.genome_config)
                    new_genome_key = self.add_organism_get_key(new_genome)
                    vals[0] = new_genome_key
                    self.set_chunk(out_mem, x, y, spot_live_radius, inds_genome, vals)
                else: 
                    new_genome = neat.DefaultGenome(str(len(self.genomes)))
                    self.genomes[genome_key].fitness = 0.0
                    self.genomes[rand_genome_key].fitness = 0.0
                    new_genome.configure_crossover(self.genomes[genome_key], self.genomes[rand_genome_key], self.neat_config)
                    new_genome_key = self.add_organism_get_key(new_genome)
                    vals[0] = new_genome_key
                    self.set_chunk(out_mem, x, y, spot_live_radius, inds_genome, vals)
        inds_2 = torch.tensor(self.substrate.windex[['infra', 'energy']], device=self.torch_device)
        vals = torch.tensor([1.0, 1.0], device=self.torch_device)
        self.set_chunk(out_mem, x, y, spot_live_radius, inds_2, vals)
        self.substrate.mem = out_mem

    @ti.kernel
    def replace_genomes(self, mem: ti.types.ndarray(), out_mem: ti.types.ndarray(),
                        genome_transitions: ti.types.ndarray(), ti_indices: ti.template()):
        inds = ti_indices[None]
        for i, j in ti.ndrange(mem.shape[2], mem.shape[3]):
            if mem[0, inds.genome, i, j] < 0:
                out_mem[i, j] = mem[0, inds.genome, i, j]
            else:
                out_mem[i, j] = genome_transitions[int(mem[0, inds.genome, i, j])]

    def reduce_population_to_threshold(self, max_population):
        print(f"REDUCING pop to max of {max_population} from current size: {len(self.genomes)}")
        if len(self.genomes) <= max_population:
            print("Population within threshold. No reduction needed.")
            return

        inds = self.substrate.ti_indices[None]
        # Combine genome indices with their cell counts and ages
        genome_info = [(i, self.substrate.mem[0, inds.genome].eq(i).sum().item(), self.ages[i]) for i in range(len(self.genomes))]
        # Sort genomes first by cell count (descending) then by age (ascending) to prioritize younger genomes
        sorted_genomes_by_info = sorted(genome_info, key=lambda x: (-x[1], -x[2]))
        new_genomes = []
        new_ages = []
        new_combined_weights = []
        new_combined_biases = []
        genome_transitions = [None] * len(self.genomes)
        for i, (index_of_genome, _, _) in enumerate(sorted_genomes_by_info):
            if i >= max_population:
                print(f"KILLING {index_of_genome}")
                genome_transitions[index_of_genome] = -1
            else:
                new_genomes.append(self.genomes[index_of_genome])
                new_ages.append(self.ages[index_of_genome])
                new_combined_weights.append(self.combined_weights[index_of_genome])
                new_combined_biases.append(self.combined_biases[index_of_genome])
                genome_transitions[index_of_genome] = len(new_genomes) - 1
        genome_transitions = torch.tensor(genome_transitions, device=self.torch_device)
        out_mem = torch.zeros_like(self.substrate.mem[0, inds.genome])
        self.replace_genomes(self.substrate.mem, out_mem, genome_transitions, self.substrate.ti_indices)
        self.substrate.mem[0, inds.genome] = out_mem

        self.genomes = new_genomes
        self.ages = new_ages
        self.combined_weights = new_combined_weights
        self.combined_biases = new_combined_biases
        self.time_last_cull = self.timestep
        print(f"\tPop size after reduction: {len(self.genomes)}")
        if len(self.genomes) == 0:
            print("NO GENOMES LEFT. REINITIALIZING")
            self.genomes = self.init_population()
            self.init_substrate(self.genomes)


    def load_checkpoint(self, folderpath):
        config_path = os.path.join(folderpath, "..", "neat_config")
        self.neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                        config_path)
        self.load_genomes(os.path.join(folderpath, "genomes"))
        with open(os.path.join(folderpath, "ages"), 'rb') as f:
            self.ages = pickle.load(f)

        self.combined_biases = []
        self.combined_weights = []
        for genome in self.genomes:
            genome.fitness = 0.0
            net = self.create_torch_net(genome)
            self.combined_biases.append(net.biases)
            self.combined_weights.append(net.weights) 
        loaded_mem = torch.load(os.path.join(folderpath, "sub_mem"))
        self.substrate.mem = loaded_mem


    def save_genomes(self, genomes, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.genomes, f)
    
    def load_genomes(self, filename):
        with open(filename, 'rb') as f:
            genomes = pickle.load(f)
        return genomes
    
    def report_if_necessary(self, timestep):
        if timestep % 300 == 0:
            timestep_dir = os.path.join(self.checkpoint_dir, f"step_{timestep}")
            os.makedirs(timestep_dir, exist_ok=True)
            self.substrate.save_mem_to_pt(filepath = os.path.join(timestep_dir, "sub_mem"))
            self.save_genomes(self.genomes, os.path.join(timestep_dir, "genomes"))
            with open(os.path.join(timestep_dir, "ages"), 'wb') as f:
                pickle.dump(self.ages, f)

        # for i in range(len(self.genomes)):
        #     # org['genome'].fitness += self.substrate.mem[0, inds.genome].eq(i).sum().item()
        #     self.genomes[i].fitness = fitness_function(self.genomes[i], i)
        # self.reporters.start_generation(self.generation)

        # # Evaluate all genomes using the user-provided function.
        # fitness_function(list(iteritems(self.population)), self.config)

        # # Gather and report statistics.
        # best = None
        # for g in itervalues(self.population):
        #     if best is None or g.fitness > best.fitness:
        #         best = g

        # self.reporters.post_evaluate(self.config, self.population, self.species, best)

        # # # Create the next generation from the current generation.
        # # self.population = self.reproduction.reproduce(self.config, self.species,
        # #                                               self.config.pop_size, self.generation)

        # # Check for complete extinction.
        # if not self.species.species:
        #     self.reporters.complete_extinction()
        #     # TODO: re-seed env

        # # Divide the new population into species.
        # self.species.speciate(self.config, self.population, self.generation)

        # self.reporters.end_generation(self.config, self.population, self.species)

        # self.generation += 1

        # if self.config.no_fitness_termination:
        #     self.reporters.found_solution(self.config, self.generation, self.best_genome)

        # return self.best_genome
    
    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)    


    def add_organism_get_key(self, genome):
        # TODO: implement culling and memory consolidation
        self.genomes.append(genome)
        net = self.create_torch_net(genome)
        self.combined_weights.append(net.weights)
        self.combined_biases.append(net.biases)
        self.ages.append(0)
        return len(self.combined_biases) - 1


    def get_energy_offset(self, timestep, repeat_steps=100, amplitude=1, positive_scale=1, negative_scale=1):
        frequency = (2 * np.pi) / repeat_steps
        value = amplitude * np.sin(frequency * timestep)
        if value > 0:
            return value * positive_scale
        else:
            return value * negative_scale


    def create_torch_net(self, genome):
        input_coords = []
        # TODO: adjust for direcitonal kernel
        for ch in range(self.n_senses):
             input_coords.append([0, 0, self.sense_chinds[ch]])
             for offset_i in range(self.dir_order.shape[0]):
                offset_x = self.dir_kernel[self.dir_order[offset_i], 0]
                offset_y = self.dir_kernel[self.dir_order[offset_i], 1]
                input_coords.append([offset_x, offset_y, self.sense_chinds[ch]])

        output_coords = []
        for ch in range(self.n_acts):
            output_coords.append([0, 0, self.act_chinds[ch]])

        net = LinearNet.create(
            genome,
            self.neat_config,
            input_coords=input_coords,
            output_coords=output_coords,
            weight_threshold=0.0,
            weight_max=3.0,
            activation=identity_activation,
            cppn_activation=identity_activation,
            device=self.torch_device,
        )
        return net
    

    def get_genome_infra_sum(self, genome_key):
        inds = self.substrate.ti_indices[None]
        infra_sum = torch.where(self.substrate.mem[0, inds.genome] == genome_key, self.substrate.mem[0, inds.infra], 0).sum()
        return infra_sum

    # def cull_genomes(self, n_cells_thresh, age_thresh):
    #     print(f"CULLING pop of size: {len(self.genomes)}")
    #     inds = self.substrate.ti_indices[None]
    #     new_genomes = []
    #     new_ages = []
    #     new_combined_weights = []
    #     new_combined_biases = []
    #     for i in range(len(self.genomes)):
    #         where_i = self.substrate.mem[0, inds.genome].eq(i)
    #         n_cells = where_i.sum().item()
    #         print(f"\tGenome {i} is {self.ages[i]} steps old and has {n_cells} cells")
    #         if n_cells == 0 or (self.ages[i] > age_thresh and n_cells < n_cells_thresh):
    #             print(f"\t\tCulling genome {i}")
    #             self.substrate.mem[0, inds.genome] = torch.where(where_i, -1, self.substrate.mem[0, inds.genome])
    #         else:
    #             self.substrate.mem[0, inds.genome]= torch.where(where_i, len(new_genomes), self.substrate.mem[0, inds.genome])
    #             new_genomes.append(self.genomes[i])
    #             new_ages.append(self.ages[i])
    #             new_combined_weights.append(self.combined_weights[i])
    #             new_combined_biases.append(self.combined_biases[i])
    #     self.genomes = new_genomes
    #     self.ages = new_ages
    #     self.combined_weights = new_combined_weights
    #     self.combined_biases = new_combined_biases
    #     print(f"\tPop size after culling: {len(self.genomes)}")
    #     if len(self.genomes) == 0:
    #         print("Population extinct")
    #         self.init_population()
    #         self.init_substrate(self.genomes)
    #         self.timestep = 0import os
import torch
import neat
import taichi as ti
from coralai.substrate.substrate import Substrate
from coralai.evolution.space_evolver import SpaceEvolver
from coralai.visualization import Visualization

class CoralVis(Visualization):
    def __init__(self, substrate, evolver, vis_chs):
        super().__init__(substrate, vis_chs)
        self.evolver = evolver
        self.next_generation = False
        self.genome_stats = []


    def render_opt_window(self):
        inds = self.substrate.ti_indices[None]
        self.substrate.mem[0, inds.genome_inv] = torch.where(
            self.substrate.mem[0, inds.genome] < 0,
            0,
            1/(self.substrate.mem[0, inds.genome]+1),
        )
        self.canvas.set_background_color((1, 1, 1))
        opt_w = min(200 / self.img_w, self.img_w)
        opt_h = min(500 / self.img_h, self.img_h * 2)
        with self.gui.sub_window("Options", 0.05, 0.05, opt_w, opt_h) as sub_w:
            self.opt_window(sub_w)
            self.next_generation = sub_w.checkbox("Next Generation", self.next_generation)
            self.chinds[0] = sub_w.slider_int(
                f"R: {self.substrate.index_to_chname(self.chinds[0])}", 
                self.chinds[0], 0, self.substrate.mem.shape[1]-1)
            self.chinds[1] = sub_w.slider_int(
                f"G: {self.substrate.index_to_chname(self.chinds[1])}", 
                self.chinds[1], 0, self.substrate.mem.shape[1]-1)
            self.chinds[2] = sub_w.slider_int(
                f"B: {self.substrate.index_to_chname(self.chinds[2])}", 
                self.chinds[2], 0, self.substrate.mem.shape[1]-1)
            current_pos = self.window.get_cursor_pos()
            pos_x = int(current_pos[0] * self.w) % self.w
            pos_y = int(current_pos[1] * self.h) % self.h
            sub_w.text(
                f"GENOME: {self.substrate.mem[0, inds.genome, pos_x, pos_y]:.2f}\n" +
                f"Energy: {self.substrate.mem[0, inds.energy, pos_x, pos_y]:.2f}\n" +
                f"Infra: {self.substrate.mem[0, inds.infra, pos_x, pos_y]:.2f}\n"
                # f"Acts: {self.substrate.mem[0, inds.acts, pos_x, pos_y]}"
            )
            sub_w.text(f"TIMESTEP: {self.evolver.timestep}")
            tot_energy = torch.sum(self.substrate.mem[0, inds.energy])
            tot_infra = torch.sum(self.substrate.mem[0, inds.infra])
            sub_w.text(f"Total Energy+Infra: {tot_energy + tot_infra}")
            sub_w.text(f"Percent Energy: {(tot_energy / (tot_energy + tot_infra)) * 100}")
            sub_w.text(f"Energy Offset: {self.evolver.energy_offset}")
            sub_w.text(f"# Infra in Genomes ({len(self.evolver.genomes)} total):")

            if self.evolver.timestep % 20 == 0:
                self.genome_stats = []
                for i in range(len(self.evolver.genomes)):
                    n_cells = self.substrate.mem[0, inds.genome].eq(i).sum().item()
                    age = self.evolver.ages[i]
                    # n_cells = self.evolver.get_genome_infra_sum(i)
                    self.genome_stats.append((i, n_cells, age))
                self.genome_stats.sort(key=lambda x: x[1], reverse=True)
            for i, n_cells, age in self.genome_stats:
                sub_w.text(f"\tG{i}: {n_cells:.2f} cells, {age:.2f} age")


def main(config_filename, channels, shape, kernel, dir_order, sense_chs, act_chs, torch_device):
    local_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(local_dir, config_filename)
    substrate = Substrate(shape, torch.float32, torch_device, channels)
    substrate.malloc()

    inds = substrate.ti_indices[None]

    space_evolver = SpaceEvolver(config_path, substrate, kernel, dir_order, sense_chs, act_chs)
    # checkpoint_dir = os.path.join(local_dir, "history", "space_evolver_run_240310-0027_01", "step_900")
    # space_evolver.load_checkpoint(folderpath=checkpoint_dir)
    vis = CoralVis(substrate, space_evolver, ["energy", "infra", "genome_inv"])
    space_evolver.run(100000000, vis, n_rad_spots = 4, radiate_interval = 20,
                      cull_max_pop=100, cull_interval=500)
    
    # checkpoint_file = os.path.join('history', 'NEAT_240308-0052_32', 'checkpoint4')
    # p = neat.Checkpointer.restore_checkpoint(checkpoint_file)
    # p.run(eval_vis, 10)


if __name__ == "__main__":
    ti.init(ti.metal)
    torch_device = torch.device("mps")
    main(
        config_filename = "coralai/instances/coral/coral_neat.config",
        channels = {
            "energy": ti.f32,
            "infra": ti.f32,
            "acts": ti.types.struct(
                invest=ti.f32,
                liquidate=ti.f32,
                explore=ti.types.vector(n=4, dtype=ti.f32) # no, forward, left, right
            ),
            "com": ti.types.struct(
                a=ti.f32,
                b=ti.f32,
                c=ti.f32,
                d=ti.f32
            ),
            "rot": ti.f32,
            "genome": ti.f32,
            "genome_inv": ti.f32
        },
        shape = (100, 100),
        kernel = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], # ccw
        dir_order = [0, -1, 1, -2], # forward (with rot), left of rot, right of rot, behind
        sense_chs = ['energy', 'infra', 'com'],
        act_chs = ['acts', 'com'],
        torch_device = torch_device
    )


import json
import warnings
import torch
import os
import taichi as ti
import numpy as np
from ..utils.ti_struct_factory import TaichiStructFactory
from .channel import Channel
from .substrate_index import SubstrateIndex


@ti.data_oriented
class Substrate:
    # TODO: Support multi-level indexing beyond 2 levels
    # TODO: Support mixed taichi and torch tensors - which will be transferred more?
    def __init__(self, shape, torch_dtype, torch_device, channels: dict = None):
        self.w = shape[0]
        self.h = shape[1]
        self.shape = (*shape, 0) # changed in malloc
        self.mem = None
        self.windex = None
        self.torch_dtype = torch_dtype
        self.torch_device = torch_device
        self.channels = {}
        if channels is not None:
            self.add_channels(channels)
        self.ti_ind_builder = TaichiStructFactory()
        self.ti_lims_builder = TaichiStructFactory()
        self.ti_indices = -1
        self.ti_lims = -1

    def save_metadata_to_json(self, filepath):
        config = {
            "shape": self.shape,
            "windex": self.windex.index_tree
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)

    def save_mem_to_pt(self, filepath):
        # Saves channels, channel metadata, dims, dtypes, etc
        torch.save(self.mem, filepath)

    def index_to_chname(self, index):
        return self.windex.index_to_chname(index)


    def add_channel(self, chid: str, ti_dtype=ti.f32, **kwargs):
        if self.mem is not None:
            raise ValueError(
                f"World: When adding channel {chid}: Cannot add channel after world memory is allocated (yet)."
            )
        self.channels[chid] = Channel(chid, self, ti_dtype=ti_dtype, **kwargs)


    def add_channels(self, channels: dict):
        if self.mem is not None:
            raise ValueError(
                f"World: When adding channels {channels}: Cannot add channels after world memory is allocated (yet)."
            )
        for chid in channels.keys():
            ch = channels[chid]
            if isinstance(ch, dict):
                self.add_channel(chid, **ch)
            else:
                self.add_channel(chid, ch)


    def check_ch_shape(self, shape):
        lshape = len(shape)
        if lshape > 3 or lshape < 2:
            raise ValueError(
                f"World: Channel shape must be 2 or 3 dimensional. Got shape: {shape}"
            )
        if shape[:2] != self.shape[:2]:
            print(shape[:2], self.shape[:2])
            raise ValueError(
                f"World: Channel shape must be (w, h, ...) where w and h are the world dimensions: {self.shape}. Got shape: {shape}"
            )
        if lshape == 2:
            return 1
        else:
            return shape[2]


    def stat(self, key):
        # Prints useful metrics about the channel(s) and contents
        minval = self[key].min()
        maxval = self[key].max()
        meanval = self[key].mean()
        stdval = self[key].std()
        shape = self[key].shape
        print(
            f"{key} stats:\n\tShape: {shape}\n\tMin: {minval}\n\tMax: {maxval}\n\tMean: {meanval}\n\tStd: {stdval}"
        )


    def _transfer_to_mem(self, mem, tensor_dict, index_tree, channel_dict):
        for chid, chindices in index_tree.items():
            if "subchannels" in chindices:
                for subchid, subchtree in chindices["subchannels"].items():
                    if tensor_dict[chid][subchid].dtype != self.torch_dtype:
                        warnings.warn(
                            f"\033[93mWorld: Casting {chid} of dtype: {tensor_dict[chid].dtype} to world dtype: {self.torch_dtype}\033[0m",
                            stacklevel=3,
                        )
                    if len(tensor_dict[chid][subchid].shape) == 2:
                        tensor_dict[chid][subchid] = tensor_dict[chid][
                            subchid
                        ].unsqueeze(2)
                    mem[:, :, subchtree["indices"]] = tensor_dict[chid][subchid].type(
                        self.torch_dtype
                    )
                    channel_dict[chid].add_subchannel(
                        subchid, ti_dtype=channel_dict[chid].ti_dtype
                    )
                    channel_dict[chid][subchid].link_to_mem(subchtree["indices"], mem)
                channel_dict[chid].link_to_mem(chindices["indices"], mem)
            else:
                if tensor_dict[chid].dtype != self.torch_dtype:
                    warnings.warn(
                        f"\033[93mWorld: Casting {chid} of dtype: {tensor_dict[chid].dtype} to world dtype: {self.torch_dtype}\033[0m",
                        stacklevel=3,
                    )
                if len(tensor_dict[chid].shape) == 2:
                    tensor_dict[chid] = tensor_dict[chid].unsqueeze(2)
                mem[:, :, chindices["indices"]] = tensor_dict[chid].type(
                    self.torch_dtype
                )
                channel_dict[chid].link_to_mem(chindices["indices"], mem)
        return mem, channel_dict


    def add_ti_inds(self, key, inds):
        if len(inds) == 1:
            self.ti_ind_builder.add_i(key, inds[0])
        else:
            self.ti_ind_builder.add_nparr_int(key, np.array(inds))


    def _index_subchannels(self, subchdict, start_index, parent_chid):
        end_index = start_index
        subch_tree = {}
        for subchid, subch in subchdict.items():
            if not isinstance(subch, torch.Tensor):
                raise ValueError(
                    f"World: Channel grouping only supported up to a depth of 2. Subchannel {subchid} of channel {parent_chid} must be a torch.Tensor. Got type: {type(subch)}"
                )
            subch_depth = self.check_ch_shape(subch.shape)
            indices = [i for i in range(end_index, end_index + subch_depth)]
            self.add_ti_inds(parent_chid + "_" + subchid, indices)
            self.ti_lims_builder.add_nparr_float(
                parent_chid + "_" + subchid, self.channels[parent_chid].lims
            )
            subch_tree[subchid] = {
                "indices": indices,
            }
            end_index += subch_depth
        return subch_tree, end_index - start_index


    def malloc(self):
        if self.mem is not None:
            raise ValueError("World: Cannot allocate world memory twice.")
        celltype = ti.types.struct(
            **{chid: self.channels[chid].ti_dtype for chid in self.channels.keys()}
        )
        tensor_dict = celltype.field(shape=self.shape[:2]).to_torch(
            device=self.torch_device
        )

        index_tree = {}
        endlayer_pointer = self.shape[2]
        for chid, chdata in tensor_dict.items():
            if isinstance(chdata, torch.Tensor):
                ch_depth = self.check_ch_shape(chdata.shape)
                indices = [
                    i for i in range(endlayer_pointer, endlayer_pointer + ch_depth)
                ]
                self.add_ti_inds(chid, indices)
                self.ti_lims_builder.add_nparr_float(chid, self.channels[chid].lims)
                index_tree[chid] = {"indices": indices}
                endlayer_pointer += ch_depth
            elif isinstance(chdata, dict):
                subch_tree, total_depth = self._index_subchannels(
                    chdata, endlayer_pointer, chid
                )
                indices = [
                    i for i in range(endlayer_pointer, endlayer_pointer + total_depth)
                ]
                self.add_ti_inds(chid, indices)
                self.ti_lims_builder.add_nparr_float(chid, self.channels[chid].lims)
                index_tree[chid] = {
                    "subchannels": subch_tree,
                    "indices": indices,
                }
                endlayer_pointer += total_depth

        self.shape = (*self.shape[:2], endlayer_pointer)
        mem = torch.zeros(self.shape, dtype=self.torch_dtype, device=self.torch_device)
        self.mem, self.channels = self._transfer_to_mem(
            mem, tensor_dict, index_tree, self.channels
        )
        self.windex = SubstrateIndex(index_tree)
        self.ti_indices = self.ti_ind_builder.build()
        self.ti_lims = self.ti_lims_builder.build()
        self.mem = self.mem.permute(2, 0, 1).unsqueeze(0).contiguous()
        self.shape = self.mem.shape


    def __getitem__(self, key):
        if self.mem is None:
            raise ValueError(f"World: World memory not allocated yet, cannot get {key}")
        val = self.mem[:, self.windex[key], :, :]
        return val
    
    def __setitem__(self, key, value):
        if self.mem is None:
            raise ValueError(f"World: World memory not allocated yet, cannot set {key}")
        raise NotImplementedError("World: Setting world values not implemented yet. (Just manipulate memory directly)")


    def get_inds_tivec(self, key):
        indices = self.windex[key]
        itype = ti.types.vector(n=len(indices), dtype=ti.i32)
        return itype(indices)


    def get_lims_timat(self, key):
        lims = []
        if isinstance(key, str):
            key = [key]
        if isinstance(key, tuple):
            key = [key[0]]
        for k in key:
            if isinstance(k, tuple):
                lims.append(self.channels[k[0]].lims)
            else:
                lims.append(self.channels[k].lims)
        if len(lims) == 1:
            lims = lims[0]
        lims = np.array(lims, dtype=np.float32)
        ltype = ti.types.matrix(lims.shape[0], lims.shape[1], dtype=ti.f32)
        return ltype(lims)


import time
import torch
import taichi as ti
from .substrate import Substrate


@ti.data_oriented
class Visualization:
    def __init__(self,
                 substrate: Substrate,
                 chids: list = None,
                 chinds: list = None,
                 name: str = None,
                 scale: int = None,):
        self.substrate = substrate
        self.w = substrate.w
        self.h = substrate.h
        self.chids = chids
        self.scale = 1 if scale is None else scale
        chinds = substrate.get_inds_tivec(chids)
        self.chinds = torch.tensor(list(chinds), device = substrate.torch_device)
        # self.name = f"Vis: {[self.substrate.index_to_chname(chindices[i]) for i in range(len(chindices))]}" if name is None else name
        self.name = "Vis"

        if scale is None:
            max_dim = max(self.substrate.w, self.substrate.h)
            desired_max_dim = 800
            scale = desired_max_dim // max_dim
            
        self.scale = scale
        self.img_w = self.substrate.w * scale
        self.img_h = self.substrate.h * scale
        self.n_channels = len(chinds)
        self.image = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.img_w, self.img_h))

        self.window = ti.ui.Window(
            f"{self.name}", (self.img_w, self.img_h), fps_limit=200, vsync=True
        )
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()
        self.paused = False
        self.brush_radius = 4
        self.mutating = False
        self.perturbation_strength = 0.1
        self.drawing = False
        self.prev_time = time.time()
        self.prev_pos = self.window.get_cursor_pos()
        self.channel_to_paint = 0
        self.val_to_paint = 0.1

    def set_channels(self, chindices):
        self.chinds = chindices

    @ti.kernel
    def add_val_to_loc(self,
            val: ti.f32,
            pos_x: ti.f32,
            pos_y: ti.f32,
            radius: ti.i32,
            channel_to_paint: ti.i32,
            mem: ti.types.ndarray()
        ):
        ind_x = int(pos_x * self.w)
        ind_y = int(pos_y * self.h)
        offset = int(pos_x) * 3
        for i, j in ti.ndrange((-radius, radius), (-radius, radius)):
            if (i**2) + j**2 < radius**2:
                mem[0, channel_to_paint, (i + ind_x) % self.w, (j + ind_y) % self.h] += val


    @ti.kernel
    def write_to_renderer(self, mem: ti.types.ndarray(), max_vals: ti.types.ndarray(), chinds: ti.types.ndarray()):
        for i, j in self.image:
            xind = (i//self.scale) % self.w
            yind = (j//self.scale) % self.h
            for k in ti.static(range(3)):
                chid = chinds[k]
                self.image[i, j][k] = mem[0, chid, xind, yind] / max_vals[k]

    def opt_window(self, sub_w):
        self.channel_to_paint = sub_w.slider_int("Paint channel: " +
                                                 f"{self.substrate.index_to_chname(self.channel_to_paint)}",
                                                 self.channel_to_paint, 0, 10)
        self.val_to_paint = sub_w.slider_float("Value to Paint", self.val_to_paint, -1.0, 1.0)
        self.val_to_paint = round(self.val_to_paint * 10) / 10
        self.brush_radius = sub_w.slider_int("Brush Radius", self.brush_radius, 1, 200)
        self.paused = sub_w.checkbox("Pause", self.paused)
        self.mutating = sub_w.checkbox("Perturb Weights", self.mutating)
        self.perturbation_strength = sub_w.slider_float("Perturbation Strength", self.perturbation_strength, 0.0, 5.0)

    def render_opt_window(self):
        self.canvas.set_background_color((1, 1, 1))
        opt_w = min(480 / self.img_w, self.img_w)
        opt_h = min(240 / self.img_h, self.img_h * 2)
        with self.gui.sub_window("Options", 0.05, 0.05, opt_w, opt_h) as sub_w:
            self.opt_window(sub_w)


    def check_events(self):
        for e in self.window.get_events(ti.ui.PRESS):
            if e.key in  [ti.ui.ESCAPE]:
                exit()
            if e.key == ti.ui.LMB and self.window.is_pressed(ti.ui.SHIFT):
                self.drawing = True
            elif e.key == ti.ui.SPACE:
                self.substrate.mem *= 0.0
        for e in self.window.get_events(ti.ui.RELEASE):
            if e.key == ti.ui.LMB:
                self.drawing = False


    def update(self):
        current_time = time.time()
        current_pos = self.window.get_cursor_pos()
        if not self.paused:
            self.check_events()
            if self.drawing and ((current_time - self.prev_time) > 0.1): # or (current_pos != self.prev_pos)):
                self.add_val_to_loc(self.val_to_paint, current_pos[0], current_pos[1], self.brush_radius, self.channel_to_paint, self.substrate.mem)
                self.prev_time = current_time  # Update the time of the last action
                self.prev_pos = current_pos

            max_vals = torch.tensor([self.substrate.mem[0, ch].max() for ch in self.chinds])
            self.write_to_renderer(self.substrate.mem, max_vals, self.chinds)
        self.render_opt_window()
        self.canvas.set_image(self.image)
        self.window.show()


import torch
import taichi as ti
import torch.nn as nn

from ...substrate.nn_lib import ch_norm


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
                ind = int((rot+dir_order[offset_m]) % dir_kernel.shape[0])
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


