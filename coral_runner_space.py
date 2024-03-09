import os
import torch
import neat
import taichi as ti
from coralai.substrate.substrate import Substrate
from coralai.evolution.space_evolver import SpaceEvolver
from coralai.substrate.visualization import Visualization

class CoralVis(Visualization):
    def __init__(self, substrate, evolver, vis_chs):
        super().__init__(substrate, vis_chs)
        self.evolver = evolver
        self.next_generation = False
        self.genome_stats = []


    def render_opt_window(self):
        inds = self.substrate.ti_indices[None]
        self.canvas.set_background_color((1, 1, 1))
        opt_w = min(200 / self.img_w, self.img_w)
        opt_h = min(500 / self.img_h, self.img_h * 2)
        with self.gui.sub_window("Options", 0.05, 0.05, opt_w, opt_h) as sub_w:
            # self.opt_window(sub_w)
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
    checkpoint_dir = os.path.join(local_dir, "history", "space_evolver_run_240309-0004_34", "step_500")
    space_evolver.load_checkpoint(folderpath=checkpoint_dir)
    vis = CoralVis(substrate, space_evolver, ["energy", "infra", "genome"])
    space_evolver.run(100000000, vis, n_rad_spots = 5, radiate_interval = 50,
                      cull_max_pop=100, cull_interval=50)
    
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
        },
        shape = (400, 400),
        kernel = [[0, 0], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]], # ccw
        dir_order = [0, -1, 1], # forward (with rot), left of rot, right of rot
        sense_chs = ['energy', 'infra', 'com'],
        act_chs = ['acts', 'com'],
        torch_device = torch_device
    )
