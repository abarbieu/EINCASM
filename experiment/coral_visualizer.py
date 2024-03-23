import torch
from coralai.visualization import Visualization

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