import torch
import taichi as ti

from coralai.reportable_neat_config import ReportableNeatConfig
from coralai.substrate.substrate import Substrate
from coralai.population import Population
from coralai.physics import Physics
from coralai.coralai import Coralai, CoralaiConfig
    

def init_substrate(substrate, genomes):
    inds = substrate.ti_indices[None]
    substrate.mem[0, inds.genome] = torch.where(
        torch.rand_like(substrate.mem[0, inds.genome]) > 0.2 ,
        torch.randint_like(substrate.mem[0, inds.genome], 0, len(genomes)),
        -1
    )
    substrate.mem[0, inds.genome, substrate.w//8:-substrate.w//8, substrate.h//8:-substrate.h//8] = -1
    substrate.mem[0, inds.energy, ...] = 0.5
    substrate.mem[0, inds.infra, ...] = 0.5
    # substrate.mem[0, inds.rot] = torch.randint_like(substrate.mem[0, inds.rot], 0, dir_kernel.shape[0])

def step(substrate: Substrate, population: Population, physics: Physics, config: CoralaiConfig):
    inds = substrate.ti_indices[None]
    physics.apply_population(substrate, population, config)
    physics.apply_physics(substrate, config)
    physics.timestep += 1

def run_experiment():
    pass

if __name__ == "__main__":
    ti.init(ti.metal)
    torch_device = torch.device("mps")

    neat_config = ReportableNeatConfig()

    substrate = Substrate(shape=(100, 100), torch_dtype=torch.float32, torch_device=torch_device,
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
                            })
    
    coralai_config = CoralaiConfig(
        kernel = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], device=torch_device), # ccw
        sense_chs = ['energy', 'infra', 'com'],
        act_chs = ['acts', 'com'],
        torch_device = torch_device
    )
    population = Population.gen_random_pop(neat_config, 100)

    dir_order = [0, -1, 1, -2], # forward (with rot), left of rot, right of rot, behind
