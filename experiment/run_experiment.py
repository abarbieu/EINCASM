import torch
import taichi as ti

from coralai.reportable_neat_config import ReportableNeatConfig
from coralai.substrate.substrate import Substrate
from coralai.population import Population
from coralai.coralai import Coralai, CoralaiConfig
    
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

    dir_order = [0, -1, 1, -2], # forward (with rot), left of rot, right of rot, behind
