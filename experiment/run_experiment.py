import os
import pickle
import torch
import taichi as ti
from typing import TypeVar, Type
from dataclasses import dataclass
from typing import Dict, Tuple, List
from coralai.population import Population

from coralai.reportable import Reportable
from coralai.reportable_neat_config import ReportableNeatConfig
from coralai.substrate.substrate import Substrate


@dataclass
class CoralaiConfig(Reportable):
    kernel: torch.Tensor
    sense_chs: List[str]
    act_chs: List[str]
    torch_device: torch.device

    report_prefix: str = "coralai_config_snap"
    
TCORALAI = TypeVar('TCORALAI', bound="Coralai")
class Coralai(Reportable):
    config: CoralaiConfig
    neat_config: ReportableNeatConfig
    substrate: Substrate
    population: Population

    def save_snapshot(self, snapshot_dir: str, report_suffix: str = None) -> str:
        """Implements Reportable.save_snapshot. Reports all components"""
        snap_name = self.get_snapshot_name(report_suffix)
        snap_path = os.path.join(snapshot_dir, snap_name)
        self.config.save_snapshot(snapshot_dir, report_suffix)
        self.neat_config.save_snapshot(snapshot_dir, report_suffix)
        self.substrate.save_snapshot(snapshot_dir, report_suffix)
        self.population.save_snapshot(snapshot_dir, report_suffix)
        return snap_path

    @classmethod
    def load_snapshot(cls: Type[TCORALAI], snapshot_path: str) -> TCORALAI:
        """Implements Reportable.load_snapshot. Loads all reportable components"""
        config = CoralaiConfig.load_snapshot(snapshot_path)
        neat_config = ReportableNeatConfig.load_snapshot(snapshot_path)
        substrate = Substrate.load_snapshot(snapshot_path)
        population = Population.load_snapshot(snapshot_path)
        return cls(config, neat_config, substrate, population)

    
if __name__ == "__main__":
    ti.init(ti.metal)
    torch_device = torch.device("mps")

    # config_filename = "coralai//coral/coral_neat.config"
    # local_dir = os.path.dirname(os.path.abspath(__file__))
    # config_path = os.path.join(local_dir, config_filename)
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
