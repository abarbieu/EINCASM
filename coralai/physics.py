import torch
import taichi as ti
from typing import Callable, Type, TypeVar
from coralai.population import Population
from coralai.reportable import Reportable
from coralai.substrate.substrate import Substrate
from coralai.coralai_config import CoralaiConfig


    

T = TypeVar('T', bound='Physics')

class Physics(Reportable):
    timestep: int = 0
    
    def apply_physics(self, substrate: Substrate, config: CoralaiConfig, timestep: int):
        pass
    
    def save_snapshot(self, snapshot_dir: str, report_suffix: str = None) -> str:
        """Save copy of physics function, save parameters separately (if time dependent)"""
        pass
    
    @classmethod
    def load_snapshot(cls: Type[T], snapshot_dir: str, torch_device: torch.DeviceObjType, report_suffix: str = None) -> T:
        pass