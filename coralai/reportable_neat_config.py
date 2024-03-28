from typing import TypeVar, Type

import pickle
import shutil
import neat
import os
from .reportable import Reportable

T = TypeVar('T', bound='ReportableNeatConfig')


class ReportableNeatConfig(Reportable, neat.Config):
    neat_config: neat.Config

    report_prefix: str = "neat_config_report"
    config_path: str = None
    has_been_reported: bool = False
    compatibility_disjoint_coefficient: float = 1.0
    compatibility_weight_coefficient: float = 0.5

    def __init__(self, config_path: str = None):
        if config_path is None:
            current_dir = os.path.dirname(__file__)
            config_path = os.path.join(current_dir, 'default_neat.config')

        super().__init__(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
        self.config_path = config_path

    def save_snapshot(self, snapshot_dir: str, report_suffix: str = None) -> str:
        """Implements Reportable.save_snapshot. Copies config file to snapshot_dir and saves this object as a pkl."""
        snap_name = self.get_snapshot_name(report_suffix)
        config_snap_dir = os.path.join(snapshot_dir, snap_name)
        os.makedirs(config_snap_dir, exist_ok=True)  # Ensure the directory exists
        og_config_fname = os.path.basename(self.config_path)
        og_config_newpath = os.path.join(config_snap_dir, f"{snap_name}_og_config_{og_config_fname}")
        shutil.copyfile(self.config_path, og_config_newpath)
        with open(os.path.join(config_snap_dir, f"{snap_name}_config.pkl"), 'wb') as file:
            pickle.dump(self, file)
        self.has_been_reported = True
        return config_snap_dir
    
    @classmethod
    def load_snapshot(cls: Type[T], snapshot_dir: str, report_suffix: str = None) -> T:
        """Implements Reportable.load_snapshot. Loads the object directly from a pkl file."""
        snap_name = cls.get_snapshot_name(report_suffix)
        config_snap_dir = os.path.join(snapshot_dir, snap_name)
        with open(os.path.join(config_snap_dir, f"{snap_name}_config.pkl"), 'rb') as file:
            return pickle.load(file)

