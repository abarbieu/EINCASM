import os
import torch
import torch.nn as nn
import taichi as ti
from typing import TypeVar, Type
from dataclasses import dataclass
from typing import List
from coralai.population import Population

from coralai.reportable import Reportable
from coralai.reportable_neat_config import ReportableNeatConfig
from coralai.substrate.substrate import Substrate
from coralai.coralai_config import CoralaiConfig


TCORALAI = TypeVar('TCORALAI', bound="Coralai")
class Coralai(Reportable):
    config: CoralaiConfig
    neat_config: ReportableNeatConfig
    substrate: Substrate
    population: Population

    report_prefix: str = "coralai_snap"

    def __init__(self, config: CoralaiConfig, neat_config: ReportableNeatConfig, substrate: Substrate, population: Population):
        self.config = config
        self.neat_config = neat_config
        self.substrate = substrate
        self.population = population

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
