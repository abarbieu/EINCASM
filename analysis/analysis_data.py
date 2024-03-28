import os
import torch
from dataclasses import dataclass
from typing import List

from coralai.population import Population
from coralai.substrate.substrate import Substrate
from coralai.reportable_neat_config import ReportableNeatConfig


def fetch_neat_config(run_dir):
    # stored as neat_config and is a .config object (but names without extension :( )
    neat_config_path = os.path.join(run_dir, "neat_config")
    return ReportableNeatConfig(neat_config_path)


@dataclass
class RunData:
    run_path: str
    run_name: str
    substrate: Substrate
    steps: List[str]
    step_path: str
    torch_device: torch.DeviceObjType
    neat_config: ReportableNeatConfig
    curr_step_index: int
    curr_step_number: int
    population: Population

    def __init__(self, run_path: str, torch_device: torch.DeviceObjType):
        self.steps = sorted([step for step in os.listdir(run_path) if step.startswith("step_")], key=lambda x: int(x.split("_")[1]))
        self.substrate = None
        self.run_path=run_path
        self.run_name=os.path.basename(run_path)
        self.curr_step_index=0
        self.curr_step_number=None
        self.step_path=None
        self.torch_device=torch_device
        self.neat_config = fetch_neat_config(run_path)
        self.load_step()

    def load_step(self):
        self.curr_step_index = self.curr_step_index % len(self.steps)
        self.step_path = os.path.join(self.run_path, self.steps[self.curr_step_index])
        self.step_number = int(self.steps[self.curr_step_index].split("_")[1])
        self.substrate = Substrate.load_snapshot_old(self.run_path, self.step_path, self.torch_device, old_substrate = self.substrate)
        self.population = Population.load_snapshot_old(self.step_path, self.neat_config)


    def next_step(self):
        self.curr_step_index += 1
        self.load_step()

    def prev_step(self):
        self.curr_step_index -= 1
        self.load_step()


@dataclass
class HistoryData:
    hist_path: str
    run_names: List[str]
    curr_run_index: int
    curr_run_path: str
    curr_run_data: RunData
    torch_device: torch.DeviceObjType

    def __init__(self, hist_path, torch_device):
        self.hist_path = hist_path
        self.run_names = [run_name for run_name in os.listdir(hist_path)]
        self.curr_run_index = 0
        self.curr_run_path = None
        self.curr_run_data = None
        self.torch_device = torch_device

        self.load_run()

    def load_run(self):
        self.curr_run_index = self.curr_run_index % len(self.run_names)
        self.curr_run_path = os.path.join(self.hist_path, self.run_names[self.curr_run_index])
        self.curr_run_data = RunData(self.curr_run_path, self.torch_device)

    def next_run(self):
        self.curr_run_index += 1
        self.load_run()

    def prev_run(self):
        self.curr_run_index -= 1
        self.load_run()

    def goto_run_name(self, run_name):
        self.curr_run_index = self.run_names.index(run_name)
        self.load_run()
