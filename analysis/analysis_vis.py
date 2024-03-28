import os
import json
import torch
from dataclasses import dataclass
from coralai.substrate.substrate import Substrate
from coralai.visualization import VisualizationData, add_channel_controls, compose_visualization

@dataclass
class AnalysisVisData(VisualizationData):
    run_dir: str
    run_name: str
    step_number: int
    next_step_clicked: bool
    prev_step_clicked: bool

    def __init__(self, substrate: Substrate, chids: list = None, window_w=800, name="Analysis Visualization"):
        super().__init__(substrate, chids, window_w, name)
        self.run_dir = None
        self.run_name = None
        self.step_number = None
        self.next_step_clicked = False
        self.prev_step_clicked = False

    def set_run_dir(self, run_dir):
        self.run_dir = run_dir
        self.run_name = os.path.basename(run_dir)


def add_analysis_controls(analysis_vis_data: AnalysisVisData, sub_window):
    sub_window.text(f"Experiment Run: {analysis_vis_data.run_name}")
    sub_window.text(f"Step: {analysis_vis_data.step_number}")
    analysis_vis_data.prev_step_clicked = sub_window.button("prev step")
    analysis_vis_data.next_step_clicked = sub_window.button("next step")
    return analysis_vis_data


def compose_analysis_vis(analysis_vis_data: AnalysisVisData):
    return compose_visualization(analysis_vis_data, [add_channel_controls, add_analysis_controls], [])
