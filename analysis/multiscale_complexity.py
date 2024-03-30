import math
import os
import time
import torch
import numpy as np
import taichi as ti
from PIL import Image
from coralai.substrate.substrate import Substrate
from coralai.visualization import compose_visualization, VisualizationData


class MSCVisData(VisualizationData):
    RG_flow: torch.Tensor # [n_renorm_groups, n_channels, w, h]
    show_diff_original: bool
    show_diff_prev_RG: bool
    sample_exp: int = 0
    prev_sample_exp: int = 0
    sample_size: int
    max_sample_size: int = 1024
    max_sample_exp: float

    def __init__(self, substrate: Substrate, chids: list = None, window_w=800, name="Default Visualization"):
        super().__init__(substrate, chids, window_w, "Downsampler")

        self.sample_size = 2 ** self.sample_exp
        self.max_sample_size = min(substrate_to_display.w, substrate_to_display.h)
        self.max_sample_exp = math.log(self.max_sample_size, 2)
        self.show_diff_original = False
        self.show_diff_prev_RG = False

def add_sampling_controls(vis_data: MSCVisData, sub_window):
    """Adds slider to subwindow to sample substrate at different scales"""
    vis_data.sample_exp = sub_window.slider_int(
        f"{int(2 ** vis_data.sample_exp)} Samples",
        vis_data.sample_exp, 0, int(vis_data.max_sample_exp))
    vis_data.sample_size = int(2 ** vis_data.sample_exp)

def add_diff_button(vis_data: MSCVisData, sub_window):
    vis_data.show_diff_original = sub_window.checkbox("Show Diffmap", vis_data.show_diff_original)
    vis_data.show_diff_prev_RG = sub_window.checkbox("Show Diffmap", vis_data.show_diff_prev_RG)
        
def load_image_to_substrate(image_path: str) -> Substrate:
    """Loads an image into a Substrate object with RGB channels, normalized to 0-1 range"""
    image = Image.open(image_path)
    image = image.convert("RGB")
    np_image = np.array(image) / 255.0  # Normalize RGB values to 0-1
    torch_image = torch.from_numpy(np_image).permute(2, 0, 1).unsqueeze(0).float()  # Convert to torch tensor and adjust dimensions
    channels = {"R": ti.f32, "G": ti.f32, "B": ti.f32} 
    substrate = Substrate((np_image.shape[1], np_image.shape[0]), torch.float32, torch.device("mps"), channels)
    substrate.malloc()
    substrate.mem = torch_image.contiguous()

    return substrate



def infer_rg_steps(pattern: torch.Tensor, scaling_factor: int = 2):
    """Infers the number of renormalization steps to apply to a patern given its dimensions

    A pattern can only be downsampled so many times before it is meaningless.
    """
    pass

def gen_rg_stack(pattern: torch.Tensor, n_rg_steps: int, scaling_factor: int = 2) -> torch.Tensor:
    """
    Generates a stack of renormalized versions of the pattern, where each step is a downsampled version of the previous step.
    Channels are kepts separate.

    Args:
        pattern: torch.Tensor, expects shapes (batch_size, n_chs, w, h)
        n_rg_steps: The number of renormalization steps, how big is the stack
        scaling_factor: The factor by which to downsample the pattern, 2 combines 2x2 metapixels into 1 recursively.

    Returns:
        rg_flow: torch.Tensor, shape (n_rg_steps, batch_size, n_chs, w, h)
    """
    rg_flow = torch.zeros((n_rg_steps, pattern.shape[0], pattern.shape[1], pattern.shape[2], pattern.shape[3]))
    rg_flow[0] = pattern
    for step in range(1, n_rg_steps):
        renormalize(rg_flow[step-1], rg_flow[step], scaling_factor)
        pattern = rg_flow[step]
    return rg_flow

def calc_overlap(pattern1: torch.Tensor, pattern2: torch.Tensor) -> float:
    """Calculates the overlap metric of two patterns. This is the partial RG flow overlap"""
    return torch.mean(calc_pixel_overlap(pattern1, pattern2))

def calc_pixel_overlap(pattern1: torch.Tensor, pattern2: torch.Tensor) -> torch.Tensor:
    """Returns new tensor representing pixel-wise difference between two patterns."""
    return torch.abs((pattern1 * pattern2) - (pattern1 * pattern1 + pattern2 * pattern2)/2.0)


@ti.kernel
def renormalize(mem: ti.types.ndarray(), out_mem: ti.types.ndarray(), sample_size: ti.i32):
    """Renormalizes all channels in memory by averaging every `sample_size * sample_size` chunk of cells into a single one.
    No overlap (each metapixel is uniquely derived from a set of subpixels).
    Edge metapixels, if cut off, are treated as smaller pixels
    Everything is rescaled to the original size
    """
    for batch, chi, sub_i, sub_j in ti.ndrange(mem.shape[0], mem.shape[1],
                                               (mem.shape[2] + sample_size - 1) // sample_size,
                                               (mem.shape[3] + sample_size - 1) // sample_size):
        sum = 0.0
        num_pixels = 0
        tl_i = sub_i*sample_size
        tl_j = sub_j*sample_size
        for super_i, super_j in ti.ndrange((tl_i, min(tl_i + sample_size, mem.shape[2])),
                                           (tl_j, min(tl_j + sample_size, mem.shape[3]))):
            sum += mem[0, chi, super_i, super_j]
            num_pixels += 1
        avg_value = sum / num_pixels
        for super_i, super_j in ti.ndrange((tl_i, min(tl_i + sample_size, mem.shape[2])),
                                           (tl_j, min(tl_j + sample_size, mem.shape[3]))):
            out_mem[0, chi, super_i, super_j] = avg_value

def display_substrate(og_substrate: Substrate, substrate_to_display: Substrate):
    """Displays the substrate using visualization tools"""
    vis_data = MSCVisData(og_substrate, substrate_to_display, chids=["R", "G", "B"], window_w = max(og_substrate.w, og_substrate.h))
    update_func = compose_visualization(vis_data, subwindow_adders=[add_sampling_controls, add_diff_button])
    diffmap = torch.zeros_like(vis_data.original_substrate.mem)
    start_time = time.time()
    repeat_after_s = 2.0
    while not vis_data.escaped and vis_data.window.running:
        time_offset = ((time.time() - start_time) % repeat_after_s) / repeat_after_s
        # vis_data.sample_exp = ((math.sin(time_offset * math.pi * 2)+1)/2.0) * vis_data.max_sample_exp
        update_func()
        renormalize(vis_data.original_substrate.mem, vis_data.substrate.mem, vis_data.sample_size)
        if vis_data.show_diffmap:
            calc_diffmap(vis_data.original_substrate.mem, vis_data.substrate.mem, diffmap)
            vis_data.substrate.mem = diffmap


if __name__ == "__main__":
    ti.init(ti.metal)
    msc_path = os.path.join(os.path.dirname(__file__), 'msc')
    images = [
        "duff_resized.jpg",
        "ferns_resized.jpg",
        "grass_resized.jpg",
        "lake_resized.jpg",
        "lichen_resized.jpg",
        "pebbles_resized.jpg",
        "redwood_resized.jpg",
        "fractals.png",
        "fractal_1.jpeg",
        "slime1.png"
    ]
    image_name = images[-1]
    og_substrate = load_image_to_substrate(os.path.join(msc_path, image_name))

    pattern = og_substrate.mem
    rg_flow = gen_rg_stack(pattern, 5)
    pass