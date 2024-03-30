import os
import torch
import numpy as np
import taichi as ti
from PIL import Image
from coralai.substrate.substrate import Substrate
from coralai.visualization import compose_visualization, VisualizationData
from matplotlib.widgets import Slider, Button
import matplotlib.pyplot as plt
from PIL import Image

@ti.kernel
def avg_neighs(mem: ti.types.ndarray(), out_mem: ti.types.ndarray()):
    # Relies on 2^n dimensions
    for batch, ch, i, j in ti.ndrange(mem.shape[0], mem.shape[1], mem.shape[2], mem.shape[3]):
        out_mem[batch, ch, i//2, j//2] += mem[batch, ch, i, j] / 4.0

def renormalize_patterns(patterns: torch.Tensor):
    out_mem = torch.zeros((patterns.shape[0], patterns.shape[1], patterns.shape[2]//2, patterns.shape[3]//2), device=patterns.device)
    avg_neighs(patterns, out_mem)
    return out_mem

def calc_rg_flow_ragged(patterns: torch.Tensor, renorm_steps: int):
    rg_flow = []
    rg_flow.append(patterns)
    for _ in range(renorm_steps):
        rg_flow.append(renormalize_patterns(rg_flow[-1]))
    return rg_flow


@ti.kernel
def upscale(pattern_coarse: ti.types.ndarray(), out_mem_fine: ti.types.ndarray()):
    for batch, ch, i, j in ti.ndrange(out_mem_fine.shape[0], out_mem_fine.shape[1], out_mem_fine.shape[2], out_mem_fine.shape[3]):
        out_mem_fine[batch, ch, i, j] = pattern_coarse[batch, ch, i//2, j//2]

def match_sizes(patterns_coarse: torch.Tensor, patterns_fine: torch.Tensor):
    coarse_upscaled = torch.zeros_like(patterns_fine)
    upscale(patterns_coarse, coarse_upscaled)
    return coarse_upscaled

def calc_pixel_complexities(patterns_coarse: torch.Tensor, patterns_fine: torch.Tensor):
    patterns_coarse_upscaled = match_sizes(patterns_coarse, patterns_fine)
    return torch.abs((patterns_coarse_upscaled * patterns_fine) -
                      ((patterns_coarse_upscaled * patterns_coarse_upscaled) +
                       (patterns_fine * patterns_fine))/2.0)
                     
def calc_pixel_complexities_ragged(rg_flow: list):
    pixel_complexities = []
    for step in range(1, len(rg_flow)):
        pixel_complexities.append(calc_pixel_complexities(rg_flow[step], rg_flow[step-1]))
    return pixel_complexities

def calc_overlaps(patterns1: torch.Tensor, patterns2: torch.Tensor):
    return torch.mean(patterns1 * patterns2, dim=(2, 3))

def calc_partial_complexities(patterns_coarse: torch.Tensor, patterns_fine: torch.Tensor):
    patterns_coarse_upscaled = match_sizes(patterns_coarse, patterns_fine)
    return torch.abs(calc_overlaps(patterns_coarse_upscaled, patterns_fine) -
                     (calc_overlaps(patterns_fine, patterns_fine) + calc_overlaps(patterns_coarse, patterns_coarse))/2.0)

def calc_all_partial_complexities(patterns: torch.Tensor, renorm_steps: int):
    """
    Patterns: (batch_size, n_chs, w, h)
    Returns: (batch_size, n_chs, renorm_steps)
    """
    all_partial_complexities = []
    scaling_factors = []
    for step in range(1, renorm_steps):
        patterns_coarse = renormalize_patterns(patterns)
        all_partial_complexities.append(calc_partial_complexities(patterns_coarse, patterns))
        scaling_factors.append((1,2**step))
        patterns = patterns_coarse
    return torch.stack(all_partial_complexities), scaling_factors

def calc_complexities(patterns: torch.Tensor, renorm_steps: int):
    """
    Patterns: (batch_size, n_chs, w, h)
    Returns: (batch_size, n_chs)
    """
    all_partial_complexities, _ = calc_all_partial_complexities(patterns, renorm_steps)
    return torch.sum(all_partial_complexities, dim=0)



if __name__ == "__main__":
    ti.init(ti.metal)
    renorm_steps = 8

    msc_path = os.path.join(os.path.dirname(__file__), 'msc')
    image_names = sorted([f for f in os.listdir(msc_path) if f.endswith('.png')])  # Sort the images
    patterns = torch.zeros((len(image_names), 3, 1024, 1024))
    for idx, image_name in enumerate(image_names):
        image = Image.open(os.path.join(msc_path, image_name))
        image = image.convert("RGB")
        np_image = np.array(image) / 255.0  # Normalize RGB values to 0-1
        torch_image = torch.from_numpy(np_image).permute(2, 0, 1).unsqueeze(0).float()  # Convert to torch tensor and adjust dimensions
        patterns[idx] = torch_image


    rg_flow = calc_rg_flow_ragged(patterns, renorm_steps)
    pixel_complexities = calc_pixel_complexities_ragged(rg_flow)
    print(pixel_complexities[0].shape)


    # Initialize figure and axes for image and heatmap display
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)

    # Display the first image and its first renormalization level as a heatmap
    img_display = ax.imshow(patterns[0].permute(1, 2, 0).numpy(), extent=[0, 1024, 0, 1024])
    heatmap_display = ax.imshow(pixel_complexities[0][0][0].numpy(), cmap='viridis', alpha=0.75, extent=[0, 1024, 0, 1024], interpolation='nearest')

    # Slider for selecting the renormalization level
    axcolor = 'lightgoldenrodyellow'
    ax_renorm = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    slider_renorm = Slider(ax_renorm, 'Renorm Level', 0, len(pixel_complexities)-1, valinit=0, valstep=1)

    # Slider for selecting the image index
    ax_img = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    slider_img = Slider(ax_img, 'Image Index', 0, len(patterns)-1, valinit=0, valstep=1)

    def update(val):
        renorm_level = int(slider_renorm.val)
        img_index = int(slider_img.val)
        img = patterns[img_index].permute(1, 2, 0).numpy()
        heatmap = pixel_complexities[renorm_level][img_index][0].numpy()

        # Update image display
        img_display.set_data(img)

        # Upscale heatmap to match original image size and update heatmap display
        heatmap_upscaled = np.kron(heatmap, np.ones((2**renorm_level, 2**renorm_level)))
        heatmap_display.set_data(heatmap_upscaled)
        heatmap_display.set_extent([0, img.shape[1], 0, img.shape[0]])

        fig.canvas.draw_idle()

    # Call update function on slider value change
    slider_renorm.on_changed(update)
    slider_img.on_changed(update)

    plt.show()

    # rg_flow = calc_rg_flow(patterns, sample_sizes)
    # pixel_complexities = calc_partial_pixel_complexities(rg_flow)
    # complexity_data = torch.sum(pixel_complexities, dim=(0,2))
    # complexity_data = complexity_data/torch.max(complexity_data)

    # n_cols = 12
    # # Calculate the number of rows needed for 3 images per row
    # num_rows = len(complexity_data)*2 // n_cols + (1 if len(complexity_data)*2 % n_cols else 0)

    # # Plot each image with its complexity, wrapping to a new row every 3 images
    # fig, axs = plt.subplots(num_rows, n_cols, figsize=(20, num_rows * 2))
    # axs = axs.flatten()  # Flatten the array to make indexing easier
    # for idx, (stacked_complexity, image_name) in enumerate(zip(complexity_data, image_names)):
    #     half_idx = idx * 2  # Calculate index for the left half (complexity)
    #     image_path = os.path.join(msc_path, image_name)
    #     image = Image.open(image_path)
        
    #     axs[half_idx].matshow(stacked_complexity.cpu().numpy(), cmap='viridis')
    #     axs[half_idx].set_title(f"{image_name} Complexity")
    #     axs[half_idx].axis('off')
        
    #     axs[half_idx + 1].imshow(image)
    #     axs[half_idx + 1].set_title(f"{image_name} Original")
    #     axs[half_idx + 1].axis('off')
    
    # # Hide any unused axes if the number of images is not a multiple of 3
    # for idx in range(len(image_names), len(axs)):
    #     axs[idx].axis('off')
    
    # plt.tight_layout()
    # plt.show()


# if __name__ == "__main__":
#     ti.init(ti.metal)
#     renorm_steps = 10

#     msc_path = os.path.join(os.path.dirname(__file__), 'msc')
#     images = sorted([f for f in os.listdir(msc_path) if f.endswith('.png')])  # Sort the images
#     patterns = torch.zeros((len(images), 3, 1024, 1024))
#     for idx, image_name in enumerate(images):
#         image = Image.open(os.path.join(msc_path, image_name))
#         image = image.convert("RGB")
#         np_image = np.array(image) / 255.0  # Normalize RGB values to 0-1
#         torch_image = torch.from_numpy(np_image).permute(2, 0, 1).unsqueeze(0).float()  # Convert to torch tensor and adjust dimensions
#         patterns[idx] = torch_image

    
#     complexities = calc_complexities(patterns, renorm_steps)

    # n_cols = 6
    # # Calculate the number of rows needed for 3 images per row
    # num_rows = len(images) // n_cols + (1 if len(images) % n_cols else 0)

    # # Plot each image with its complexity, wrapping to a new row every 3 images
    # fig, axs = plt.subplots(num_rows, n_cols, figsize=(20, num_rows * 2))
    # axs = axs.flatten()  # Flatten the array to make indexing easier
    # for idx, (image_name, complexity) in enumerate(zip(images, complexities)):
    #     image_path = os.path.join(msc_path, image_name)
    #     image = Image.open(image_path)
    #     axs[idx].imshow(image)
    #     axs[idx].set_title(f"{image_name}\nComplexity: R: {complexity[0].item():.2f}, G: {complexity[1].item():.2f}\n, B: {complexity[2].item():.2f}")
    #     axs[idx].axis('off')
    
    # # Hide any unused axes if the number of images is not a multiple of 3
    # for idx in range(len(images), len(axs)):
    #     axs[idx].axis('off')
    
    # plt.tight_layout()
    # plt.show()




    # all_partial_complexities, scaling_factors = calc_all_partial_complexities(patterns, renorm_steps)
    # all_partial_complexities_tensor = torch.stack(all_partial_complexities)
    # complexities_first_channel = all_partial_complexities_tensor[:, :, 0]

    # # Extract the scaling factors for plotting on the x-axis
    # x_values = [factor[1] for factor in scaling_factors]

    # # Create a figure and axis for the plot
    # fig, ax = plt.subplots(figsize=(10, 6))

    # # Plot the partial complexity for each image with dotted lines and circle markers
    # for idx, image_name in enumerate(images):
    #     # Extract the partial complexities for the current image across all renormalization steps
    #     y_values = all_partial_complexities_tensor[:, idx, 0].numpy()
    #     # Plot the line for the current image with dotted lines and circle markers
    #     ax.plot(x_values, y_values, 'o:', label=image_name)  # 'o:' creates dotted lines with circle markers

    # # Set the plot title and labels
    # ax.set_title('Partial Complexity Across Scaling Factors')
    # ax.set_xlabel('Scaling Factor')
    # ax.set_ylabel('Partial Complexity')

    # # Enable the legend
    # ax.legend()

    # # Make the plot layout tight
    # plt.tight_layout()

    # # Show the plot
    # plt.show()








# def load_image_to_substrate(image_path: str) -> Substrate:
#     """Loads an image into a Substrate object with RGB channels, normalized to 0-1 range"""
#     image = Image.open(image_path)
#     image = image.convert("RGB")
#     np_image = np.array(image) / 255.0  # Normalize RGB values to 0-1
#     torch_image = torch.from_numpy(np_image).permute(2, 0, 1).unsqueeze(0).float()  # Convert to torch tensor and adjust dimensions
#     channels = {"R": ti.f32, "G": ti.f32, "B": ti.f32} 
#     substrate = Substrate((np_image.shape[1], np_image.shape[0]), torch.float32, torch.device("mps"), channels)
#     substrate.malloc()
#     substrate.mem = torch_image.contiguous()

#     return substrate


# class MSCVisData(VisualizationData):
#     RG_k: int = 1
#     show_overlap: bool = False


#     def __init__(self, substrate: Substrate, chids: list = None, window_w=800, name="RG Flow"):
#         super().__init__(substrate, chids, window_w, name)


# def add_rg_slider(vis_data: MSCVisData, sub_window):
#     """Adds slider to subwindow to sample substrate at different scales"""
#     vis_data.RG_k = sub_window.slider_int(
#         f"{vis_data.RG_k} Renormalizations",
#         vis_data.RG_k, 1, 10)

# def add_diff_button(vis_data: MSCVisData, sub_window):
#     vis_data.show_overlap = sub_window.checkbox("Show Overlap", vis_data.show_overlap)


# def display_substrate(substrate: Substrate, rg_flow: torch.Tensor):
#     """Substrate is used as vis memory, overwritten"""
#     vis_data = MSCVisData(substrate, chids=["R", "G", "B"], window_w = max(substrate.w, substrate.h))
#     update_func = compose_visualization(vis_data, subwindow_adders=[add_rg_slider, add_diff_button])
#     # start_time = time.time()
#     # repeat_after_s = 2.0
#     while not vis_data.escaped and vis_data.window.running:
#         # time_offset = ((time.time() - start_time) % repeat_after_s) / repeat_after_s
#         # vis_data.sample_exp = ((math.sin(time_offset * math.pi * 2)+1)/2.0) * vis_data.max_sample_exp
#         update_func()
#         if vis_data.show_overlap:
#             substrate.mem = calc_pixel_overlap(rg_flow[vis_data.RG_k], rg_flow[vis_data.RG_k-1])
#         else:
#             substrate.mem = rg_flow[vis_data.RG_k]


# def calc_partial_overlap(rg_flow: torch.Tensor):
#     partial_overlaps = []
#     scaling_factors = []
#     scaling_factor = 2
#     for step in range(1, rg_flow.shape[0]):
#         partial_overlaps.append(calc_overlap(rg_flow[step], rg_flow[step-1]))
#         scaling_factors.append(scaling_factor)
#         scaling_factor *= 2
#     return partial_overlaps, scaling_factors




# def infer_rg_steps(pattern: torch.Tensor, scaling_factor: int = 2):
#     """Infers the number of renormalization steps to apply to a patern given its dimensions

#     A pattern can only be downsampled so many times before it is meaningless.
#     """
#     pass

# def gen_rg_stack(pattern: torch.Tensor, n_rg_steps: int, scaling_factor: int = 2) -> torch.Tensor:
#     """
#     Generates a stack of renormalized versions of the pattern, where each step is a downsampled version of the previous step.
#     Channels are kept separate. Displays the result of renormalization using matplotlib or PIL each time it is created.

#     Args:
#         pattern: torch.Tensor, expects shapes (batch_size, n_chs, w, h)
#         n_rg_steps: The number of renormalization steps, how big is the stack
#         scaling_factor: The factor by which to downsample the pattern, 2 combines 2x2 metapixels into 1 recursively.

#     Returns:
#         rg_flow: torch.Tensor, shape (n_rg_steps, batch_size, n_chs, w, h)
#     """
#     rg_flow = torch.zeros((n_rg_steps, pattern.shape[0], pattern.shape[1], pattern.shape[2], pattern.shape[3]))
#     rg_flow[0] = pattern
#     for step in range(1, n_rg_steps):
#         renormalize_upscale(rg_flow[step-1], rg_flow[step], scaling_factor)
#         scaling_factor *= 2
#     return rg_flow


# @ti.kernel
# def renormalize_upscale(mem: ti.types.ndarray(), out_mem: ti.types.ndarray(), sample_size: ti.i32):
#     """Renormalizes all channels in memory by averaging every `sample_size * sample_size` chunk of cells into a single one.
#     No overlap (each metapixel is uniquely derived from a set of subpixels).
#     Edge metapixels, if cut off, are treated as smaller pixels
#     Everything is rescaled to the original size
#     """
#     for batch, chi, sub_i, sub_j in ti.ndrange(mem.shape[0], mem.shape[1],
#                                                (mem.shape[2] + sample_size - 1) // sample_size,
#                                                (mem.shape[3] + sample_size - 1) // sample_size):
#         sum = 0.0
#         num_pixels = 0
#         tl_i = sub_i*sample_size
#         tl_j = sub_j*sample_size
#         for super_i, super_j in ti.ndrange((tl_i, min(tl_i + sample_size, mem.shape[2])),
#                                            (tl_j, min(tl_j + sample_size, mem.shape[3]))):
#             sum += mem[0, chi, super_i, super_j]
#             num_pixels += 1
#         avg_value = sum / num_pixels
#         for super_i, super_j in ti.ndrange((tl_i, min(tl_i + sample_size, mem.shape[2])),
#                                            (tl_j, min(tl_j + sample_size, mem.shape[3]))):
#             out_mem[0, chi, super_i, super_j] = avg_value



# def calc_partial_complexity(pattern1: torch.Tensor, pattern2: torch.Tensor) -> torch.Tensor:
#     """Returns new tensor representing pixel-wise difference between two patterns."""
#     return torch.abs((pattern1 * pattern2) - ((pattern1 * pattern1 + pattern2 * pattern2)/2.0))

# def calc_overlap(pattern1: torch.Tensor, pattern2: torch.Tensor):
#     return pattern1 * pattern2  


# def match_shapes(pattern_fine: torch.Tensor, pattern_coarse: torch.Tensor) -> torch.Tensor:
#     resized_coarse_pattern = torch.zeros_like(pattern_fine)
#     upscale(pattern_coarse, resized_coarse_pattern)
#     return resized_coarse_pattern

# def calc_self_overlaps(patterns_same_size: torch.Tensor):
#     # shape: (batch_size, n_chs, w, h)
#     # just returns avg of each pattern times itself
#     return torch.sum(patterns_same_size * patterns_same_size, dim=(2, 3)) / (patterns_same_size.shape[2]*patterns_same_size.shape[3])

# def calc_renorm_overlaps(patterns_coarse: torch.Tensor, patterns_fine: torch.Tensor) -> float:
#     # expand patterns_coarse to the size of patterns_fine by repeating neighboring values
#     overlaps = torch.zeros((patterns_coarse.shape[0], patterns_coarse.shape[1]), device=patterns_coarse.device)
#     calc_renorm_overlaps_ti(patterns_coarse, patterns_fine, overlaps)
#     return overlaps 

# @ti.kernel
# def calc_renorm_overlaps_ti(patterns_coarse: ti.types.ndarray(), patterns_fine: ti.types.ndarray(), overlaps: ti.types.ndarray()):
#     num_pixels = (patterns_fine.shape[2] * patterns_fine.shape[3])
#     for i, j in ti.ndrange(patterns_fine.shape[2], patterns_fine.shape[3]):
#         for batch, ch in ti.ndrange(patterns_coarse.shape[0], patterns_coarse.shape[1]):
#             overlaps[batch, ch] += (patterns_coarse[batch, ch, i//2, j//2] * patterns_fine[batch, ch, i, j])








# SLOW -------------------------------

# @ti.kernel
# def renormalize_upscale(mem: ti.types.ndarray(), out_mem: ti.types.ndarray(), sample_size: ti.i32):
#     """Renormalizes all channels in memory by averaging every `sample_size * sample_size` chunk of cells into a single one.
#     No overlap (each metapixel is uniquely derived from a set of subpixels).
#     Edge metapixels, if cut off, are treated as smaller pixels
#     Everything is rescaled to the original size
#     """
#     for batch, ch, sub_i, sub_j in ti.ndrange(mem.shape[0], mem.shape[1],
#                                                (mem.shape[2] + sample_size - 1) // sample_size,
#                                                (mem.shape[3] + sample_size - 1) // sample_size):
#         sum = 0.0
#         num_pixels = 0
#         tl_i = sub_i*sample_size
#         tl_j = sub_j*sample_size
#         for super_i, super_j in ti.ndrange((tl_i, min(tl_i + sample_size, mem.shape[2])),
#                                            (tl_j, min(tl_j + sample_size, mem.shape[3]))):
#             sum += mem[batch, ch, super_i, super_j]
#             num_pixels += 1
#         avg_value = sum / num_pixels
#         for super_i, super_j in ti.ndrange((tl_i, min(tl_i + sample_size, mem.shape[2])),
#                                            (tl_j, min(tl_j + sample_size, mem.shape[3]))):
#             out_mem[batch, ch, super_i, super_j] = avg_value

# def calc_rg_flow(patterns, sample_sizes):
#     rg_flow = torch.zeros((len(sample_sizes), *patterns.shape), device=patterns.device)
#     for i, sample_size in enumerate(sample_sizes):
#         renormalize_upscale(patterns, rg_flow[i], sample_size)
#     return rg_flow

# def pixel_overlap(patterns1: torch.Tensor, patterns2: torch.Tensor):
#     return patterns1 * patterns2

# def calc_partial_pixel_complexities(rg_flow):
#     partial_pixel_complexities = torch.zeros_like(rg_flow)
#     for i in range(1, rg_flow.shape[0]):
#         partial_pixel_complexities[i] = torch.abs(
#             pixel_overlap(rg_flow[i], rg_flow[i-1]) -
#             (pixel_overlap(rg_flow[i], rg_flow[i]) + pixel_overlap(rg_flow[i-1], rg_flow[i-1]))/2.0
#         )
#     return partial_pixel_complexities

# --------------------------------------------------------------