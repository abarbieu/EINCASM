import torch
import taichi as ti

from analysis_vis_mpl import visualize_run_data

if __name__ == "__main__":
    ti.init(ti.metal)
    torch_device = torch.device("mps")

    # run_dirs = os.listdir("./history/")
    hist_dir = "./history/"

    start_run_name = "space_evolver_run_240310-0013_40"
    visualize_run_data(hist_dir, start_run_name, torch_device)
