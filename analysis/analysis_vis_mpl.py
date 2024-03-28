import numpy as np

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button

from analysis_data import HistoryData

def update_plots(axs, history_data, inds):
    run_data = history_data.curr_run_data
    # Clear previous plots to ensure old data doesn't stick around
    for ax in axs:
        ax.clear()
    
    genome_data = run_data.substrate.mem[0, inds["genome"]].detach().cpu().numpy()
    genome_data[genome_data == -1] = np.nan
    # Create a copy of the viridis colormap and set NaN values to black
    viridis_cmap = plt.cm.viridis.copy()
    viridis_cmap.set_bad(color='black')
    axs[0].imshow(genome_data, cmap=viridis_cmap, vmin=0)
    axs[0].set_title('Genome')

    energy_data = run_data.substrate.mem[0, inds["energy"]].detach().cpu().numpy()
    # axs[1].imshow(energy_data, cmap="Reds")
    # axs[1].set_title('Energy')

    infra_data = run_data.substrate.mem[0, inds["infra"]].detach().cpu().numpy()
    # axs[2].imshow(infra_data, cmap="Greens")
    # axs[2].set_title('Infra')

    # Combine energy, infra, and genome data into an RGB image
    # Normalize each channel by its maximum value
    rgb_image = np.stack([
        energy_data / np.nanmax(energy_data),
        infra_data / np.nanmax(infra_data),
        genome_data / np.nanmax(genome_data)
    ], axis=-1)
    # Handle NaNs after normalization
    rgb_image = np.nan_to_num(rgb_image)
    
    # Update the first three plots as before
    # axs[0].imshow(genome_data, cmap="viridis", vmin=0)
    # axs[0].set_title('Genome')
    # axs[1].imshow(energy_data, cmap="Reds")
    # axs[1].set_title('Energy')
    # axs[2].imshow(infra_data, cmap="Greens")
    # axs[2].set_title('Infra')
    
    # Display the new RGB image
    axs[1].imshow(rgb_image)
    axs[1].set_title('Combined RGB')

    # Update figure title with the number of steps in the current run
    plt.suptitle(f"Run: {history_data.curr_run_index + 1}/{len(history_data.run_names)}, Step: {history_data.curr_run_data.curr_step_index + 1}/{len(history_data.curr_run_data.steps)}")

    plt.draw()


def visualize_run_data(hist_dir: str, start_run_name: str, torch_device: str):
    history_data = HistoryData(hist_dir, torch_device)
    history_data.goto_run_name(start_run_name)

    # Increase figure size for better spacing
    fig, axs = plt.subplots(1, 2, figsize=(20, 8.5), sharex=True, sharey=True)
    
    # Use constrained layout to automatically adjust subplot params
    plt.subplots_adjust(bottom=0.25, left=0.05, right=0.95, top=0.85)
    fig.set_constrained_layout(True)

    inds = history_data.curr_run_data.substrate.ti_indices[None]

    # Initial plot
    update_plots(axs, history_data, inds)

    # Adjust the size and positioning of the sliders and buttons
    slider_step_pos = [0.2, 0.15, 0.65, 0.03]
    slider_run_pos = [0.2, 0.1, 0.65, 0.03]
    button_width = 0.1
    button_height = 0.04
    button_spacing = 0.02
    bottom_button_pos = 0.02
    top_button_pos = 0.2

    ax_slider_step = plt.axes(slider_step_pos, facecolor='lightgoldenrodyellow')
    slider_steps = Slider(ax_slider_step, 'Step', 0, len(history_data.curr_run_data.steps)-1, valinit=0, valstep=1)

    ax_slider_run = plt.axes(slider_run_pos, facecolor='lightgoldenrodyellow')
    slider_runs = Slider(ax_slider_run, 'Run', 0, len(history_data.run_names)-1, valinit=0, valstep=1)

    # Position buttons with consistent spacing
    ax_next_step = plt.axes([0.85, bottom_button_pos, button_width, button_height])
    btn_next_step = Button(ax_next_step, 'Next Step')
    ax_prev_step = plt.axes([0.85 - button_width - button_spacing, bottom_button_pos, button_width, button_height])
    btn_prev_step = Button(ax_prev_step, 'Prev Step')

    ax_next_run = plt.axes([0.85, top_button_pos, button_width, button_height])
    btn_next_run = Button(ax_next_run, 'Next Run')
    ax_prev_run = plt.axes([0.85 - button_width - button_spacing, top_button_pos, button_width, button_height])
    btn_prev_run = Button(ax_prev_run, 'Prev Run')

    ax_plot_genome_distance_matrix = plt.axes([0.85, bottom_button_pos + button_height + button_spacing, button_width, button_height])
    btn_plot_genome_distance_matrix = Button(ax_plot_genome_distance_matrix, 'Plot Genome Distances')
    
    ax_plot_genome_knn_graph = plt.axes([0.85, bottom_button_pos + 2 * (button_height + button_spacing), button_width, button_height])
    btn_plot_genome_knn_graph = Button(ax_plot_genome_knn_graph, 'Plot Genome KNN Graph')


    def plot_genome_distance_matrix(event):
        history_data.curr_run_data.population.plot_genome_distance_matrix(title = f"{history_data.curr_run_data.run_name} Step {history_data.curr_run_data.curr_step_number}")

    def plot_genome_knn_graph(event):
        history_data.curr_run_data.population.plot_knn_net(k=2,
                                                           substrate=history_data.curr_run_data.substrate,
                                                           title=f"{history_data.curr_run_data.run_name} Step {history_data.curr_run_data.curr_step_number}")

    def update_step(val):
        history_data.curr_run_data.curr_step_index = int(slider_steps.val)
        history_data.curr_run_data.load_step()
        update_plots(axs, history_data, inds)

    def update_run(val):
        history_data.curr_run_index = int(slider_runs.val)
        history_data.load_run()
        slider_steps.valmax = len(history_data.curr_run_data.steps) - 1
        slider_steps.reset()  # Reset to first step of the new run, also triggers update_step
        update_plots(axs, history_data, inds)  # Ensure plots are updated immediately after run change

    def next_step(event):
        history_data.curr_run_data.next_step()
        slider_steps.set_val(history_data.curr_run_data.curr_step_index)
    
    def prev_step(event):
        history_data.curr_run_data.prev_step()
        slider_steps.set_val(history_data.curr_run_data.curr_step_index)
    
    def next_run(event):
        history_data.next_run()
        slider_runs.set_val(history_data.curr_run_index)
        update_plots(axs, history_data, inds) # Update plots for the new run
    
    def prev_run(event):
        history_data.prev_run()
        slider_runs.set_val(history_data.curr_run_index)
        update_plots(axs, history_data, inds) # Update plots for the new run
    
    slider_steps.on_changed(update_step)
    slider_runs.on_changed(update_run)
    btn_next_step.on_clicked(next_step)
    btn_prev_step.on_clicked(prev_step)
    btn_next_run.on_clicked(next_run)
    btn_prev_run.on_clicked(prev_run)
    btn_plot_genome_distance_matrix.on_clicked(plot_genome_distance_matrix)
    btn_plot_genome_knn_graph.on_clicked(plot_genome_knn_graph)

    plt.show()

