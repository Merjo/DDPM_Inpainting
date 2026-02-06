# Save Scripts

Outputs of runs (see [run description](../run/README.md)) are managed by an instance of the so called output manager class, found in [`output_manager.py`](output_manager.py). This class is responsible for managing the output of the model runs, including saving the model checkpoints, training plots, and evaluation results. It also manages the output folder structure and naming conventions for the different runs.

The scrips that produce some sort of output are:
- [`save_plots.py`](save_plots.py) The main script for plots used by the DDPM during training, but also other modules. It contains the code for saving the training plots, such as the loss curve, the histogram and RAPSD curve of the sampled data, and the inpainting results at random timesteps. The plots are usually directly in the folder of the corresponding run in the output folder.
- [`save_model.py`](save_model.py) Contains the code for saving the model checkpoints during training, which are stored in the output folder under the corresponding run.
- [`plot_station_inpainting.py`](plot_station_inpainting.py) Plots the station, inpainted and ground truth data at random timesteps, including the corresponding HYRAS/REGNIE interpolations when looking at daily data.
- [`spectra.py`](spectra.py) Contains the code for calculating the RAPSD curves based on the work by Hess et al., see the [Master's thesis](docs/master.hosak.pdf) for more details.
- [`plot_training_log`.py](plot_training_log.py) Contains the code for plotting the training log and validation loss for the results section of the thesis.

