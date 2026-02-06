# Evaluation Code

For model evaluation the workflow is as follows. After training the model, the model is inpainted with station data using [`run_station_inpainting.py`](run_station_inpainting.py) and the inpainted results are stored in a cache folder.

Then, the evaluation is prepared using [`prepare_evaluation.py`](prepare_evaluation.py),  transforms the data to an easier format and handles the different DDPM modes (Monte Carlo + Aggregate).

 The output of prepare evaluation is stored in a cache folder as well and can be used for the plotting routine [`plot_station_inpainting.py`](plot_station_inpainting.py), which visualizes the inpainting at random timesteps..

Lastly, the metrics are calculated with [`evaluate_final.py`](evaluate_final.py) and the resulting plots are stored in the output folder.
