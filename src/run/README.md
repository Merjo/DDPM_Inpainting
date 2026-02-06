 # Run Scripts

 The run folder contains scripts that use a DDPM model instance for training, sampling or inpainting. Various run types have been used in this investigation that are listed below.

 - [`run_normal.py`](run_normal.py): Loads a model with parameters defined in the config file and trains it with the given training configurations.
 - [`run_best.py`](run_best.py): Loads the best model trained so far and continues training it.
 - [`run_optuna.py`](run_optuna.py): Runs the hyperparameter optimization with Optuna, which trains multiple models with different hyperparameters and evaluates them to find the best hyperparameters.
 - [`run_optuna_best.py`](run_optuna_best.py): Starts an optuna run and afterwards trains the best model found with optuna
 - [`run_inpainting.py`](run_inpainting.py): Runs the inpainting routine for a trained model by applying a random mask on the input data and using the model to inpaint the missing values.
 - [`run_extensive_inpainting.py`](run_extensive_inpainting.py): Runs the inpainting routine several times for different lambda values in diffusion posterior sampling (DPS) and compares those.
- [`run_sampling.py`](run_sampling.py): Runs the sampling routine for a trained model by starting from pure noise and using the model to sample new data.
- [`run_extensive_sampling.py`](run_extensive_sampling.py): Runs the sampling routine a high number of times to obtain a plausible histogram and RAPSD curve
- [`run_station_inpainting.py`](run_station_inpainting.py): Runs the inpainting routine for a trained model by using station data as conditioning information and using the model to inpaint the missing values between the station locations.

Lastly there is the [`prepare_run.py`](prepare_run.py) script, which initiateds a model with a specific set of parameters.