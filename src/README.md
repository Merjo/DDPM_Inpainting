# Source Code

This files provides an overview of how the source code is structured in general and how the code configuration is implemented. Each folder within the source folder has its own README file that explains how the different aspects of the project are implemented in more detail.

## Folder Structure
- [`data/`](data): Contains the data processing code, including the dataset class and data loading functions.
- [`evaluate/`](evaluate): Contains the code for evaluating the model, including the evaluation metrics and inpainting scripts.
- [`model/`](model): Contains the code for the DDPM model class as well as the U-Net functions and schedulers.
- [`run/`](run): Contains the code for training and using the DDPM model, including also the inpainting code and the hyperparameter optimization with Optuna.
- [`save/`](save): Contains the code for saving aspects of the model like the model itself, plots of the models and other plotting routines.
- [`scripts/`](scripts): Contains scripts for training, data reading and evaluation used to run slurm jobs on the high performance cluster (HPC) of the Potsdam Institute for Climate Impact Research (PIK).
- [`utils/`](utils): Contains utility functions such as the output manager that manages the output of the model runs, but also helper functions in general.


## The Config File

The source coded is structured around the [`config.py`](config.py) file, which contains all the configuration parameters for the model, training, data processing, and evaluation. 

The configurations are realized via a configuration class with the attributes being the different configuration parameters. This allows for the configurations to be dynamically be updated or calculated via properties.

It consists of configurations in the following categories:
- General
- Mode
- Efficiency
- Data
- Importance Sampling
- Training
- Output
- Plotting
- Inpainting
- Model
- Optuna / Hyperparameter Optimization

Further, it contains properties that load data or aspects of the data once and storing them in an attribute for other functions to use like a cache.

Lastly, it contains some simple methods for managing the output or calculating the clamp range.

An instance, `cfg`, of the config class is generated and inported in all files that need configuration information or data access.