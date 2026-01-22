from src.run.run_best import load_best_model, load_model, find_best_saved_model
from src.config import cfg
from src.utils.output_manager import OutputManager

import torch
import itertools
import csv
import datetime
from src.run.run_normal import run_model_normal
from src.config import cfg
from src.save.save_plot import plot_random


def run_sampling(param_file,
                 model_file,
                 epoch = -1,
                 n = cfg.n_hist_samples,
                 width=None,
                 height=None,
                 verbose=True):

    best_loss = float(param_file.split('/')[-1].split("_")[2])  # TODO make more stable in case of param file name changes

    diffusion, unet, params, optimizer, scheduler = load_model(param_file=param_file,
                                                               model_file=model_file,
                                                               verbose=verbose)
    samples = diffusion.sample(n_samples=n, width=width, height=height, chunk_size=8, verbose=verbose)

    diffusion.plot_samples(samples[:cfg.n_samples], epoch=epoch, sample_info="Run Sampling")
    loader = [l for l in cfg.val_loaders if l.height == cfg.patch_size][0]
    diffusion.plot_histogram(loader=loader, epoch=epoch, sample_info="Run Sampling", samples=samples)

    return best_loss, unet, params


def plot_real():
    loader = [l for l in cfg.val_loaders if l.height == cfg.patch_size][0]
    data_iter = iter(loader)
    real_samples, _ = next(data_iter)
    plot_random(real_samples[:8], n=8, title='Real Hourly Samples', out_dir='real_samples')

if __name__=='__main__':
    plot_real()
    output = OutputManager(run_type="sampling")

    #param_file = 'output_new/normal_Jan02_2100_256_0.0/dailyoptuna_params_0.02305_256_0.1_4_150_normal.csv'
    #model_file = 'output_new/normal_Jan02_2100_256_0.0/model_0.02305.pkl'

    #param_file, model_file, best_loss = find_best_saved_model()

    #model_file = "output_new/0.0358_normal_Jan08_0328_256_0.0/model_0.035797.pkl" # Hourly Best at 250 / 233 -> Run 6381732
    #model_file = "output_new/0.0358_normal_Jan08_0328_256_0.0/model_0.0393005.pkl" # Hourly Best at 50 -> Run 6382157
    #model_file = "output_new/0.0358_normal_Jan08_0328_256_0.0/model_0.0371571.pkl"  # Hourly Best at 150 -> Run 6382158
    model_file = "output_new/0.0358_normal_Jan08_0328_256_0.0/model_0.0399487.pkl"  # Hourly Best at 10 -> Run 6382165
    param_file = "output_new/0.0358_normal_Jan08_0328_256_0.0/best_params_0.0358_256_0.0_8_250_normal.csv"

    #model_file = "output_new/0.02194_normal_Jan03_2011_256_0.0/model_0.02194.pkl" # Daily Best at 181 -> Run 6225127
    #model_file = "output_new/0.02194_normal_Jan03_2011_256_0.0/model_0.02358.pkl" # Daily Best at 10 
    #model_file = "output_new/0.02194_normal_Jan03_2011_256_0.0/model_0.02298.pkl" # Daily Best at 50
    #model_file = "output_new/0.02194_normal_Jan03_2011_256_0.0/model_0.02326.pkl" # Daily Best at 100
    #param_file = "output_new/0.02194_normal_Jan03_2011_256_0.0/best_params_0.02194_256_0.0_8_250_normal.csv"

    epoch = 10

    best_loss, unet, params = run_sampling(param_file=param_file, model_file=model_file, epoch=epoch)

    output.finalize(best_loss, unet, epochs=cfg.epochs, params=params)
