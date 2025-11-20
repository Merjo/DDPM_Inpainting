from src.run.run_best import load_best_model, load_model, find_best_saved_model
from src.config import cfg
from src.utils.output_manager import OutputManager

import torch
import itertools
import csv
import datetime
from src.run.run_normal import run_model_normal
from src.config import cfg


def run_sampling(param_file,
                 model_file):
    output = OutputManager(run_type="sampling")

    best_loss = float(param_file.split('/')[-1].split("_")[2])  # TODO make more stable in case of param file name changes

    diffusion, unet, params, optimizer, scheduler = load_model(param_file=param_file,
                                                               model_file=model_file)
    samples = diffusion.sample(n_samples=cfg.n_hist_samples, chunk_size=8, verbose=True)
    diffusion.plot_samples(samples[:cfg.n_samples], epoch=1, sample_info="Run Sampling")
    diffusion.plot_histogram(loader=cfg.loader, epoch=1, sample_info="Run Sampling", samples=samples)

    output.finalize(best_loss, unet, epochs=cfg.epochs, params=params)

if __name__=='__main__':

    param_file = 'output_new/0.04387_normal_Nov07_1908_256_0.1/best_params_0.04387_256_0.1_4_150_normal.csv'
    model_file = 'output_new/0.04387_normal_Nov07_1908_256_0.1/model_0.04387.pkl'

    #param_file, model_file, best_loss = find_best_saved_model()

    run_sampling(param_file=param_file, model_file=model_file)
