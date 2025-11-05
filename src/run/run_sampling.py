from src.run.run_best import load_best_model
from src.config import cfg
from src.utils.output_manager import OutputManager

import torch
import itertools
import csv
import datetime
from src.run.run_normal import run_model_normal
from src.config import cfg


def run_sampling():
    diffusion, unet, params, optimizer, scheduler = load_best_model(epochs=cfg.epochs)
    samples = diffusion.sample(n_samples=8, chunk_size=8, verbose=True)
    diffusion.plot_samples(samples, epoch=1, sample_info="Run Sampling")
    diffusion.plot_histogram(loader=cfg.loader, epoch=1, sample_info="Run Sampling", samples=samples)
    return samples

if __name__=='__main__':
    output = OutputManager(run_type="sampling")

    run_sampling()

    #output.finalize(best_loss, unet, epochs=cfg.optuna_epochs, params=params)
