from src.run.run_best import run_best
from src.run.run_optuna import run_optuna
from src.data.loader import get_loader
from src.save.save_plot import plot_random
from src.save.save_model import save_model
import os
import glob
import pandas as pd
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from src.data.precipitation_dataset import PrecipitationPatchDataset


@torch.no_grad()
def save_generated_samples(diff, outdir="samples", n=8, step_name="final", labels=None):
    os.makedirs(outdir, exist_ok=True)
    diff.model.eval()

    # Generate samples
    samples = diff.sample(n_samples=n)  # returns tensor in [0,1]

    samples = samples.detach().cpu().numpy()

    for i in range(min(n, samples.shape[0])):
        plt.imsave(f"{outdir}/gen_{step_name}_{i}.png", samples[i,0], cmap="viridis")
        
@torch.no_grad()
def save_real_examples(loader, outdir="samples", n=8):
    os.makedirs(outdir, exist_ok=True)
    batch = next(iter(loader))
    x, _ = batch if isinstance(batch, (list, tuple)) else (batch, None)

    x = x[:n].detach().cpu().numpy()

    for i in range(min(n, x.shape[0])):
        plt.imsave(f"{outdir}/real_{i}.png", x[i,0], cmap="viridis")


def plot_samples_grid(real_dir, gen_dir, n=8, save_path="comparison.png"):
    fig, axes = plt.subplots(2, n, figsize=(2*n, 4))  # 2 rows: real + generated
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in range(n):
        # Real samples
        real_path = os.path.join(real_dir, f"real_{i}.png")
        real_img = np.array(Image.open(real_path))
        axes[0, i].imshow(real_img, cmap="viridis")
        axes[0, i].axis('off')
        if i == n//2:
            axes[0, i].set_title("Real", fontsize=14)

        # Generated samples
        gen_path = os.path.join(gen_dir, f"gen_final_{i}.png")
        gen_img = np.array(Image.open(gen_path))
        axes[1, i].imshow(gen_img, cmap="viridis")
        axes[1, i].axis('off')
        if i == n//2:
            axes[1, i].set_title("Generated", fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)



def run_best_optuna(n_trials=30,
                    max_optuna_epochs=20,
                    max_optuna_patience=3,
                    max_epochs=50,
                    max_patience=3):
    param_filename, model_filename = run_optuna(n_trials=n_trials, 
                                                 max_epochs=max_optuna_epochs, 
                                                 patience=max_optuna_patience)
    
    # run best automatically saves best model
    diffusion, unet, best_loss = run_best(param_file=param_filename,
                                          model_file=model_filename,
                                          epochs=max_epochs,
                                          patience=max_patience)

    return diffusion, unet, best_loss

if __name__=='__main__':
    n_trials = 50
    max_optuna_epochs = 10
    max_optuna_patience = 2
    max_epochs = 150
    max_patience = 4

    diffusion, unet, best_loss = run_best_optuna(n_trials=n_trials,
                                                 max_optuna_epochs=max_optuna_epochs,
                                                 max_optuna_patience=max_optuna_patience,
                                                 max_epochs=max_epochs,
                                                 max_patience=max_patience)
    
    save_generated_samples(diffusion, outdir="samples", n=8, step_name="final")

    # Save real examples
    loader = get_loader()
    save_real_examples(loader, outdir="samples", n=8)
    # Plot comparison
    plot_samples_grid("samples", "samples", n=8, save_path="samples_comparison.png")    


