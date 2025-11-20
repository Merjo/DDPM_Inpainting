import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import glob
import os
import pandas as pd
import torch
import torch
import datetime
import numpy as np

from src.config import cfg
from src.model.song.song_unet import SongUNet
from src.model.diffusion import Diffusion
from src.model.schedulers.warmup_cosine import WarmupCosineScheduler
from src.save.save_model import save_model
from src.save.save_plot import plot_inpainting, plot_inpainting_mse_curves
from src.run.run_best import load_best_model, load_model, find_best_saved_model
import torch.nn.functional as F
from src.utils.output_manager import OutputManager
from src.run.run_inpainting import test_inpainting


def test_extensive_inpainting(diffusion, unet, loader, n=20):
    coverage_levels = [0.001, 0.005, 0.01]  # [0.999, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001]
    lam_levels = [0.001,0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1,2,5, 10,20,50,100,200,500,1000,2000,5000,10000,20000,50000,100000]#[0.0, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1, 3, 5, 10]

    results = []   # list of dicts → will become DataFrame rows

    for pct in coverage_levels:
        for lam in lam_levels:
            print(f"\nTesting inpainting with {pct*100}% known data and lambda {lam}:")
            mse_loss = test_inpainting(diffusion, unet, loader, pct=pct, n=n, lam=lam)

            results.append({
                "coverage": pct,
                "lambda": lam,
                "mse": mse_loss
            })

            print(f"MSE Loss at {pct*100}% known data and lambda {lam}: {mse_loss:.6f}")

    df = pd.DataFrame(results)

    plot_inpainting_mse_curves(df)

    # Optional: save results to disk
    out_dir = f"{cfg.current_output}/inpainting_results"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "inpainting_mse_results.csv"), index=False)

    print("\nSaved MSE matrix to inpainting_mse_results.csv")
    return df




def run_extensive_inpainting(param_file=None, 
                   model_file=None):
    
    output = OutputManager(run_type="extensive_inpainting")

    loader = cfg.loader
    if param_file is None or model_file is None:
        param_file, model_file, best_loss = find_best_saved_model()
    best_loss = float(param_file.split('/')[-1].split("_")[2])  # TODO make more stable in case of param file name changes

    diffusion, unet, params, optimizer, scheduler = load_model(param_file=param_file,
                                                               model_file=model_file)

    test_extensive_inpainting(diffusion, unet, loader)


    output.finalize(best_loss, unet, epochs=cfg.epochs, params=params)  # TODO epochs might be wrong here

    return diffusion, unet


if __name__=='__main__':
    #param_file = 'output_new/0.04387_normal_Nov07_1908_256_0.1/best_params_0.04387_256_0.1_4_150_normal.csv'
    #model_file = 'output_new/0.04387_normal_Nov07_1908_256_0.1/model_0.04387.pkl'

    param_file = 'output_new/0.04342_normal_Nov10_1851_256_0.1/best_params_0.04342_256_0.1_4_200_normal.csv'
    model_file = 'output_new/0.04342_normal_Nov10_1851_256_0.1/model_0.04342.pkl'

    run_extensive_inpainting(param_file=param_file, model_file=model_file)
    
    
