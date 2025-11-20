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
from src.save.save_plot import plot_inpainting
from src.run.run_best import load_best_model, load_model, find_best_saved_model
import torch.nn.functional as F
from src.utils.output_manager import OutputManager


def test_inpainting(diffusion, unet, loader, pct=cfg.inpainting_test_coverage, n=3, lam=cfg.dps_lam):
    x_known = loader.get_samples(n_samples=n)
    #mask = torch.rand_like(x_known) < pct

    mask = (torch.rand_like(x_known) < pct).float().to(cfg.device)  # TODO necessary?

    x_inpainted = diffusion.inpaint_dps(x_known, mask)

    # Evaluate only in the missing areas (where mask == 0)
    missing = (1 - mask).bool()

    mse_loss = F.mse_loss(x_inpainted[missing], x_known[missing])
    mae_loss = F.l1_loss(x_inpainted[missing], x_known[missing])

    print(f"Inpainting MSE (masked region): {mse_loss.item():.6f}")
    print(f"Inpainting MAE (masked region): {mae_loss.item():.6f}")

    masked_imgs = []
    for x, m in zip(x_known, mask):
        arr = x.squeeze().cpu().numpy()
        m_arr = m.squeeze().cpu().numpy()
        arr_masked = np.where(m_arr > 0.5, arr, np.nan)  # NaN where missing → white in plot
        masked_imgs.append(arr_masked)

    plot_inpainting(original=[x.squeeze().cpu().numpy() for x in x_known],
                    masked=masked_imgs,
                    inpainted=[x.squeeze().cpu().numpy() for x in x_inpainted],
                    pct=pct,
                    lam=lam,
                    title=f'Inpainting Results (pct={pct}, lam={lam})')
    return mse_loss.item()

def run_inpainting(param_file=None, 
                   model_file=None):
    
    output = OutputManager(run_type="inpainting")

    loader = cfg.loader
    if param_file is None or model_file is None:
        param_file, model_file, best_loss = find_best_saved_model()
    best_loss = float(param_file.split('/')[-1].split("_")[2])  # TODO make more stable in case of param file name changes

    diffusion, unet, params, optimizer, scheduler = load_model(param_file=param_file,
                                                               model_file=model_file)

    test_inpainting(diffusion, unet, loader)

    output.finalize(best_loss, unet, epochs=cfg.epochs, params=params)  # TODO epochs might be wrong here

    return diffusion, unet


if __name__=='__main__':
    #param_file = 'output_new/0.04387_normal_Nov07_1908_256_0.1/best_params_0.04387_256_0.1_4_150_normal.csv'
    #model_file = 'output_new/0.04387_normal_Nov07_1908_256_0.1/model_0.04387.pkl'

    param_file = 'output_new/0.04342_normal_Nov10_1851_256_0.1/best_params_0.04342_256_0.1_4_200_normal.csv'
    model_file = 'output_new/0.04342_normal_Nov10_1851_256_0.1/model_0.04342.pkl'

    run_inpainting(param_file=param_file, model_file=model_file)
    
    
