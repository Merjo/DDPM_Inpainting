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
from src.save.save_plot import plot_inpainting, plot_inpainting3
from src.run.run_best import load_best_model, load_model, find_best_saved_model
import torch.nn.functional as F


def test_inpainting(diffusion, unet, loader, pct=0.3, n=3):
    x_known = loader.get_samples(n_samples=n)
    #mask = torch.rand_like(x_known) < pct

    mask = (torch.rand_like(x_known) < pct).float().to(cfg.device)  # TODO necessary?

    x_inpainted = diffusion.inpaint(x_known, mask)

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
                    title=f'Inpainting Results (pct={pct})')
    
    plot_inpainting3(original=[x.squeeze().cpu().numpy() for x in x_known],
                    masked=masked_imgs,
                    inpainted=[x.squeeze().cpu().numpy() for x in x_inpainted],
                    title=f'Inpainting3 Results (pct={pct})')


def run_inpainting(param_file=None, 
                   model_file=None):
    loader = cfg.loader
    if param_file is None or model_file is None:
        param_file, model_file, best_loss = find_best_saved_model()
    diffusion, unet, params, optimizer, scheduler = load_model(param_file=param_file,
                                                                          model_file=model_file)

    test_inpainting(diffusion, unet, loader)

    return diffusion, unet


if __name__=='__main__':
    param_file = "output_new/0.0213_best_Oct28_1639_256_0.1/best_params_0.0213_256_0.1_4_200_best.csv"
    model_file = "output_new/0.0213_best_Oct28_1639_256_0.1/model_0.0213.pkl"
    date_str = datetime.datetime.now().strftime("%b%d_%H%M")
    output_dir = f"output_inpainting/{date_str}_inpainting_test"
    cfg.update_output_path(output_dir)
    run_inpainting(param_file=param_file, model_file=model_file)