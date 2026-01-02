import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import numpy as np

from src.config import cfg
from src.save.save_plot import plot_station_inpainting
from src.run.run_best import load_model, find_best_saved_model
import torch.nn.functional as F
from src.utils.output_manager import OutputManager


def test_station_inpainting(diffusion, data, n=3, lam=cfg.dps_lam):
    station_samples, original_samples, timestamps = data[:n]
    mask = (~torch.isnan(station_samples)).float() 
    station_samples = torch.nan_to_num(station_samples, nan=0.0)  # replace NaN with 0

    station_samples = station_samples.to(cfg.device)
    mask = mask.to(cfg.device)
    original_samples = original_samples.to(cfg.device)

    x_inpainted = diffusion.inpaint_dps(station_samples, mask, lam = lam, chunk_size=1)

    # Evaluate only in the missing areas (where mask == 0) where original data exists
    missing = (1 - mask).bool()
    orginal_nan = torch.isnan(original_samples)
    loss_mask = missing & (~orginal_nan)

    mse_loss = F.mse_loss(
        x_inpainted.cpu()[loss_mask.cpu()],
        original_samples.cpu()[loss_mask.cpu()]
    )

    mae_loss = F.l1_loss(
        x_inpainted.cpu()[missing.cpu()],
        original_samples.cpu()[missing.cpu()]
    )

    print(f"Inpainting MSE (masked region): {mse_loss.item():.6f}")
    print(f"Inpainting MAE (masked region): {mae_loss.item():.6f}")

    plot_station_inpainting(
                    radar=[x.squeeze().cpu().numpy() for x in original_samples],
                    station=[x.squeeze().cpu().numpy() for x in station_samples],
                    inpainted=[x.squeeze().cpu().numpy() for x in x_inpainted],
                    timestamps = timestamps,
                    lam=lam,
                    title=f'Inpainting Results, lam={lam})')
    return mse_loss.item()

def run_station_inpainting(param_file=None, 
                            model_file=None):
    
    output = OutputManager(run_type="station_inpainting")

    data = cfg.station_val_data
    if param_file is None or model_file is None:
        param_file, model_file, best_loss = find_best_saved_model()
    best_loss = float(param_file.split('/')[-1].split("_")[2])  # TODO make more stable in case of param file name changes

    diffusion, unet, params, optimizer, scheduler = load_model(param_file=param_file,
                                                               model_file=model_file)

    test_station_inpainting(diffusion, data)

    output.finalize(best_loss, unet, epochs=cfg.epochs, params=params)  # TODO epochs might be wrong here

    return diffusion, unet


if __name__=='__main__':
    #param_file = 'output_new/0.04387_normal_Nov07_1908_256_0.1/best_params_0.04387_256_0.1_4_150_normal.csv'
    #model_file = 'output_new/0.04387_normal_Nov07_1908_256_0.1/model_0.04387.pkl'

    #param_file = 'output_new/0.04342_normal_Nov10_1851_256_0.1/best_params_0.04342_256_0.1_4_200_normal.csv'
    #model_file = 'output_new/0.04342_normal_Nov10_1851_256_0.1/model_0.04342.pkl'

    param_file = 'output_new/0.01942_normal_daily_Dec26_1010_256_0.0/best_params_0.01942_256_0.0_8_250_normal_daily.csv'
    model_file = 'output_new/0.01942_normal_daily_Dec26_1010_256_0.0/model_0.01942.pkl'

    
    #data = read_station_data()

    run_station_inpainting(param_file=param_file, model_file=model_file)
    
    
