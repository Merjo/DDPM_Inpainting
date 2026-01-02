import os
from src.config import cfg
from src.utils.output_manager import OutputManager
from src.run.run_best import find_best_saved_model, load_model
import torch.nn.functional as F
import torch

def inpaint_stations_daily(diffusion, data, lam = cfg.dps_lam, n=cfg.inpainting_chunk_size):
    print('\n\nLIMITING STATIONS FOR DAILY EVAL!!!\n\n')
    station_samples, original_samples, timestamps = data[:n]
    mask = (~torch.isnan(station_samples)).float() 
    station_samples = torch.nan_to_num(station_samples, nan=0.0)  # replace NaN with 0

    x_inpainted = diffusion.inpaint_dps(station_samples, mask, lam = lam, chunk_size=cfg.inpainting_chunk_size)

    # Evaluate only in the missing areas (where mask == 0) where original data exists
    missing = (1 - mask).bool()
    orginal_nan = torch.isnan(original_samples)
    loss_mask = missing & (~orginal_nan)

    mse_loss = F.mse_loss(
        x_inpainted.cpu()[loss_mask.cpu()],
        original_samples.cpu()[loss_mask.cpu()]
    )

    print(f"Inpainting MSE (masked region): {mse_loss.item():.6f}")

    return x_inpainted, mse_loss.item(), timestamps

def evaluate_stations_daily(param_file=None, 
                            model_file=None,
                            notes=''):
    
    output = OutputManager(run_type="evaluate_stations_daily")

    data = cfg.station_val_data

    if param_file is None or model_file is None:
        param_file, model_file, best_loss = find_best_saved_model()

    diffusion, unet, params, optimizer, scheduler = load_model(param_file=param_file,
                                                               model_file=model_file)

    x_inpainted, mse_loss, timestamps = inpaint_stations_daily(diffusion, data)

    run_dir = output.get_run_dir()

    path = os.path.join(run_dir, f"data/inpainted_{mse_loss:.4g}{'' if notes is None else f'_{notes}'}.pt")
    torch.save(x_inpainted, path)

    import pandas as pd

    ts_path = path.replace(".pt", "_timestamps.csv")
    pd.Series(timestamps).to_csv(ts_path, index=False, header=False)

    #timestamps = pd.read_csv(ts_path, header=None, squeeze=True)
    #timestamps = pd.to_datetime(timestamps[0])



    output.finalize(mse_loss)

    return diffusion, unet


if __name__=='__main__':
    #param_file = 'output_new/0.04387_normal_Nov07_1908_256_0.1/best_params_0.04387_256_0.1_4_150_normal.csv'
    #model_file = 'output_new/0.04387_normal_Nov07_1908_256_0.1/model_0.04387.pkl'

    #param_file = 'output_new/0.04342_normal_Nov10_1851_256_0.1/best_params_0.04342_256_0.1_4_200_normal.csv'
    #model_file = 'output_new/0.04342_normal_Nov10_1851_256_0.1/model_0.04342.pkl'

    param_file = 'output_new/0.01942_normal_daily_Dec26_1010_256_0.0/best_params_0.01942_256_0.0_8_250_normal_daily.csv'
    model_file = 'output_new/0.01942_normal_daily_Dec26_1010_256_0.0/model_0.01942.pkl'

    notes = ''

    evaluate_stations_daily(param_file=param_file, model_file=model_file, notes=notes)