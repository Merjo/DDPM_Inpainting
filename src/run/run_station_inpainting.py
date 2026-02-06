import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import numpy as np
import pandas as pd

from src.config import cfg
from src.run.run_best import load_model, find_best_saved_model
import torch.nn.functional as F
from src.save.output_manager import OutputManager
from datetime import datetime


def station_inpainting(diffusion, data, mode, timesteps=None, subset=None, lam=cfg.dps_lam, timesteps_offset=10, monte_carlo=False, daily_aggregate=False):
    if monte_carlo and not daily_aggregate:
        if mode=='daily':
            subset = np.arange(0, 360, 10)
        else:
            indices = np.arange(0, 360, 10)
            subset = subset[indices]
    if timesteps is not None:
        idx = slice(timesteps_offset, timesteps_offset+timesteps)
        station_samples, radar_samples, timestamps = data[idx]
    elif subset is not None:
        # subset is np.array of arbitrary indices
        idx = np.array(subset)
        station_samples = data.station_data[idx].cpu()
        radar_samples = data.radar_data[idx].cpu()
        timestamps = list(np.array(data.timestamps)[idx])

        print(f'\nTimestamps: \n{timestamps}\n')
    else:
        station_samples, radar_samples, timestamps = data[slice(None)]

    mask = (~torch.isnan(station_samples)).float() 
    station_samples = torch.nan_to_num(station_samples, nan=0.0)  # replace NaN with 0

    if mode == 'daily' or True:  # Run normal for all
        if monte_carlo:
            x_samples = []
            repeats = 10
            if daily_aggregate:
                repeats = 5  # Reduce for daily due to memory
            for _ in range(repeats):
                x = diffusion.inpaint_dps(station_samples, mask, lam=lam, chunk_size=cfg.inpainting_chunk_size)
                x_samples.append(x)
            x_inpainted = torch.stack(x_samples, dim=0).mean(dim=0)
        else:
            x_inpainted = diffusion.inpaint_dps(station_samples, mask, lam = lam, chunk_size=cfg.inpainting_chunk_size)
    elif mode=='hourly':
        x_inpainted = diffusion.inpaint_dps_cpu(station_samples, mask, lam = lam, chunk_size=cfg.inpainting_chunk_size)
    else:
        raise Exception('Mode should be daily or hourly')

    # Evaluate only in the missing areas (where mask == 0) where original data exists
    missing = (1 - mask).bool()
    orginal_nan = torch.isnan(radar_samples)
    loss_mask = missing & (~orginal_nan)
    loss_mask = loss_mask.cpu()

    mse_loss = F.mse_loss(
        x_inpainted.cpu()[loss_mask],
        radar_samples.cpu()[loss_mask]
    )

    mae_loss = F.l1_loss(
        x_inpainted.cpu()[loss_mask],
        radar_samples.cpu()[loss_mask]
    )

    print(f"Inpainting MSE (masked region): {mse_loss.item():.6f}")
    print(f"Inpainting MAE (masked region): {mae_loss.item():.6f}")

    return mse_loss.item(), radar_samples, station_samples, x_inpainted, timestamps



def pick_hourly_subset_indices_monthly(seed=cfg.seed, leap_year=False):
    rng = np.random.default_rng(seed)

    # Month lengths for a non-leap year
    month_lengths = [
        31,  # Jan
        28,  # Feb
        31,  # Mar
        30,  # Apr
        31,  # May
        30,  # Jun
        31,  # Jul
        31,  # Aug
        30,  # Sep
        31,  # Oct
        30,  # Nov
        31,  # Dec
    ]

    if leap_year:
        month_lengths[1] = 29
    

    # Extra random hours per month (chosen to match month length)
    # 24 + extra = number of days in month
    extra_hours_per_month = [m-24 for m in month_lengths]

    indices = []
    day_offset = 0

    for days_in_month, extra in zip(month_lengths, extra_hours_per_month):
        # Base hours: 0..23
        base_hours = np.arange(24)

        # Extra random hours
        random_hours = rng.integers(0, 24, size=extra)

        # Pool + shuffle
        hour_pool = np.concatenate([base_hours, random_hours])
        rng.shuffle(hour_pool)

        # Assign one hour per day
        for i in range(days_in_month):
            hour = hour_pool[i]
            timestep = (day_offset + i) * 24 + hour
            indices.append(timestep)

        day_offset += days_in_month
    return np.array(indices)

def pick_hourly_subset_indices_aggregate(seed=cfg.seed, leap_year=False):
    rng = np.random.default_rng(seed)

    # Month lengths for a non-leap year
    month_lengths = [
        31,  # Jan
        28,  # Feb
        31,  # Mar
        30,  # Apr
        31,  # May
        30,  # Jun
        31,  # Jul
        31,  # Aug
        30,  # Sep
        31,  # Oct
        30,  # Nov
        31,  # Dec
    ]

    if leap_year:
        month_lengths[1] = 29

    indices = []
    day_offset = 0

    for days_in_month in month_lengths:
        # Randomly pick one day in the month
        day = rng.integers(0, days_in_month)

        # Add all 24 hourly timesteps for that day
        base_timestep = (day_offset + day) * 24
        indices.extend(base_timestep + np.arange(24))

        day_offset += days_in_month

    return np.array(indices)



def run_station_inpainting(mode,
                           data=None,
                           param_file=None,
                           model_file=None,
                           years=cfg.val_inpainting_years,
                           timesteps=None,
                           filippou = False,
                           reduce_hourly=True,
                           daily_aggregate=False,
                           monte_carlo=False):

    if data is None:
        data = cfg.station_data(years, mode, filippou=filippou)
    if param_file is None or model_file is None:
        param_file, model_file, best_loss = find_best_saved_model()

    print(f'\nRUNNING STATION INPAINTING\n\nmode={mode}, reduce hourly={reduce_hourly}, filippou={filippou}, years={years}\n\tmodel file={model_file}\n\tparam file={param_file}')

    subset=None
    if mode=='hourly' and reduce_hourly:
        if len(years)>1:
            raise NotImplementedError('Not Implemented for multiple years')
        leap_year = years[0]%4==0
        if daily_aggregate:
            subset = pick_hourly_subset_indices_aggregate(leap_year=leap_year)
        else:
            subset = pick_hourly_subset_indices_monthly(leap_year=leap_year)
        

    diffusion, unet, params, optimizer, scheduler = load_model(param_file=param_file,
                                                               model_file=model_file)

    mse_loss, radar_samples, station_samples, x_inpainted, timestamps = station_inpainting(diffusion, data, mode, timesteps=timesteps, subset=subset, monte_carlo=monte_carlo, daily_aggregate=daily_aggregate)

    # Save Inpainted Data

    date_str = datetime.now().strftime("%b%d_%H%M")
    dir_name = f'{date_str}_{mode}_{years[0]}'
    run_dir = cfg.output_cache_path
    inpainted_path = f'{run_dir}/{date_str}_{mse_loss:.6g}_{mode}_inpainted_stations_{years[0]}_{years[-1]}{"" if timesteps is None else "_ts"+str(timesteps)}{"_filippou" if filippou else ""}.pt'
    torch.save(x_inpainted.cpu(), inpainted_path)
    print(f"[Run Station Inpainting] Saved inpainted data to {inpainted_path}")

    ts_path = inpainted_path.replace(".pt", "_timestamps.csv")
    pd.Series(timestamps).to_csv(ts_path, index=False, header=False)

    return x_inpainted, mse_loss, params


if __name__=='__main__':
    #param_file = 'output_new/0.04387_normal_Nov07_1908_256_0.1/best_params_0.04387_256_0.1_4_150_normal.csv'
    #model_file = 'output_new/0.04387_normal_Nov07_1908_256_0.1/model_0.04387.pkl'

    #param_file = 'output_new/0.04342_normal_Nov10_1851_256_0.1/best_params_0.04342_256_0.1_4_200_normal.csv'
    #model_file = 'output_new/0.04342_normal_Nov10_1851_256_0.1/model_0.04342.pkl'

    #param_file = 'output_new/0.01942_normal_daily_Dec26_1010_256_0.0/best_params_0.01942_256_0.0_8_250_normal_daily.csv'
    #model_file = 'output_new/0.01942_normal_daily_Dec26_1010_256_0.0/model_0.01942.pkl'

    #param_file = 'output_new/0.01985_normal_Dec31_1447_256_0.0/best_params_0.01985_256_0.0_8_250_normal.csv'
    #model_file = 'output_new/normal_Jan02_2100_256_0.0/model_0.02305.pkl'

    output = OutputManager(run_type="station_inpainting")

    #param_file = 'output_new/normal_Jan03_2011_256_0.0/dailyOptuna_params_0.02358_256_0.0_8_250_normal.csv'
    #model_file = 'output_new/normal_Jan03_2011_256_0.0/model_0.02358.pkl'
    
    #data = read_station_data()

    years = cfg.test_years
    timesteps = None
    mode = cfg.model_type
    filippou = cfg.filippou_mode
    reduce_hourly = True
    daily_aggregate = False
    monte_carlo = True

    if mode=='daily':
        model_file = "output_new/0.02194_normal_Jan03_2011_256_0.0/model_0.02194.pkl"
        param_file = "output_new/0.02194_normal_Jan03_2011_256_0.0/best_params_0.02194_256_0.0_8_250_normal.csv"

       # model_file = "output_new/0.02194_normal_Jan03_2011_256_0.0/model_0.02249.pkl"  # Run Epoch 220 2nd best

        # model_file = "output_new/0.02194_normal_Jan03_2011_256_0.0/model_0.02326.pkl"  # Run Epoch 100 3? Best good His
    elif mode == 'hourly':
        model_file = "output_new/0.0358_normal_Jan08_0328_256_0.0/model_0.035797.pkl"
        param_file = "output_new/0.0358_normal_Jan08_0328_256_0.0/best_params_0.0358_256_0.0_8_250_normal.csv"
    else:
        raise Exception(f'Mode should be daily or hourly.')
    

    x_inpainted, mse_loss, params = run_station_inpainting(mode, param_file=param_file, model_file=model_file, years = years, timesteps=timesteps, filippou=filippou, reduce_hourly=reduce_hourly, daily_aggregate=daily_aggregate, monte_carlo=monte_carlo)

    output.finalize(mse_loss, params=params)

    
    
