from matplotlib import pyplot as plt
import torch
from src.save.save_plot import scale_back_numpy
from src.config import cfg
import os
from datetime import datetime
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path
import pandas as pd
from src.run.run_station_inpainting import station_inpainting
from src.utils.evaluate_utils import transfer_to_hyras
from src.data.read_data import read_raw_data
from src.data.read_hyras_data import load_hyras
from src.data.read_data import read_raw_data
from skimage.metrics import structural_similarity as ssim


def prepare_evaluation(years, mode, timesteps, filippou=False, timesteps_offset=10):  # Often, first few radar samples are nan for some reason, therefore introduced offset 
    data = cfg.station_data(years, mode=mode)
    station_samples, radar_samples, timestamps = data[:] if timesteps is None else data[timesteps_offset:timesteps_offset+timesteps]
    inpainted_path_ending = f'{mode}_inpainted_stations_{years[0]}_{years[-1]}{"" if timesteps is None else "_ts"+str(timesteps)}{"_filippou" if filippou else ""}.pt'
    data_path = cfg.output_cache_path
    inpainted_files = [f for f in os.listdir(data_path) if f.endswith(inpainted_path_ending)]
    if not inpainted_files:
        raise NotImplementedError('Not implemented yet :/')
    else: 
        latest_file = max(inpainted_files, key=lambda f: os.path.getmtime(os.path.join(data_path, f)))
        inpainted_path = os.path.join(data_path, latest_file)
        inpainted = torch.load(inpainted_path)
        if mode=='hourly':
            ts_path = inpainted_path.replace(".pt", "_timestamps.csv")
            timestamps = pd.read_csv(ts_path, header=None).iloc[:, 0]
            timestamps = pd.to_datetime(timestamps)
            raw_ref = read_raw_data(years=years)
            radar_times = pd.to_datetime(raw_ref["time"].values)
            hour_indices = pd.Index(radar_times).get_indexer(timestamps)

            if (hour_indices < 0).any():
                missing = timestamps[hour_indices < 0]
                raise ValueError(
                    f"{len(missing)} timestamps not found in radar data.\n"
                    f"First few missing:\n{missing[:5]}"
                )

            assert hour_indices.max() < radar_samples.shape[0]
            assert len(hour_indices) == inpainted.shape[0]

            radar_subset = radar_samples[hour_indices]
            station_subset = station_samples[hour_indices]

            print("Inpainted:", inpainted.shape)
            print("Radar subset:", radar_subset.shape)
            print("Station subset:", station_subset.shape)

            radar_samples = radar_subset
            station_samples = station_subset
        print(f"[Plot] Loaded inpainted data from {latest_file}")
        loss = latest_file.split('_')[1]
        mse_loss = float(loss)

    date_str = datetime.now().strftime("%b%d_%H%M")

    # Tranfer data to HYRAS grid for plotting
    raw_data = read_raw_data(years=years, aggregate_daily=cfg.model_type=='daily', delete_coords=False, time_slices=1)
    lon = raw_data['lon'][38:1200-138, 30:1100-46]
    lat = raw_data['lat'][38:1200-138, 30:1100-46]
    radar = transfer_to_hyras(years=years, data=radar_samples.squeeze(), data_lon=lon, data_lat=lat, do_mask=True, limit_to_values=False, return_xarray=False)
    station = transfer_to_hyras(years=years, data=station_samples.squeeze(), data_lon=lon, data_lat=lat, do_mask=True, limit_to_values=False, return_xarray=False)
    inpainted = transfer_to_hyras(years=years, data=inpainted.squeeze(), data_lon=lon, data_lat=lat, do_mask=True, limit_to_values=False, return_xarray=False)


    radar_np = [x.squeeze().cpu().numpy() for x in radar]
    station_np = [x.squeeze().cpu().numpy() for x in station]
    inpainted_np = [x.squeeze().cpu().numpy() for x in inpainted]

    # Scale back
    radar = np.array([scale_back_numpy(x, cfg.scaler) for x in radar_np])
    station = np.array([scale_back_numpy(x, cfg.scaler) for x in station_np])
    inpainted = np.array([scale_back_numpy(x, cfg.scaler) for x in inpainted_np])


    hyras = None
    if mode=='daily':
        hyras = load_hyras(years=years)
        hyras = torch.from_numpy(hyras.values)
        hyras = hyras if timesteps is None else hyras[timesteps_offset:timesteps_offset+timesteps]
        hyras_np = [x.squeeze().cpu().numpy() for x in hyras]
        hyras = np.array(hyras_np)

    save_dir_name = f"final_{date_str}_{mode}_{years[0]}_{timesteps}_filippou{filippou}"
    save_dir = Path(cfg.output_cache_path) / save_dir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save arrays
    np.save(save_dir / "radar.npy", radar)
    np.save(save_dir / "station.npy", station)
    np.save(save_dir / "inpainted.npy", inpainted)

    if hyras is not None:
        np.save(save_dir / "hyras.npy", hyras)

    # Save timestamps
    timestamps_out = pd.to_datetime(timestamps)
    timestamps_out = pd.Series(timestamps_out)
    timestamps_out.to_csv(save_dir / "timestamps.csv", index=False, header=False)

    # Optional: save minimal metadata (very useful later)
    with open(save_dir / "meta.txt", "w") as f:
        f.write(f"mode: {mode}\n")
        f.write(f"years: {years}\n")
        f.write(f"timesteps: {timesteps}\n")
        f.write(f"filippou: {filippou}\n")
        f.write(f"mse_loss: {mse_loss}\n")
        f.write(f"samples: {len(timestamps_out)}\n")

    print(f"[Save] Aligned data saved to {save_dir}")
    
    return

def listify_numpy(data):
    data = [data[i] for i in range(data.shape[0])]
    return data

def load_final(path, hyras = False, listify=False):
    path = Path(path)
    radar = np.load(path / "radar.npy")
    station = np.load(path / "station.npy")
    inpainted = np.load(path / "inpainted.npy")
    timestamps = pd.read_csv(path / "timestamps.csv", header=None)[0]
    timestamps = pd.to_datetime(timestamps)
    if listify:
        radar = listify_numpy(radar)
        station = listify_numpy(station)
        inpainted = listify_numpy(inpainted)
    if hyras:
        hyras = np.load(path / 'hyras.npy')
        if listify:
            hyras = listify_numpy(hyras)
        return radar, station, inpainted, timestamps, hyras
    else:
        return radar, station, inpainted, timestamps


if __name__ == "__main__":
    years = range(2018, 2019)
    timesteps = None
    mode = cfg.model_type
    filippou = cfg.filippou_mode
    
    prepare_evaluation(years=years, mode=mode, timesteps=timesteps, filippou = filippou)