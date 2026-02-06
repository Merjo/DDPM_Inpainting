import torch
from src.save.save_plot import scale_back_numpy
from src.config import cfg
import os
from datetime import datetime
import numpy as np
from pathlib import Path
import pandas as pd

from src.data.read_data import read_raw_data
from src.data.read_hyras_data import load_hyras
from src.data.read_data import read_raw_data

import numpy as np
from src.data.read_hyras_data import load_hyras
from src.config import cfg

import xarray as xr
from scipy.spatial import cKDTree
import torch

def prepare_evaluation(years, mode, timesteps, filippou=False, timesteps_offset=10, daily_aggregate_mode=False, multiple_mode=False):  # Often, first few radar samples are nan for some reason, therefore introduced offset 
    data = cfg.station_data(years, mode=mode, filippou=filippou)
    station_samples, radar_samples, timestamps = data[:] if timesteps is None else data[timesteps_offset:timesteps_offset+timesteps]
    inpainted_path_ending = f'{mode}_inpainted_stations_{years[0]}_{years[-1]}{"" if timesteps is None else "_ts"+str(timesteps)}{"_filippou" if filippou else ""}.pt'
    data_path = cfg.output_cache_path
    inpainted_files = [f for f in os.listdir(data_path) if f.endswith(inpainted_path_ending)]
    if not inpainted_files:
        raise NotImplementedError('Not implemented yet :/')
    else: 
        latest_file = max(inpainted_files, key=lambda f: os.path.getmtime(os.path.join(data_path, f)))
        if daily_aggregate_mode:
            latest_file = 'Jan15_1111_1.06977_hourly_inpainted_stations_2018_2018.pt'  # 24 times hourly to day
        if multiple_mode:
            if mode=='daily':
                latest_file = 'Jan15_1707_0.381863_daily_inpainted_stations_2018_2018.pt'  # Daily 10 times same timestamp
            else:
                if filippou:
                    latest_file = 'Jan15_1656_0.97097_hourly_inpainted_stations_2018_2018_filippou.pt'  # With aggregated values
                    #latest_file = 'Jan16_0837_1.01311_hourly_inpainted_stations_2018_2018.pt'  # Without aggregate

        #latest_file = 'Jan27_1647_0.654592_hourly_inpainted_stations_2018_2018_filippou.pt'
        latest_file = 'Jan12_0931_1.18552_hourly_inpainted_stations_2018_2018_filippou.pt'

        inpainted_path = os.path.join(data_path, latest_file)
        inpainted = torch.load(inpainted_path)
        ts_path = inpainted_path.replace(".pt", "_timestamps.csv")
        timestamps = pd.read_csv(ts_path, header=None).iloc[:, 0]
        timestamps = pd.to_datetime(timestamps)
        if mode=='hourly':
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
        print(f"[Prepare Evaluation] Loaded inpainted data from {latest_file}")
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
    if daily_aggregate_mode or mode=='daily':
        hyras = load_hyras(years=years)

        if daily_aggregate_mode:
            timestamps_daily = timestamps.iloc[::24].dt.normalize() + pd.Timedelta(hours=6)
            hyras_time = pd.to_datetime(hyras.time.values)#
            hyras_day_indices = hyras_time.get_indexer(timestamps_daily)

            if (hyras_day_indices < 0).any():
                missing = timestamps_daily[hyras_day_indices < 0]
                raise ValueError(
                    f"{len(missing)} HYRAS days not found.\n"
                    f"First missing:\n{missing[:5]}"
                )

            hyras_subset = hyras.isel(time=hyras_day_indices)

            hyras = hyras_subset
            print(f'Hyras new shape {hyras_subset.shape}')

        hyras = torch.from_numpy(hyras.values)
        hyras = hyras if timesteps is None else hyras[timesteps_offset:timesteps_offset+timesteps]
        hyras_np = [x.squeeze().cpu().numpy() for x in hyras]
        hyras = np.array(hyras_np)
    
    if daily_aggregate_mode:
        T, Y, X = inpainted.shape
        assert T % 24 == 0
        n_days = T // 24
        inpainted = inpainted.reshape(n_days, 24, Y, X).sum(axis=1)
        station = station.reshape(n_days, 24, Y, X).sum(axis=1)
        radar = radar.reshape(n_days, 24, Y, X).sum(axis=1)
        print("First inpainted day:", timestamps_daily.iloc[0])
        print("First HYRAS day:", hyras_subset.time.values[0])

        assert len(inpainted) == hyras_subset.sizes["time"]

        timestamps = timestamps.iloc[::24].reset_index(drop=True)

        print(f'Final Timestamps= {timestamps}')

    if multiple_mode and inpainted.shape!=radar.shape:
        radar = radar[::10][:-1]
        station = station [::10][:-1]
        if hyras is not None:
            hyras = hyras[::10][:-1]


    save_dir_name = f"final_{date_str}_{mode}_{years[0]}_{timesteps}_filippou{filippou}"
    save_dir = Path(cfg.output_cache_path) / save_dir_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f'Radar shape = {radar.shape}')
    print(f'Station shape = {station.shape}')
    print(f'Inpainted shape = {inpainted.shape}')

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

def transfer_to_hyras(years,
                      data,
                      hyras=None,
                      do_mask=True,
                      data_lon=None,
                      data_lat=None,
                      return_hyras=False,
                      limit_to_values = True,
                      return_xarray=True):
    if hyras is None:
        hyras = load_hyras(years=years)

    # Use first timestep only (static grid + mask)
    hyras0 = hyras.isel(time=0)

    # Source lon/lat
    if data_lon is None:
        data_lon = data['lon']
    if data_lat is None:
        data_lat = data['lat']

    # Flatten source grid
    data_coords = np.column_stack([
        data_lon.values.ravel(),
        data_lat.values.ravel()
    ])

    # KDTree on source grid
    tree = cKDTree(data_coords)

    # Flatten HYRAS grid (static)
    hyras_coords = np.column_stack([
        hyras0['lon'].values.ravel(),
        hyras0['lat'].values.ravel()
    ])

    # Nearest-neighbour lookup
    _, indices = tree.query(hyras_coords)

    ny, nx = data_lon.shape
    data_y_idx = indices // nx
    data_x_idx = indices % nx

    # Map all timesteps at once
    
    data_values = data.values if limit_to_values else data # (time, y, x)
    data_flat = data_values[:, data_y_idx, data_x_idx]

    hyras_shape = (len(hyras0['y']), len(hyras0['x']))
    data_on_hyras_array = data_flat.reshape(
        (data.shape[0], *hyras_shape) # Old version: (len(data['time']), *hyras_shape)
    )

    if return_xarray:
        # Wrap in xarray
        data_on_hyras = xr.DataArray(
            data_on_hyras_array,
            coords={
                'time': data['time'],
                'y': hyras0['y'],
                'x': hyras0['x'],
            },
            dims=('time', 'y', 'x'),
            name=data.name
        )

        # Apply static Germany mask
        if do_mask:
            germany_mask = hyras0.notnull()
            data_on_hyras = data_on_hyras.where(germany_mask)
    else:
        data_on_hyras = data_on_hyras_array
        if do_mask:
            germany_mask = hyras0.notnull().values
            mask_torch = torch.from_numpy(germany_mask)
            data_on_hyras[:, ~mask_torch] = torch.nan

    print("Mapping to HYRAS complete!")

    if return_hyras:
        return data_on_hyras, hyras0

    return data_on_hyras

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
    daily_aggregate = False
    multiple_mode = False
    
    prepare_evaluation(years=years, mode=mode, timesteps=timesteps, filippou = filippou, daily_aggregate_mode=daily_aggregate, multiple_mode=multiple_mode)