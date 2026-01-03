import numpy as np
from src.data.read_data import read_raw_data
from src.data.read_filippou import read_filippou
from src.data.read_hyras_data import load_hyras
from src.config import cfg

import xarray as xr
from scipy.spatial import cKDTree

import matplotlib.pyplot as plt

def plot_comparison(data1_on_hyras, data2_on_hyras, data2_original, n=12, transpose_data2 = True, data1_name = 'HYRAS', data2_name='Radar'):
    
    time_len = len(data1_on_hyras['time'])
    time_indices = np.random.choice(time_len, size=n, replace=False)
    
    fig, axes = plt.subplots(n, 3, figsize=(18, 6*n))
    
    if n == 1:
        axes = axes[np.newaxis, :]  # ensure 2D indexing
    
    for i, t_idx in enumerate(time_indices):
        # HYRAS
        im0 = axes[i, 0].imshow(data1_on_hyras.isel(time=t_idx), origin='lower', cmap='viridis')
        axes[i, 0].set_title(f'{data1_name} (on HYRAS) (time={t_idx})')
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
        
        # Radar on HYRAS
        im1 = axes[i, 1].imshow(data2_on_hyras.isel(time=t_idx), origin='lower', cmap='viridis')
        axes[i, 1].set_title(f'{data2_name} on HYRAS (time={t_idx})')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        # Original radar
        # Resample radar time if radar has more/less timesteps than HYRAS
        data2_time_idx = min(t_idx, len(data2_original['time'])-1)
        data2_patch = data2_original.isel(time=data2_time_idx)
        if transpose_data2:
            data2_patch = data2_patch.T
        im2 = axes[i, 2].imshow(data2_patch, origin='lower', cmap='viridis')
        axes[i, 2].set_title(f'Original {data2_name} (time={data2_time_idx})')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
    
    #plt.tight_layout()
    path = f'output_new/random/plot_{data1_name.lower()}_comparison_{data2_name.lower()}.png'
    plt.savefig(path)
    print(f'[Plot Comparison] plotted comparison to {path}')

def transfer_to_hyras(years,
                      data,
                      hyras=None,
                      do_mask=True,
                      data_lon=None,
                      data_lat=None,
                      return_hyras=False):
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
    data_values = data.values  # (time, y, x)
    data_flat = data_values[:, data_y_idx, data_x_idx]

    hyras_shape = (len(hyras0['y']), len(hyras0['x']))
    data_on_hyras_array = data_flat.reshape(
        (len(data['time']), *hyras_shape)
    )

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

    print("Mapping to HYRAS complete!")

    if return_hyras:
        return data_on_hyras, hyras0

    return data_on_hyras

def save_output(data, fname):
    if data.name is None:
        data = data.rename("data")
    path = f'{cfg.output_cache_path}/{fname}'
    if path[-3:]!='.nc':
        path = f'{path}.nc'
    data.to_netcdf(path, mode='w')
    print(f'[Save Output] Saved dataset to {path}')
    return data

def load_output(fname):
    path = f'{cfg.output_cache_path}/{fname}'
    if path[-3:]!='.nc':
        path = f'{path}.nc'
    ds = xr.open_dataset(path)
    data = ds[list(ds.data_vars)[0]]  # take the first variable

    print(f'[Load Output] Loaded dataset from {path}')
    return data


def prepare_hyras_validation(years=cfg.test_years):
    radar = read_raw_data(years, aggregate_daily=True, delete_coords=False, time_slices=None)
    hyras = load_hyras(years=years)
    radar_on_hyras = transfer_to_hyras(years, radar, hyras)
    plot_comparison(hyras, radar_on_hyras, radar, n=5)

    save_output(radar_on_hyras, fname=f'radar_on_hyras_daily_{years[0]}_{years[-1]}.nc')
    save_output(hyras, fname=f'hyras_{years[0]}_{years[-1]}.nc')

    return

def prepare_filippou_validation(years=cfg.test_years):
    filippou = read_filippou(years)
    hyras = load_hyras(years=years)
    filippou_on_hyras = transfer_to_hyras(years, filippou, hyras)
    radar_hourly = read_raw_data(years, aggregate_daily=False, delete_coords=False, time_slices=None)
    radar_on_hyras_hourly = transfer_to_hyras(years, radar_hourly, hyras)

    plot_comparison(radar_on_hyras_hourly, filippou_on_hyras, filippou, transpose_data2=False, data1_name='Radar', data2_name='Filippou')

    filippou_on_hyras_daily = filippou_on_hyras.resample(time="1D").sum(skipna=False)
    filippou_daily = filippou.resample(time="1D").sum(skipna=False)

    plot_comparison(hyras, filippou_on_hyras_daily, filippou_daily, transpose_data2=False, data1_name='HYRAS', data2_name='filippou')

    save_output(filippou_on_hyras, fname=f'filippou_on_hyras_{years[0]}_{years[-1]}.nc')
    save_output(radar_on_hyras_hourly, fname=f'radar_on_hyras_hourly_{years[0]}_{years[-1]}.nc')
    save_output(filippou_on_hyras_daily, fname=f'filippou_on_hyras_daily_{years[0]}_{years[-1]}.nc')

    #save_output(hyras, fname=f'hyras_{years[0]}_{years[-1]}.nc')

    return

if __name__=='__main__':
    #prepare_hyras_validation()
    prepare_filippou_validation()
