import sys
import os
import joblib
import random

import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import cfg
from src.save.save_plot import plot_random
from src.data.precipitation_dataset import PrecipitationPatchDataset


def read_one_year(reload=False, time_slices=cfg.default, scaler=None,
              patch_size=128, min_coverage=0.0,
              cache_folder="cache", year=2020):
    """
    Load a PrecipitationPatchDataset from cache, or rebuild and cache it.
    """
    if time_slices==cfg.default:
        time_slices=cfg.time_slices
    cache_path = f'{cache_folder}/precip_dataset_{year}_{patch_size}_{time_slices}_{int(min_coverage * 100)}%.pkl'
    cache_available = os.path.exists(cache_path)

    if reload or not cache_available:
        path = f"data/pr_RADKLIM-1km_v1.0_{year}0101_{year}1231.nc"
        all_data = xr.open_mfdataset(path)
        """nc_files = glob.glob("data/*.nc") # <-- collects all .nc files in the folder 
        
        all_data = xr.open_mfdataset(nc_files,
                                        combine="nested", 
                                        concat_dim="time", 
                                        data_vars="minimal", # don’t try to align all variables
                                        coords="minimal", # don’t try to align all coords
                                        compat="override", # allow overwriting attributes
                                        chunks={"time": 100})"""



        data_array = all_data["RR"].transpose("time", "x", "y").reset_coords(drop=True)
        data_array = data_array.isel(time=slice(0, time_slices))

        ds = PrecipitationPatchDataset(data_array,
                                       patch_size=patch_size,
                                       stride=patch_size,
                                       augment=True,
                                       scaler=scaler,
                                       min_coverage=min_coverage)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            joblib.dump(ds, f)

        print(f"\n[Cache] Saved dataset to {cache_path}\n")
        return ds

    else:
        with open(cache_path, "rb") as f:
            ds = joblib.load(f)
            print(f"\n[Cache] Loaded dataset from {cache_path}\n")         
              
        return ds


def read_data(reload=False, scaler=None, patch_size=cfg.patch_size, min_coverage=cfg.min_coverage,
                   cache_folder="cache", years=range(2001, 2018), reload_years=False):
    time_slices=cfg.time_slices
    cache_path = f'{cache_folder}/precip_dataset_{years[0]}-{years[-1]}_{patch_size}_{time_slices}_{int(min_coverage * 100)}%.pkl'
    cache_available = os.path.exists(cache_path)

    if reload or not cache_available:
        datasets = []
        for year in years:
            print(f'Loading year {year}')
            ds_year = read_one_year(reload=reload_years, time_slices=cfg.default, scaler=None, patch_size=patch_size,
                                cache_folder=cache_folder, year=year, min_coverage=min_coverage)
            datasets.append(ds_year)

        ds_all = PrecipitationPatchDataset.concat(*datasets)
        print(f"\n[Info] Combined dataset from years {years.start}-{years.stop - 1}, total patches = {len(ds_all)}\n")

        with open(cache_path, "wb") as f:
            joblib.dump(ds_all, f)

            print(f"\n[Cache] Saved dataset to {cache_path}\n")

    else:
        with open(cache_path, "rb") as f:
            ds_all = joblib.load(f)
            print(f"\n[Cache] Loaded dataset from {cache_path}\n")  
    return ds_all
        
if __name__=='__main__': 
    import src.data.read_data as this_module
    sys.modules["src.data.read_data"] = this_module
    ds = read_data(reload=True, reload_years=False, scaler=cfg.scaler(), patch_size=cfg.patch_size, min_coverage=cfg.min_coverage)
    print(f'\nDataset shape: {ds.data.shape}\n')
    plot_random(ds)
