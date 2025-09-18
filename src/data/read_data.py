import sys
import os
import pickle
import random

import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import cfg
from src.visualization.save_plot import plot_random


class PrecipitationPatchDataset(Dataset):
    """
    Precompute precipitation patches from a DataArray and store them in memory.
    - Filters NaN/all-zero patches once.
    - Optionally applies augmentation (flip/rotate) once during preprocessing.
    - Optionally stores scaled version of patches.
    - __getitem__ simply returns a patch (scaled or raw).
    """

    def __init__(self, data_array,
                 patch_size=128, stride=128,
                 augment=True, scaler=cfg.default,
                 do_drop_nan=True, do_drop_zero=True):
        if scaler==cfg.default:
            scaler=cfg.scaler()

        if data_array is None:
            raise ValueError("data_array must be provided")

        self.ps = patch_size
        self.stride = stride
        self.augment = augment
        self.scaler = scaler
        self.use_scaled = scaler is not None  # default: use scaled data if available

        self.do_drop_nan = do_drop_nan
        self.do_drop_zero = do_drop_zero

        self.T = data_array.sizes["time"]
        self.X = data_array.sizes["x"]
        self.Y = data_array.sizes["y"]

        self.data_raw = self._create_patches(data_array)
        self.data_scaled = None

        if self.use_scaled:
            # Fit scaler on dataset and transform
            self.scaler.fit(dataset=self.data_raw)
            self.data_scaled = self.scaler.encode(self.data_raw)
        
        self.data = self.data_scaled if self.use_scaled else self.data_raw
    
    def _create_patches(self, data_array):
        patches = []
        dropped_nan = 0
        dropped_zero = 0

        xs = list(range(0, self.X - self.ps + 1, self.stride))
        ys = list(range(0, self.Y - self.ps + 1, self.stride))

        for t in range(self.T):
            for x0 in xs:
                for y0 in ys:
                    patch = data_array.isel(
                        time=t, x=slice(x0, x0+self.ps), y=slice(y0, y0+self.ps)
                    ).values.astype("float32")

                    if self.do_drop_nan and np.isnan(patch).any():
                        dropped_nan += 1
                        continue
                    elif self.do_drop_zero and np.all(patch == 0):
                        dropped_zero += 1
                        continue

                    patch = torch.from_numpy(patch).unsqueeze(0)  # [1,H,W]

                    if self.augment:
                        patch = self._augment_patch(patch)

                    patches.append(patch)
        
        self.dropped_nan = dropped_nan
        self.dropped_zero = dropped_zero

        print(f"[Dataset] total potential patches = {len(patches) + dropped_nan + dropped_zero}")
        print(f"[Dataset] total patches kept = {len(patches)}")
        print(f"[Dataset] patches dropped (NaN)  = {self.dropped_nan}")
        print(f"[Dataset] patches dropped (zeros)= {self.dropped_zero}")

        return torch.stack(patches) if patches else torch.empty(0)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def _augment_patch(self, patch):
        """Random flip/rotate a patch (done once during preprocessing)."""
        if torch.rand(()) < 0.5:
            patch = torch.flip(patch, dims=[2])
        if torch.rand(()) < 0.5:
            patch = torch.flip(patch, dims=[1])
        if torch.rand(()) < 0.25:
            patch = patch.rot90(1, dims=[1, 2])
        return patch

    def switch_scaled(self, use_scaled=True):
        """Switch between returning scaled vs raw data in __getitem__."""
        if use_scaled == self.use_scaled:
            print(f'Use scaled was already {self.use_scaled}, no switch needed.')
        else:
            self.data = self.data_scaled if use_scaled else self.data_raw
            self.use_scaled = use_scaled


def read_data(reload=False, time_slices=cfg.default, scaler=None,
              cache_folder="cache"):
    """
    Load a PrecipitationPatchDataset from cache, or rebuild and cache it.
    """
    if time_slices==cfg.default:
        time_slices=cfg.time_slices
    cache_path = f'{cache_folder}/precip_dataset_{time_slices}.pkl'
    cache_available = os.path.exists(cache_path)

    if reload or not cache_available:
        path = "data/pr_RADKLIM-1km_v1.0_20200101_20201231.nc"
        all_data = xr.open_mfdataset(path)
        data_array = all_data["RR"].transpose("time", "x", "y").reset_coords(drop=True)
        data_array = data_array.isel(time=slice(0, time_slices))

        ds = PrecipitationPatchDataset(data_array,
                                       patch_size=128,
                                       stride=128,
                                       augment=True,
                                       scaler=scaler)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(ds, f)

        print(f"\n[Cache] Saved dataset to {cache_path}\n")
        return ds

    else:
        with open(cache_path, "rb") as f:
            ds = pickle.load(f)
            print(f"\n[Cache] Loaded dataset from {cache_path}\n")         
              
        return ds

        
if __name__=='__main__': 
    import src.data.read_data as this_module
    sys.modules["src.data.read_data"] = this_module
    ds = read_data(reload=True, scaler=cfg.scaler()) 
    print(f'\nDataset shape: {ds.data.shape}\n')
    plot_random(ds) 