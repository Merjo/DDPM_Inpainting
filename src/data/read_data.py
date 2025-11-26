import os
import joblib

import xarray as xr

from src.config import cfg
from src.save.save_plot import plot_random
from src.data.precipitation_dataset import PrecipitationPatchDataset
from src.data.precipitation_patch_diffusion_dataset import PrecipitationPatchDiffusionDataset


def read_one_year(reload=False, scaler=cfg.default,
              patch_size=cfg.patch_size, min_coverage=cfg.min_coverage,
              cache_folder="cache", year=2020):
    """
    Load a PrecipitationPatchDataset from cache, or rebuild and cache it.
    """
    
    if cfg.do_patch_diffusion:
        #cache_path = f'{cache_folder}/precip_patch_dataset_{year}_isp{cfg.do_importance_sampling}_dna{cfg.drop_na}_ag{cfg.augment}_multi_patch.pkl'
        if cfg.do_patch_diffusion:
            cache_path = f"{cache_folder}/precip_patch_dataset_{year}" \
                        f"{'_isp' if cfg.do_importance_sampling else ''}" \
                        f"{'_dna' if cfg.drop_na else ''}" \
                        f"{'_ag' if cfg.augment else ''}_multi_patch.pkl"

    else:
        cache_path = f'{cache_folder}/precip_dataset_{year}_{patch_size}_{8760}_{int(min_coverage * 100)}%.pkl'
    cache_available = os.path.exists(cache_path)

    if reload or not cache_available:
        path = f"../../../../p/tmp/merlinho/diffusion_data/data/pr_RADKLIM-1km_v1.0_{year}0101_{year}1231.nc"
        all_data = xr.open_mfdataset(path)

        data_array = all_data["RR"].transpose("time", "x", "y").reset_coords(drop=True)

        #data_array = data_array.isel(time=slice(0, 100)) # TODO DELETE
        #print('\n\n[Info] Only selected first 100 timesteps for debugging purposes.\n\n') # TODO DELETE


        if cfg.do_patch_diffusion:
            ds = PrecipitationPatchDiffusionDataset(data_array, scaler=scaler)
        else:
            ds = PrecipitationPatchDataset(data_array, scaler=scaler)



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
                   cache_folder="cache", years=cfg.years, reload_years=False):
    if cfg.do_patch_diffusion:
        #cache_path = f'{cache_folder}/precip_patch_dataset_{years[0]}-{years[-1]}_isp{cfg.do_importance_sampling}_dna{cfg.drop_na}_ag{cfg.augment}_multi_patch.pkl'
        cache_path = f"{cache_folder}/precip_patch_dataset_{years[0]}-{years[-1]}" \
                    f"{'_isp' if cfg.do_importance_sampling else ''}" \
                    f"{'_dna' if cfg.drop_na else ''}" \
                    f"{'_ag' if cfg.augment else ''}_multi_patch.pkl"

    else:
        cache_path = f'{cache_folder}/precip_dataset_{years[0]}-{years[-1]}_{patch_size}_{8760}_{int(min_coverage * 100)}%.pkl'
    cache_available = os.path.exists(cache_path)

    if reload or not cache_available:
        datasets = []
        for year in years:
            print(f'Loading year {year}')
            ds_year = read_one_year(reload=reload_years, scaler=scaler, patch_size=patch_size,
                                cache_folder=cache_folder, year=year, min_coverage=min_coverage)
            datasets.append(ds_year)

        if cfg.do_patch_diffusion:
            ds_all = PrecipitationPatchDiffusionDataset.concat(*datasets)
        else:
            ds_all = PrecipitationPatchDataset.concat(*datasets)
        print(f"\n[Info] Combined dataset from years {years.start}-{years.stop - 1}, total patches = {len(ds_all)}\n")

        import io
        buffer = io.BytesIO()
        joblib.dump(ds_all, buffer, compress=0)
        size_mb = buffer.tell() / 1e6
        print(f"[Info] Estimated joblib dump size: {size_mb:.2f} MB")

        with open(cache_path, "wb") as f:
            joblib.dump(ds_all, f)

        print(f"\n[Cache] Saved dataset to {cache_path}\n")

    else:
        with open(cache_path, "rb") as f:
            ds_all = joblib.load(f)
            print(f"\n[Cache] Loaded dataset from {cache_path}\n")  
    return ds_all
        
if __name__=='__main__': 
    ds = read_data(reload=cfg.reload, reload_years=cfg.reload, scaler=cfg.scaler_class(), patch_size=cfg.patch_size, min_coverage=cfg.min_coverage, years=cfg.years)
    print(f'\nDataset shapes: {[dataset.shape for dataset in ds.data]}\n')
    cfg.update_output_path('random')
    plot_random(ds, n=32, title='Real samples')
