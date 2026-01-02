import os
import io
import joblib

import xarray as xr

from src.config import cfg
from src.save.save_plot import plot_random
from src.data.multi_patch_dataset import MultiPatchDataset


def read_raw_data(years, 
                  aggregate_daily = cfg.model_type=='daily', 
                  delete_coords=True,
                  time_slices = cfg.time_slices,
                  check_negative = True):
    # if years is int -> make it a list
    if isinstance(years, int):
        years = [years]
    paths = [f"../../../../p/tmp/merlinho/diffusion_data/data/pr_RADKLIM-1km_v1.0_{year}0101_{year}1231.nc" for year in years]
    all_data = xr.open_mfdataset(paths, parallel=False, engine="netcdf4")
    data_array = all_data['RR']

    data_array = data_array.transpose("time", "x", "y")

    if delete_coords:
        data_array = data_array.reset_coords(drop=True)

    if aggregate_daily:
        data_array = data_array.resample(time="1D").sum(skipna=False)

    if time_slices is not None:
        data_array = data_array.isel(time=slice(0, cfg.time_slices)) 

    if check_negative:
        # check for negative values
        if (data_array < 0).any():
            data_array = data_array.where(data_array >= 0)
            raise RuntimeWarning(f"Data in years {years} contains negative precipitation values, replaced with NA!")

    return data_array


def read_one_year(reload=False, scaler=None,
              patch_size=cfg.patch_size, min_coverage=cfg.min_coverage,
              cache_folder=cfg.cache_path, year=2020, return_importance_prob=False):
    """
    Load a PrecipitationPatchDataset from cache, or rebuild and cache it.
    """
    if scaler is None:
        scaler = cfg.scaler
    
    if True or cfg.do_patch_diffusion:
        ps_string = "_".join(str(x) for x in cfg.isp_patch_sizes)
        shares_string = "_".join(str(int(s * 100)) for s in cfg.isp_shares)
        cache_path = f"{cache_folder}/precip_patch_dataset_{cfg.model_type}_{year}_{ps_string}_{shares_string}" \
                    f"{f'_isp_{cfg.isp_s}_{cfg.isp_m}_{cfg.isp_q_min}' if cfg.do_importance_sampling else ''}" \
                    f"{'_dna' if cfg.drop_na else ''}" \
                    f"{'_pd' if cfg.do_patch_diffusion else ''}" \
                    f"{'_prop' if cfg.constrain_proportions else ''}" \
                    f"{'_qn' if return_importance_prob else ''}" \
                    f"{'_lim1024' if cfg.do_limit_1024 else ''}" \
                    f"_{cfg.stride}" \
                    f"{'_ag' if cfg.augment else ''}_multi_patch{'' if cfg.time_slices is None else f'_{cfg.time_slices}'}.pkl"

    else:
        cache_path = f'{cache_folder}/precip_dataset_{year}_{patch_size}_8760_{int(min_coverage * 100)}%.pkl'
    cache_available = os.path.exists(cache_path)

    if reload or not cache_available:
        data_array = read_raw_data(years=year)

        if cfg.do_patch_diffusion:
            ds = MultiPatchDataset(data_array, 
                                    scaler=scaler,
                                    patch_sizes=cfg.isp_patch_sizes,
                                    shares=cfg.isp_shares,
                                    do_drop_nan=cfg.drop_na,
                                    min_coverage=None if cfg.do_importance_sampling else cfg.min_coverage,
                                    return_importance_prob=return_importance_prob)
        else:
            ds = MultiPatchDataset(data_array, 
                                    scaler=scaler, 
                                    patch_sizes=[cfg.patch_size], 
                                    shares=[1], 
                                    do_drop_nan=cfg.drop_na, 
                                    min_coverage=None if cfg.do_importance_sampling else cfg.min_coverage,
                                    return_importance_prob=return_importance_prob)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            joblib.dump(ds, f)

        print(f"\n[Cache] Saved dataset to {cache_path}\n")
        return ds

    else:
        with open(cache_path, "rb") as f:
            ds = joblib.load(f)
            print(f"\n[Cache] Loaded datasets from {cache_path}\n")         
              
        return ds


def read_data(reload=False, scaler=None, patch_size=cfg.patch_size, min_coverage=cfg.min_coverage,
                   cache_folder=cfg.cache_path, years=cfg.years, reload_years=False, return_importance_prob=False):
    if True or cfg.do_patch_diffusion:
        #cache_path = f'{cache_folder}/precip_patch_dataset_{years[0]}-{years[-1]}_isp{cfg.do_importance_sampling}_dna{cfg.drop_na}_ag{cfg.augment}_multi_patch.pkl'
        ps_string = "_".join(str(x) for x in cfg.isp_patch_sizes)
        shares_string = "_".join(str(int(s * 100)) for s in cfg.isp_shares)
        cache_path = f"{cache_folder}/precip_patch_dataset_{cfg.model_type}_{years[0]}-{years[-1]}_{ps_string}_{shares_string}" \
                        f"{f'_isp_{cfg.isp_s}_{cfg.isp_m}_{cfg.isp_q_min}' if cfg.do_importance_sampling else f'mc{int(cfg.min_coverage*100)}'}" \
                        f"{'_dna' if cfg.drop_na else ''}" \
                        f"{'_pd' if cfg.do_patch_diffusion else ''}" \
                        f"{'_prop' if cfg.constrain_proportions else ''}" \
                        f"{'_qn' if return_importance_prob else '_'}" \
                        f"{'_lim1024' if cfg.do_limit_1024 else ''}" \
                        f"_{cfg.stride}" \
                        f"{'_ag' if cfg.augment else ''}_multi_patch{'' if cfg.time_slices is None else f'_{cfg.time_slices}'}.pkl"

    else:
        cache_path = f'{cache_folder}/precip_dataset_{years[0]}-{years[-1]}_{patch_size}_{8760}_{int(min_coverage * 100)}%.pkl'
    cache_available = os.path.exists(cache_path)

    if scaler is None:
        scaler = cfg.scaler

    if reload or not cache_available:
        ds_years = []
        for year in years:
            print(f'Loading year {year}')
            ds_year = read_one_year(reload=reload_years, scaler=scaler, patch_size=patch_size,
                                cache_folder=cache_folder, year=year, min_coverage=min_coverage, return_importance_prob=return_importance_prob)
            ds_years.append(ds_year)
        
        data = MultiPatchDataset.concat(ds_years)

        ds_lengths = [len(ds) for ds in data.datasets]
        print(f"\n[Info] Combined dataset from years {years.start}-{years.stop - 1}, total patches = {sum(ds_lengths)} ({ds_lengths})\n")

        buffer = io.BytesIO()
        joblib.dump(data, buffer, compress=0)
        size_mb = buffer.tell() / 1e6
        print(f"[Info] Estimated joblib dump size: {size_mb:.2f} MB")

        with open(cache_path, "wb") as f:
            joblib.dump(data, f)

        print(f"\n[Cache] Saved dataset to {cache_path}\n")

    else:
        with open(cache_path, "rb") as f:
            data = joblib.load(f)
            print(f"\n[Cache] Loaded dataset from {cache_path}\n")  
    return data
        
if __name__=='__main__': 
    data = read_data(reload=cfg.reload, reload_years=cfg.reload, scaler=cfg.scaler, patch_size=cfg.patch_size, min_coverage=cfg.min_coverage, years=cfg.train_years)
    if cfg.do_patch_diffusion:
        shapes = [tuple(dataset.data.shape) for dataset in data.datasets]
        print(f'\nDataset shapes: {shapes}\n')
    cfg.update_output_path('random')
    plot_random(data, n=32, title='Real samples')
