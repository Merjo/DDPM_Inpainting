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
from src.evaluate.prepare_evaluation import load_final


def evaluate_fields(gt, pred, data_range=None):
    """
    Evaluate pred against gt for a single timestep.
    NaNs in either array are ignored.

    gt, pred: 2D numpy arrays (H, W)
    data_range: required for SSIM if using float data
    """

    mask = np.isfinite(gt) & np.isfinite(pred)

    if mask.sum() == 0:
        return {
            "rmse": np.nan,
            "bias": np.nan,
            "corr": np.nan,
            "ssim": np.nan,
        }

    gt_v = gt[mask]
    pred_v = pred[mask]

    # RMSE
    rmse = np.sqrt(np.mean((pred_v - gt_v) ** 2))

    # Bias
    bias = np.mean(pred_v - gt_v)

    # Correlation
    if gt_v.size < 2:
        corr = np.nan
    else:
        corr = np.corrcoef(gt_v, pred_v)[0, 1]

    # SSIM (needs full 2D fields)
    # Fill invalid pixels with mean of GT (neutral choice)
    gt_f = gt.copy()
    pred_f = pred.copy()

    fill = np.nanmean(gt_v)
    gt_f[~mask] = fill
    pred_f[~mask] = fill

    ssim_val = ssim(
        gt_f,
        pred_f,
        data_range=data_range if data_range is not None else gt_v.max() - gt_v.min(),
    )

    return {
        "rmse": rmse,
        "bias": bias,
        "corr": corr,
        "ssim": ssim_val,
    }

def evaluate_timeseries(gt_list, pred_list, data_range=None):
    """
    gt_list, pred_list: lists of 2D numpy arrays (length T)
    """
    metrics = {
        "rmse": [],
        "bias": [],
        "corr": [],
        "ssim": [],
    }

    for gt, pred in zip(gt_list, pred_list):
        m = evaluate_fields(gt, pred, data_range=data_range)
        for k in metrics:
            metrics[k].append(m[k])

    return metrics


if __name__ == "__main__":
    #folder_name = 'final_Jan12_1221_hourly_2018_None_filippouFalse'
    folder_name = 'final_Jan12_1252_daily_2018_4_filippouFalse'
    final_dir = f'{cfg.output_cache_path}/{folder_name}'
    hyras = True
    if hyras:
        radar, station, inpainted, timestamps, hyras = load_final(final_dir, hyras=True, listify=True)
    else:
        radar, station, inpainted, timestamps = load_final(final_dir, hyras=False, listify=True)
        hyras = None

    test = evaluate_timeseries(radar, inpainted)
    print(test)