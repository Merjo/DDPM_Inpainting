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
from src.random.spectra import tqdm, rapsd
from src.evaluate.prepare_evaluation import load_final

from src.random.spectra import mean_rapsd_numpy


def plot_rapsd_radar_inpainted_hyras(
    radar,
    inpainted,
    hyras,
    name,
    title="RAPSD: Radar vs Inpainted vs HYRAS",
    out_dir=None,
    exclude_dc=False
):
    """
    Plot radially averaged power spectral density (RAPSD)
    for Radar, Inpainted, and HYRAS fields.
    """


    # === Stack to [T, H, W] ===
    radar_arr = filter_valid_timesteps(
        np.stack([r.squeeze() for r in radar])
    )
    inpainted_arr = filter_valid_timesteps(
        np.stack([i.squeeze() for i in inpainted])
    )
    hyras_arr = filter_valid_timesteps(
        np.stack([h.squeeze() for h in hyras])
    )

    print(
        f"[RAPSD] Valid timesteps — "
        f"Radar: {radar_arr.shape[0]}, "
        f"Inpainted: {inpainted_arr.shape[0]}, "
        f"HYRAS: {hyras_arr.shape[0]}"
    )

    radar_arr = np.stack([r.squeeze() for r in radar_arr])
    inpainted_arr = np.stack([i.squeeze() for i in inpainted_arr])
    hyras_arr = np.stack([h.squeeze() for h in hyras_arr])


    # === Compute mean RAPSD ===
    radar_psd, freq = mean_rapsd_numpy(radar_arr)
    inpainted_psd, _ = mean_rapsd_numpy(inpainted_arr)
    hyras_psd, _ = mean_rapsd_numpy(hyras_arr)

    # Normalize for plotting (exclude freq=0)
    def normalize_psd_skip_dc(psd):
        psd_norm = psd.copy()
        if exclude_dc:
            total = np.sum(psd[1:])  # exclude DC
            if total > 0:
                psd_norm[1:] /= total
        else:
            total = np.sum(psd)
            if total > 0:
                psd_norm /= total
        return psd_norm


    if exclude_dc:
        radar_psd = normalize_psd_skip_dc(radar_psd)[1:]
        inpainted_psd = normalize_psd_skip_dc(inpainted_psd)[1:]
        hyras_psd = normalize_psd_skip_dc(hyras_psd)[1:]
        freq = freq[1:]
    else:
        radar_psd = normalize_psd_skip_dc(radar_psd)
        inpainted_psd = normalize_psd_skip_dc(inpainted_psd)
        hyras_psd = normalize_psd_skip_dc(hyras_psd)

    print("Radar area:", np.trapz(radar_psd, freq))
    print("Inp area:", np.trapz(inpainted_psd, freq))
    print("HYRAS area:", np.trapz(hyras_psd, freq))

    print("Radar sum:", np.nansum(radar_psd))
    print("Inp sum:", np.nansum(inpainted_psd))
    print("HYRAS sum:", np.nansum(hyras_psd))

    #print(f'\nArea difference (Radar vs Inpainted): {np.trapz(np.abs(radar_psd - inpainted_psd), freq)}')
    #print(f'Area difference (Radar vs HYRAS): {np.trapz(np.abs(radar_psd - hyras_psd), freq)}\n')
    print(f'Sum difference (Radar vs Inpainted): {np.nansum(np.abs(radar_psd - inpainted_psd))}')
    print(f'Sum difference (Radar vs HYRAS): {np.nansum(np.abs(radar_psd - hyras_psd))}\n')

    # === Output directory ===
    if out_dir is None:
        out_dir = "rapsd"
    os.makedirs(out_dir, exist_ok=True)

    existing = [
        f for f in os.listdir(out_dir)
        if f.startswith("rapsd_radar_inpainted_hyras") and f.endswith(".png")
    ]
    nums = [
        int(f.split("_")[-1].replace(".png", ""))
        for f in existing
        if f.split("_")[-1].replace(".png", "").isdigit()
    ]
    next_num = max(nums, default=0) + 1

    filename = os.path.join(
        out_dir, f"rapsd_radar_inpainted_hyras_{name}_{next_num}.png"
    )


    # === Plot ===
    plt.figure(figsize=(7, 5))

    plotter = plt.loglog if exclude_dc else plt.loglog

    plotter(freq, radar_psd, label="Radar", color="black", linewidth=2)
    plotter(freq, inpainted_psd, label="Inpainted", color="tab:blue", linewidth=2)
    plotter(
        freq,
        hyras_psd,
        label="HYRAS",
        color="tab:orange",
        #linestyle=":",
        linewidth=2,
    )

    plt.xlabel("Spatial frequency")
    plt.ylabel("Power spectral density")
    plt.title(f"{title}\n({datetime.now().strftime('%m-%d %H:%M')})")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"[Plot] Saved RAPSD comparison to {filename}")



def filter_valid_timesteps(arr):
    """
    Keep only timesteps with at least one finite value.
    arr: [T, H, W]
    """
    mask = np.any(np.isfinite(arr), axis=(1, 2))
    return arr[mask]



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
        #win_size=21
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

def plot_metrics_by_time_and_month(evaluation, timestamps, radar, name):
    """
    Plots average evaluation metrics by hour of day and month,
    with mean radar rain intensity as secondary axis.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # Compute rain intensity
    rain_intensity = compute_mean_intensity(radar)

    # Build DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'rmse': evaluation['rmse'],
        'bias': evaluation['bias'],
        'corr': evaluation['corr'],
        'ssim': evaluation['ssim'],
        'rain': rain_intensity,
    })

    df['hour'] = df['timestamp'].dt.hour
    df['month'] = df['timestamp'].dt.month

    metrics = ['rmse', 'bias', 'corr', 'ssim']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    # ---------- Hour of day ----------
    fig, ax1 = plt.subplots(figsize=(10, 5))

    for m, c in zip(metrics, colors):
        mean_hour = df.groupby('hour')[m].mean()
        label_name = m.upper()
        if label_name == 'CORR':
            label_name = 'Correlation'
        elif label_name == 'BIAS':
            label_name = 'Bias'
        ax1.plot(mean_hour.index, mean_hour.values, label=label_name, color=c, marker='o')

    ax1.set_xlabel('Hour of day')
    ax1.set_ylabel('Metric value')
    ax1.grid(True)

    # Secondary axis: rain intensity
    ax2 = ax1.twinx()
    mean_rain_hour = df.groupby('hour')['rain'].mean()
    ax2.plot(
        mean_rain_hour.index,
        mean_rain_hour.values,
        color='darkgrey',
        linestyle='--',
        linewidth=2,
        label='Mean rain\nintensity'
    )
    ax2.set_ylabel('mm/h (rain intensity)')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, 
               labels1 + labels2, 
               loc='upper right',            # pick corner you want as reference
               bbox_to_anchor=(1, 0.75))

    plt.title('Model Performance by Time of Day')
    plt.tight_layout()
    plt.savefig(f'evaluation_metrics_by_hour_{name}.png')
    plt.close()

    print(f'Saved evaluation metrics by hour of day to evaluation_metrics_by_hour_{name}.png')

    # ---------- Month ----------
    fig, ax1 = plt.subplots(figsize=(10, 5))

    for m, c in zip(metrics, colors):
        mean_month = df.groupby('month')[m].mean()
        label_name = m.upper()
        if label_name == 'CORR':
            label_name = 'Correlation'
        elif label_name == 'BIAS':
            label_name = 'Bias'
        ax1.plot(mean_month.index, mean_month.values, label=label_name, color=c, marker='o')

    ax1.set_xlabel('Month')
    ax1.set_ylabel('Metric value')
    ax1.grid(True)

    ax2 = ax1.twinx()
    mean_rain_month = df.groupby('month')['rain'].mean()
    ax2.plot(
        mean_rain_month.index,
        mean_rain_month.values,
        color='darkgrey',
        linestyle='--',
        linewidth=2,
        label='Mean rain\nintensity'
    )
    ax2.set_ylabel('mm/h (rain intensity)')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, 
               labels1 + labels2, 
               loc='upper right',            # pick corner you want as reference
               bbox_to_anchor=(0.93, 0.95))

    plt.title('Model Performance by Month')
    plt.tight_layout()
    plt.savefig(f'evaluation_metrics_by_month_{name}.png')
    plt.close()

    print(f'Saved evaluation metrics by month to evaluation_metrics_by_month_{name}.png')

def compute_mean_intensity(radar_fields):
    """
    radar_fields: list of 2D numpy arrays
    Returns: 1D numpy array of mean rain intensity per timestep
    """
    intensities = []
    for r in radar_fields:
        if np.isfinite(r).any():
            intensities.append(np.nanmean(r))
        else:
            intensities.append(np.nan)
    return np.array(intensities)


def plot_monthly_comparison_normal_vs_hyras(
    eval_normal,
    eval_hyras,
    timestamps,
    radar,
    out_prefix="evaluation_metrics_by_month"
):
    """
    Monthly plots comparing normal evaluation (solid)
    vs HYRAS evaluation (dashed).
    """

    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    # --- Rain intensity ---
    rain_intensity = compute_mean_intensity(radar)

    # --- Build DataFrame ---
    df = pd.DataFrame({
        "timestamp": timestamps,
        "rmse_n": eval_normal["rmse"],
        "bias_n": eval_normal["bias"],
        "corr_n": eval_normal["corr"],
        "ssim_n": eval_normal["ssim"],
        "rmse_h": eval_hyras["rmse"],
        "bias_h": eval_hyras["bias"],
        "corr_h": eval_hyras["corr"],
        "ssim_h": eval_hyras["ssim"],
        "rain": rain_intensity,
    })

    df["month"] = df["timestamp"].dt.month

    # --- Monthly means ---
    m = df.groupby("month").mean()

    # ==========================================================
    # Plot 1: RMSE + Bias (with rain intensity)
    # ==========================================================
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # RMSE
    ax1.plot(
        m.index, m["rmse_n"],
        color="tab:blue", marker="o",
        label="RMSE (DDPM)"
    )
    ax1.plot(
        m.index, m["rmse_h"],
        color="tab:blue", linestyle="--", marker="o",
        label="RMSE (HYRAS)"
    )

    # Bias
    ax1.plot(
        m.index, m["bias_n"],
        color="gold", marker="o",
        label="Bias (DDPM)"
    )
    ax1.plot(
        m.index, m["bias_h"],
        color="gold", linestyle="--", marker="o",
        label="Bias (HYRAS)"
    )

    ax1.set_xlabel("Month")
    ax1.set_ylabel("mm/day")
    ax1.grid(True)

    # Secondary axis: rain intensity
    #ax2 = ax1.twinx()
    ax1.plot(
        m.index, m["rain"],
        color="darkgrey", linestyle="--", linewidth=2,
        label="Mean rain\nintensity"
    )
    #ax2.set_ylabel("mm (rain intensity)")

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    #lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1, #+ lines2,
        labels1, #+ labels2,
        loc="upper right"
    )

    plt.title("Monthly Performance: RMSE & Bias")
    #plt.tight_layout()
    plt.savefig(f"{out_prefix}_rmse_bias.png")
    plt.close()

    print(f"Saved {out_prefix}_rmse_bias.png")

    # ==========================================================
    # Plot 2: Correlation + SSIM
    # ==========================================================
    fig, ax = plt.subplots(figsize=(10, 5))

    # SSIM
    ax.plot(
        m.index, m["ssim_h"],
        color="tab:red", linestyle="--", marker="o",
        label="SSIM (HYRAS)"
    )

    ax.plot(
        m.index, m["ssim_n"],
        color="tab:red", marker="o",
        label="SSIM (DDPM)"
    )

    # Correlation
    ax.plot(
        m.index, m["corr_h"],
        color="tab:green", linestyle="--", marker="o",
        label="Correlation (HYRAS)"
    )
    ax.plot(
        m.index, m["corr_n"],
        color="tab:green", marker="o",
        label="Correlation (DDPM)"
    )


    ax.set_xlabel("Month")
    ax.set_ylabel("Metric value")
    ax.grid(True)
    ax.legend(loc="best")

    plt.title("Monthly Performance: Correlation & SSIM")
    #plt.tight_layout()
    plt.savefig(f"{out_prefix}_corr_ssim.png")
    plt.close()

    print(f"Saved {out_prefix}_corr_ssim.png")


def rmse_3d(a, b):
    return np.sqrt(np.nanmean((a - b) ** 2))


def bias_3d(a, b):
    return np.nanmean(a - b)


def corr_3d(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if np.sum(mask) == 0:
        return np.nan
    return np.corrcoef(a[mask].ravel(), b[mask].ravel())[0, 1]

def prdiff_3d(output, ground_truth):
    return np.nansum(ground_truth) - np.nansum(output)

if __name__ == "__main__":

    #folder_name = 'final_Jan12_1259_hourly_2018_None_filippouFalse' # Hourly Normal Final
    #folder_name = 'final_Jan26_1602_hourly_2018_None_filippouFalse'  # Hourly Monte Carlo and Hourly24
    
    #folder_name = 'final_Jan13_1546_daily_2018_None_filippouFalse' # Daily Final 181

    #folder_name = 'final_Jan22_0827_daily_2018_None_filippouFalse' # Daily 220
    #folder_name = 'final_Jan22_0833_daily_2018_None_filippouFalse' # Daily 100
    #folder_name = 'final_Jan16_1814_hourly_2018_None_filippouFalse' # Daily Hourly24
    #folder_name = 'final_Jan16_1843_daily_2018_None_filippouFalse' # Daily Monte Carlo


    #folder_name = 'final_Jan26_1558_hourly_2018_None_filippouTrue'  # Filippou Monte Carlo and Hourly24
    #folder_name = 'final_Jan21_1406_hourly_2018_None_filippouTrue' # Filippou Hourly24 Aggregated
    #folder_name = 'final_Jan16_1857_hourly_2018_None_filippouTrue'  # Filippou Monte Carlo Old 
    #folder_name = 'final_Jan27_1738_hourly_2018_None_filippouTrue'  # Filippou Monte Carlo New
    #folder_name = 'final_Jan13_1548_hourly_2018_None_filippouTrue' # Filippou Final
    folder_name = 'final_Jan28_1034_hourly_2018_None_filippouTrue'  # Filippou Final New

    hyras_flag = False
    name = 'filippou'

    final_dir = f'{cfg.output_cache_path}/{folder_name}'


    if hyras_flag:
        radar, station, inpainted, timestamps, hyras = load_final(
            final_dir, hyras=True, listify=False
        )

        # === HYRAS vs RADAR (3D metrics) ===
        rmse_hyras = rmse_3d(hyras, radar)
        bias_hyras = bias_3d(hyras, radar)
        prdiff_hyras = prdiff_3d(hyras, radar)
        corr_hyras = corr_3d(hyras, radar)

        # SSIM: 2D per-timestep → mean
        evaluation_hyras = evaluate_timeseries(radar, hyras)
        ssim_hyras = np.nanmean(evaluation_hyras["ssim"])



        print(
            "HYRAS Evaluation Results:\n"
            f"  RMSE (3D): {rmse_hyras:.4f}\n"
            f"  Corr (3D): {corr_hyras:.4f}\n"
            f"  Bias (3D): {bias_hyras:.4f}\n"
            f"  PRDiff: {prdiff_hyras:.4f}\n"
            f"  SSIM (mean over time): {ssim_hyras:.4f}"

            f"  \nHYRAS RMSE Hourly (3D): {(rmse_hyras)/24:.4f}\n"
            f"  Bias Hourly (3D): {(bias_hyras/24):.4f}\n"
            f"  PRDiff Hourly (3D): {(prdiff_hyras/24):.4f}\n"
        )

    else:
        radar, station, inpainted, timestamps = load_final(
            final_dir, hyras=False, listify=False
        )
        hyras = None


    # === INPAINTED vs RADAR (3D metrics) ===
    rmse_inp = rmse_3d(inpainted, radar)
    bias_inp = bias_3d(inpainted, radar)
    prdiff_inp = prdiff_3d(inpainted, radar)
    corr_inp = corr_3d(inpainted, radar)

    # SSIM again from 2D evaluation
    evaluation = evaluate_timeseries(radar, inpainted)
    ssim_inp = np.nanmean(evaluation["ssim"])

    rmse_t = np.array(evaluation["rmse"])
    bias_t = np.array(evaluation["bias"])
    corr_t = np.array(evaluation["corr"])
    ssim_t = np.array(evaluation["ssim"])

    rmse_mean = np.nanmean(rmse_t)
    bias_mean = np.nanmean(bias_t)
    corr_mean = np.nanmean(corr_t)
    ssim_mean = np.nanmean(ssim_t)    

    rmse_std = np.nanstd(rmse_t)
    bias_std = np.nanstd(bias_t)
    corr_std = np.nanstd(corr_t)
    ssim_std = np.nanstd(ssim_t)

    print(
        "Inpainted Evaluation Results:\n"
        f"  RMSE (3D): {rmse_inp:.4f}\n"
        f"  Corr (3D): {corr_inp:.4f}\n"
        f"  Bias (3D): {bias_inp:.4f}\n"
        f"  PRDiff: {prdiff_inp:.4f}\n"
        f"  SSIM (mean over time): {ssim_inp:.4f}"
    )

    print(
        "Inpainted Evaluation Results (per-timestep std):\n"
        f"  RMSE mean: {rmse_mean:.4f} std: {rmse_std:.4f}\n"
        f"  Corr mean: {corr_mean:.4f} std: {corr_std:.4f}\n"
        f"  Bias mean: {bias_mean:.4f} std: {bias_std:.4f}\n"
        f"  SSIM mean: {ssim_mean:.4f} std: {ssim_std:.4f}\n"
    )

    # === Plotting (still uses per-time metrics) ===
    if hyras is not None:
        print(
            "Inpainted Hourly Evaluation Results:\n"
            f"  RMSE Hourly (3D): {(rmse_inp)/24:.4f}\n"
            f"  Bias Hourly (3D): {(bias_inp/24):.4f}\n"
            f"  PRDiff Hourly: {(prdiff_inp/24):.4f}\n"
        )

        rmse_inp_hyras = rmse_3d(inpainted, hyras)
        bias_inp_hyras = bias_3d(inpainted, hyras)
        prdiff_inp_hyras = prdiff_3d(inpainted, hyras)
        corr_inp_hyras = corr_3d(inpainted, hyras)

        print(
            "Inpainted vs HYRAS Evaluation Results:\n"
            f"  RMSE (3D): {rmse_inp_hyras:.4f}\n"
            f"  Corr (3D): {corr_inp_hyras:.4f}\n"
            f"  Bias (3D): {bias_inp_hyras:.4f}\n"
            f"  PRDiff: {prdiff_inp_hyras:.4f}\n"

            f"  \nInpainted vs HYRAS \nRMSE Hourly (3D): {(rmse_inp_hyras)/24:.4f}\n"
            f"  Bias Hourly (3D): {(bias_inp_hyras/24):.4f}\n"
            f"  PRDiff Hourly (3D): {(prdiff_inp_hyras/24):.4f}\n"
        )

        plot_rapsd_radar_inpainted_hyras(
            radar,
            inpainted,
            hyras,
            name,
            title="RAPSD Comparison (Daily, 2018)",
        )
        plot_monthly_comparison_normal_vs_hyras(
            evaluation,
            evaluation_hyras,
            timestamps,
            radar,
            out_prefix=f"evaluation_metrics_by_month_comparison_vs_hyras_{name}"
        )

    plot_metrics_by_time_and_month(evaluation, timestamps, radar, name)
