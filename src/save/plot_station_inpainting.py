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

def expand_stations(field, min_val=0.3, high_radius=3, small_radius=1):
    """
    Expand non-NaN points into (2*radius+1)x(2*radius+1) blocks.
    Does NOT overwrite existing non-NaN values.
    
    field: 2D numpy array (H, W)
    radius: 1 -> 3x3, 2 -> 5x5, 3-> 7x7
    """
    H, W = field.shape
    out = field.copy()

    # Locations of stations
    ys, xs = np.where(~np.isnan(field))

    for y, x in zip(ys, xs):
        val = field[y, x]
        radius = high_radius if val>= min_val else small_radius
        y0 = max(0, y - radius)
        y1 = min(H, y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(W, x + radius + 1)

        block = out[y0:y1, x0:x1]

        # Only fill NaNs
        mask = np.isnan(block)
        block[mask] = val

    return out

def plot_station_inpainting(radar, station, inpainted, hyras, timestamps, title='Inpainting Results', out_dir=None, filename='station_inpainting.png', do_expand_stations=True):
    """Plot original, masked, and inpainted images side by side with masked areas in white.
       Uses separate color scales for input data and inpainted data."""

    radar = [np.clip(x, 0.1, None) for x in radar]
    station = [np.clip(x, 0.1, None) for x in station]
    inpainted = [np.clip(x, 0.1, None) for x in inpainted]

    if hyras is not None:
        hyras = [np.clip(x, 0.1, None) for x in hyras]

    if do_expand_stations:
        station = [expand_stations(x) for x in station]  # 5x5 blocks

    if out_dir is None:
        out_dir = f"output_new/output_inpainting"
    os.makedirs(out_dir, exist_ok=True)

    date_str = datetime.now().strftime("%m-%d %H:%M")
    #title_full = f"{title}\n({date_str})"
    title_full = title

    n = len(radar)
    n_subplots = 3 if hyras is None else 4
    fig, axes = plt.subplots(n, n_subplots, figsize=(11, 3*n))

    # Handle n=1 case
    if n == 1:
        axes = np.expand_dims(axes, 0)

    # Set colormap: turbo with white for masked
    cmap = plt.get_cmap('turbo').copy()
    cmap.set_bad('white')

    # Plot
    for i in range(n):
        axrow = axes[i]

        axrow[0].text(
                -200,  # slightly left of the image
                200,
                pd.Timestamp(timestamps[i]).strftime("%b %d, %H:00"),  # human-readable
                fontsize=12,
                rotation=90
            )

        # Stations
        station_plot = np.ma.masked_invalid(np.squeeze(station[i]))
        axrow[0].imshow(station_plot, origin='lower', cmap=cmap, norm=mcolors.LogNorm(vmin=0.1, vmax=cfg.vmax))
        axrow[0].set_title('Stations')
        axrow[0].axis('off')

        # Inpaintings
        inpaint_plot = np.ma.masked_invalid(np.squeeze(inpainted[i]))
        
        axrow[1].imshow(inpaint_plot, origin='lower', cmap=cmap, norm=mcolors.LogNorm(vmin=0.1, vmax=cfg.vmax))
        axrow[1].set_title('Inpainted')
        axrow[1].axis('off')

        # Radar
        radar_plot = np.ma.masked_invalid(np.squeeze(radar[i]))
        axrow[2].imshow(radar_plot, origin='lower', cmap=cmap, norm=mcolors.LogNorm(vmin=0.1, vmax=cfg.vmax))
        axrow[2].set_title('Radar')
        axrow[2].axis('off')
        
        if hyras is not None:
            # Radar
            hyras_plot = np.ma.masked_invalid(np.squeeze(hyras[i]))
            axrow[3].imshow(hyras_plot, origin='lower', cmap=cmap, norm=mcolors.LogNorm(vmin=0.1, vmax=cfg.vmax))
            axrow[3].set_title('Hyras')
            axrow[3].axis('off')


    # === Colorbar on the right ===
    norm = mcolors.LogNorm(vmin=0.1, vmax=cfg.vmax)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm),
        cax=cbar_ax
    )
    cbar.set_label("Precipitation")


    # Title and layout
    plt.suptitle(title_full)
    plt.subplots_adjust(top=0.9, bottom=0.12, wspace=0.05)

    filepath = os.path.join(out_dir, filename)
    plt.savefig(filepath, dpi=400)
    plt.close(fig)

    print(f"[Plot] Saved inpainting results to {filepath}")



if __name__ == "__main__":
    #folder_name = 'final_Jan12_1259_hourly_2018_None_filippouFalse' # Hourly Normal Final
    #folder_name = 'final_Jan12_1252_daily_2018_4_filippouFalse'  # Daily 4 Timesteps
    #folder_name = 'final_Jan13_1548_hourly_2018_None_filippouTrue' # Hourly Filippou Final
    folder_name = 'final_Jan13_1546_daily_2018_None_filippouFalse'  # Daily Final

    folder_name = 'final_Jan16_1843_daily_2018_None_filippouFalse'  # 10 time mean Daily

    n=6
    final_dir = f'{cfg.output_cache_path}/{folder_name}'
    hyras = True  # TODO Implement Filippou Comparison?
    if hyras:
        radar, station, inpainted, timestamps, hyras = load_final(final_dir, hyras=True, listify=True)
    else:
        radar, station, inpainted, timestamps = load_final(final_dir, hyras=False, listify=True)
        hyras = None

    date_str = datetime.now().strftime("%b%d_%H%M")
    filename = f'{folder_name}.png'

    total_samples = len(radar)

    if n < total_samples:  # only subsample if n < total available
        rng = np.random.default_rng(seed=42)  # fixed seed for reproducibility
        selected_idx = rng.choice(total_samples, size=n, replace=False)
        selected_idx = np.sort(selected_idx)
        # Subset arrays
        radar = [radar[i] for i in selected_idx] if isinstance(radar, list) else radar[selected_idx]
        station = [station[i] for i in selected_idx] if isinstance(station, list) else station[selected_idx]
        inpainted = [inpainted[i] for i in selected_idx] if isinstance(inpainted, list) else inpainted[selected_idx]
        if hyras is not None:
            hyras = [hyras[i] for i in selected_idx] if isinstance(hyras, list) else hyras[selected_idx]

        timestamps = [timestamps[i] for i in selected_idx] if isinstance(timestamps, list) else timestamps[selected_idx]

        print(f"[Info] Subsampled {n} random indices for plotting")
    else:
        print(f"[Info] Using all {total_samples} samples for plotting")

    plot_station_inpainting(radar=radar,
                            station=station,
                            inpainted=inpainted,
                            hyras = hyras,
                            timestamps = list(timestamps),
                            title=f'Inpainting Results',#, lam={cfg.dps_lam}',
                            out_dir='output_new/output_inpainting',
                            filename=filename)