import os
import random
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from src.config import cfg
from src.random.spectra import mean_rapsd_numpy  # or adjust the import to where your PSD code lives

import matplotlib.colors as mcolors
import torch
from matplotlib.colors import LogNorm


def scale_back_numpy(arr, scaler):
    t = torch.as_tensor(arr)            # numpy → tensor
    t_dec = scaler.decode(t)            # decode
    t_dec = t_dec.clamp(min=0.0)          # clamp negatives to 0
    return t_dec.detach().cpu().numpy() 

def remove_nans(arr_list):
    arr = np.array(arr_list)
    arr = arr[np.isfinite(arr)]
    return arr


def plot_inpainting(original, masked, inpainted, pct, lam, title='Inpainting Results', out_dir=None):
    """Plot original, masked, and inpainted images side by side with masked areas in white.
       Uses separate color scales for input data and inpainted data."""
    # Scale back
    original = [scale_back_numpy(x, cfg.scaler) for x in original]
    masked = [scale_back_numpy(x, cfg.scaler) for x in masked]
    inpainted = [scale_back_numpy(x, cfg.scaler) for x in inpainted]

    original = [np.clip(x, 0.1, None) for x in original]
    masked = [np.clip(x, 0.1, None) for x in masked]
    inpainted = [np.clip(x, 0.1, None) for x in inpainted]
    
    if out_dir is None:
        out_dir = f"{cfg.current_output}/inpainting_results"
    os.makedirs(out_dir, exist_ok=True)

    date_str = datetime.now().strftime("%m-%d %H:%M")
    title_full = f"{title}\n({date_str})"

    n = len(original)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3*n))

    # Handle n=1 case
    if n == 1:
        axes = np.expand_dims(axes, 0)

    # Set colormap: turbo with white for masked
    cmap = plt.get_cmap('turbo').copy()
    cmap.set_bad('white')

    # Plot
    for i in range(n):
        axrow = axes[i]

        # Original
        orig = np.ma.masked_invalid(np.squeeze(original[i]))
        #axrow[0].imshow(orig, cmap=cmap, vmin=cfg.vmin, vmax=cfg.vmax)
        axrow[0].imshow(orig, cmap=cmap, norm=mcolors.LogNorm(vmin=0.1, vmax=cfg.vmax))
        axrow[0].set_title('Original')
        axrow[0].axis('off')

        # Masked
        masked_plot = np.ma.masked_invalid(np.squeeze(masked[i]))
        #axrow[1].imshow(masked_plot, cmap=cmap, vmin=cfg.vmin, vmax=cfg.vmax)
        axrow[1].imshow(masked_plot, cmap=cmap, norm=mcolors.LogNorm(vmin=0.1, vmax=cfg.vmax))
        axrow[1].set_title('Masked')
        axrow[1].axis('off')

        # Inpainted
        #inpaint_plot = np.ma.masked_invalid(np.squeeze(inpainted[i]))

        # Restore original pixels in inpainted output (so full image is shown)
        inpaint_plot = np.squeeze(inpainted[i]).copy()
        orig = np.squeeze(original[i])
        mask = np.isnan(masked[i])  # True where mask was missing (white)
        inpaint_plot[~mask] = orig[~mask]  # put original values back

        
        #axrow[2].imshow(inpaint_plot, cmap=cmap, vmin=cfg.vmin, vmax=cfg.vmax)
        axrow[2].imshow(inpaint_plot, cmap=cmap, norm=mcolors.LogNorm(vmin=0.1, vmax=cfg.vmax))
        axrow[2].set_title('Inpainted')
        axrow[2].axis('off')


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
    filename = os.path.join(out_dir, f"inpainting_{pct}_{lam}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(filename, dpi=200)
    plt.close(fig)

    print(f"[Plot] Saved inpainting results to {filename}")


def plot_station_inpainting(radar, station, inpainted, timestamps, lam, title='Inpainting Results', out_dir=None):
    """Plot original, masked, and inpainted images side by side with masked areas in white.
       Uses separate color scales for input data and inpainted data."""
    # Scale back
    radar = [scale_back_numpy(x, cfg.scaler) for x in radar]
    station = [scale_back_numpy(x, cfg.scaler) for x in station]
    inpainted = [scale_back_numpy(x, cfg.scaler) for x in inpainted]

    radar = [np.clip(x, 0.1, None) for x in radar]
    station = [np.clip(x, 0.1, None) for x in station]
    inpainted = [np.clip(x, 0.1, None) for x in inpainted]
    
    if out_dir is None:
        out_dir = f"{cfg.current_output}/inpainting_results"
    os.makedirs(out_dir, exist_ok=True)

    date_str = datetime.now().strftime("%m-%d %H:%M")
    title_full = f"{title}\n({date_str})"

    n = len(radar)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3*n))

    # Handle n=1 case
    if n == 1:
        axes = np.expand_dims(axes, 0)

    # Set colormap: turbo with white for masked
    cmap = plt.get_cmap('turbo').copy()
    cmap.set_bad('white')

    # Plot
    for i in range(n):
        axrow = axes[i]

        axrow[0].set_ylabel(
            str(timestamps[i]),
            rotation=0,
            fontsize=9,
            labelpad=40,
            va="center"
        )

        # Radar
        radar_plot = np.ma.masked_invalid(np.squeeze(radar[i]))
        axrow[0].imshow(radar_plot.T, origin='lower', cmap=cmap, norm=mcolors.LogNorm(vmin=0.1, vmax=cfg.vmax))
        axrow[0].set_title('Radar')
        axrow[0].axis('off')

        # Stations
        station_plot = np.ma.masked_invalid(np.squeeze(station[i]))
        axrow[1].imshow(station_plot.T, origin='lower', cmap=cmap, norm=mcolors.LogNorm(vmin=0.1, vmax=cfg.vmax))
        axrow[1].set_title('Stations')
        axrow[1].axis('off')

        inpaint_plot = np.ma.masked_invalid(np.squeeze(inpainted[i]))
        
        axrow[2].imshow(inpaint_plot.T, origin='lower', cmap=cmap, norm=mcolors.LogNorm(vmin=0.1, vmax=cfg.vmax))
        axrow[2].set_title('Inpainted')
        axrow[2].axis('off')

    """for i, idx in enumerate(indices):
        #axes[i].imshow(images[i], cmap="turbo", vmin=cfg.vmin, vmax=cfg.vmax)
        images[i][images[i] < 0.0001] = 0.0001  #  avoid 0, make sure they are seen in log scale
        axes[i].imshow(
            images[i].T,        
            origin="lower", 
            cmap="turbo",
            norm=mcolors.LogNorm(vmin=0.0001, vmax=cfg.vmax)
        )
        axes[i].axis("off")"""

    """fig.colorbar(
        plt.cm.ScalarMappable(cmap="turbo", norm=mcolors.LogNorm(vmin=0.0001, vmax=cfg.vmax)),
        ax=axes, orientation='vertical', fraction=0.02, pad=0.01
    )"""

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
    filename = os.path.join(out_dir, f"station_inpainting_{lam}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(filename, dpi=400)
    plt.close(fig)

    print(f"[Plot] Saved inpainting results to {filename}")



def get_new_filename(title, out_dir, folder, no_log=False):
    if out_dir is None:
        out_dir = f"{cfg.current_output}/{folder}"
    os.makedirs(out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%m-%d %H:%M")
    title = f"{title}\n({date_str})"

    # Find the next available plot number
    existing = [f for f in os.listdir(out_dir) if f.startswith("plot") and f.endswith(".png")]
    numbers = []
    for f in existing:
        try:
            numbers.append(int(f[4:-4]))  # extract number from "plotX.png"
        except ValueError:
            continue
    next_num = max(numbers, default=0) + 1
    filename = os.path.join(out_dir, f"plot{next_num}{f'_nolog' if no_log else ''}.png")

    return filename

def plot_random(dataset, n=6, title='Random Samples', out_dir=None, folder='samples'):
    plot_random_specific(dataset, n=n, title=title, out_dir=out_dir, folder=folder, do_log=True)
    plot_random_specific(dataset, n=n, title=title, out_dir=out_dir, folder=folder, do_log=False)

def plot_random_specific(dataset, n=6, title='Random Samples', out_dir=None, folder='samples', do_log=True):
    """Save a random selection of dataset images into out_dir as plot#.png."""
    
    filename = get_new_filename(title, out_dir=out_dir, folder=folder, no_log = not do_log)


    # Pick random indices
    indices = random.sample(range(len(dataset)), n)

    # Select images
    images = [scale_back_numpy(dataset[idx], cfg.scaler).squeeze() for idx in indices]

    # Create figure
    max_per_row = 8
    cols = min(n, max_per_row)
    rows = (n + max_per_row - 1) // max_per_row  # ceiling division

    fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
    axes = np.array(axes).reshape(-1)  # flatten to 1D list

    # Hide any unused axes (e.g., if 10 samples → 8 + 2 empty)
    for j in range(n, len(axes)):
        axes[j].axis("off")

    for i, idx in enumerate(indices):
        #axes[i].imshow(images[i], cmap="turbo", vmin=cfg.vmin, vmax=cfg.vmax)
        if do_log:
            images[i][images[i] < 0.1] = 0.1  #  avoid 0, make sure they are seen in log scale
            norm = mcolors.LogNorm(vmin=0.1, vmax=cfg.vmax)
        else:
            norm = mcolors.Normalize(vmin=0.0, vmax=cfg.vmax)
        axes[i].imshow(
            images[i].T,        
            origin="lower", 
            cmap="turbo",
            norm=norm
        )
        axes[i].axis("off")
    norm_cb = mcolors.LogNorm(vmin=0.1, vmax=cfg.vmax) if do_log else mcolors.Normalize(vmin=0.0, vmax=cfg.vmax)

    fig.colorbar(
        plt.cm.ScalarMappable(cmap="turbo", norm=norm_cb),
        ax=axes, orientation='vertical', fraction=0.02, pad=0.01
    )

    plt.suptitle(title)
    #plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    plt.savefig(filename, dpi=150)
    plt.close(fig)

    print(f"[Plot] Saved {filename}")

def plot_histogram(real, generated, title='Histogram', bins=100, out_dir=None):
    if out_dir is None:
        out_dir = f"{cfg.current_output}/histograms"
    plot_rapsd_comparison(real, generated, title='RAPSD Comparison', out_dir=out_dir)
    real = real.flatten()
    generated = generated.flatten()
    plot_histogram_normal(real, generated, title=title, bins=bins, out_dir=out_dir)
    plot_histogram_log(real, generated, title=title, bins=bins, out_dir=out_dir)


def plot_histogram_normal(real, generated, title='Histogram', bins=100, out_dir=None):
    """
    Compare histogram of real vs generated samples.
    """
    # Scale back
    #real = [scale_back_numpy(x, cfg.scaler) for x in real]
    #generated = [scale_back_numpy(x, cfg.scaler) for x in generated]

    real = remove_nans([scale_back_numpy(x, cfg.scaler) for x in real])
    generated = remove_nans([scale_back_numpy(x, cfg.scaler) for x in generated])


    if out_dir is None:
        out_dir = f"{cfg.current_output}/histograms"

    os.makedirs(out_dir, exist_ok=True)

    # Find the next available plot number
    existing = [f for f in os.listdir(out_dir) if f.startswith("hist_normal_") and f.endswith(".png")]
    numbers = []
    for f in existing:
        try:
            num_str = f.split("_")[-1].replace(".png", "")  # get the number after last underscore
            numbers.append(int(num_str))
        except ValueError:
            continue
    next_num = max(numbers, default=0) + 1
    filename = os.path.join(out_dir, f"hist_normal_{next_num}.png")


    # --- Plot histograms ---
    plt.figure(figsize=(8, 5))

    # Compute common bin edges
    combined = np.concatenate([real, generated])
    bins_edges = np.histogram_bin_edges(combined, bins=bins)  # e.g., bins=50

    # Plot histograms using the same edges
    plt.hist(real, bins=bins_edges, alpha=0.6, label='Real', density=True)
    plt.hist(generated, bins=bins_edges, alpha=0.6, label='Generated', density=True)

    date_str = datetime.now().strftime("%m-%d %H:%M")
    fin_title = f"{title}\n({date_str})"


    plt.legend()
    plt.title(fin_title)
    plt.xlabel("Pixel / Value")
    plt.ylabel("Density")
    #plt.tight_layout()

    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[Plot] Saved histogram comparison to {filename}")


def plot_histogram_log(real, generated, title='Histogram', bins=100, out_dir=None):
    """
    Compare histogram of real vs generated samples.
    """
    # Scale back
    #real = [scale_back_numpy(x, cfg.scaler) for x in real]
    #generated = [scale_back_numpy(x, cfg.scaler) for x in generated]

    real = remove_nans([scale_back_numpy(x, cfg.scaler) for x in real])
    generated = remove_nans([scale_back_numpy(x, cfg.scaler) for x in generated])


    os.makedirs(out_dir, exist_ok=True)

    # Find the next available plot number
    existing = [f for f in os.listdir(out_dir) if f.startswith("hist_log_") and f.endswith(".png")]
    numbers = []
    for f in existing:
        try:
            num_str = f.split("_")[-1].replace(".png", "")
            numbers.append(int(num_str))
        except ValueError:
            continue
    next_num = max(numbers, default=0) + 1
    filename = os.path.join(out_dir, f"hist_log_{next_num}.png")


    # --- Plot histograms ---
    plt.figure(figsize=(8, 5))

    # Compute common bin edges
    combined = np.concatenate([real, generated])
    bins_edges = np.histogram_bin_edges(combined, bins=bins)

    # Calculate histogram values manually to plot side-by-side
    real_hist, _ = np.histogram(real, bins=bins_edges, density=True)
    gen_hist, _ = np.histogram(generated, bins=bins_edges, density=True)

    # Compute bar centers and width
    bin_centers = 0.5 * (bins_edges[:-1] + bins_edges[1:])
    width = (bins_edges[1] - bins_edges[0]) * 0.4  # side-by-side spacing

    # Plot side-by-side bars
    plt.bar(bin_centers - width/2, real_hist, width=width, alpha=0.7, label='Real')
    plt.bar(bin_centers + width/2, gen_hist, width=width, alpha=0.7, label='Generated')

    # Logarithmic axes
    #plt.xscale('log')
    plt.yscale('log')

    date_str = datetime.now().strftime("%m-%d %H:%M")
    title = f"{title}\n({date_str})"


    plt.legend()
    plt.title(title)
    plt.xlabel("Pixel / Value")
    plt.ylabel("Density")
    #plt.tight_layout()

    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[Plot] Saved histogram comparison to {filename}")

def plot_rapsd_comparison(real, generated, title='RAPSD Comparison', out_dir=None):
    """
    Plot the radially averaged power spectral density (RAPSD) for real vs generated samples.
    """

    # === Prepare data ===
    real = [scale_back_numpy(x, cfg.scaler) for x in real]
    generated = [scale_back_numpy(x, cfg.scaler) for x in generated]

    # Check and crop spatial size if needed
    real_shape = real[0].shape[-2:]   # (H, W) of first real image
    gen_shape = generated[0].shape[-2:]

    if gen_shape != real_shape:
        print(f"[RAPSD] Warning: Generated images shape {gen_shape} "
              f"does not match real images shape {real_shape}. Cropping generated images.")
        gen_cropped = []
        h, w = real_shape
        for g in generated:
            gh, gw = g.shape[-2:]
            # center crop
            start_h = (gh - h) // 2
            start_w = (gw - w) // 2
            gen_cropped.append(g[..., start_h:start_h+h, start_w:start_w+w])
        generated = gen_cropped

    # Stack into numpy arrays of shape [time, lat, lon]
    real_arr = np.stack([r.squeeze() for r in real])
    gen_arr = np.stack([g.squeeze() for g in generated])

    # === Compute mean RAPSD ===
    real_psd, freq = mean_rapsd_numpy(real_arr)
    gen_psd, _ = mean_rapsd_numpy(gen_arr)

    # === Plot ===
    if out_dir is None:
        out_dir = f"{cfg.current_output}/rapsd"

    # Filename
    existing = [f for f in os.listdir(out_dir) if f.startswith("rapsd_") and f.endswith(".png")]
    numbers = [int(f.split("_")[-1].replace(".png", "")) for f in existing if f.split("_")[-1].replace(".png", "").isdigit()]
    next_num = max(numbers, default=0) + 1
    filename = os.path.join(out_dir, f"rapsd_{next_num}.png")

    # Plot PSD
    plt.figure(figsize=(7, 5))
    plt.loglog(freq, real_psd, label='Real', linewidth=2)
    plt.loglog(freq, gen_psd, label='Generated', linewidth=2)
    plt.xlabel("Spatial frequency")
    plt.ylabel("Power spectral density")
    plt.title(f"{title}\n({datetime.now().strftime('%m-%d %H:%M')})")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.4)

    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"[Plot] Saved RAPSD comparison to {filename}")


def plot_inpainting_mse_curves(df, out_dir=None, title="MSE vs Coverage for Different λ"):
    """
    Plots MSE curves and a heatmap with coverage vs lambda.
    df must contain columns ['coverage', 'lambda', 'mse'].
    """

    if out_dir is None:
        out_dir = f"{cfg.current_output}/inpainting_results"
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # 1. Line plot
    # -------------------------

    coverages = sorted(df["coverage"].unique())

    plt.figure(figsize=(10, 6))
    for cov in coverages:
        df_cov = df[df["coverage"] == cov].sort_values("lambda")
        plt.plot(
            df_cov["lambda"],
            df_cov["mse"],
            marker="o",
            label=f"Coverage = {cov*100:.3g}%"
        )

    plt.xscale("log")

    plt.xlabel("LAMBDA (λ)")
    plt.ylabel("MSE (log scale)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale("log")

    fname_curve = os.path.join(out_dir, "inpainting_mse_curve.png")
    plt.savefig(fname_curve, dpi=200)
    plt.close()
    print(f"[Plot] Saved MSE curve plot to {fname_curve}")

    # -------------------------
    # 2. Heatmap
    # -------------------------
    # Get sorted unique values
    coverage_vals = np.sort(df["coverage"].unique())
    lambda_vals = np.sort(df["lambda"].unique())

    # Create grid
    heatmap_array = np.empty((len(coverage_vals), len(lambda_vals)))
    heatmap_array[:] = np.nan  # fill missing cells with NaN

    # Fill grid with MSE
    for i, cov in enumerate(coverage_vals):
        for j, lam in enumerate(lambda_vals):
            match = df[(df["coverage"] == cov) & (df["lambda"] == lam)]
            if not match.empty:
                heatmap_array[i, j] = match["mse"].values[0]

    plt.figure(figsize=(12, 6))
    im = plt.imshow(
        heatmap_array,
        origin="lower",
        aspect="auto",
        interpolation="none",
        norm=LogNorm(),  # logarithmic color scale
        cmap="viridis"
    )

    # Set ticks
    plt.xticks(ticks=np.arange(len(lambda_vals)), labels=lambda_vals)
    plt.yticks(ticks=np.arange(len(coverage_vals)), labels=np.round(coverage_vals, 3))

    plt.xlabel("λ")
    plt.ylabel("Coverage")
    plt.title("Heatmap of Inpainting MSE")
    plt.colorbar(im, label="MSE (log scale)")

    fname_heatmap = os.path.join(out_dir, "inpainting_mse_heatmap.png")
    plt.savefig(fname_heatmap, dpi=200)
    plt.close()
    print(f"[Plot] Saved MSE heatmap to {fname_heatmap}")

    return fname_curve, fname_heatmap


