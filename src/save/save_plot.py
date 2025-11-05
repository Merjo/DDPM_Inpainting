import os
import random
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from src.config import cfg

import matplotlib.colors as mcolors
import torch


def scale_back_numpy(arr, scaler):
    t = torch.as_tensor(arr)            # numpy → tensor
    t_dec = scaler.decode(t)            # decode
    return t_dec.detach().cpu().numpy() 


def plot_inpainting(original, masked, inpainted, title='Inpainting Results', out_dir=None):
    """Plot original, masked, and inpainted images side by side with masked areas in white.
       Uses separate color scales for input data and inpainted data."""
    # Scale back
    original = [scale_back_numpy(x, cfg.scaler) for x in original]
    masked = [scale_back_numpy(x, cfg.scaler) for x in masked]
    inpainted = [scale_back_numpy(x, cfg.scaler) for x in inpainted]
    
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
        inpaint_plot = np.ma.masked_invalid(np.squeeze(inpainted[i]))
        #axrow[2].imshow(inpaint_plot, cmap=cmap, vmin=cfg.vmin, vmax=cfg.vmax)
        axrow[2].imshow(inpaint_plot, cmap=cmap, norm=mcolors.LogNorm(vmin=0.1, vmax=cfg.vmax))
        axrow[2].set_title('Inpainted')
        axrow[2].axis('off')

    # === Colorbars ===
    # Left colorbar (for Original & Masked)
    norm = mcolors.LogNorm(vmin=0.1, vmax=cfg.vmax)
    cbar1 = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=axes[:, :2].ravel().tolist(),
        orientation='horizontal',
        fraction=0.05,
        pad=0.05
    )
    cbar1.set_label("Precipitation")


    # Title and layout
    plt.suptitle(title_full)
    plt.subplots_adjust(top=0.9, bottom=0.12, wspace=0.05)
    filename = os.path.join(out_dir, f"inpainting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(filename, dpi=200)
    plt.close(fig)

    print(f"[Plot] Saved inpainting results to {filename}")



def plot_inpainting3(original, masked, inpainted, title='Inpainting Results', out_dir=None):
    """Plot original, masked, and inpainted images side by side with masked areas in white."""
    # Scale back
    original = [scale_back_numpy(x, cfg.scaler) for x in original]
    masked = [scale_back_numpy(x, cfg.scaler) for x in masked]
    inpainted = [scale_back_numpy(x, cfg.scaler) for x in inpainted]

    if out_dir is None:
        out_dir = f"{cfg.current_output}/inpainting_results"
    os.makedirs(out_dir, exist_ok=True)

    date_str = datetime.now().strftime("%m-%d %H:%M")
    title_full = f"{title}\n({date_str})"

    n = len(original)
    fig, axes = plt.subplots(n, 3, figsize=(10, 3*n))
    
    # Flatten axes if n=1
    if n == 1:
        axes = np.expand_dims(axes, 0)

    # Convert masked pixels to NaN so they appear white
    masked_imgs = []
    for img, m in zip(masked, masked):
        arr = np.squeeze(img)
        arr = np.where(np.isnan(arr), np.nan, arr)  # ensure NaNs stay NaN
        masked_imgs.append(arr)


    # Colormap: turbo with white for masked
    cmap = plt.get_cmap('turbo').copy()
    cmap.set_bad('white')

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
        inpaint_plot = np.ma.masked_invalid(np.squeeze(inpainted[i]))
        #axrow[2].imshow(inpaint_plot, cmap=cmap, vmin=cfg.vmin, vmax=cfg.vmax)
        axrow[2].imshow(inpaint_plot, cmap=cmap, norm=mcolors.LogNorm(vmin=0.1, vmax=cfg.vmax))
        axrow[2].set_title('Inpainted')
        axrow[2].axis('off')

    # Shared colorbar below the plots
    norm = mcolors.LogNorm(vmin=0.1, vmax=cfg.vmax)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=norm),
        ax=axes.ravel().tolist(),
        orientation='horizontal',
        fraction=0.05,
        pad=0.05
    )
    cbar.set_label("Precipitation (normalized units)")

    plt.suptitle(title_full)
    #plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # leave space for suptitle
    filename = os.path.join(out_dir, f"inpainting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"[Plot] Saved inpainting results to {filename}")


def plot_inpainting2(original, masked, inpainted, title='Inpainting Results', out_dir=None):
    # Scale back
    original = [scale_back_numpy(x, cfg.scaler) for x in original]
    masked = [scale_back_numpy(x, cfg.scaler) for x in masked]
    inpainted = [scale_back_numpy(x, cfg.scaler) for x in inpainted]


    if out_dir is None:
        out_dir = f"{cfg.current_output}/inpainting_results"
    os.makedirs(out_dir, exist_ok=True)
    date_str = datetime.now().strftime("%m-%d %H:%M")
    title = f"{title}\n({date_str})"

    n = len(original)
    fig, axes = plt.subplots(n, 3, figsize=(10, 3*n))
    
    
    # Custom colormap: blue to red, white for masked (NaN)
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad(color='white')

    for i in range(n):
        # Convert to masked arrays so NaNs appear white
        orig = np.squeeze(original[i])
        maskd = np.squeeze(masked[i])
        inpaint = np.squeeze(inpainted[i])

        # Mask NaNs for plotting (if you set them where mask==0)
        orig = np.ma.masked_invalid(orig)
        maskd = np.ma.masked_invalid(maskd)
        inpaint = np.ma.masked_invalid(inpaint)

        axrow = axes[i] if n > 1 else axes
        ims = []
        ims.append(axrow[0].imshow(orig, cmap=cmap, norm=mcolors.LogNorm(vmin=0.1, vmax=cfg.vmax)))
        ims.append(axrow[1].imshow(maskd, cmap=cmap, norm=mcolors.LogNorm(vmin=0.1, vmax=cfg.vmax)))
        ims.append(axrow[2].imshow(inpaint, cmap=cmap, norm=mcolors.LogNorm(vmin=0.1, vmax=cfg.vmax)))

        for ax, title_ in zip(axrow, ['Original', 'Masked', 'Inpainted']):
            ax.set_title(title_)
            ax.axis('off')

    # Add a shared colorbar
    cbar = fig.colorbar(ims[0], ax=axes.ravel().tolist(), shrink=0.6, orientation='horizontal', pad=0.05)
    cbar.set_label("Precipitation (normalized units)")

    plt.suptitle(title)
    #plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    filename = os.path.join(out_dir, f"inpainting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(filename, dpi=200)
    plt.close(fig)
    print(f"[Plot] Saved inpainting results to {filename}")


def plot_random(dataset, n=6, title='Random Samples', out_dir=None):
    """Save a random selection of dataset images into out_dir as plot#.png."""
    # Scale back
    dataset = [scale_back_numpy(x, cfg.scaler) for x in dataset]
 
    
    if out_dir is None:
        out_dir = f"{cfg.current_output}/samples"
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
    filename = os.path.join(out_dir, f"plot{next_num}.png")

    # Pick random indices
    indices = random.sample(range(len(dataset)), n)

    # Compute global min/max across all selected images
    images = [dataset[idx].squeeze() for idx in indices]  # convert to numpy

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
        images[i][images[i] < 0.1] = 0.1  #  avoid 0, make sure they are seen in log cale
        axes[i].imshow(images[i], cmap="turbo", norm=mcolors.LogNorm(vmin=0.1, vmax=cfg.vmax))
        axes[i].axis("off")

    fig.colorbar(
        plt.cm.ScalarMappable(cmap="turbo", norm=mcolors.LogNorm(vmin=0.1, vmax=cfg.vmax)),
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
    plot_histogram_normal(real, generated, title=title, bins=bins, out_dir=out_dir)
    plot_histogram_log(real, generated, title=title, bins=bins, out_dir=out_dir)


def plot_histogram_normal(real, generated, title='Histogram', bins=100, out_dir=None):
    """
    Compare histogram of real vs generated samples.
    """
    # Scale back
    real = [scale_back_numpy(x, cfg.scaler) for x in real]
    generated = [scale_back_numpy(x, cfg.scaler) for x in generated]

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
    real = [scale_back_numpy(x, cfg.scaler) for x in real]
    generated = [scale_back_numpy(x, cfg.scaler) for x in generated]

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
