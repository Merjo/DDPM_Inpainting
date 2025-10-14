import os
import random
import matplotlib.pyplot as plt
import numpy as np

def plot_random(dataset, n=6, title='Random Samples', out_dir="output/plots"):
    """Save a random selection of dataset images into out_dir as plot#.png."""
    os.makedirs(out_dir, exist_ok=True)

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
        image = dataset[idx].squeeze()  # remove channel dim
        axes[i].imshow(image, cmap="turbo")
        axes[i].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)

    print(f"[Plot] Saved {filename}")


def plot_histogram(real, generated, title='Histogram', bins=50, out_dir="output/plots"):
    """
    Compare histogram of real vs generated samples.
    """

    os.makedirs(out_dir, exist_ok=True)

    # Find the next available plot number
    existing = [f for f in os.listdir(out_dir) if f.startswith("hist") and f.endswith(".png")]
    numbers = []
    for f in existing:
        try:
            numbers.append(int(f[4:-4]))  # extract number from "plotX.png"
        except ValueError:
            continue
    next_num = max(numbers, default=0) + 1
    filename = os.path.join(out_dir, f"hist{next_num}.png")


    # --- Plot histograms ---
    plt.figure(figsize=(8, 5))
    plt.hist(real, bins=bins, alpha=0.6, label='Real', density=True)
    plt.hist(generated, bins=bins, alpha=0.6, label='Generated', density=True)
    plt.legend()
    plt.title(title)
    plt.xlabel("Pixel / Value")
    plt.ylabel("Density")
    plt.tight_layout()

    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"[Plot] Saved histogram comparison to {filename}")
