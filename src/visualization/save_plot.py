import os
import random
import matplotlib.pyplot as plt

def plot_random(dataset, n=6, title='Random Samples', out_dir="output/plots"):
    print(os.getcwd)
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
    fig, axes = plt.subplots(1, n, figsize=(2*n, 2))
    if n == 1:
        axes = [axes]  # ensure iterable

    for i, idx in enumerate(indices):
        image = dataset[idx].squeeze()  # remove channel dim
        axes[i].imshow(image, cmap="turbo")
        axes[i].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close(fig)

    print(f"[Plot] Saved {filename}")
