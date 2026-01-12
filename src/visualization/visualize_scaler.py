from matplotlib import pyplot as plt
from src.config import cfg
import torch

def visualize_scaler():
    train_data = cfg.train_data

    # Collect flattened tensors
    raw_list = [ds.data_raw.flatten() for ds in train_data.datasets]
    scaled_list = [ds.data_scaled.flatten() for ds in train_data.datasets]

    # Concatenate
    raw_all = torch.cat(raw_list)
    scaled_all = torch.cat(scaled_list)

    print(raw_all.shape, scaled_all.shape)

    raw_np = raw_all.detach().cpu().numpy()
    scaled_np = scaled_all.detach().cpu().numpy()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(raw_np, bins=200, density=True)
    plt.title("Raw data distribution")
    plt.xlabel("Value")
    plt.ylabel("Density (log)")
    plt.yscale("log")

    plt.subplot(1, 2, 2)
    plt.hist(scaled_np, bins=200, density=True)
    plt.title("Scaled data distribution")
    plt.xlabel("Value")
    plt.ylabel("Density (log)")
    plt.yscale("log")

    plt.savefig('src/visualization/scalers.png')


    return

if __name__ == "__main__":
    visualize_scaler()