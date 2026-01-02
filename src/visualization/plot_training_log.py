import re
import matplotlib.pyplot as plt

def plot_training_log(log_path, out_path=None):
    """
    Parse a training log and plot train/val loss over epochs.

    Args:
        log_path (str): Path to the text log file.
        out_path (str, optional): If given, saves plot to this path instead of showing it.
    """
    epochs, train_loss, val_loss = [], [], []

    # Regex pattern to match the loss lines
    pattern = re.compile(r"Epoch (\d+)/\d+ - train_loss: ([0-9.]+) - val_loss: ([0-9.]+)")

    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                e = int(match.group(1))
                train = float(match.group(2))
                val = float(match.group(3))
                epochs.append(e)
                train_loss.append(train)
                val_loss.append(val)

    if not epochs:
        print("No epochs found in log — check your log format.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Train Loss", marker="o", markersize=3)
    plt.plot(epochs, val_loss, label="Validation Loss", marker="o", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved loss curve to {out_path}")
        plt.close()
    else:
        plt.show()


if __name__=='__main__':
    #path = "output_new/0.04342_normal_Nov10_1851_256_0.1/run_normal_5358000.out"
    path = "logs/run_normal_5964510.out"
    plot_training_log(log_path=path, out_path="src/visualization/plots")