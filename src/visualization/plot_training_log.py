import re
import matplotlib.pyplot as plt
from matplotlib import gridspec

def compute_required_patience(val_loss):
    """Compute the minimum EarlyStopping patience needed based on validation loss."""
    best = float("inf")
    since_improve = 0
    max_since_improve = 0

    for v in val_loss:
        if v < best:
            best = v
            since_improve = 0
        else:
            since_improve += 1
            max_since_improve = max(max_since_improve, since_improve)

    return max_since_improve

def plot_training_log(log_path, out_path=None, do_y_log=False, break_axis=True, skip_first=False):
    """
    Parse a training log and plot train/val loss over epochs.

    Args:
        log_path (str): Path to the text log file.
        out_path (str, optional): If given, saves plot to this path instead of showing it.
        break_axis (bool): Whether to break the y-axis for early extreme values.
        skip_first (bool): Skip first epoch for plotting (ignored if break_axis is True).
    """
    epochs, train_loss, val_loss = [], [], []

    # Regex pattern to match the loss lines
    pattern = re.compile(r"Epoch (\d+)/\d+ - train_loss: ([0-9.]+) - val_loss: ([0-9.]+)")

    # Parse the log
    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                e = int(match.group(1))
                train = float(match.group(2))
                val = float(match.group(3))
                if skip_first and e < 2:
                    continue
                epochs.append(e)
                train_loss.append(train)
                val_loss.append(val)

    if not epochs:
        print("No epochs found in log — check your log format.")
        return

    required_patience = compute_required_patience(val_loss)
    print(f"[EarlyStopping] Minimum patience needed: {required_patience}")

    # --- Find minimum validation loss ---
    min_val = min(val_loss)
    min_idx = val_loss.index(min_val)
    min_epoch = epochs[min_idx]

    val_loss_copy = val_loss.copy()

    # Remove the first minimum
    val_loss_copy.remove(min(val_loss_copy))

    # Find the second minimum
    second_min_val = min(val_loss_copy)

    # Find its index in the original list
    second_min_idx = val_loss.index(second_min_val)
    second_min_epoch = epochs[second_min_idx]

    # Broken axis plotting
    if break_axis and len(val_loss) > 2:
        # Main cluster (ignore first two epochs for focus)
        main_val = val_loss[2:]
        main_train = train_loss[2:]

        # Compute main cluster limits
        bottom_min = min(main_val + main_train) * 0.95
        bottom_max = max(main_val + main_train) * 1.075

        # Identify all points outside the main cluster
        all_values = val_loss + train_loss
        outliers = [v for v in all_values if v < bottom_min or v > bottom_max]
        if not outliers:
            outliers = [max(all_values)]
        top_min = min(outliers) * 0.8
        top_max = max(outliers) * 1.2

        # Create broken axis layout
        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3], hspace=0.05)
        ax_top = fig.add_subplot(gs[0])
        ax_bottom = fig.add_subplot(gs[1], sharex=ax_top)

        # Plot all points on both axes
        ax_top.plot(epochs, train_loss, 'o-', label="Train Loss", markersize=3)
        ax_top.plot(epochs, val_loss, 'o-', label="Validation Loss", markersize=3)
        ax_bottom.plot(epochs, train_loss, 'o-', label="Train Loss", markersize=3)
        ax_bottom.plot(epochs, val_loss, 'o-', label="Validation Loss", markersize=3)

        max_epoch = max(epochs)
        xticks = list(range(0, max_epoch + 1, 50))
        ax_bottom.set_xticks(xticks)
        ax_bottom.tick_params(
            axis="x",
            which="both",
            bottom=True,
            labelbottom=True
        )
        ax_bottom.set_xlim(min(epochs)-5, max_epoch+5)

        # Set y-limits
        ax_top.set_ylim(top_min, top_max)
        ax_bottom.set_ylim(bottom_min, bottom_max)

        # Hide spines and ticks for clean broken axis
        ax_top.spines['bottom'].set_visible(False)
        ax_bottom.spines['top'].set_visible(False)
        #ax_top.tick_params(labelbottom=False)
        ax_top.grid(False)  # remove gridlines in top axis
        #ax_top.xaxis.set_ticks([])  # remove x-ticks entirely
        #ax_top.xaxis.set_ticklabels([])  

        ax_top.tick_params(
            axis="x",
            which="both",
            bottom=False,
            top=False,
            labelbottom=False
        )

        # Draw diagonal break marks
        d = .015
        kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs.update(transform=ax_bottom.transAxes)
        ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

        # --- Mark minimum validation loss(es) ---
        ax_bottom.annotate(
            f"Epoch {min_epoch}\nLoss {min_val:.4g}",
            xy=(min_epoch, min_val),
            xytext=(-30, 50),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", lw=1),
            fontsize=9,
            ha="left"
        )

        ax_bottom.annotate(
            f"Epoch {second_min_epoch}\nLoss {second_min_val:.4g}",
            xy=(second_min_epoch, second_min_val),
            xytext=(-30, 50),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", lw=1),
            fontsize=9,
            ha="left"
        )


        # Labels and legend
        ax_bottom.set_xlabel("Epoch")
        ax_bottom.set_ylabel("Loss")
        #ax_top.set_ylabel("Loss")
        ax_top.legend()
        if do_y_log:
            ax_bottom.set_yscale("log")
            ax_top.set_yscale("log")

    else:
        # Standard single-axis plot
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss, label="Train Loss", marker="o", markersize=3)
        plt.plot(epochs, val_loss, label="Validation Loss", marker="o", markersize=3)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        if do_y_log:
            plt.yscale("log")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)

    # Save or show
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[Plot] Saved loss curve to {out_path}")
        plt.close()
    else:
        plt.show()


if __name__=='__main__':
    mode = 'daily'

    if mode=='daily':
        path = "output_new/0.02194_normal_Jan03_2011_256_0.0/run_normal_6225127.out"
    else:
        path = "output_new/0.0358_normal_Jan08_0328_256_0.0/run_normal_6311792.out"

    plot_training_log(
        log_path=path,
        out_path=f"src/visualization/training_plot_{mode}.png"
    )


"""import re
import matplotlib.pyplot as plt

def compute_required_patience(val_loss):
    best = float("inf")
    since_improve = 0
    max_since_improve = 0

    for v in val_loss:
        if v < best:
            best = v
            since_improve = 0
        else:
            since_improve += 1
            max_since_improve = max(max_since_improve, since_improve)

    return max_since_improve

def plot_training_log(log_path, out_path=None, do_y_log=False, skip_first=False):
    
    epochs, train_loss, val_loss = [], [], []

    # Regex pattern to match the loss lines
    pattern = re.compile(r"Epoch (\d+)/\d+ - train_loss: ([0-9.]+) - val_loss: ([0-9.]+)")

    with open(log_path, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                e = int(match.group(1))
                if skip_first and e<2:
                    continue
                train = float(match.group(2))
                val = float(match.group(3))
                epochs.append(e)
                train_loss.append(train)
                val_loss.append(val)

    if not epochs:
        print("No epochs found in log — check your log format.")
        return

    required_patience = compute_required_patience(val_loss)
    print(f"[EarlyStopping] Minimum patience needed: {required_patience}")

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="Train Loss", marker="o", markersize=3)
    plt.plot(epochs, val_loss, label="Validation Loss", marker="o", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    if do_y_log:
        plt.yscale("log")
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
    #path = "logs/run_normal_5964510.out"
    
    mode = 'hourly'

    if mode=='daily':
        path = "output_new/0.02194_normal_Jan03_2011_256_0.0/run_normal_6225127.out"  # final daily model
    else:
        path = "output_new/0.0358_normal_Jan08_0328_256_0.0/run_normal_6311792.out" # final hourly model

    plot_training_log(log_path=path, out_path=f"src/visualization/training_plot_{mode}.png")"""