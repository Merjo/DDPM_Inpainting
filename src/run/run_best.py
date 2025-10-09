import glob
import os
import pandas as pd
import torch

from src.config import cfg
from src.model.song.song_unet import SongUNet
from src.model.diffusion import Diffusion
from src.model.schedulers.warmup_cosine import WarmupCosineScheduler
from src.data.loader import get_loader
from src.save.save_model import save_model

def train_best_model(param_file, model_file, loader, epochs=10, patience=3, device="cuda"):
    """Reload best UNet+Diffusion and continue training with new epochs."""

    # --- Load params ---
    params = pd.read_csv(param_file).iloc[0].to_dict()

    # Map back params that were categorical keys
    channel_mult_options = {
        124: [1, 2, 4],
        1224: [1, 2, 2, 4],
        1248: [1, 2, 4, 8],
        1124: [1, 1, 2, 4],
    }
    channel_mult = channel_mult_options[params["channel_mult"]]

    channel_mult = channel_mult_options[params["channel_mult"]]

    attn_options = {
        "none": [],
        "last": [-1],
        "last_two": [-2, -1],
    }
    attn_stages = attn_options[params["attn_config"]]
    resolutions = [128 // (2**i) for i in range(len(channel_mult))]
    attn_resolutions = [resolutions[i] for i in attn_stages]

    # --- Rebuild UNet ---
    unet = SongUNet(
        img_resolution=128,
        in_channels=1,
        out_channels=1,
        model_channels=int(params["model_channels"]),
        channel_mult=channel_mult,
        num_blocks=int(params["num_blocks"]),
        attn_resolutions=attn_resolutions,
        label_dim=0,
        dropout=float(params["dropout"]),
        encoder_type=params["downsample_type"],
    ).to(device)

    # --- Load weights ---
    unet.load_state_dict(torch.load(model_file, map_location=device, weights_only=True))


    # --- Rebuild Diffusion ---
    diffusion = Diffusion(
        model=unet,
        img_size=128,
        channels=1,
        timesteps=int(params["timesteps"]),
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule=params["beta_schedule"],
        loss_type=params["loss"],
        device=device,
    )

    # --- Optimizer + Scheduler ---
    if params["optimizer"] == "Adam":
        OptimizerClass = torch.optim.Adam
    else:
        OptimizerClass = torch.optim.AdamW

    lr = float(params["lr"])
    optimizer = OptimizerClass(unet.parameters(), lr=lr)

    if params["scheduler"] == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    else:
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps=3, total_steps=epochs)

    # --- Continue training ---
    best_loss, epoch_losses = diffusion.train(loader, optimizer, epochs=epochs, scheduler=scheduler, patience=patience, log_every_epoch=True, sample_every=cfg.sample_every)

    params_filename, model_filename = save_model(params, unet, best_loss)


    return diffusion, unet, best_loss


def find_best_saved_model():
    """
    Finds the best saved model by scanning all best_params_*.csv files in `directory`
    and returning the param + model filenames with the lowest loss_value.
    
    Returns:
        best_params_file (str): Path to the best params CSV
        best_model_file (str): Path to the best model checkpoint
        best_loss (float): Lowest loss value found
    """
    param_files = glob.glob(os.path.join("output/params", "best_params_*.csv"))
    if not param_files:
        raise FileNotFoundError("No best_params_*.csv files found.")

    best_loss = float("inf")
    best_params_file, best_model_file = None, None

    for pf in param_files:
        try:
            df = pd.read_csv(pf)
            loss_value = float(df["loss_value"].iloc[0])
        except Exception as e:
            print(f"Skipping {pf}, error reading: {e}")
            continue

        if loss_value < best_loss:
            best_loss = loss_value
            best_params_file = pf

            # model filename should match the timestamp in params filename
            base = os.path.basename(pf).replace("best_params_", "").replace(".csv", "")
            candidate_model = os.path.join("output/models", f"best_model_{base}.pt")

            if os.path.exists(candidate_model):
                best_model_file = candidate_model
            else:
                print(f"Warning: matching model file not found for {pf}")

    if best_params_file is None:
        raise RuntimeError("No valid best_params_*.csv files found.")

    print(f"Best model: {best_model_file} (loss={best_loss:.6f})")
    return best_params_file, best_model_file, best_loss


def run_best(param_file=None, 
             model_file=None,
             epochs=1,
             patience=3,
             device=cfg.device):
    loader = get_loader()
    if param_file is None or model_file is None:
        param_file, model_file, best_loss = find_best_saved_model()
    diffusion, unet, best_loss = train_best_model(param_file=param_file,
                                                  model_file=model_file,
                                                  loader=loader,
                                                  epochs=epochs,
                                                  patience=patience,
                                                  device=device)

    return diffusion, unet, best_loss


if __name__=='__main__':
    epochs = 1
    patience = 3
    run_best(epochs=epochs, patience=patience)