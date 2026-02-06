import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import glob
import os
import pandas as pd
import torch

from src.config import cfg
from src.model.song.song_unet import SongUNet
from src.model.diffusion import Diffusion
from src.model.warmup_cosine import WarmupCosineScheduler
from src.save.save_model import save_model

from src.save.output_manager import OutputManager

def load_model(param_file, model_file, epochs=cfg.epochs, verbose=True):
    """Reload best UNet+Diffusion and continue training with new epochs."""
    device = cfg.device

    # --- Load params ---
    df = pd.read_csv(param_file)
    if df.shape[1] == 2 and set(df.columns) == {"param", "value"}:
        # key-value format
        params = pd.Series(df.value.values, index=df.param).to_dict()
    else:
        # wide format
        params = df.iloc[0].to_dict()

    # Cast numeric values (optional but useful)
    for k, v in params.items():
        try:
            params[k] = float(v)
            if params[k].is_integer():
                params[k] = int(params[k])
        except (ValueError, AttributeError):
            pass

    # Map back params that were categorical keys
    channel_mult_options = {
        124: [1, 2, 4],
        1224: [1, 2, 2, 4],
        1248: [1, 2, 4, 8],
        1124: [1, 1, 2, 4],
    }
    channel_mult = channel_mult_options[params["channel_mult"]]

    attn_options = {
        "none": [],
        "last": [-1],
        "last_two": [-2, -1],
    }
    attn_stages = attn_options[params["attn_config"]]
    resolutions = [cfg.patch_size // (2**i) for i in range(len(channel_mult))]
    attn_resolutions = [resolutions[i] for i in attn_stages]
    
    if verbose:
        print("UNet constructor args:")
        print("model_channels:", int(params["model_channels"]))
        print("num_blocks:", int(params["num_blocks"]))
        print("channel_mult:", channel_mult)
        print("attn_resolutions:", attn_resolutions)
        print("encoder_type:", params["downsample_type"])
        print("dropout:", float(params["dropout"]))
        print("patch_size:", cfg.patch_size)

    # --- Rebuild UNet ---
    unet = SongUNet(
        img_resolution=cfg.patch_size,
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

    checkpoint = torch.load(model_file, map_location=device, weights_only=False)

    # If your checkpoint contains a 'state_dict', extract it:
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Handle DataParallel prefix if present
    if any(k.startswith("module.") for k in state_dict.keys()):
        print("Removing 'module.' prefix from state dict keys...")
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}


    # --- Load weights ---
    unet.load_state_dict(state_dict=state_dict)


    # --- Rebuild Diffusion ---
    diffusion = Diffusion(
        model=unet,
        img_size=cfg.patch_size,
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


    return diffusion, unet, params, optimizer, scheduler


def find_best_saved_model():
    base_dir = cfg.output_base_dir
    run_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

    best_loss = float("inf")
    best_path = None

    for folder in run_folders:
        try:
            # Extract prefix as float (folder name starts with mse)
            prefix = folder.split("_")[0]
            loss_value = float(prefix)
        except ValueError:
            # Skip folders that don’t start with a float
            continue

        # If this folder’s loss is better than current best
        if loss_value < best_loss:
            best_loss = loss_value
            best_path = os.path.join(base_dir, folder)

    if best_path is None:
        raise RuntimeError("No valid run folders found.")
    
    print(f"Best model found in: {best_path} (loss={best_loss:.6f})")

    # Construct param and model filenames

    param_files = glob.glob(os.path.join(best_path, "*.csv"))
    if not param_files:
        raise FileNotFoundError(f"*.csv file found in {best_path}.")
    if len(param_files) > 1:
        raise Warning(f"Multiple *.csv files found in {best_path}.\n"
                      f"Using the first one found: {param_files[0]}")
    param_file = param_files[0]  # Assume only one such file
    print(f"Using params file: {param_file}")

    model_files = glob.glob(os.path.join(best_path, "*.pt"))

    if not model_files:
        raise FileNotFoundError(f"*.pt file found in {best_path}.")
    if len(model_files) > 1:
        raise Warning(f"Multiple *.pt files found in {best_path}.\n"
                      f"Using the first one found: {model_files[0]}")
    model_file = model_files[0]  # Assume only one such file
    print(f"Using model file: {model_file}")
    
    return param_file, model_file, best_loss

def load_best_model(epochs):
    param_file, model_file, best_loss = find_best_saved_model()
    diffusion, unet, params, optimizer, scheduler = load_model(param_file=param_file,
                                                                          model_file=model_file,
                                                                          epochs=epochs)
    return diffusion, unet, params, optimizer, scheduler


def run_best(param_file=None, 
             model_file=None,
             epochs=1,
             patience=3,
             device=cfg.device):
    if param_file is None or model_file is None:
        param_file, model_file, best_loss = find_best_saved_model()
    diffusion, unet, params, optimizer, scheduler = load_model(param_file=param_file,
                                                                          model_file=model_file,
                                                                          epochs=epochs)
                            
    # --- Continue training ---
    best_loss = diffusion.train(optimizer, train_loaders=cfg.train_loaders, val_loaders=cfg.val_loaders, epochs=epochs, scheduler=scheduler, patience=patience, log_every_epoch=True, sample_every=cfg.sample_every)

    return diffusion, unet, best_loss, params  # TODO: need unet/diffusion?


if __name__=='__main__':
    epochs = cfg.epochs
    patience = cfg.patience

    output = OutputManager(run_type="best")

    diffusion, unet, best_loss, params = run_best(epochs=epochs, patience=patience)


    output.finalize(best_loss, unet, epochs=epochs, params=params)