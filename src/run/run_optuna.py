import torch
import time
from datetime import datetime
import optuna
import pandas as pd
from collections import defaultdict
import os

from src.data.loader import get_loader
from src.model.schedulers.warmup_cosine import WarmupCosineScheduler
from src.model.song.song_unet import SongUNet
from src.model.diffusion import Diffusion
from src.config import cfg
from src.save.save_model import save_model

def objective(trial, loader, run_dir, max_epochs=20, patience=3):
    trial.set_user_attr("duration", float('inf'))
    start_time = time.time()
    
    # --- UNET hyperparameters ---
    model_channels = trial.suggest_categorical("model_channels", [64, 128, 256])
    num_blocks = trial.suggest_int("num_blocks", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)

    downsample_type = trial.suggest_categorical("downsample_type", ["residual", "standard"])
    
    channel_mult_options = {
        "124": [1, 2, 4],
        "1224": [1, 2, 2, 4],
        "1248": [1, 2, 4, 8],
        "1124": [1, 1, 2, 4],
    }
    channel_mult_key = trial.suggest_categorical("channel_mult", list(channel_mult_options.keys()))
    channel_mult = channel_mult_options[channel_mult_key]

    attn_options = {
        "none": [],
        "last": [-1],        
        "last_two": [-2, -1],
    }
    attn_key = trial.suggest_categorical("attn_config", list(attn_options.keys()))
    attn_stages = attn_options[attn_key]
    
    resolutions = [128 // (2**i) for i in range(len(channel_mult))]
    attn_resolutions = [resolutions[i] for i in attn_stages]

    # --- Build UNet ---
    unet = SongUNet(
        img_resolution=128,
        in_channels=1,
        out_channels=1,
        model_channels=model_channels,
        channel_mult=channel_mult,
        num_blocks=num_blocks,
        attn_resolutions=attn_resolutions,
        label_dim=0,
        dropout=dropout,
        encoder_type=downsample_type,
    ).to(cfg.device)
    
    # --- Diffusion hyperparameters ---
    timesteps = trial.suggest_categorical("timesteps", [250, 500, 1000])
    beta_schedule = trial.suggest_categorical("beta_schedule", ["linear", "quadratic", "exponential", "cosine"])
    loss_type = trial.suggest_categorical("loss", ["mse", "l1", "huber"])

    diffusion = Diffusion(
        model=unet,
        img_size=128,
        channels=1,
        timesteps=timesteps,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule=beta_schedule,
        loss_type=loss_type,
        device=cfg.device,
    )
    
    # --- Training hyperparameters ---
    optimizer_type = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    if optimizer_type == "Adam":
        OptimizerClass = torch.optim.Adam
    else:
        OptimizerClass = torch.optim.AdamW

    # Learning rate search space depends on scheduler
    scheduler_type = trial.suggest_categorical("scheduler", ["ExponentialLR", "WarmupCosine"])
    if scheduler_type == "WarmupCosine":
        lr = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    else:
        lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)

    optimizer = OptimizerClass(unet.parameters(), lr=lr)

    if scheduler_type == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    else:
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps=3, total_steps=max_epochs)

    # --- Training ---
    best_rmse, epoch_losses = diffusion.train(
        loader, optimizer, epochs=max_epochs, scheduler=scheduler, trial=trial, patience=patience, sample_every=cfg.optuna_sample_every
    )

    # --- Save checkpoint for this trial ---
    ckpt_name = f"{run_dir}/unet_trial_{trial.number}.pt"
    torch.save(unet.state_dict(), ckpt_name)
    trial.set_user_attr("checkpoint", ckpt_name)
    
    # --- Save duration ---
    duration = time.time() - start_time
    trial.set_user_attr("duration", duration)

    if cfg.cuda:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return best_rmse

def run_optuna(n_trials=25, max_epochs=15, patience=2, resume=False):
    loader = get_loader()
    #study = optuna.create_study(direction="minimize")
    study = optuna.create_study(
        direction='minimize',
        study_name = "rain_diffusion" if resume else f"rain_diffusion_{int(time.time())}",
        storage="sqlite:///optuna_rain.db",
        load_if_exists=resume,
    )

    base_dir = "output/trials"
    existing_runs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("optuna_run_")]
    run_numbers = [int(d.split("_")[-1]) for d in existing_runs if d.split("_")[-1].isdigit()]
    next_run_number = max(run_numbers, default=0) + 1

    # Create new folder for this Optuna run
    run_dir = os.path.join(base_dir, f"optuna_run_{next_run_number}")
    os.makedirs(run_dir, exist_ok=True)


    study.optimize(lambda trial: objective(trial, loader=loader, run_dir=run_dir, max_epochs=max_epochs, patience=patience), n_trials=n_trials)

    # --- Best trial ---
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value
    best_model = torch.load(best_trial.user_attrs["checkpoint"])

    params_filename, model_filename = save_model(best_params, best_model, best_value)
    
    # --- Parameter-wise ranking summary ---
    summary = defaultdict(lambda: defaultdict(list))

    for trial in study.trials:
        if trial.value is None:
            continue
        for param, val in trial.params.items():
            for param, val in trial.params.items():
                duration = trial.user_attrs.get("duration", float("inf"))
                summary[param][val].append((trial.value, duration))

    print("\n=== Parameter-wise Summary ===")
    for param, choices in summary.items():
        print(f"\nParameter: {param}")
        for val, results in choices.items():
            avg_value = sum(v for v, _ in results) / len(results)
            avg_time = sum(t for _, t in results) / len(results)
            print(f"  {val}: avg RMSE={avg_value:.4f}, avg time={avg_time:.1f}s over {len(results)} trials")


    return params_filename, model_filename


if __name__=='__main__':
    max_epochs = 15
    patience = 2
    n_trials = 60

    params_file, model_file = run_optuna(n_trials=n_trials, max_epochs=max_epochs, patience=patience)

    print(params_file)
    print(model_file)