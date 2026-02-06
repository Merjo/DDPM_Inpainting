import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import time
from datetime import datetime
import optuna
import pandas as pd
from collections import defaultdict
import os

from src.model.warmup_cosine import WarmupCosineScheduler
from src.model.song.song_unet import SongUNet
from src.model.diffusion import Diffusion
from src.config import cfg

from src.save.output_manager import OutputManager

def get_unet_parameters(trial):
    # --- UNET hyperparameters ---
    if 'model_channels' in cfg.optuna_search_space.keys():
        model_channels = trial.suggest_categorical("model_channels", cfg.optuna_search_space['model_channels'])
    else:
        model_channels = cfg.model_channels

    if 'num_blocks' in cfg.optuna_search_space.keys():
        num_blocks_range = cfg.optuna_search_space['num_blocks']
        num_blocks = trial.suggest_int("num_blocks", num_blocks_range[0], num_blocks_range[1])
    else:
        num_blocks = cfg.num_blocks

    if 'dropout' in cfg.optuna_search_space.keys():
        dropout_range = cfg.optuna_search_space['dropout']
        dropout = trial.suggest_float("dropout", dropout_range[0], dropout_range[1])
    else:
        dropout = cfg.dropout

    if 'downsample_type' in cfg.optuna_search_space.keys():
        downsample_type = trial.suggest_categorical("downsample_type", cfg.optuna_search_space['downsample_type'])
    else:
        downsample_type = cfg.downsample_type

    if 'channel_mult' in cfg.optuna_search_space.keys():
        channel_mult = trial.suggest_categorical("channel_mult", cfg.optuna_search_space['channel_mult'])
    else:
        channel_mult = cfg.channel_mult

    channel_mult = [int(ch) for ch in channel_mult]


    attn_options = {
            "none": [],
            "last": [-1],        
            #"last_two": [-2, -1], #  TODO: Decide whether to keep last two, can be very intense with 256 patch size
        }
    if 'attn_config' in cfg.optuna_search_space.keys():

        attn_key = trial.suggest_categorical("attn_config", list(attn_options.keys()))
    else:
        attn_key = cfg.attn_config

    attn_stages = attn_options[attn_key]
    resolutions = [cfg.patch_size // (2**i) for i in range(len(channel_mult))]
    attn_resolutions = [resolutions[i] for i in attn_stages]

    return model_channels, num_blocks, dropout, downsample_type, channel_mult, attn_resolutions


def get_diffusion_parameters(trial):
    # --- Diffusion hyperparameters ---
    if 'timesteps' in cfg.optuna_search_space.keys():
        timesteps = trial.suggest_categorical("timesteps", cfg.optuna_search_space['timesteps'])
    else:
        timesteps = cfg.timesteps

    if 'beta_schedule' in cfg.optuna_search_space.keys():
        beta_schedule = trial.suggest_categorical("beta_schedule", cfg.optuna_search_space['beta_schedule'])
    else:
        beta_schedule = cfg.beta_schedule
    if 'loss' in cfg.optuna_search_space.keys():
        loss_type = trial.suggest_categorical("loss", cfg.optuna_search_space['loss'])
    else:
        loss_type = cfg.loss

    return timesteps, beta_schedule, loss_type

def get_training_parameters(trial):
    # --- Training hyperparameters ---
    if 'optimizer' in cfg.optuna_search_space.keys():
        optimizer_type = trial.suggest_categorical("optimizer", cfg.optuna_search_space['optimizer'])
    else:
        optimizer_type = cfg.optimizer
    if 'scheduler' in cfg.optuna_search_space.keys():
        scheduler_type = trial.suggest_categorical("scheduler", cfg.optuna_search_space['scheduler'])
    else:
        scheduler_type = cfg.scheduler
    
    if 'lr' in cfg.optuna_search_space:
        if scheduler_type == "WarmupCosine":
            lr_range = cfg.optuna_search_space['lr_wc']
        elif scheduler_type == "ExponentialLR":
            lr_range = cfg.optuna_search_space['lr_xr']
        else:
            lr_range = cfg.optuna_search_space['lr']
        lr = trial.suggest_float("lr", lr_range[0], lr_range[1], log=True)
    else:
        lr = cfg.lr
    return optimizer_type, scheduler_type, lr


def objective(trial, loaders, run_dir, max_epochs=cfg.optuna_epochs, patience=cfg.optuna_patience):
    #device_id = trial.number % torch.cuda.device_count()
    #torch.cuda.set_device(device_id)

    trial.set_user_attr("duration", float('inf'))  # TODO fix
    start_time = time.time()

    # --- Build UNet ---

    model_channels, num_blocks, dropout, downsample_type, channel_mult, attn_resolutions = get_unet_parameters(trial)

    unet = SongUNet(
        img_resolution=cfg.patch_size,
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

    # --- Build Diffusion ---
    
    timesteps, beta_schedule, loss_type = get_diffusion_parameters(trial)

    diffusion = Diffusion(
        model=unet,
        img_size=cfg.patch_size,
        channels=1,
        timesteps=timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        beta_schedule=beta_schedule,
        loss_type=loss_type,
        device=cfg.device
    )
    
    # --- Training hyperparameters ---
    optimizer_type, scheduler_type, lr = get_training_parameters(trial)

    if optimizer_type == "Adam":
        OptimizerClass = torch.optim.Adam
    elif optimizer_type == "AdamW":
        OptimizerClass = torch.optim.AdamW
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    optimizer = OptimizerClass(unet.parameters(), lr=lr)

    if scheduler_type == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.xlr_scheduler_gamma)
    else:
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps=cfg.wcs_scheduler_steps, total_steps=max_epochs)

    # --- Training ---
    best_loss = diffusion.train(
        optimizer, train_loaders=cfg.train_loaders, val_loaders=cfg.val_loaders, epochs=max_epochs, scheduler=scheduler, trial=trial, patience=patience, sample_every=cfg.optuna_sample_every, sample_info=f'Trial {trial.number}'
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

    return best_loss

def run_optuna(n_trials=cfg.optuna_n_trials, max_epochs=cfg.optuna_epochs, patience=cfg.optuna_patience, resume=False, run_dir=None, return_unet=False):
    if run_dir is None:
        run_dir = f'{cfg.current_output}/trials'
    loaders = cfg.val_loaders
    study = optuna.create_study(
        direction='minimize',
        study_name = "rain_diffusion" if resume else f"rain_diffusion_{int(time.time())}",
        storage="sqlite:///optuna_rain.db",
        load_if_exists=resume,
    )

    study.optimize(lambda trial: objective(trial, loaders=loaders, run_dir=run_dir, max_epochs=max_epochs, patience=patience), n_trials=n_trials, n_jobs=1)#torch.cuda.device_count())

    # --- Best trial ---
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value
    best_model = torch.load(best_trial.user_attrs["checkpoint"])

    
    # --- Parameter-wise ranking summary ---
    summary = defaultdict(lambda: defaultdict(list))

    for trial in study.trials:
        if trial.value is None:
            continue
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

    if return_unet:
        model_channels, num_blocks, dropout, downsample_type, channel_mult, attn_resolutions = \
            get_unet_parameters(best_trial)

        best_unet = SongUNet(
            img_resolution=cfg.patch_size,
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

        # --- Load weights ---
        state_dict = torch.load(
            best_trial.user_attrs["checkpoint"],
            weights_only=True
        )
        best_unet.load_state_dict(state_dict)
        best_model = best_unet

    return best_model, best_value, best_params


if __name__=='__main__':
    if not cfg.optuna_mode:
        raise Exception('Config should be in Optuna Mode')
    output = OutputManager(run_type="optuna")

    unet, best_loss, params = run_optuna(return_unet=True)

    output.finalize(best_loss, unet, epochs=cfg.optuna_epochs, params=params)