from src.config import cfg
from src.model.warmup_cosine import WarmupCosineScheduler
from src.save.output_manager import OutputManager
from src.model.song.song_unet import SongUNet
from src.model.diffusion import Diffusion
import torch

def prepare_run(epochs=cfg.epochs,
                model_channels=cfg.model_channels, 
                num_blocks=cfg.num_blocks, 
                dropout=cfg.dropout, 
                downsample_type=cfg.downsample_type, 
                channel_mult=cfg.channel_mult, 
                attn_config=cfg.attn_config,
                timesteps=cfg.timesteps, 
                beta_schedule=cfg.beta_schedule, 
                loss=cfg.loss, 
                optimizer_type=cfg.optimizer, 
                scheduler_type=cfg.scheduler, 
                lr=cfg.lr,
                model_file=None):

    channel_mult_options = {
        124: [1, 2, 4],
        1224: [1, 2, 2, 4],
        1248: [1, 2, 4, 8],
        1124: [1, 1, 2, 4],
    }
    channel_mult = channel_mult_options[int(channel_mult)]

    attn_options = {
        "none": [],
        "last": [-1],
        "last_two": [-2, -1],
    }
    attn_stages = attn_options[attn_config]
    resolutions = [cfg.patch_size // (2**i) for i in range(len(channel_mult))]
    attn_resolutions = [resolutions[i] for i in attn_stages]

    unet_class = SongUNet

    # --- Rebuild UNet ---
    unet = unet_class(
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

    if model_file is not None:
        # --- Load weights ---
        unet.load_state_dict(torch.load(model_file, map_location=cfg.device, weights_only=True))
    else:
        print("[Prepare Run] No model file provided, initialized new, random model.")


    # --- Rebuild Diffusion ---
    diffusion = Diffusion(
        model=unet,
        img_size=cfg.patch_size,
        channels=1,
        timesteps=timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        beta_schedule=beta_schedule,
        loss_type=loss
    )

    # --- Optimizer + Scheduler ---
    if optimizer_type == "Adam":
        OptimizerClass = torch.optim.Adam
    else:
        OptimizerClass = torch.optim.AdamW

    optimizer = OptimizerClass(unet.parameters(), lr=lr)

    if scheduler_type == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.xlr_scheduler_gamma)
    else:
        scheduler = WarmupCosineScheduler(optimizer, warmup_steps=cfg.wcs_scheduler_steps, total_steps=epochs)

    return diffusion, optimizer, scheduler
