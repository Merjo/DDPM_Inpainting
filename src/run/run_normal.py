import torch
from src.model.song.song_unet import SongUNet
from src.model.diffusion import Diffusion
from src.data.loader import get_loader
from src.config import cfg
from src.model.schedulers.warmup_cosine import WarmupCosineScheduler

def run_model_normal(epochs, patience):
    loader = get_loader()

    n_channels = 128
    channel_mult = [1, 2, 4]

    attn_options = {
        "none": [],
        "last": [-1],
        "last_two": [-2, -1],
    }
    attn_stages = attn_options['none']
    resolutions = [cfg.patch_size // (2**i) for i in range(len(channel_mult))]
    attn_resolutions = [resolutions[i] for i in attn_stages]


    # Instantiate UNet
    unet = SongUNet(
        img_resolution=cfg.patch_size,
        in_channels=1,
        out_channels=1,
        model_channels=n_channels,
        channel_mult=channel_mult,
        num_blocks=2,
        attn_resolutions=attn_resolutions,
        dropout=0.175,
        encoder_type='residual',
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet = unet.to(device)

    # Diffusion model
    diffusion = Diffusion(unet, img_size=128, channels=1, timesteps=1000, beta_schedule='linear', loss_type='mse',)
    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=2e-4)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=3, total_steps=epochs)

    # Train
    best_rmse, epoch_losses = diffusion.train(
        loader, 
        optimizer, 
        epochs=epochs,
        patience=patience, 
        scheduler=scheduler, 
        trial=None, 
        log_every_epoch=True,
        sample_every=1,
        sample_info=f'Normal run, {n_channels} channels'
        )
    print(best_rmse)

if __name__=='__main__':
    run_model_normal(epochs=200,
                     patience=4)