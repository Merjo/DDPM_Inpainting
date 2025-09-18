import torch
from src.model.song.song_unet import SongUNet
from src.model.diffusion import Diffusion
from src.data.loader import get_loader

def run_model_normal():
    loader = get_loader()

    # Instantiate UNet
    unet = SongUNet(
        img_resolution=128,
        in_channels=1,
        out_channels=1,
        model_channels=128,
        channel_mult=[1, 2, 2, 4],
        num_blocks=3,
        attn_resolutions=[16],
        dropout=0.0916512675648172,
        encoder_type='residual',
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet = unet.to(device)

    # Diffusion model
    diffusion = Diffusion(unet, img_size=128, channels=1, timesteps=1000, beta_schedule='linear', loss_type='mse',)
    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=3.693821697838651e-05)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # Train
    best_rmse, epoch_losses = diffusion.train(
        loader, 
        optimizer, 
        epochs=150,
        patience=3, 
        scheduler=scheduler, 
        trial=None, 
        log_every_epoch=True,
        sample_every=1
        )
    print(best_rmse)

if __name__=='__main__':
    run_model_normal()