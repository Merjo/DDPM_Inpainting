from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

def run_model_pl(epochs, patience):
    loader = get_loader()  # same as before

    # UNet same as before
    unet = SongUNet(
        img_resolution=cfg.patch_size,
        in_channels=1,
        out_channels=1,
        model_channels=256,
        channel_mult=[1,2,4,8],
        num_blocks=2,
        attn_resolutions=[cfg.patch_size//2],
        dropout=0.175,
        encoder_type='residual',
    )

    model = DiffusionPL(unet, timesteps=1000, beta_schedule='linear', loss_type='mse', lr=2e-4)

    # Callbacks
    early_stop = EarlyStopping(monitor="train_loss", patience=patience, mode="min")
    checkpoint = ModelCheckpoint(monitor="train_loss", mode="min")

    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",  # CPU/GPU/TPU auto
        devices=1,
        callbacks=[early_stop, checkpoint],
        log_every_n_steps=1,
    )

    trainer.fit(model, loader)
