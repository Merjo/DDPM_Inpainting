import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from src.model.song.song_unet import SongUNet
from src.model.diffusion import Diffusion
from src.config import cfg
from src.model.schedulers.warmup_cosine import WarmupCosineScheduler
from src.utils.output_manager import OutputManager
from src.run.prepare_run import prepare_run

def run_model_normal(epochs, 
                     patience,
                     model_channels,
                     num_blocks,
                     dropout,
                     downsample_type,
                     channel_mult,
                     attn_config,
                     timesteps,
                     beta_schedule,
                     loss,
                     optimizer,
                     scheduler,
                     lr):
    # Prepare model, diffusion, optimizer, scheduler
    diffusion, optimizer, scheduler = prepare_run(epochs,
                                                  model_channels,
                                                  num_blocks,
                                                  dropout,
                                                  downsample_type,
                                                  channel_mult,
                                                  attn_config,
                                                  timesteps,
                                                  beta_schedule,
                                                  loss,
                                                  optimizer,
                                                  scheduler,
                                                  lr,
                                                  model_file=None)

    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=cfg.wcs_scheduler_steps, total_steps=epochs)

    # Train
    best_rmse = diffusion.train(
        optimizer, 
        train_loaders=cfg.train_loaders, 
        val_loaders=cfg.val_loaders, 
        epochs=epochs,
        patience=patience, 
        scheduler=scheduler, 
        trial=None, 
        log_every_epoch=True,
        sample_every=cfg.sample_every,
        sample_info=f'Normal run, {model_channels} channels'
        )
    print(f'Best Loss: {best_rmse}')
    return best_rmse, diffusion

if __name__=='__main__':
    epochs=cfg.epochs
    patience=cfg.patience

    params = {
        'model_channels': cfg.model_channels,
        'num_blocks': cfg.num_blocks,
        'dropout': cfg.dropout,
        'downsample_type': cfg.downsample_type,
        'channel_mult': cfg.channel_mult,
        'attn_config': cfg.attn_config,
        'timesteps': cfg.timesteps,
        'beta_schedule': cfg.beta_schedule,
        'loss': cfg.loss,
        'optimizer': cfg.optimizer,
        'scheduler': cfg.scheduler,
        'lr': cfg.lr
    }

    output = OutputManager(run_type=f"normal_{cfg.model_type}")
    best_rmse, diffusion = run_model_normal(epochs=epochs,
                                            patience=patience,
                                            model_channels=params['model_channels'],
                                            num_blocks=params['num_blocks'],
                                            dropout=params['dropout'],
                                            downsample_type=params['downsample_type'],
                                            channel_mult=params['channel_mult'],
                                            attn_config=params['attn_config'],
                                            timesteps=params['timesteps'],
                                            beta_schedule=params['beta_schedule'],
                                            loss=params['loss'],
                                            optimizer=params['optimizer'],
                                            scheduler=params['scheduler'],
                                            lr=params['lr'])
    output.finalize(best_rmse, diffusion.model, epochs=epochs, params=params)
