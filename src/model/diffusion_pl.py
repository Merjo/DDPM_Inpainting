import pytorch_lightning as pl
import torch
import torch.nn.functional as F

class DiffusionPL(pl.LightningModule):
    def __init__(self, model, timesteps=1000, beta_schedule="linear", loss_type="mse", lr=2e-4, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.T = timesteps
        self.lr = lr

        # Beta schedule
        self.beta = self.make_beta_schedule(beta_schedule)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        # Loss
        if loss_type == "mse":
            self.criterion = F.mse_loss
        elif loss_type == "l1":
            self.criterion = F.l1_loss
        else:
            raise ValueError(loss_type)

    def make_beta_schedule(self, schedule):
        if schedule == "linear":
            return torch.linspace(1e-4, 0.02, self.T)
        else:
            raise NotImplementedError(schedule)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise

    def p_losses(self, x0, t):
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = self.model(xt, t)
        return self.criterion(pred, noise)

    # -------------------
    # Lightning Hooks
    # -------------------
    def forward(self, x, t):
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        imgs = batch
        t = torch.randint(0, self.T, (imgs.size(0),), device=imgs.device)
        loss = self.p_losses(imgs, t)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        # Scheduler can also be returned as a dict for PL
        return optimizer
