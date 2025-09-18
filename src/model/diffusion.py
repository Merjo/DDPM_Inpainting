import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import optuna
from src.visualization.save_plot import plot_random

class Diffusion:
    def __init__(
        self,
        model: nn.Module,
        img_size=128,
        channels=1,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        beta_schedule="linear",
        loss_type="mse",
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.img_size = img_size
        self.channels = channels
        self.T = timesteps

        # Beta schedule
        self.beta = self.make_beta_schedule(beta_start, beta_end, self.T, beta_schedule)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        # Loss function
        if loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "l1":
            self.criterion = nn.L1Loss()
        elif loss_type == "huber":
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        
    def make_beta_schedule(self, beta_start, beta_end, T, schedule):
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, T, device=self.device)
        elif schedule == "quadratic":
            return torch.linspace(beta_start**0.5, beta_end**0.5, T, device=self.device) ** 2
        elif schedule == "exponential":
            return beta_start * (beta_end / beta_start) ** (torch.arange(T, device=self.device) / (T-1))
        elif schedule == "cosine":
            # Cosine schedule based on Nichol & Dhariwal 2021
            s = 0.008
            steps = torch.arange(T+1, device=self.device, dtype=torch.float64)
            alphas_cumprod = torch.cos(((steps / T) + s) / (1 + s) * torch.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return betas.float()
        else:
            raise ValueError(f"Unknown beta schedule: {schedule}")



    def q_sample(self, x0, t, noise=None):
        """Forward diffusion q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise
    
    
    def p_losses(self, x0, t, loss_function=None):
        if loss_function is None:
            loss_function = self.criterion
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = self.model(xt, t)  # UNet predicts the noise
        loss = loss_function(pred, noise)  # Compare predicted noise vs actual noise
        return loss


            
    def train(
        self,
        dataloader,
        optimizer,
        epochs=10,
        scheduler=None,
        trial=None,
        patience=None,
        log_every_epoch=False,
        sample_every=None
    ):
        """
        Train the model with optional logging per epoch.

        Args:
            dataloader: PyTorch DataLoader.
            optimizer: Optimizer instance.
            epochs: Number of epochs.
            scheduler: Optional LR scheduler.
            trial: Optional Optuna trial for pruning.
            patience: Optional early stopping patience.
            log_every_epoch: If True, prints average loss and RMSE each epoch.

        Returns:
            best_rmse: Lowest RMSE achieved.
            epoch_losses: List of average losses per epoch.
        """
        self.model.train()
        best_rmse = float("inf")
        best_loss = float("inf")
        bad_epochs = 0
        epoch_losses = []

        for epoch in range(epochs):
            total_loss = 0.0
            total_rmse = 0.0
            count = 0
            for imgs in dataloader:
                imgs = imgs.to(self.device)
                t = torch.randint(0, self.T, (imgs.size(0),), device=self.device)
                loss = self.p_losses(imgs, t)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                count += 1

                # Compute RMSE for trial reporting
                with torch.no_grad():
                    rmse = self.p_losses(imgs, t, loss_function=F.mse_loss)
                    total_rmse += rmse.item()

            avg_loss = total_loss / count
            avg_rmse = total_rmse / count
            epoch_losses.append(avg_loss)

            # Logging
            if log_every_epoch:
                print(f"Epoch {epoch+1}/{epochs} - avg_loss: {avg_loss:.6f}, avg_rmse: {avg_rmse:.6f}")

            # Track best RMSE
            if avg_rmse < best_rmse:
                best_rmse = avg_rmse
                
                
            if avg_loss < best_loss:
                bad_epochs = 0  # reset early stopping counter
                best_loss = avg_loss
            else:
                if patience is not None:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        if log_every_epoch:
                            print(f"Early stopping at epoch {epoch+1}")
                        break

            # Report to Optuna
            if trial is not None:
                trial.report(avg_rmse, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if scheduler is not None:
                scheduler.step()
            
            if sample_every is not None and (epoch + 1) % sample_every == 0:
                samples = self.sample()  # shape: (n, c, h, w)
                self.plot_samples(samples, epoch + 1)  

        return best_rmse, epoch_losses
    
    def plot_samples(self, samples, epoch):
        samples = samples.detach().cpu()
        n = samples.size(0)
        plot_random(samples, n=n, title = f"Samples at epoch {epoch}")


    @torch.no_grad()
    def sample(self, n_samples=8):
        self.model.eval()
        x = torch.randn(n_samples, self.channels, self.img_size, self.img_size, device=self.device)

        for t in reversed(range(self.T)):
            t_batch = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
            pred_noise = self.model(x, t_batch)

            alpha_t = self.alpha[t]
            alpha_hat_t = self.alpha_hat[t]

            # DDPM reverse update
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * pred_noise)

            if t > 0:
                z = torch.randn_like(x)
                beta_t = self.beta[t]
                x = x + torch.sqrt(beta_t) * z

        return x.clamp(0, 1)
