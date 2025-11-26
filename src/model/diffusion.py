import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
import optuna
from src.save.save_plot import plot_random, plot_histogram
from src.config import cfg
import datetime
import copy

class Diffusion:
    def __init__(
        self,
        model: nn.Module,
        img_size=cfg.patch_size,
        channels=1,
        timesteps=cfg.timesteps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        beta_schedule=cfg.beta_schedule,
        loss_type=cfg.loss,
        device=cfg.device,
        plot_dir=None,
        hist_dir=None,
    ):
        if plot_dir is None:
            plot_dir = f'{cfg.current_output}/samples'
        if hist_dir is None:
            hist_dir = f'{cfg.current_output}/histograms'
        if cfg.cuda and torch.cuda.device_count()>1:
            #model = torch.nn.DataParallel(model)
            model = model # TODO Decide
        self.model = model.to(device)
        self.device = device
        self.img_size = img_size
        self.channels = channels
        self.T = timesteps
        self.plot_dir = plot_dir
        self.hist_dir = hist_dir

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
        optimizer,
        train_loader=cfg.train_loader,
        val_loader=cfg.val_loader,
        epochs=cfg.epochs,
        scheduler=None,
        trial=None,
        patience=None,
        log_every_epoch=cfg.log_every_epoch,
        sample_every=cfg.sample_every,
        sample_info=None,
        min_patience_delta=cfg.min_patience_delta,
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
        """
        self.model.train()
        best_loss = float("inf")
        best_epoch = -1
        bad_epochs = 0
        last_epoch = epochs

        n_batches = len(train_loader)

        best_model = copy.deepcopy(self.model)

        do_save_model_regular = cfg.output_manager is not None and cfg.do_save_model_regular

        for epoch in range(epochs):
            datestr = datetime.datetime.now().strftime("%b%d_%H%M")
            print(f"[{datestr}] Starting epoch {epoch+1}/{epochs}...")
            total_loss = 0.0

            for imgs in train_loader:
                imgs = imgs.to(self.device)
                t = torch.randint(0, self.T, (imgs.size(0),), device=self.device)
                loss = self.p_losses(imgs, t)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()


            avg_loss = total_loss / n_batches
                
            val_loss = self.compute_val_loss(val_loader)
            
            if log_every_epoch:
                print(f"Epoch {epoch+1}/{epochs} - train_loss: {avg_loss:.6f} - val_loss: {val_loss:.6f}")

            if val_loss < best_loss - min_patience_delta:
                bad_epochs = 0
                best_loss = val_loss
                best_model = copy.deepcopy(self.model)
                best_epoch = epoch
            else:
                bad_epochs += 1
                if patience is not None and bad_epochs >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    last_epoch = epoch
                    break

            # Report to Optuna
            if trial is not None and val_loader is not None:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if scheduler is not None:
                scheduler.step()
            
            if sample_every is not None and (epoch + 1) % sample_every == 0 and (epoch+1)!=epochs:
                samples = self.sample(n_samples=cfg.n_hist_samples_regular)  # shape: (n, c, h, w)
                self.plot_samples(samples[:cfg.n_samples_regular], epoch + 1, sample_info)  
                if cfg.do_regular_hist:
                    self.plot_histogram(loader=train_loader, epoch=epoch, sample_info=sample_info, samples=samples)
                if do_save_model_regular:
                    cfg.output_manager.save_model(self.model, val_loss)
        
        self.model = best_model

        sample_info = f'{sample_info}\nBest Epoch: {best_epoch+1}, Val Loss: {best_loss:.6f}'

        samples = self.sample(n_samples=cfg.n_hist_samples)  # shape: (n, c, h, w)
        self.plot_samples(samples[:cfg.n_samples], last_epoch, sample_info)  
        self.plot_histogram(loader=train_loader, epoch=last_epoch, sample_info=sample_info, samples=samples)

        return best_loss
    
    def compute_val_loss(self, val_loader):
        self.model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for imgs in val_loader:
                imgs = imgs.to(self.device)
                t = torch.randint(0, self.T, (imgs.size(0),), device=self.device)
                loss = self.p_losses(imgs, t)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        self.model.train()
        return avg_val_loss

    
    @torch.no_grad()
    def plot_samples(self, samples, epoch, sample_info=None):
        samples = samples.detach().cpu()
        n = samples.size(0)
        title = f'Samples at epoch {epoch}'
        if sample_info is not None:
            title = sample_info + '\n' + title
        plot_random(samples, n=n, title = title, out_dir=self.plot_dir)

    @torch.no_grad()
    def plot_histogram(self, loader, epoch, sample_info=None, samples=None, n_samples=128):
        # --- Get generated samples ---
        if samples is None:
            generated = self.sample(n_samples=n_samples)
        else:
            generated = samples
            n_samples = samples.size(0)
        generated = generated.detach().cpu().numpy()

        # --- Get real samples ---
        real_batch = next(iter(loader))
        real = real_batch[:n_samples].to(self.device)
        real = real.detach().cpu().numpy()

        title = f'Histogram at epoch {epoch}'
        if sample_info is not None:
            title = sample_info + '\n' + title

        plot_histogram(real, generated, title, out_dir=self.hist_dir)
    

    @torch.no_grad()
    def sample(self, n_samples=16, chunk_size=16, verbose=True):
        """
        Sample n_samples images in memory-safe chunks.
        Returns tensor on CPU.
        """
        self.model.eval()
        device = self.device
        all_samples = []

        for start in range(0, n_samples, chunk_size):
            cur = min(chunk_size, n_samples - start)
            if verbose:
                datestr = datetime.datetime.now().strftime("%b%d_%H%M")
                print(f"[{datestr}] Sampling {start+1}-{start+cur} / {n_samples}")

            # Initialize noise for this chunk
            x = torch.randn(cur, self.channels, self.img_size, self.img_size, device=device)

            for t in reversed(range(self.T)):
                t_batch = torch.full((cur,), t, device=device, dtype=torch.long)
                pred_noise = self.model(x, t_batch)

                alpha_t = self.alpha[t]
                alpha_hat_t = self.alpha_hat[t]

                # Reverse diffusion step
                x = (1 / torch.sqrt(alpha_t)) * (
                    x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * pred_noise
                )

                if t > 0:
                    z = torch.randn_like(x)
                    beta_t = self.beta[t]
                    x = x + torch.sqrt(beta_t) * z

                if t > cfg.n_skip_clamp:  # Optional: skip last few timesteps
                    clamp_range = cfg.clamp_range_t(t, total_timesteps=self.T)
                    x = x.clamp(clamp_range[0], clamp_range[1])

            # Move finished samples to CPU and clean up GPU memory
            all_samples.append(x.detach().cpu())
            del x, t_batch, pred_noise
            gc.collect()
            torch.cuda.empty_cache()

        samples = torch.cat(all_samples, dim=0)

        # Check for NaNs
        if torch.isnan(samples).any():
            raise ValueError("NaN detected in samples!")

        return samples
    
    def inpaint_dps(self, x_known, mask, n_steps=None, chunk_size=4, lam=cfg.dps_lam, verbose=True):
        print(f'Starting inpainting with DPS, lambda={lam}')
        self.model.eval()
        T = n_steps or self.T
        device = x_known.device

        n_total = x_known.size(0)
        all_outputs = []

        for start in range(0, n_total, chunk_size):
            cur = min(chunk_size, n_total - start)
            if verbose:
                datestr = datetime.datetime.now().strftime("%b%d_%H%M")
                print(f"[{datestr}] Sampling {start+1}-{start+cur} / {n_total}")

            end = min(start + chunk_size, n_total)
            x_known_batch = x_known[start:end]
            mask_batch = mask[start:end]

            x = torch.randn_like(x_known_batch, requires_grad=True)  # start from noise
            alpha = self.alpha
            alpha_hat = self.alpha_hat
            beta = self.beta
            sqrt_alpha_hat = torch.sqrt(alpha_hat)
            sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat)
            for t in reversed(range(T)):
                if t % 50 == 0:
                    print(f'Inpainting, {(T - t) / 10}%')

                t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)

                # Z3 Predict Noise
                pred_noise = self.model(x, t_batch)

                # Z4 Compute x0_pred
                # from paper: x0_pred = (1 / sqrt_alpha_hat[t]) * (x + (1 - alpha_hat[t]) * pred_noise)
                x0_pred = (x - sqrt_one_minus_alpha_hat[t] * pred_noise) / sqrt_alpha_hat[t]

                # Z5 Compute z
                z = torch.randn_like(x)

                """# Z6 Update x (& Calc Gradient for Z7)
                term1 = ((sqrt_alpha[t]*(1 - alpha_hat[t-1])) / (1 - alpha_hat[t])) * x
                term2 = ((sqrt_alpha_hat[t-1]*beta[t]) / (1-alpha_hat[t])) * x0_pred
                term3 = sqrt_beta[t] * z

                x_dash = term1 + term2 + term3"""

                # Z6 Revision Update x

                mu = (1 / torch.sqrt(alpha[t])) * (
                    x - ((1 - alpha[t]) / sqrt_one_minus_alpha_hat[t]) * pred_noise
                )

                if t > 0:
                    x_dash = mu + torch.sqrt(beta[t]) * z
                else:
                    x_dash = mu

                # Z7 DPS Guidance
                residual = mask_batch * (x_known_batch - x0_pred)
                loss = (residual ** 2).sum()
                grad = torch.autograd.grad(loss, x)[0]
                guidance = lam * grad # * (1 - alpha_hat[t]) # Optional scaling TODO Decide
                x = (x_dash - guidance).detach().requires_grad_(True)

                # Optional clamping
                if t > cfg.n_skip_clamp:
                    clamp_range = cfg.clamp_range_t(t, total_timesteps=self.T)
                    x = x.clamp(clamp_range[0], clamp_range[1])

            x = x.detach()
            all_outputs.append(x.cpu())
            del x, x_known_batch, mask_batch, pred_noise, grad
            torch.cuda.empty_cache()

        result = torch.cat(all_outputs, dim=0)

        return result


    @torch.no_grad()
    def inpaint_dps_old(self, x_known, mask, n_steps=None, lam=cfg.dps_lam):
        self.model.eval()
        T = n_steps or self.T
        device = x_known.device
        x = torch.randn_like(x_known)  # start from noise
        for t in reversed(range(T)):
            t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)

            alpha_t = self.alpha[t] 
            alpha_hat_t = self.alpha_hat[t]
            beta_t = self.beta[t]
            # Z3 Predict Noise
            pred_noise = self.model(x, t_batch)

            # 2. DDPM mean
            mu_theta = (1 / torch.sqrt(alpha_t)) * (
                x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * pred_noise)
            # 3. Predict x0 from x_t
            x0_pred = (x - torch.sqrt(1 - alpha_hat_t) * pred_noise) / torch.sqrt(alpha_hat_t)
            # 4. Compute DPS guidance (nudging predicted x0 toward known pixels)
            guidance = lam * (1 - alpha_hat_t) * mask * (x_known - x0_pred)
            # 5. Update mean with DPS
            mu_dps = mu_theta + guidance

            # 6. Add noise for t>0
            x = mu_dps if t == 0 else mu_dps + torch.sqrt(beta_t) * torch.randn_like(x)

            # 7. Optional clamping
            if t > cfg.n_skip_clamp:
                clamp_range = cfg.clamp_range_t(t, total_timesteps=self.T)
                x = x.clamp(clamp_range[0], clamp_range[1])

        return x



    @torch.no_grad()
    def inpaint_old(self, x_known, mask, n_steps=None, lam=cfg.dps_lam, do_use_dps=cfg.do_use_dps):
        """
        Perform inpainting using the DDPM model.
        """
        self.model.eval()
        T = n_steps or self.T
        x = torch.randn_like(x_known)

        lam = lam if do_use_dps else 1

        for t in reversed(range(T)):
            if t % 50 == 0:
                print(f'Inpainting, {(T - t) / 10}%')

            t_batch = torch.full((x.size(0),), t, device=self.device, dtype=torch.long)
            pred_noise = self.model(x, t_batch)

            alpha_t = self.alpha[t]
            alpha_hat_t = self.alpha_hat[t]

            # DDPM reverse step
            x = (1 / torch.sqrt(alpha_t)) * (
                x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * pred_noise
            )

            if t > 0:
                z = torch.randn_like(x)
                beta_t = self.beta[t]
                x = x + torch.sqrt(beta_t) * z

            if t > cfg.n_skip_clamp:  # Optional: skip last few timesteps
                    clamp_range = cfg.clamp_range_t(t, total_timesteps=self.T)
                    x = x.clamp(clamp_range[0], clamp_range[1])

            # Inpainting step: enforce known pixels
            if t > T * cfg.dps_hard_overwrite:
                # Early steps → soft guidance
                x = x + lam * mask * (x_known - x)
            else:
                # Late steps → hard overwrite
                x = mask * x_known + (1 - mask) * x

        return x

    