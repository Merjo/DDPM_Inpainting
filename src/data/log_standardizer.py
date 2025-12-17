import torch
import joblib
import os
import time


def progress_log(i, total, start_time, prefix=""):
    """Lightweight HPC-friendly progress logger."""
    pct = 100 * (i / total)
    elapsed = time.time() - start_time
    rate = i / elapsed if elapsed > 0 else 0
    eta = (total - i) / rate if rate > 0 else float("inf")

    print(f"{prefix} [{i}/{total}]  {pct:5.1f}% | "
          f"elapsed: {elapsed:6.1f}s | ETA: {eta:6.1f}s")


class LogStandardizerStreaming:
    """
    Fully streaming LogStandardizer for GPU.
    Computes mean/std and best c without loading entire dataset.
    """
    def __init__(self, c=None, eps=1e-6, chunk_size=24, device=None):
        self.c = c
        self.eps = eps
        self.chunk_size = chunk_size
        self.fitted = False
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.std = None
        self.mean = None

    def iterate_chunks(self, data):
        T = data.sizes["time"]
        for start in range(0, T, self.chunk_size):
            end = min(start + self.chunk_size, T)
            chunk = torch.tensor(
                data.isel(time=slice(start, end)).values.astype("float32"),
                device=self.device
            )
            yield chunk

    @staticmethod
    def compute_score(y):
        """
        Compute |skew| + |kurtosis - 3| in a fully vectorized way
        y: torch.Tensor on correct device
        """
        n = y.numel()
        if n == 0:
            return torch.tensor(float("nan"), device=y.device)

        mean = torch.mean(y)
        diff = y - mean
        var = torch.mean(diff ** 2) + 1e-12
        std = torch.sqrt(var)

        skew_val = torch.mean(diff ** 3) / (std**3 + 1e-12)
        kurt_val = torch.mean(diff ** 4) / (var**2 + 1e-12)

        score = torch.abs(skew_val) + torch.abs(kurt_val - 3.0)
        return score

    def select_c(self, data):
        print("[Scaler] Selecting best c (streaming on GPU)...")
        start_time = time.time()
        candidate_c = [2 ** i for i in range(-10, 11, 2)]  # from 1/1024 to 1024

        scores = {c: [] for c in candidate_c}

        T = data.sizes["time"]
        total_chunks = (T + self.chunk_size - 1) // self.chunk_size

        for i, chunk in enumerate(self.iterate_chunks(data)):
            chunk = chunk[torch.isfinite(chunk)]
            if chunk.numel() == 0:
                continue

            for c in candidate_c:
                y = torch.log(chunk / c + 1.0)
                scores[c].append(self.compute_score(y))

            if i % 25 == 0:
                progress_log(i, total_chunks, start_time, prefix="[c-select]")

        # average scores over chunks
        avg_scores = {}
        for c in candidate_c:
            if len(scores[c]) > 0:
                avg_scores[c] = torch.stack(scores[c]).mean().item()
            else:
                avg_scores[c] = float("inf")  # fallback if empty

        for c, s in avg_scores.items():
            print(f"   c={c} → score={s:.6f}")

        best_c = min(avg_scores, key=avg_scores.get)
        self.c = best_c
        print(f"[Scaler] Selected c={self.c}")
        return best_c

    def fit(self, data):
        print("[Scaler] START streaming fit (GPU)")
        if self.c is None:
            self.select_c(data)

        print("[Scaler] Computing mean/std (streaming, GPU)...")
        start_time = time.time()

        count = 0
        mean = torch.tensor(0.0, device=self.device)
        m2 = torch.tensor(0.0, device=self.device)

        T = data.sizes["time"]
        total_chunks = (T + self.chunk_size - 1) // self.chunk_size

        for i, chunk in enumerate(self.iterate_chunks(data)):
            chunk = chunk[torch.isfinite(chunk)]
            if chunk.numel() == 0:
                continue

            y = torch.log(chunk / self.c + 1.0)

            # Vectorized Welford batch update
            n = y.numel()
            batch_mean = torch.mean(y)
            batch_var = torch.var(y, unbiased=False)

            delta = batch_mean - mean
            new_count = count + n

            m2 = m2 + batch_var * n + delta**2 * count * n / new_count
            mean = mean + delta * n / new_count
            count = new_count

            if i % 25 == 0:
                progress_log(i, total_chunks, start_time, prefix="[fit]")

        variance = m2 / (count - 1)
        self.mean = float(mean)
        self.std = float(torch.sqrt(variance) + self.eps)

        self.fitted = True
        print(f"[Scaler] DONE: mean={self.mean:.6f}, std={self.std:.6f}")

        return self

    def encode(self, x):
        x = x.to(self.device) if isinstance(x, torch.Tensor) else torch.tensor(x, device=self.device)
        y = torch.log(x / self.c + 1.0)
        return (y - self.mean) / self.std

    def decode(self, y):
        y = y.to(self.device) if isinstance(y, torch.Tensor) else torch.tensor(y, device=self.device)
        z = y * self.std + self.mean
        return (torch.exp(z) - 1.0) * self.c

    def __eq__(self, other):
        if not isinstance(other, LogStandardizerStreaming):
            return False
        
        # Compare scalars
        attrs_equal = (
            self.c == other.c and
            self.eps == other.eps and
            self.chunk_size == other.chunk_size and
            self.device == other.device and
            self.fitted == other.fitted
        )

        # Compare mean/std if fitted
        if self.fitted and other.fitted:
            attrs_equal = attrs_equal and (abs(self.mean - other.mean) < 1e-6) and (abs(self.std - other.std) < 1e-6)
        elif self.fitted != other.fitted:
            # One is fitted, the other not
            return False

        return attrs_equal


def create_scaler(years):
    print("[Scaler] Reading dataset (lazy xarray)")
    from src.data.read_data import read_raw_data
    data = read_raw_data(years=years)

    scaler = LogStandardizerStreaming(c=None, chunk_size=24)
    scaler.fit(data)
    return scaler


def load_scaler(reload, cache_path, years, time_slices):
    cache_path = f"{cache_path}/scaler_{years[0]}_{years[-1]}{'' if time_slices is None else f'_{time_slices}'}.pkl"

    if reload or not os.path.exists(cache_path):
        scaler = create_scaler(years)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        joblib.dump(scaler, cache_path)
        print(f"[Scaler] Saved scaler to {cache_path}")
        return scaler

    print(f"[Scaler] Loaded scaler from cache: {cache_path}")
    return joblib.load(cache_path)


if __name__ == "__main__":
    load_scaler(reload=True,
                cache_path="../../../../p/tmp/merlinho/cache",
                years=range(2001, 2018),
                time_slices=None)
    

import torch
from torch.utils.data import Dataset
import numpy as np

import torch


from scipy.stats import skew, kurtosis

import numpy as np
import torch
from torch.utils.data import Dataset

# --- Old LogStandardizer for reference --- #
class LogStandardizer:
    """
    y = (log(x/c + 1) - mean) / std
    Inverse: x = (exp(y * std + mean) - 1) * c

    If c=None, automatically finds the optimal value to reduce skew/kurtosis.
    """
    def __init__(self, c=None, eps: float = 1e-6):
        self.c = c
        self.eps = eps
        self.fitted = False

    @torch.no_grad()
    def fit(self, dataset: Dataset, num_samples: int = 1024, batch_size: int = 32):
        # --- Sample patches ---
        idxs = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
        vals = []
        for i in range(0, len(idxs), batch_size):
            batch_idxs = idxs[i:i+batch_size]
            batch = [dataset[j].squeeze(0) for j in batch_idxs]
            x = torch.stack(batch, dim=0)
            vals.append(x)
        big = torch.cat(vals, dim=0).numpy().flatten()

        # --- Automatic c search if c is None ---
        if self.c is None:
            candidate_c = [0.1, 0.25, 0.5, 1.0, 2.0]
            best_score = float('inf')
            best_c = None

            for c in candidate_c:
                y = np.log(big / c + 1.0)
                s = abs(skew(y)) + abs(kurtosis(y, fisher=False) - 3)
                if s < best_score:
                    best_score = s
                    best_c = c

            self.c = best_c
            print(f"[LogScaler] Auto-selected c = {self.c:.4f}")

        # --- Fit mean and std with chosen c ---
        y = np.log(big / self.c + 1.0)
        self.mean = y.mean()
        self.std = y.std() + self.eps
        self.fitted = True

        print(f"[LogScaler] mean={self.mean:.4f}, std={self.std:.4f}")
        
        
        print("Raw values stats:")
        print("min:", np.nanmin(big), "max:", np.nanmax(big))
        print("NaNs:", np.isnan(big).sum(), "Infs:", np.isinf(big).sum())


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        assert self.fitted, "Call fit() first"
        y = torch.log(x / self.c + 1.0)
        return (y - self.mean) / self.std

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        z = y * self.std + self.mean
        return (torch.exp(z) - 1.0) * self.c