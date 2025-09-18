import torch
from torch.utils.data import Dataset
import numpy as np

import torch


from scipy.stats import skew, kurtosis

import numpy as np
import torch
from torch.utils.data import Dataset


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
