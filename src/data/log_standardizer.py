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

        self.clamp_low = None
        self.clamp_high = None
    
    def iterate_chunks(self, data):
        T = data.sizes["time"]
        for start in range(0, T, self.chunk_size):
            end = min(start + self.chunk_size, T)
            chunk = torch.tensor(
                data.isel(time=slice(start, end)).values.astype("float32"),
                device=self.device
            )
            yield chunk

    def compute_clamp_from_data(self, data, q_low=0.001, q_high=0.999, sample_frac=0.03):
        if self.clamp_low is not None and self.clamp_high is not None:
            print(f'[Scaler] Using reference clamp values low: {self.clamp_low}, high: {self.clamp_high}')
            return self.clamp_low, self.clamp_high

        zs = []

        for chunk in self.iterate_chunks(data):
            chunk = chunk[torch.isfinite(chunk)]
            if chunk.numel() == 0:
                continue

            y = torch.log(chunk / self.c + 1.0)
            z = (y - self.mean) / (self.std + self.eps)

            n_sample = max(1, int(z.numel() * sample_frac))
            idx = torch.randperm(z.numel(), device=z.device)[:n_sample]
            zs.append(z.flatten()[idx].cpu())

        z_all = torch.cat(zs)
        self.clamp_low = torch.quantile(z_all, q_low).item()
        self.clamp_high = torch.quantile(z_all, q_high).item()

        print(f"[Scaler] clamp range = ({self.clamp_low:.3f}, {self.clamp_high:.3f})")

        return self.clamp_low, self.clamp_high



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

        var_penalty = torch.abs(torch.log(std + 1e-6))

        score = torch.abs(skew_val) + torch.abs(kurt_val - 3.0) + 0.5 * var_penalty
        return score

    def select_c(self, data):
        print("[Scaler] Selecting best c (streaming on GPU)...")
        start_time = time.time()
        candidate_c = [2 ** i for i in range(-7,3)]  # from 1/1024 to 1024

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


def create_scaler(years, model_type, clamp_high_pct=None):
    print("[Scaler] Reading dataset (lazy xarray)")
    from src.data.read_data import read_raw_data
    data = read_raw_data(years=years, aggregate_daily=model_type=="daily")

    scaler = LogStandardizerStreaming(c=None, chunk_size=24)
    scaler.fit(data)
    if clamp_high_pct is not None:
        scaler.compute_clamp_from_data(data, q_high=clamp_high_pct)
    return scaler


def load_scaler(reload, cache_path, years, time_slices, model_type, clamp_high_pct = None):
    cache_path = f"{cache_path}/scaler_{model_type}_{years[0]}_{years[-1]}{'' if time_slices is None else f'_{time_slices}'}.pkl"

    if reload or not os.path.exists(cache_path):
        scaler = create_scaler(years, model_type=model_type, clamp_high_pct=clamp_high_pct)
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
    