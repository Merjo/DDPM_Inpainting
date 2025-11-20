import torch

print(torch.cuda.is_available())  # Test if GPU is available (important! should be TRUE!)
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))



# TODO delete / refactor this script

import torch
from src.config import cfg

import torch
# Ensure dataset and scaler exist
_ = cfg.train_data
scaler = cfg.scaler
dataset = cfg.train_data.dataset  # unwrap Subset

# Encode all data once (or use precomputed scaled data)
if dataset.data_scaled is not None:
    vals = dataset.data_scaled
else:
    vals = scaler.encode(dataset.data_raw)

# Use CPU (quantile runs fine there and avoids GPU OOM)
vals_flat = vals.flatten().cpu()

# Subsample if too large
n = vals_flat.numel()
print(n)
max_n = 10_000_000  # 2 million values is plenty
if n > max_n:
    idx = torch.randint(0, n, (max_n,))
    vals_flat = vals_flat[idx]

# Compute 0.5% and 99.5% quantiles
q_low, q_high = torch.quantile(vals_flat, torch.tensor([0.005, 0.995]))
print(f"Recommended clamp range: ({q_low.item():.3f}, {q_high.item():.3f})")