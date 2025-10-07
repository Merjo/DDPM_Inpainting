from src.data.log_standardizer import LogStandardizer
import torch

class Config:
    def __init__(self):
        self.default = object()
        self.scaler = LogStandardizer
        self.time_slices = 8760 # for one year #175320 or None for 20/all years # e.g., hourly data for one year 2020 (366 days)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_every = 5
        self.optuna_sample_every = 10
        self.patch_size = 256  # Patch size for dataset
        self.min_coverage = 0.2  # Minimum coverage for patches (0.0 to 1.0)

cfg = Config()
