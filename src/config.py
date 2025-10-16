from src.data.log_standardizer import LogStandardizer
import torch

class Config:
    def __init__(self):
        self.default = object()
        self.scaler = LogStandardizer
        self.time_slices = 8760 # for one year #175320 or None for 20/all years # e.g., hourly data for one year 2020 (366 days)
        self.cuda = torch.cuda.is_available()
        self.device = "cuda" if self.cuda else "cpu"
        self.sample_every = 5
        self.batch_size = 8
        self.optuna_sample_every = 10
        self.patch_size = 128  # Patch size for dataset
        self.min_coverage = 0.1  # Minimum coverage for patches (0.0 to 1.0)

cfg = Config()
