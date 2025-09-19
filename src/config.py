from src.data.log_standardizer import LogStandardizer
import torch

class Config:
    def __init__(self):
        self.default = object()
        self.scaler = LogStandardizer
        self.time_slices = 8784  # e.g., hourly data for one year 2020 (366 days)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sample_every = 5
        self.optuna_sample_every = 10

cfg = Config()
