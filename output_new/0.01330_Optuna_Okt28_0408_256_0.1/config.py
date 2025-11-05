from src.data.log_standardizer import LogStandardizer
import torch

class Config:
    def __init__(self):
        self.default = object()
        self.scaler = LogStandardizer
        self.time_slices = 8760 # for one year #175320 or None for 20/all years # e.g., hourly data for one year 2020 (366 days)
        self.cuda = torch.cuda.is_available()
        print("Torch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
        if not self.cuda:
            raise(Exception('Cuda not available.'))
        self.device = "cuda" if self.cuda else "cpu"
        self.sample_every = 5
        self.batch_size = 4
        self.optuna_sample_every = 10
        self.patch_size = 256  # Patch size for dataset
        self.min_coverage = 0.1  # Minimum coverage for patches (0.0 to 1.0)

cfg = Config()
