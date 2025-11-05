from src.data.log_standardizer import LogStandardizer
import torch
import os

class Config:
    def __init__(self):
        self.check_cuda()
        self.default = object()
        self.scaler = LogStandardizer
        self.start_year = 2001
        self.end_year = 2002
        self.time_slices = 8760 # for one year #175320 or None for 20/all years # e.g., hourly data for one year 2020 (366 days)
        self.device = "cuda" if self.cuda else "cpu"
        self.sample_every = 1
        self.batch_size = 4
        self.optuna_sample_every = 10
        self.patch_size = 256  # Patch size for dataset
        self.min_coverage = 0.1  # Minimum coverage for patches (0.0 to 1.0)
        self.output_base_dir = "output_new"
        self.output_path = None
        self.n_samples = 8
        self.n_hist_samples = 32
    
    def check_cuda(self):
        self.cuda = torch.cuda.is_available()
        if not self.cuda:
            raise(Exception('Cuda not available.'))
        print("Torch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

    def update_output_path(self, run_name):
        self.output_path = os.path.join(self.output_base_dir, run_name)
        print(f"Output path set to: {self.output_path}")

    @property
    def current_output(self):
        if self.output_path is None:
            raise ValueError("Output path is not set. Please set it using update_output_path().")
        return self.output_path

    @property
    def years(self):
        return range(self.start_year, self.end_year+1)

cfg = Config()
