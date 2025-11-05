from src.data.log_standardizer import LogStandardizer
import torch
import os

class Config:
    def __init__(self):

        # General parameters
        self.cuda = torch.cuda.is_available()
        self.device = "cuda" if self.cuda else "cpu"
        self.check_cuda()
        self.default = object()
        self.log_every_epoch = True

        # Data parameters

        self.patch_size = 256  # Patch size for dataset
        self.min_coverage = 0.1  # Minimum coverage for patches (0.0 to 1.0)
        self.scaler = LogStandardizer
        self.start_year = 2001
        self.end_year = 2002
        self.time_slices = 8760 # for one year #175320 or None for 20/all years # e.g., hourly data for one year 2020 (366 days)

        # Training parameters
        self.epochs = 1#200
        self.patience = 10
        self.optuna_epochs = 1#10
        self.optuna_sample_every = 5
        self.optuna_n_trials = 2#75
        self.batch_size = 4

        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.xlr_scheduler_gamma = 0.99
        self.wcs_scheduler_steps = 3
        self.min_patience_delta = 1e-4

        # Output parameters

        self.n_samples = 8
        self.n_hist_samples = 128

        self.sample_every = 5
        self.optuna_sample_every = 10

        self.do_regular_hist = True

        self.output_base_dir = "output_new"
        self.output_path = None
        
        self.dpi = 300

        # Inpainting / DPS Parameters

        self.do_use_dps = True
        self.dps_lam = 0.1
        self.dps_hard_overwrite = 0.2  # TODO -> maybe just in the last step?

        # Normal Parameters

        self.model_channels = 256
        self.num_blocks = 3
        self.dropout = 0.15
        self.downsample_type = 'standard'
        self.channel_mult = '1224'
        self.attn_config = 'last'
        self.timesteps = 1000
        self.beta_schedule = 'cosine'
        self.loss = 'mse'
        self.optimizer = 'AdamW'
        self.scheduler = 'WarmupCosine'
        self.lr = 2.5e-04

        # Optuna Search Spaces

        self.optuna_search_space = {
            "model_channels": [64, 128, 256],
            "num_blocks": (1, 3),  # use int range
            "dropout": (0.0, 0.3),  # float range
            "downsample_type": ['residual', 'standard'],
            "channel_mult": ['124', '1224', '1248', '1124'],
            "attn_config": ['none', 'last'],#, 'last_two'],
            "timesteps": [250, 500, 1000],
            "beta_schedule": ['linear', 'quadratic', 'exponential', 'cosine'],
            "loss": ['mse', 'l1', 'huber'],
            "optimizer": ['Adam', 'AdamW'],
            "scheduler": ['ExponentialLR', 'WarmupCosine'],
            "lr": (1e-5, 5e-4),
            "lr_xr": (1e-5, 5e-4),
            "lr_wc": (1e-5, 5e-3)
        }
        
    
    def check_cuda(self):
        if not self.cuda:
            raise(Exception('Cuda not available.'))
        print("Torch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        print("CUDA device count:", torch.cuda.device_count())
        print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

    def update_output_path(self, run_dir_name):
        self.output_path = os.path.join(self.output_base_dir, run_dir_name)
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
