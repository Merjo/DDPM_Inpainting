from src.data.log_standardizer import LogStandardizer
import torch
import os

class Config:
    def __init__(self):

        # General parameters
        self.cuda = torch.cuda.is_available()
        self.device = "cuda" if self.cuda else "cpu"
        self.check_cuda()
        self.log_every_epoch = True

        # Data

        self.data_ref = None
        self.loader_ref = None
        self.scaler_ref = None
        self.default = object()
        self.scaler_class = LogStandardizer

        self.train_data_ref = None
        self.val_data_ref = None
        self.train_loader_ref = None
        self.val_loader_ref = None
        self.val_fraction = 0.15

        # Data parameters

        self.patch_size = 256  # Patch size for dataset
        self.min_coverage = 0.1  # Minimum coverage for patches (0.0 to 1.0)
        
        #self.scaler = LogStandardizer
        self.start_year = 2001
        self.end_year = 2002
        self.time_slices = 8760 # for one year #175320 or None for 20/all years # e.g., hourly data for one year 2020 (366 days)
        self.reload = False

        # Training parameters
        self.epochs = 2#50
        self.patience = 150
        self.optuna_epochs = 15#10
        self.optuna_patience = 8
        self.optuna_sample_every = 5
        self.optuna_n_trials = 50#50
        self.batch_size = 4

        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.xlr_scheduler_gamma = 0.99
        self.wcs_scheduler_steps = 3
        self.min_patience_delta = 1e-4

        # Output parameters

        self.n_samples_regular = 8
        self.n_hist_samples_regular = 16

        self.n_samples = 32
        self.n_hist_samples = 128

        self.sample_every = 10
        self.optuna_sample_every = 10

        self.do_regular_hist = True

        self.output_base_dir = "output_new"
        self.output_path = None

        # Plotting parameters
        self.vmin_ref = None
        self.vmax_ref = None
        
        self.dpi = 300

        # Inpainting / DPS Parameters

        self.do_use_dps = True
        self.dps_lam = 0.1
        self.dps_hard_overwrite = 0.2  # TODO -> maybe just in the last step?

        # Normal Parameters

        self.model_channels = 128
        self.num_blocks = 3
        self.dropout = 0.15
        self.downsample_type = 'standard'
        self.channel_mult = '1224'
        self.attn_config = 'last'
        self.timesteps = 1000
        self.beta_schedule = 'linear'
        self.loss = 'l1'
        self.optimizer = 'Adam'
        self.scheduler = 'WarmupCosine'
        self.lr = 0.00017898129466371347

        #{'model_channels': 128, 'dropout': 0.22737977360150574, 'beta_schedule': 'linear', 'optimizer': 'Adam', 'scheduler': 'WarmupCosine', 'lr': 0.00017898129466371347}. Best is trial 2 with value: 0.049538634445891384.

        # Optuna Search Spaces

        self.optuna_search_space = {
            "model_channels": [128, 256],  # 64 ? 
            #"num_blocks": (1, 3),  # use int range
            "dropout": (0.0, 0.3),  # float range
            #"downsample_type": ['residual', 'standard'],
            #"channel_mult": ['124', '1224', '1248', '1124'],
            #"attn_config": ['none', 'last'],#, 'last_two'],
            #"timesteps": [250, 500, 1000],
            "beta_schedule": ['linear', 'exponential'],#, 'quadratic', 'cosine'],
            #"loss": ['mse', 'l1', 'huber'],
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
    def data(self):
        if self.data_ref is None:
            from src.data.read_data import read_data
            self.data_ref = read_data(reload=self.reload, scaler=self.scaler_class(), patch_size=self.patch_size, min_coverage=self.min_coverage, years=self.years)
            self.scaler_ref = self.data_ref.scaler
        return self.data_ref

    @property
    def scaler(self):
        if self.scaler_ref is None:
            from src.data.read_data import read_data
            self.data_ref = read_data(reload=self.reload, scaler=self.scaler_class(), patch_size=self.patch_size, min_coverage=self.min_coverage, years=self.years)
            self.scaler_ref = self.data_ref.scaler
        return self.scaler_ref

    @property
    def loader(self):
        if self.loader_ref is None:
            from src.data.loader import get_loader
            self.loader_ref = get_loader(self.data)
        return self.loader_ref
    

    @property
    def train_data(self):
        if self.train_data_ref is None:
            self.train_data_ref, self.val_data_ref = self.data.train_val_split(val_fraction=self.val_fraction, seed=42)
        return self.train_data_ref

    @property
    def val_data(self):
        if self.val_data_ref is None:
            self.train_data_ref, self.val_data_ref = self.data.train_val_split(val_fraction=self.val_fraction, seed=42)
        return self.val_data_ref

    @property
    def train_loader(self):
        if self.train_loader_ref is None:
            from src.data.loader import get_loader
            self.train_loader_ref = get_loader(self.train_data)
        return self.train_loader_ref

    @property
    def val_loader(self):
        if self.val_loader_ref is None:
            from src.data.loader import get_loader
            self.val_loader_ref = get_loader(self.val_data)
        return self.val_loader_ref

    @property
    def current_output(self):
        if self.output_path is None:
            raise ValueError("Output path is not set. Please set it using update_output_path().")
        return self.output_path

    @property
    def vmin(self):
        if self.vmin_ref is None:
            self.vmin_ref = self.data.data_raw.min().item() * 1.1  # assume 10 % higher is ok
        return self.vmin_ref

    @property
    def vmax(self):
        if self.vmax_ref is None:
            self.vmax_ref = self.data.data_raw.max().item()
        return self.vmax_ref

    @property
    def years(self):
        return range(self.start_year, self.end_year+1)

cfg = Config()
