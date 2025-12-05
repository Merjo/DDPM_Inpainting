import torch
import os

class Config:
    def __init__(self):

        # General parameters
        self.cuda = torch.cuda.is_available()
        self.device = "cuda" if self.cuda else "cpu"
        self.check_cuda()
        self.log_every_epoch = True
        self.output_manager = None
        self.seed = 42
        self.model_type = 'daily'  # choices: ['hourly', 'daily']

        # Efficiency
        self.do_mixed_precision = False  # TODO Not working yet
        self.do_checkpointing = False  # TODO Not working yet

        # Data

        self.cache_path = "../../../../p/tmp/merlinho/cache"
        self.data_ref = None
        self.scaler_ref = None
        self.do_reload_scaler = False
        self.default = object()

        self.train_data_ref = None
        self.val_data_ref = None
        self.train_loaders_ref = None
        self.val_loaders_ref = None
        self.val_fraction = 0.15  # TODO Delete

        # Data parameters

        self.patch_size = 256  # Patch size for dataset
        self.min_coverage_ref = 0.1  # Minimum coverage for patches (0.0 to 1.0)

        self.start_year = 2001
        self.end_year = 2017  # Inclusive
        self.train_years = range(2001, 2012)
        self.val_years = range(2012, 2018)
        self.test_years = range(2018, 2019)
        self.time_slices = None # 250  # TODO None
        self.reload = False
        self.drop_na = False
        self.augment = True
        self.do_patch_diffusion = True

        # Importance Sampling parameters

        self.do_importance_sampling = self.do_patch_diffusion
        self.isp_patch_sizes = [64, 128, 256, 448]#512, 896]  # TODO Decide 896 -> potentially range until 1152, with 52 padding each side?
        self.isp_shares = [0.25,0.25,0.25,0.25]  #[0.2,0.2,0.2,0.2,0.2]
        self.isp_s = 0.05
        self.isp_m = 2
        self.isp_q_min = 2e-4

        # Training parameters
        self.epochs = 200
        self.patience = 200
        self.optuna_epochs = 15#10
        self.optuna_patience = 8
        self.optuna_sample_every = 15
        self.optuna_n_trials = 70#50
        self.batch_size = 4

        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.xlr_scheduler_gamma = 0.99
        self.wcs_scheduler_steps = 3
        self.min_patience_delta = 1e-4

        # Output / Sampling parameters

        self.clamp_range = (-3.5, 5)
        self.clamp_range_end_ref = (-0.598, 3.388) # None
        self.do_adapt_clamp_range = True
        self.clamp_high_pct = 0.995
        self.n_skip_clamp = 3  # number of last timesteps to skip clamping

        self.n_samples_regular = 16
        self.n_hist_samples_regular = 64

        self.n_samples = 32
        self.n_hist_samples = 128

        self.sample_every = 10
        self.optuna_sample_every = 10

        self.do_regular_hist = True
        self.do_save_model_regular = True

        self.output_base_dir = "output_new"
        self.output_path = None

        # Plotting parameters
        self.vmin_ref = None
        self.vmax_ref = None
        
        self.dpi = 600

        # Inpainting / DPS Parameters

        self.inpainting_test_coverage = 0.001 # 0.03

        self.do_use_dps = True
        self.dps_lam = 0.02
        self.dps_hard_overwrite = 0.0  # TODO -> maybe just in the last step?

        # Normal Parameters

        self.model_channels = 256
        self.num_blocks = 2
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
            "num_blocks": (1, 3),  # use int range
            "dropout": (0.0, 0.3),  # float range
            "downsample_type": ['residual', 'standard'],
            "channel_mult": ['124', '1224', '1248', '1124'],
            "attn_config": ['none', 'last'],#, 'last_two'],
            #"timesteps": [250, 500, 1000],
            "beta_schedule": ['linear', 'exponential', 'quadratic'],#, 'cosine'],
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
            self.data_ref = read_data(reload=self.reload, scaler=self.scaler, patch_size=self.patch_size, min_coverage=self.min_coverage, years=self.years)
        return self.data_ref
    
    @property
    def train_data(self):
        if self.train_data_ref is None:
            from src.data.read_data import read_data
            self.train_data_ref = read_data(reload=self.reload, scaler=self.scaler, patch_size=self.patch_size, min_coverage=self.min_coverage, years=self.train_years, return_importance_prob=False)
        return self.train_data_ref

    @property
    def val_data(self):
        if self.val_data_ref is None:
            from src.data.read_data import read_data
            self.val_data_ref = read_data(reload=self.reload, scaler=self.scaler, patch_size=self.patch_size, min_coverage=self.min_coverage, years=self.val_years, return_importance_prob=True)
        return self.val_data_ref
    
    @property
    def test_data(self):
        if self.test_data_ref is None:
            from src.data.read_data import read_data
            raise RuntimeError("Use test data only when absolutely sure!!")
            self.test_data_ref = read_data(reload=self.reload, scaler=self.scaler, patch_size=self.patch_size, min_coverage=self.min_coverage, years=self.test_years, return_importance_prob=True)
        return self.test_data_ref


    @property
    def scaler(self):
        if self.scaler_ref is None:
            from src.data.log_standardizer import load_scaler
            self.scaler_ref = load_scaler(reload=self.do_reload_scaler,
                                          cache_path=self.cache_path,
                                          years=self.years,
                                          time_slices=self.time_slices)
        return self.scaler_ref
    
    @property 
    def min_coverage(self):
        if self.do_importance_sampling or self.do_patch_diffusion:
            print("[Config] Min coverage is not used with importance sampling or patch diffusion.")
            return 0.0
        else:   
            return self.min_coverage_ref

    @property
    def train_loaders(self):
        if self.train_loaders_ref is None:
            from src.data.loader import get_loaders
            self.train_loaders_ref = get_loaders(self.train_data)
        return self.train_loaders_ref

    @property
    def val_loaders(self):
        if self.val_loaders_ref is None:
            from src.data.loader import get_loaders
            self.val_loaders_ref = get_loaders(self.val_data)
        return self.val_loaders_ref

    @property
    def current_output(self):
        if self.output_path is None:
            raise ValueError("Output path is not set. Please set it using update_output_path().")
        return self.output_path

    @property
    def vmin(self):
        if self.vmin_ref is None:
            self.vmin_ref = min([d.data_raw[torch.isfinite(d.data_raw)].min() for d in self.data.datasets]).item() * 1.1  # TODO Decide 1.1
        return self.vmin_ref

    @property
    def vmax(self):
        if self.vmax_ref is None:
            self.vmax_ref = max([d.data_raw[torch.isfinite(d.data_raw)].max() for d in self.data.datasets]).item()
        return self.vmax_ref

    @property
    def years(self):
        return range(self.start_year, self.end_year+1)
    
    @property
    def clamp_range_end(self):
        """
        Automatically determine clamp range based on real (scaled) data.

        The lower bound corresponds to the minimum of the scaled dataset 
        (typically representing 0 precipitation after decoding).
        The upper bound is chosen as the high_pct quantile (e.g., 99.5%) 
        of the scaled data distribution.

        This ensures that ~99.5% of the data lies within the clamp range.
        """
        if self.clamp_range_end_ref is None:

            # Ensure data and scaler are loaded
            dataset = self.data
            if dataset.data_scaled is None:
                raise ValueError("Dataset must be scaled before computing clamp range.")

            vals = dataset.data_scaled  # [N, 1, H, W]
            vals_flat = vals.view(-1)

            # Sample if tensor is too large for quantile computation
            if vals_flat.numel() > 5_000_000:
                idx = torch.randperm(vals_flat.numel(), device=vals_flat.device)[:5_000_000]
                vals_flat = vals_flat[idx]

            # Compute quantiles safely on CPU
            vals_flat = vals_flat.detach().cpu()
            vals_flat = vals_flat[torch.isfinite(vals_flat)]
            low = vals_flat.min().item()
            high = torch.quantile(vals_flat, torch.tensor(self.clamp_high_pct)).item()

            print(f"[AutoClamp] Recommended clamp range: ({low:.3f}, {high:.3f})")
            self.clamp_range_end_ref = (low, high)
            return((low, high))
        else:
            return(self.clamp_range_end_ref)
    
    def clamp_range_t(self, t, total_timesteps=None, factor=0.1):
        """
        Get the clamp range at timestep t during diffusion sampling/training.

        The clamp range is linearly interpolated between the initial clamp range 
        and the final clamp range over the diffusion timesteps.

        Args:
            t (int): Current timestep.
            total_timesteps (int): Total number of diffusion timesteps.
            factor (float): Factor to scale the end clamp range.

        Returns:
            tuple: (clamp_low, clamp_high) at timestep t.
        """
        if not self.do_adapt_clamp_range:
            return self.clamp_range
        if total_timesteps is None:
            total_timesteps = self.timesteps
        
        t = total_timesteps - t  # Reverse t for interpolation from start to end

        low_start, high_start = self.clamp_range
        low_end, high_end = self.clamp_range_end
        low_end *= (1+factor)
        high_end *= (1+factor)

        alpha = t / total_timesteps
        clamp_low = low_start + alpha * (low_end - low_start)
        clamp_high = high_start + alpha * (high_end - high_start)

        return (clamp_low, clamp_high)
    
    def set_output_manager(self, output_manager):
        self.output_manager = output_manager

cfg = Config()