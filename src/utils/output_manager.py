import os
import shutil
import datetime
import json
import csv
from src.config import cfg

class OutputManager:
    def __init__(self, run_type, base_dir="output_new"):
        self.base_dir = base_dir
        self.run_type = run_type

        # Temporary folder name (so SLURM can already write into it)
        date_str = datetime.datetime.now().strftime("%b%d_%H%M")
        self.run_dir_name = f"{self.run_type}_{date_str}_{cfg.patch_size}_{cfg.min_coverage}"
        self.run_dir = os.path.join(self.base_dir, self.run_dir_name)
        os.makedirs(self.run_dir, exist_ok=False)
        cfg.update_output_path(run_dir_name=self.run_dir_name)

        cfg.set_output_manager(self)

        # Prepare subfolders
        subfolders = ["samples", "histograms", "models", "rapsd", "data"]
        if run_type in ['optuna', 'optuna_best']:
            subfolders.append("trials")

        for folder in subfolders:
            os.makedirs(os.path.join(self.run_dir, folder), exist_ok=False)

        # Copy config file for reproducibility
        self.copy_config()

    def copy_config(self, config_path="src/config.py"):
        shutil.copy(config_path, os.path.join(self.run_dir, "config.py"))

    def save_best_params(self, params: dict, mse_val, epochs):
        mse_str = f"{mse_val:.4g}"
        filename = f"best_params_{mse_str}_{cfg.patch_size}_{cfg.min_coverage}_{cfg.batch_size}_{epochs}_{self.run_type}.csv"
        path = os.path.join(self.run_dir, filename)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["param", "value"])
            for k, v in params.items():
                writer.writerow([k, v])

        return path

    def save_model(self, model, mse_val):
        path = os.path.join(self.run_dir, f"model_{mse_val:.6g}.pkl")
        import torch
        torch.save(model.state_dict(), path)
        return path

    def get_run_dir(self):
        return self.run_dir

    def rename_folder(self, mse_val):
        mse_str = f"{mse_val:.4g}"
        new_name = f"{mse_str}_{self.run_dir_name}"
        new_dir = os.path.join(self.base_dir, new_name)
        os.rename(self.run_dir, new_dir)
        self.run_dir = new_dir
        print(f"Run directory renamed: {new_dir}")
        return new_dir
    
    def move_logs(self):
        slurm_out = f"logs/run_{self.run_type}_{os.environ.get('SLURM_JOB_ID')}.out"
        slurm_err = f"logs/run_{self.run_type}_{os.environ.get('SLURM_JOB_ID')}.err"

        if os.path.exists(slurm_out):
            shutil.move(slurm_out, os.path.join(self.run_dir, os.path.basename(slurm_out)))
        else:
            print(f"SLURM output log not found: {slurm_out}")

        if os.path.exists(slurm_err):
            shutil.move(slurm_err, os.path.join(self.run_dir, os.path.basename(slurm_err)))
        else:
            print(f"SLURM error log not found: {slurm_err}")

    def finalize(self, mse_val, model=None, params=None, epochs=0):
        if model is not None:
            self.save_model(model, mse_val)
        if params is not None:
            self.save_best_params(params, mse_val, epochs)
        final_dir = self.rename_folder(mse_val)
        self.move_logs()

        print(f"Run complete. Results stored in {final_dir}")

        return final_dir

    
