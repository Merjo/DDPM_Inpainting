from src.utils.evaluate_utils import load_output
from src.config import cfg
import xarray as xr
import numpy as np

def evaluate_filippou_on_hyras(years):
    filippou_on_hyras = load_output(f'filippou_on_hyras_daily_{years[0]}_{years[-1]}.nc').values
    hyras = load_output(f'hyras_{years[0]}_{years[-1]}.nc').values
    # Ensure alignment (just in case)
    mse = np.nanmean((filippou_on_hyras - hyras) ** 2)

    print(f"[Evaluate HYRAS] MSE = {mse:.4f}")
    print(f'[Evaluate HYRAS] RMSE = {np.sqrt(mse):.4f}')
    print(f'[Evaluate HYRAS] MAE = {np.nanmean(np.abs(filippou_on_hyras - hyras)):.4f}')
    rmse_hourly = np.sqrt(mse) / np.sqrt(24)
    print(f"[Evaluate HYRAS] Estimated hourly RMSE: {rmse_hourly:.4f} mm/hr")
    print(f'[Evaluate HYRAS] HYRAS mean: {np.nanmean(hyras):.4f}, max: {np.nanmax(hyras):.4f}, min: {np.nanmin(hyras):.4f}')
    print(f'[Evaluate HYRAS] Filippou-on-HYRAS mean: {np.nanmean(filippou_on_hyras):.4f}, max: {np.nanmax(filippou_on_hyras):.4f}, min: {np.nanmin(filippou_on_hyras):.4f}')

    return mse

if __name__ == "__main__":
    evaluate_filippou_on_hyras(cfg.test_years)
