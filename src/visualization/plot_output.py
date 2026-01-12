from matplotlib import pyplot as plt
import torch
import pandas as pd
import glob
from src.config import cfg
from src.save.save_plot import plot_station_inpainting

def load_timestamps(data_path):
    ts_files = glob.glob(f'{data_path}/*timestamps.csv')
    if not ts_files:
        raise FileNotFoundError("No timestamps CSV file found in data directory.")
    ts_path = ts_files[0]

    timestamps = pd.read_csv(ts_path, header=None)
    timestamps = pd.to_datetime(timestamps[0])

    return timestamps

def load_inpainted_data(data_path):
    inpainted_files = glob.glob(f'{data_path}/*inpainted_*.pt')
    if not inpainted_files:
        raise FileNotFoundError("No inpainted .pt file found in data directory.")
    inpainted_path = inpainted_files[0]

    inpainted_data = torch.load(inpainted_path)

    return inpainted_data

def plot_output(output_path, years=cfg.test_years, mode=cfg.model_type, filippou=False):
    timestamps_inpainted = load_timestamps(output_path)
    inpainted_data = load_inpainted_data(output_path)

    data = cfg.station_data(years=years,mode=mode,filippou=filippou)

    station_data, original_data, timestamps = data[:inpainted_data.shape[0]]

    # get station indices of timestamps_inpainted
    timestamp_indices = [timestamps.index(ts) for ts in timestamps_inpainted]
    station_data = station_data[timestamp_indices]
    original_data = original_data[timestamp_indices]

    plot_station_inpainting(station_data, original_data, inpainted_data, timestamps_inpainted, lam = cfg.dps_lam,title="Station Inpainting Results", out_dir='output_new/output_inpainting')


    return

if __name__ == "__main__":
    output_path = 'output_new/0.0201_evaluate_stations_daily_Jan01_2230_256_0.0/data'
    #output_path = '../../../../p/tmp/merlinho/cache/output_cache/Jan08_0116_hourly_2018'

    years = cfg.val_inpainting_years
    mode = 'hourly'
    filippou= False

    plot_output(output_path, years=years, mode=mode, filippou=filippou)