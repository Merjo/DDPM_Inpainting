import numpy as np
from src.data.read_data import read_raw_data
from src.data.read_hyras_data import load_hyras
from src.config import cfg

def evaluate_hyras(years):
    radar = read_raw_data(years, aggregate_daily=True, time_slices=None)
    hyras = load_hyras

    print(radar.shape)
    print(hyras.shape)
    print('whats goign on?')

    print(radar.x.min().item(), radar.x.max().item())
    print(radar.y.min().item(), radar.y.max().item())

    print(hyras.x.min().item(), hyras.x.max().item())
    print(hyras.y.min().item(), hyras.y.max().item())

    print(np.diff(radar.x[:2]))
    print(np.diff(hyras.x[:2]))

    radar_lon = radar['lon'].load()  # returns an xarray DataArray backed by NumPy
    radar_lat = radar['lat'].load()

    print(radar_lon.min().item(), radar_lon.max().item())
    print(radar_lat.min().item(), radar_lat.max().item())

    print(hyras.lon.min().item(), hyras.lon.max().item())
    print(hyras.lat.min().item(), hyras.lat.max().item())

    print(np.diff(radar.x[:2]))
    print(np.diff(hyras.x[:2]))


    import numpy as np
    import xarray as xr
    from scipy.spatial import cKDTree

    # Flatten radar lon/lat coordinates
    radar_lon_flat = radar['lon'].values.ravel()
    radar_lat_flat = radar['lat'].values.ravel()
    radar_coords = np.column_stack((radar_lon_flat, radar_lat_flat))

    # Build KDTree
    tree = cKDTree(radar_coords)

    # Flatten HYRAS lon/lat coordinates
    hyras_lon_flat = hyras['lon'].values.ravel()
    hyras_lat_flat = hyras['lat'].values.ravel()
    hyras_coords = np.column_stack((hyras_lon_flat, hyras_lat_flat))

    # Query nearest radar point for each HYRAS grid point
    distances, indices = tree.query(hyras_coords)

    # Map indices back to 2D radar array indices
    ny, nx = radar['lon'].shape
    radar_y_idx = indices // nx
    radar_x_idx = indices % nx

    # Radar values as NumPy array (time, y, x)
    radar_values = radar['RR'].values

    # Use advanced indexing with broadcasting to map all timesteps at once
    # radar_y_idx and radar_x_idx are 1D arrays of size (y*x)
    # We'll flatten the HYRAS grid and then reshape later
    radar_flat = radar_values[:, radar_y_idx, radar_x_idx]  # shape: (time, y*x)
    hyras_shape = (len(hyras['y']), len(hyras['x']))
    radar_on_hyras_array = radar_flat.reshape((len(hyras['time']), *hyras_shape))

    # Wrap as xarray DataArray
    radar_on_hyras_lonlat = xr.DataArray(
        radar_on_hyras_array,
        coords={
            'time': hyras['time'],
            'y': hyras['y'],
            'x': hyras['x']
        },
        dims=('time', 'y', 'x')
    )

    print("Mapping complete!")




    return

def evaluate_hyras_validation():
    return evaluate_hyras(cfg.val_years)

if __name__=='__main__':
    mse = evaluate_hyras_validation()
    print(f'[Evaluate HYRAS] Evaluated HYRAS, MSE is: {mse}')


def _test(distances, indices, hyras, hyras_lat_flat, hyras_lon_flat, radar, radar_lat_flat, radar_lon_flat, radar_on_hyras, radar_on_hyras_lonlat):
    mean_dist = distances.mean()
    median_dist = np.median(distances)
    max_dist = distances.max()

    print("Mean NN distance (deg):", mean_dist)
    print("Median NN distance (deg):", median_dist)
    print("Max NN distance (deg):", max_dist)


    mean_lat = 50.0  # approx Germany mid-latitude
    deg_to_m_lat = 111_000
    deg_to_m_lon = 111_000 * np.cos(np.deg2rad(mean_lat))  # ~71 km

    # Approx meters
    mean_m = np.sqrt((0.00468*deg_to_m_lon)**2 + (0.00468*deg_to_m_lat)**2)
    max_m  = np.sqrt((0.00929*deg_to_m_lon)**2 + (0.00929*deg_to_m_lat)**2)

    print("Mean distance ~", mean_m, "m")
    print("Max distance ~", max_m, "m")

    import matplotlib.pyplot as plt

    # Reshape distances to HYRAS grid
    dist_grid = distances.reshape(len(hyras['y']), len(hyras['x']))

    plt.figure(figsize=(10,6))
    plt.imshow(dist_grid, origin='lower', cmap='viridis')
    plt.colorbar(label='Distance to nearest radar pixel (deg)')
    plt.title('Nearest neighbor distances from HYRAS to radar')
    plt.xlabel('x index')
    plt.ylabel('y index')
    plt.show()


    import numpy as np

    mean_lat = 50.0
    deg_to_m_lat = 111_000
    deg_to_m_lon = 111_000 * np.cos(np.deg2rad(mean_lat))

    dlat = hyras_lat_flat - radar_lat_flat[indices]
    dlon = hyras_lon_flat - radar_lon_flat[indices]
    dist_m = np.sqrt((dlat*deg_to_m_lat)**2 + (dlon*deg_to_m_lon)**2)
    dist_grid_m = dist_m.reshape(len(hyras['y']), len(hyras['x']))

    plt.figure(figsize=(10,6))
    plt.imshow(dist_grid_m, origin='lower', cmap='viridis')
    plt.colorbar(label='Distance to nearest radar pixel (m)')
    plt.title('Nearest neighbor distances (meters)')
    plt.show()


    import matplotlib.pyplot as plt
    import numpy as np

    # Number of random timesteps to plot
    n_days = 5
    np.random.seed(42)  # for reproducibility
    time_indices = np.random.choice(len(hyras['time']), n_days, replace=False)

    # Create figure
    fig, axes = plt.subplots(n_days, 3, figsize=(18, 4 * n_days))

    for i, t in enumerate(time_indices):
        # HYRAS
        ax = axes[i, 0]
        hyras.isel(time=t).plot(ax=ax)
        ax.set_title(f"HYRAS (t={t})")

        # X/Y nearest neighbor (your previous method)
        ax = axes[i, 1]
        radar_on_hyras.isel(time=t).plot(ax=ax)
        ax.set_title(f"X/Y nearest neighbor (t={t})")

        # Lon/Lat nearest neighbor
        ax = axes[i, 2]
        radar_on_hyras_lonlat.isel(time=t).plot(ax=ax)
        ax.set_title(f"Lon/Lat nearest neighbor (t={t})")

    plt.tight_layout()
    plt.show()


    rmse_xy = np.sqrt(((radar_on_hyras - hyras) ** 2).mean(skipna=True))
    rmse_lonlat = np.sqrt(((radar_on_hyras_lonlat - hyras) ** 2).mean(skipna=True))
    print("RMSE XY:", rmse_xy.compute().item())
    print("RMSE lon/lat:", rmse_lonlat.compute().item())


    print("Any NaNs in radar_on_hyras_lonlat:", np.isnan(radar_on_hyras_lonlat).any().item())
    print("Number of NaNs:", np.isnan(radar_on_hyras_lonlat).sum().item())
