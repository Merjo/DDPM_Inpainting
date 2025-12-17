import os
import tempfile
import requests
import zipfile

import numpy as np
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt

# ---------------------------------------------------
# 1. Create temporary folder + download ZIP
# ---------------------------------------------------
download_url = "https://userpage.fu-berlin.de/soga/data/raw-data/spatial/srtm_germany_dtm.zip"

temp_dir = tempfile.mkdtemp()
zip_path = os.path.join(temp_dir, "srtm_germany_dtm.zip")

print("Downloading...")
with requests.get(download_url, stream=True) as r:
    r.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

print("Download completed:", zip_path)

# ---------------------------------------------------
# 2. Unzip
# ---------------------------------------------------
print("Unzipping...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(temp_dir)

tif_path = os.path.join(temp_dir, "srtm_germany_dtm.tif")
print("Extracted:", tif_path)

# ---------------------------------------------------
# 3. Check file size
# ---------------------------------------------------
file_size_mb = os.path.getsize(tif_path) / 1e6
print(f"The file size is {file_size_mb:.2f} MB")

# ---------------------------------------------------
# 4. Load raster with rasterio
# ---------------------------------------------------
with rasterio.open(tif_path) as src:
    data = src.read(1)  # read first band
    profile = src.profile

print("Raster loaded. Shape:", data.shape)

# ---------------------------------------------------
# 5. Aggregate by factor 20 (downscale)
# ---------------------------------------------------
factor = 20

with rasterio.open(tif_path) as src:
    # scale height and width by factor
    new_height = src.height // factor
    new_width = src.width // factor

    aggregated = src.read(
        out_shape=(1, new_height, new_width),
        resampling=Resampling.average
    )[0]

print("Aggregated raster shape:", aggregated.shape)

# ---------------------------------------------------
# 6. Plot result
# ---------------------------------------------------
plt.figure(figsize=(10, 8))
plt.imshow(aggregated, cmap="terrain")
plt.title("Downscaled SRTM Germany DTM (factor=20)")
plt.colorbar(label="Elevation (m)")
plt.save('random/test.png')
