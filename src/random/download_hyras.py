import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import time
from src.config import cfg

# Make sure target folder exists
os.makedirs(cfg.hyras_path, exist_ok=True)

# Base URL
base_url = "https://opendata.dwd.de/climate_environment/CDC/grids_germany/daily/hyras_de/precipitation/"

# Get HTML of the page
resp = requests.get(base_url)
resp.raise_for_status()
soup = BeautifulSoup(resp.text, "html.parser")

# Find all NetCDF files (files ending with .nc)
links = [urljoin(base_url, a["href"]) for a in soup.find_all("a") if a["href"].endswith(".nc")]

print(f"Found {len(links)} files.")

# Download files into cfg.hyras_path
for i, link in enumerate(links, 1):
    filename = os.path.join(cfg.hyras_path, os.path.basename(link))
    if os.path.exists(filename):
        print(f"[{i}/{len(links)}] Already exists: {filename}")
        continue
    print(f"[{i}/{len(links)}] Downloading {filename} ...")
    with requests.get(link, stream=True) as r:
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    time.sleep(0.1)  # polite pause
