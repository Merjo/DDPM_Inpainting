import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os
import time
import zipfile
from src.config import cfg

path = None

# Determine base URL based on model_type
if cfg.model_type == "daily":
    base_url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/more_precip/historical/"
    path = cfg.stations_daily_path
elif cfg.model_type == "hourly":
    base_url = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/precipitation/historical/"
    path = cfg.stations_hourly_path
else:
    raise ValueError(f"Unknown model_type: {cfg.model_type}")


# Ensure output directory exists
os.makedirs(path, exist_ok=True)

def list_files_recursive(url):
    """Recursively crawl a directory on the DWD OpenData server and return direct file URLs."""
    files = []
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for a in soup.find_all("a"):
        href = a.get("href")
        if href in [None, "../"]:
            continue

        full_url = urljoin(url, href)

        if href.endswith("/"):
            files.extend(list_files_recursive(full_url))
        else:
            files.append(full_url)

    return files

print("Scanning directory tree… this may take a few seconds.")
links = list_files_recursive(base_url)
print(f"Found {len(links)} files to download.")

# Download and unzip files
for i, link in enumerate(links, 1):
    filename = os.path.join(path, os.path.basename(link))
    
    if os.path.exists(filename):
        print(f"[{i}/{len(links)}] Already exists: {filename}")
    else:
        print(f"[{i}/{len(links)}] Downloading {filename} ...")
        with requests.get(link, stream=True) as r:
            r.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        time.sleep(0.1)  # polite pause

    # Unpack ZIP files
    if filename.endswith(".zip"):
        print(f"Unpacking {filename} ...")
        with zipfile.ZipFile(filename, "r") as z:
            z.extractall(path)
