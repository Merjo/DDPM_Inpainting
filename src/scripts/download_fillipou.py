import gdown

# Replace with your file ID from the Drive share link:
file_id = "1RLrXSdgzKor2HFo3jj7T_Y6c6q78L3MR"
download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

output_path = "radolan_fromstations_2018.nc"  # choose the name you want
gdown.download(download_url, output_path, quiet=False)

print(f"Downloaded to: {output_path}")
