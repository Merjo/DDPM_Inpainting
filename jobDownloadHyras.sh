#!/bin/bash
#SBATCH --job-name=hyras_download
#SBATCH --output=logs/hyras_download_%j.out
#SBATCH --error=logs/hyras_download_%j.err
#SBATCH --partition=io
#SBATCH --qos=io
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00  # adjust as needed


# Load modules
module load anaconda/2025

# Activate conda env
source activate diffusion_rain


# Ensure imports work (src/ will be discoverable)
export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# Run your Python download script
python src/random/download_hyras.py
