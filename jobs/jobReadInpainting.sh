#!/bin/bash
#SBATCH --job-name=read_inpainting_data
#SBATCH --output=logs/read_inpainting_data_%j.out
#SBATCH --error=logs/read_inpainting_data_%j.err
#SBATCH --qos=gpushort
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G

# Load modules
module load anaconda/2025

# Activate conda env
source activate diffusion_rain


# Ensure imports work (src/ will be discoverable)
export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u src/data/read_inpainting_data.py
