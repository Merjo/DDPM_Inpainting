#!/bin/bash
#SBATCH --job-name=run_inpaint
#SBATCH --output=logs/run_inpaint_%j.out
#SBATCH --error=logs/run_inpaint_%j.err
#SBATCH --qos=gpushort
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Load modules
module load anaconda/2025

# Activate conda env
source activate diffusion_rain


# Ensure imports work (src/ will be discoverable)
export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u src/run/run_normal.py
