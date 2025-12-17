#!/bin/bash
#SBATCH --job-name=normal_run
#SBATCH --output=logs/run_normal_%j.out
#SBATCH --error=logs/run_normal_%j.err
#SBATCH --qos=gpumedium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1          # start with 1 GPU to match interactive test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G              # realistic for your interactive run

# Make logs folder
mkdir -p logs

# Load modules
module load anaconda/2025

# Initialize and activate conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate diffusion_rain

# Ensure imports work
export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Optional: make GPUs visible if needed
export CUDA_VISIBLE_DEVICES=0

# Run
python -u src/run/run_normal.py
