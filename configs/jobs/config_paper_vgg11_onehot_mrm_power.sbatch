#!/bin/bash
#SBATCH --job-name=onn_fsdp
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --output=onn_%j.out

set -euo pipefail

CONFIG=configs/config_paper_vgg11_onehot_mrm_power.yaml

echo "Starting job with config=${CONFIG}"
module load conda/25.7.0
conda activate onn-torch
srun python onn_main.py --config-file "${CONFIG}" --enable-fsdp
