#!/bin/bash
#SBATCH --job-name=onn_fsdp
#SBATCH --partition=hpg-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --output=onn_%j.out

set -euo pipefail

CONFIG=configs/config_paper_vgg11_onehot_pd.yaml
OUT_DIR=runs/ftvgg11_cifar10_onehot_pd
BATCH=24

echo "Starting job with config=${CONFIG}"
module load conda/25.7.0
conda activate onn-torch
NPROC=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-4}}
srun torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} onn_main.py --config-file "${CONFIG}" --enable-fsdp --output-dir "${OUT_DIR}" --batch-size "${BATCH}"
