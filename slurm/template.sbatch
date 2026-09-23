#!/usr/bin/env bash
#SBATCH --job-name=onn_resnet_ddp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=256gb
#SBATCH --time=1-00:00:00
#SBATCH --output=onn_resnet_ddp_%j.out

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
cd "${REPO_ROOT}"

CONFIG_FILE="${CONFIG_FILE:-configs/config_ideal.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/slurm_template_resnet18}"
MODEL_ARCH="${MODEL_ARCH:-resnet18}"
CONV_BACKEND="${CONV_BACKEND:-jtc_emulation}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-1}"
# With per-chunk activation checkpointing, peak memory is bounded by one
# chunk's pipeline intermediates, so large chunks are safe and much faster
# (measured 4.4x step time between 4096 and 65536 on resnet11/A100).
JTC_MAX_SHOTS="${JTC_MAX_SHOTS:-65536}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
MEMORY_REPORT_INTERVAL="${MEMORY_REPORT_INTERVAL:-0}"
GPUS="${GPUS:-4}"

# Set these in the submit environment when the site needs Conda activation.
if [[ -n "${ONN_CONDA_MODULE:-}" ]]; then
  module load "${ONN_CONDA_MODULE}"
fi
if [[ -n "${CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "Training ONN model"
echo "  config: ${CONFIG_FILE}"
echo "  output: ${OUTPUT_DIR}"
echo "  model_arch: ${MODEL_ARCH}"
echo "  conv_backend: ${CONV_BACKEND}"
echo "  batch_size_per_gpu: ${BATCH_SIZE}"
echo "  num_epochs: ${NUM_EPOCHS}"
echo "  gpus: ${GPUS}"
echo "  jtc_max_shots: ${JTC_MAX_SHOTS}"
echo "  dataloader_num_workers_per_rank: ${DATALOADER_NUM_WORKERS}"
echo "  memory_report_interval: ${MEMORY_REPORT_INTERVAL}"

# One batch allocation owns all GPUs; torchrun starts the per-GPU workers.
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${GPUS}" \
  onn_main.py \
  --config-file "${CONFIG_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-arch "${MODEL_ARCH}" \
  --conv-backend "${CONV_BACKEND}" \
  --batch-size "${BATCH_SIZE}" \
  --num-epochs "${NUM_EPOCHS}" \
  --checkpoint-interval "${CHECKPOINT_INTERVAL}" \
  --jtc-max-shots "${JTC_MAX_SHOTS}" \
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS}" \
  --dataloader-pin-memory false \
  --dataloader-persistent-workers false \
  --memory-report-interval "${MEMORY_REPORT_INTERVAL}" \
  --enable-ddp true \
  --enable-jtc-activation-checkpointing true \
  --run-pretrain-tests false \
  --run-full-strength-inference false \
  --show-progress false
