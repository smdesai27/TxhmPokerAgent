#!/bin/bash
#SBATCH --job-name=ah_strong_fchpa_gpu
#SBATCH --output=logs/strong_fchpa/slurm/%x_%j.out
#SBATCH --error=logs/strong_fchpa/slurm/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# S7: GPU-scaled "strong head-to-head" FCHPA run (vectorized collector + corrected PFSP + Trinal-Clip).
# CPUs capped at 12 = the norm-gpu per-user QOS limit (cpu=12, gres/gpu=2, mem=192G); vectorized
# self-play is CPU-encode bound so we take the max allowed. GPU=1, mem=64G are within QOS.
# WANDB offline (no API key needed). Override CONFIG_NAME for the canary:
#   CONFIG_NAME=strong_fchpa_gpu_canary sbatch scripts/slurm_strong_fchpa_gpu_train.slurm
set -euo pipefail

echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})  Node: ${SLURM_NODELIST}  Date: $(date)"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/strong_fchpa/slurm checkpoints/snapshots/strong_fchpa

module load python/3.11.11-5e66
module load cuda/11.8.0-kuhf
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

CONFIG_NAME="${CONFIG_NAME:-strong_fchpa_gpu}"
CONFIG_FILE="${CONFIG_FILE:-configs/training_configs.yaml}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONDONTWRITEBYTECODE=1

echo "=== GPU ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "=== config=${CONFIG_NAME} (vectorized collector + PFSP=loss + Trinal-Clip) ==="
# background GPU-util sampler so the .out shows collection is actually feeding the GPU (>0% util)
( for i in $(seq 1 60); do
    u=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1)
    echo "[gpum t+$((i*30))s] ${u}"; sleep 30
  done ) &
UTIL_PID=$!

python3 scripts/debug_training.py --config_file "${CONFIG_FILE}" --config_name "${CONFIG_NAME}"

kill "${UTIL_PID}" 2>/dev/null || true

if [[ -f checkpoints/latest.pt ]]; then
  SNAP="checkpoints/snapshots/strong_fchpa/${CONFIG_NAME}_${SLURM_JOB_ID}.pt"
  cp -f checkpoints/latest.pt "${SNAP}" && echo "Snapshot: ${SNAP}"
fi
echo "STRONG_FCHPA_TRAIN_DONE $(date)"
