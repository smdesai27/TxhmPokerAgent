#!/bin/bash
#SBATCH --job-name=ah_stage_c_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node: ${SLURM_NODELIST}"
echo "Date: $(date)"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-$PWD}"
echo "-------------------"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs checkpoints

module load python/3.11.11-5e66
module load cuda/11.8.0-kuhf

source venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

CONFIG_NAME="${CONFIG_NAME:-quadro_stage_c_fcpa}"
CONFIG_FILE="${CONFIG_FILE:-configs/training_configs.yaml}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-alpha-holdem-poker}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-stage_c_fcpa}"
WANDB_TAGS="${WANDB_TAGS:-stage_c,fcpa}"

export WANDB_MODE
export WANDB_PROJECT
export WANDB_ENTITY
export WANDB_RUN_GROUP
export WANDB_TAGS

if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_MODE=online requires WANDB_API_KEY in the SLURM environment."
  echo "Set it with: export WANDB_API_KEY=... before sbatch."
  exit 2
fi

echo "Running Stage C training with config: ${CONFIG_NAME}"
echo "W&B: mode=${WANDB_MODE}, project=${WANDB_PROJECT}, entity=${WANDB_ENTITY}, group=${WANDB_RUN_GROUP}"
python scripts/debug_training.py --config_file "${CONFIG_FILE}" --config_name "${CONFIG_NAME}"
