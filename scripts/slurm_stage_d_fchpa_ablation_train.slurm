#!/bin/bash
#SBATCH --job-name=ah_stage_d_fchpa_ablate_train
#SBATCH --output=logs/stage_d/slurm/%x_%j.out
#SBATCH --error=logs/stage_d/slurm/%x_%j.err
#SBATCH --time=18:00:00
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
mkdir -p logs/stage_d/slurm logs/stage_d/eval logs/stage_d/metrics checkpoints/snapshots/stage_d/ablations

module load python/3.11.11-5e66
module load cuda/11.8.0-kuhf

source venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

CONFIG_NAME="${CONFIG_NAME:-quadro_stage_d_fchpa_recover_explore_ablate}"
CONFIG_FILE="${CONFIG_FILE:-configs/training_configs.yaml}"
STRICT_ABSTRACTION="${STRICT_ABSTRACTION:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-alpha-holdem-poker}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-stage_d_fchpa_ablation}"
WANDB_TAGS="${WANDB_TAGS:-stage_d,fchpa,ablation}"
SNAPSHOT_COPY="${SNAPSHOT_COPY:-true}"
SNAPSHOT_PATH="${SNAPSHOT_PATH:-}"

if [[ -z "${SNAPSHOT_PATH}" ]]; then
  SNAPSHOT_PATH="checkpoints/snapshots/stage_d/ablations/${CONFIG_NAME}_${SLURM_JOB_ID}.pt"
fi

export STRICT_ABSTRACTION
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

echo "Running Stage D FCHPA ablation training:"
echo "  config=${CONFIG_NAME}"
echo "  strict_abstraction=${STRICT_ABSTRACTION}"
echo "  wandb_group=${WANDB_RUN_GROUP}"
python scripts/debug_training.py --config_file "${CONFIG_FILE}" --config_name "${CONFIG_NAME}"

if [[ "${SNAPSHOT_COPY,,}" == "true" && -f checkpoints/latest.pt ]]; then
  mkdir -p "$(dirname "${SNAPSHOT_PATH}")"
  cp -f checkpoints/latest.pt "${SNAPSHOT_PATH}"
  echo "Copied ablation checkpoint snapshot: ${SNAPSHOT_PATH}"
fi

python poker_rl_agent/scripts/checkpoint_manager.py index --root checkpoints --apply || true
