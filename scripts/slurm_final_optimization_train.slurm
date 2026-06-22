#!/bin/bash
#SBATCH --job-name=ah_final_optimization_train
#SBATCH --output=logs/final_opt/slurm/%x_%j.out
#SBATCH --error=logs/final_opt/slurm/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --signal=B:USR1@300

set -euo pipefail

echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node: ${SLURM_NODELIST}"
echo "Date: $(date)"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-$PWD}"
echo "-------------------"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/final_opt/slurm checkpoints/snapshots/final_opt

module load python/3.11.11-5e66
module load cuda/11.8.0-kuhf

source venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

CONFIG_NAME="${CONFIG_NAME:-final_optimization}"
CONFIG_FILE="${CONFIG_FILE:-configs/training_configs.yaml}"
STRICT_ABSTRACTION="${STRICT_ABSTRACTION:-true}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-alpha-holdem-poker}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-final_optimization}"
WANDB_TAGS="${WANDB_TAGS:-final_opt,fchpa,interview}"
RUN_ID="${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)"

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

# Resume logic: detect existing checkpoint
RESUME_FLAG=""
if [[ -f checkpoints/latest.pt ]]; then
  echo "Resuming from existing checkpoint: checkpoints/latest.pt"
  RESUME_FLAG="--resume_from checkpoints/latest.pt"
else
  echo "No checkpoint found at checkpoints/latest.pt; starting fresh."
fi

# Signal handler: on USR1 (5 min before timeout), snapshot and exit
TIMEOUT_SNAPSHOT="checkpoints/snapshots/final_opt/timeout_${RUN_ID}.pt"
handle_timeout() {
  echo "Received USR1 signal — saving timeout snapshot..."
  if [[ -f checkpoints/latest.pt ]]; then
    cp -f checkpoints/latest.pt "${TIMEOUT_SNAPSHOT}"
    echo "Timeout snapshot saved: ${TIMEOUT_SNAPSHOT}"
  else
    echo "No checkpoint to snapshot."
  fi
  # Let the background training process exit naturally
  if [[ -n "${TRAIN_PID:-}" ]]; then
    kill -TERM "${TRAIN_PID}" 2>/dev/null || true
    wait "${TRAIN_PID}" 2>/dev/null || true
  fi
  exit 0
}
trap handle_timeout USR1

echo "Running final optimization training:"
echo "  config=${CONFIG_NAME}"
echo "  strict_abstraction=${STRICT_ABSTRACTION}"
echo "  wandb_group=${WANDB_RUN_GROUP}"
echo "  run_id=${RUN_ID}"

python3 scripts/debug_training.py \
  --config_file "${CONFIG_FILE}" \
  --config_name "${CONFIG_NAME}" &
TRAIN_PID=$!
wait ${TRAIN_PID}
TRAIN_EXIT=$?
echo "Training exited with code: ${TRAIN_EXIT}"

# Post-training snapshot
FINAL_SNAPSHOT="checkpoints/snapshots/final_opt/${RUN_ID}.pt"
if [[ -f checkpoints/latest.pt ]]; then
  cp -f checkpoints/latest.pt "${FINAL_SNAPSHOT}"
  echo "Final snapshot saved: ${FINAL_SNAPSHOT}"
fi

python3 poker_rl_agent/scripts/checkpoint_manager.py index --root checkpoints --apply || true

exit ${TRAIN_EXIT}
