#!/bin/bash
#SBATCH --job-name=ah_stage_d_fchpa_prep_winner
#SBATCH --output=logs/stage_d/slurm/%x_%j.out
#SBATCH --error=logs/stage_d/slurm/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node: ${SLURM_NODELIST}"
echo "Date: $(date)"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-$PWD}"
echo "-------------------"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/stage_d/slurm checkpoints/snapshots/stage_d/recovery

module load python/3.11.11-5e66
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

WINNER_ENV_PATH="${WINNER_ENV_PATH:-}"
TARGET_PATH="${TARGET_PATH:-checkpoints/snapshots/stage_d/recovery/winner_19k.pt}"

if [[ -z "${WINNER_ENV_PATH}" ]]; then
  echo "WINNER_ENV_PATH must be provided."
  exit 2
fi
if [[ ! -f "${WINNER_ENV_PATH}" ]]; then
  echo "Winner env file not found: ${WINNER_ENV_PATH}"
  exit 2
fi

# shellcheck source=/dev/null
source "${WINNER_ENV_PATH}"

if [[ -z "${WINNER_CHECKPOINT:-}" ]]; then
  echo "WINNER_CHECKPOINT missing in ${WINNER_ENV_PATH}"
  exit 2
fi
if [[ ! -f "${WINNER_CHECKPOINT}" ]]; then
  echo "Winner checkpoint not found: ${WINNER_CHECKPOINT}"
  exit 2
fi

mkdir -p "$(dirname "${TARGET_PATH}")"
cp -f "${WINNER_CHECKPOINT}" "${TARGET_PATH}"
echo "Prepared winner checkpoint:"
echo "  source=${WINNER_CHECKPOINT}"
echo "  target=${TARGET_PATH}"

python3 poker_rl_agent/scripts/checkpoint_manager.py index --root checkpoints --apply || true
