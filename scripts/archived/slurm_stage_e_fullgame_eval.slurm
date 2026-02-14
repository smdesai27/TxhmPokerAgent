#!/bin/bash
#SBATCH --job-name=ah_stage_e_fullgame_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

echo "Job: ${SLURM_JOB_NAME} (${SLURM_JOB_ID})"
echo "Node: ${SLURM_NODELIST}"
echo "Date: $(date)"
echo "Submit dir: ${SLURM_SUBMIT_DIR:-$PWD}"
echo "-------------------"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

module load python/3.11.11-5e66
module load cuda/11.8.0-kuhf

source venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/latest.pt}"
CONFIG_NAME="${CONFIG_NAME:-quadro_stage_e_fullgame_progressive}"
CONFIG_FILE="${CONFIG_FILE:-configs/training_configs.yaml}"
STRICT_ABSTRACTION="${STRICT_ABSTRACTION:-true}"
EPISODES_PER_SEED="${EPISODES_PER_SEED:-2500}"
BASE_SEED="${BASE_SEED:-42}"
HOLDOUT_SEED_BASE="${HOLDOUT_SEED_BASE:-30042}"
HOLDOUT_SEED_COUNT="${HOLDOUT_SEED_COUNT:-5}"
PROFILE="${PROFILE:-standard}"
OUTPUT_JSON="${OUTPUT_JSON:-logs/eval_stage_e_fullgame_${SLURM_JOB_ID}.json}"

export STRICT_ABSTRACTION

echo "Running Stage E fullgame complete evaluation:"
echo "  checkpoint=${CHECKPOINT_PATH}"
echo "  config=${CONFIG_NAME}"
echo "  strict_abstraction=${STRICT_ABSTRACTION}"
echo "  profile=${PROFILE}"
echo "  episodes_per_seed=${EPISODES_PER_SEED}"
echo "  base_seed=${BASE_SEED}"
echo "  holdout_seed_base=${HOLDOUT_SEED_BASE}"
echo "  holdout_seed_count=${HOLDOUT_SEED_COUNT}"
echo "  output=${OUTPUT_JSON}"

python poker_rl_agent/scripts/evaluate_complete.py \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config_file "${CONFIG_FILE}" \
  --config_name "${CONFIG_NAME}" \
  --episodes_per_seed "${EPISODES_PER_SEED}" \
  --profile "${PROFILE}" \
  --seed "${BASE_SEED}" \
  --verify_solver_training \
  --holdout_seed_base "${HOLDOUT_SEED_BASE}" \
  --holdout_seed_count "${HOLDOUT_SEED_COUNT}" \
  --output_json "${OUTPUT_JSON}"
