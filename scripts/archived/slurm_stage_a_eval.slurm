#!/bin/bash
#SBATCH --job-name=ah_stage_a_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=08:00:00
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
mkdir -p logs

module load python/3.11.11-5e66
module load cuda/11.8.0-kuhf

source venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/latest.pt}"
CONFIG_NAME="${CONFIG_NAME:-quadro_stage_a}"
CONFIG_FILE="${CONFIG_FILE:-configs/training_configs.yaml}"
EPISODES_PER_SEED="${EPISODES_PER_SEED:-2000}"
BASELINE_ITERS="${BASELINE_ITERS:-3000}"
BASELINE_SEEDS="${BASELINE_SEEDS:-3}"
BASE_SEED="${BASE_SEED:-42}"
OUTPUT_JSON="${OUTPUT_JSON:-logs/eval_stage_a_${SLURM_JOB_ID}.json}"

echo "Running Stage A evaluation:"
echo "  checkpoint=${CHECKPOINT_PATH}"
echo "  config=${CONFIG_NAME}"
echo "  episodes_per_seed=${EPISODES_PER_SEED}"
echo "  baseline_iters=${BASELINE_ITERS}"
echo "  baseline_seeds=${BASELINE_SEEDS}"
echo "  seed=${BASE_SEED}"
echo "  output=${OUTPUT_JSON}"

python poker_rl_agent/scripts/evaluate.py \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config_file "${CONFIG_FILE}" \
  --config_name "${CONFIG_NAME}" \
  --episodes_per_seed "${EPISODES_PER_SEED}" \
  --baseline_iters "${BASELINE_ITERS}" \
  --baseline_seeds "${BASELINE_SEEDS}" \
  --seed "${BASE_SEED}" \
  --output_json "${OUTPUT_JSON}"
