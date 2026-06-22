#!/bin/bash
#SBATCH --job-name=ah_stage_b_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
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
CONFIG_NAME="${CONFIG_NAME:-quadro_stage_b}"
CONFIG_FILE="${CONFIG_FILE:-configs/training_configs.yaml}"
EPISODES_PER_SEED="${EPISODES_PER_SEED:-5000}"
BASE_SEED="${BASE_SEED:-42}"

OUT_5K="${OUT_5K:-logs/eval_stage_b_5k.json}"
OUT_10K="${OUT_10K:-logs/eval_stage_b_10k.json}"

echo "Running Stage B primary benchmark (5k MCCFR-ES)"
python poker_rl_agent/scripts/evaluate.py \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config_file "${CONFIG_FILE}" \
  --config_name "${CONFIG_NAME}" \
  --episodes_per_seed "${EPISODES_PER_SEED}" \
  --baseline_iters 5000 \
  --baseline_seeds 5 \
  --seed "${BASE_SEED}" \
  --output_json "${OUT_5K}"

echo "Running Stage B robustness benchmark (10k MCCFR-ES)"
python poker_rl_agent/scripts/evaluate.py \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config_file "${CONFIG_FILE}" \
  --config_name "${CONFIG_NAME}" \
  --episodes_per_seed "${EPISODES_PER_SEED}" \
  --baseline_iters 10000 \
  --baseline_seeds 3 \
  --seed "${BASE_SEED}" \
  --output_json "${OUT_10K}"

echo "Wrote benchmark reports:"
echo "  ${OUT_5K}"
echo "  ${OUT_10K}"
