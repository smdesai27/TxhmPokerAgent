#!/bin/bash
#SBATCH --job-name=ah_stage_d_fchpa_cont12k_cert
#SBATCH --output=logs/stage_d/slurm/%x_%j.out
#SBATCH --error=logs/stage_d/slurm/%x_%j.err
#SBATCH --time=30:00:00
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
mkdir -p logs/stage_d/slurm logs/stage_d/eval

module load python/3.11.11-5e66
module load cuda/11.8.0-kuhf

source venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/latest.pt}"
CONFIG_NAME="${CONFIG_NAME:-quadro_stage_d_fchpa_continue_12k_tuned}"
CONFIG_FILE="${CONFIG_FILE:-configs/training_configs.yaml}"
STRICT_ABSTRACTION="${STRICT_ABSTRACTION:-true}"
EPISODES_PER_SEED="${EPISODES_PER_SEED:-10000}"
BASE_SEED="${BASE_SEED:-42}"
HOLDOUT_SEED_BASE="${HOLDOUT_SEED_BASE:-50042}"
HOLDOUT_SEED_COUNT="${HOLDOUT_SEED_COUNT:-5}"
PROFILE="${PROFILE:-standard}"
REQUIRE_ROBUST_CI="${REQUIRE_ROBUST_CI:-true}"
VERIFY_SOLVER_TRAINING="${VERIFY_SOLVER_TRAINING:-true}"
OUTPUT_JSON="${OUTPUT_JSON:-logs/stage_d/eval/eval_stage_d_fchpa_continue_12k_cert_${SLURM_JOB_ID}.json}"

export STRICT_ABSTRACTION

EXTRA_ARGS=()
if [[ "${REQUIRE_ROBUST_CI,,}" == "true" ]]; then
  EXTRA_ARGS+=(--require_robust_ci)
fi
if [[ "${VERIFY_SOLVER_TRAINING,,}" == "true" ]]; then
  EXTRA_ARGS+=(--verify_solver_training)
else
  EXTRA_ARGS+=(--no-verify_solver_training)
fi

echo "Running Stage D FCHPA continuation CERT evaluation:"
echo "  checkpoint=${CHECKPOINT_PATH}"
echo "  config=${CONFIG_NAME}"
echo "  strict_abstraction=${STRICT_ABSTRACTION}"
echo "  profile=${PROFILE}"
echo "  episodes_per_seed=${EPISODES_PER_SEED}"
echo "  holdout_seed_base=${HOLDOUT_SEED_BASE}"
echo "  holdout_seed_count=${HOLDOUT_SEED_COUNT}"
echo "  require_robust_ci=${REQUIRE_ROBUST_CI}"
echo "  output=${OUTPUT_JSON}"

python poker_rl_agent/scripts/evaluate_complete.py \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config_file "${CONFIG_FILE}" \
  --config_name "${CONFIG_NAME}" \
  --episodes_per_seed "${EPISODES_PER_SEED}" \
  --profile "${PROFILE}" \
  --seed "${BASE_SEED}" \
  --holdout_seed_base "${HOLDOUT_SEED_BASE}" \
  --holdout_seed_count "${HOLDOUT_SEED_COUNT}" \
  --output_json "${OUTPUT_JSON}" \
  "${EXTRA_ARGS[@]}"
