#!/bin/bash
#SBATCH --job-name=ah_stage_d_fchpa_select
#SBATCH --output=logs/stage_d/slurm/%x_%j.out
#SBATCH --error=logs/stage_d/slurm/%x_%j.err
#SBATCH --time=00:20:00
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
mkdir -p logs/stage_d/slurm logs/stage_d/recovery

module load python/3.11.11-5e66
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

CANDIDATE_A_NAME="${CANDIDATE_A_NAME:-probe_a}"
CANDIDATE_A_EVAL_JSON="${CANDIDATE_A_EVAL_JSON:-}"
CANDIDATE_B_NAME="${CANDIDATE_B_NAME:-probe_b}"
CANDIDATE_B_EVAL_JSON="${CANDIDATE_B_EVAL_JSON:-}"
OUTPUT_JSON="${OUTPUT_JSON:-logs/stage_d/recovery/selection_${SLURM_JOB_ID}.json}"
OUTPUT_ENV="${OUTPUT_ENV:-logs/stage_d/recovery/selection_${SLURM_JOB_ID}.env}"
MIN_CI_FLOOR="${MIN_CI_FLOOR:-760.0}"
MAX_ALLIN_FREQ="${MAX_ALLIN_FREQ:-0.05}"
MIN_CALL_CHECK_FREQ="${MIN_CALL_CHECK_FREQ:-0.40}"
MIN_HALF_POT_FREQ="${MIN_HALF_POT_FREQ:-0.01}"
MIN_RAISE_TOTAL_FREQ="${MIN_RAISE_TOTAL_FREQ:-0.08}"
MIN_POT_CHOOSE_GIVEN_LEGAL_FREQ="${MIN_POT_CHOOSE_GIVEN_LEGAL_FREQ:-}"
MIN_FOLD_FREQ="${MIN_FOLD_FREQ:-0.35}"
MAX_FOLD_FREQ="${MAX_FOLD_FREQ:-0.55}"
MIN_ENTROPY_BITS="${MIN_ENTROPY_BITS:-1.1}"
MAX_ENTROPY_BITS="${MAX_ENTROPY_BITS:-1.8}"
EXPLOIT_WEIGHT="${EXPLOIT_WEIGHT:-0.20}"
EXPLOIT_CLIP="${EXPLOIT_CLIP:-500.0}"

if [[ -z "${CANDIDATE_A_EVAL_JSON}" || -z "${CANDIDATE_B_EVAL_JSON}" ]]; then
  echo "CANDIDATE_A_EVAL_JSON and CANDIDATE_B_EVAL_JSON must be provided."
  exit 2
fi

EXTRA_ARGS=()
if [[ -n "${MIN_POT_CHOOSE_GIVEN_LEGAL_FREQ}" ]]; then
  EXTRA_ARGS+=(--min_pot_choose_given_legal_freq "${MIN_POT_CHOOSE_GIVEN_LEGAL_FREQ}")
fi

python3 poker_rl_agent/scripts/select_stage_d_recovery_winner.py \
  --candidate "${CANDIDATE_A_NAME}=${CANDIDATE_A_EVAL_JSON}" \
  --candidate "${CANDIDATE_B_NAME}=${CANDIDATE_B_EVAL_JSON}" \
  --output_json "${OUTPUT_JSON}" \
  --output_env "${OUTPUT_ENV}" \
  --min_ci_floor "${MIN_CI_FLOOR}" \
  --max_allin_freq "${MAX_ALLIN_FREQ}" \
  --min_call_check_freq "${MIN_CALL_CHECK_FREQ}" \
  --min_half_pot_freq "${MIN_HALF_POT_FREQ}" \
  --min_raise_total_freq "${MIN_RAISE_TOTAL_FREQ}" \
  --min_fold_freq "${MIN_FOLD_FREQ}" \
  --max_fold_freq "${MAX_FOLD_FREQ}" \
  --min_entropy_bits "${MIN_ENTROPY_BITS}" \
  --max_entropy_bits "${MAX_ENTROPY_BITS}" \
  --exploit_weight "${EXPLOIT_WEIGHT}" \
  --exploit_clip "${EXPLOIT_CLIP}" \
  "${EXTRA_ARGS[@]}"

echo "Selection outputs:"
echo "  json=${OUTPUT_JSON}"
echo "  env=${OUTPUT_ENV}"
