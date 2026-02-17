#!/bin/bash
#SBATCH --job-name=ah_stage_d_fchpa_corrective_screen
#SBATCH --output=logs/stage_d/slurm/%x_%j.out
#SBATCH --error=logs/stage_d/slurm/%x_%j.err
#SBATCH --time=16:00:00
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
CONFIG_NAME="${CONFIG_NAME:-quadro_stage_d_fchpa_recover_diverse_corrective_17k}"
CONFIG_FILE="${CONFIG_FILE:-configs/training_configs.yaml}"
STRICT_ABSTRACTION="${STRICT_ABSTRACTION:-true}"
EPISODES_PER_SEED="${EPISODES_PER_SEED:-5000}"
PROFILE="${PROFILE:-standard}"
BASE_SEED="${BASE_SEED:-42}"
HOLDOUT_SEED_BASE="${HOLDOUT_SEED_BASE:-91042}"
HOLDOUT_SEED_COUNT="${HOLDOUT_SEED_COUNT:-5}"
VERIFY_SOLVER_TRAINING="${VERIFY_SOLVER_TRAINING:-true}"
REQUIRE_BEHAVIOR_EXTENDED="${REQUIRE_BEHAVIOR_EXTENDED:-}"
OUTPUT_JSON="${OUTPUT_JSON:-}"
WINNER_ENV_PATH="${WINNER_ENV_PATH:-}"
WINNER_JSON_PATH="${WINNER_JSON_PATH:-}"
SELECTED_CONFIG_CONSERVATIVE="${SELECTED_CONFIG_CONSERVATIVE:-quadro_stage_d_fchpa_recover_selected_20k_conservative}"
SELECTED_CONFIG_BALANCED="${SELECTED_CONFIG_BALANCED:-quadro_stage_d_fchpa_recover_selected_20k_balanced}"

if [[ "${CONFIG_NAME,,}" == "auto" ]]; then
  CONFIG_NAME="$(
    python3 poker_rl_agent/scripts/resolve_stage_d_selected_config.py \
      --winner_env_path "${WINNER_ENV_PATH}" \
      --winner_json_path "${WINNER_JSON_PATH}" \
      --selected_config_conservative "${SELECTED_CONFIG_CONSERVATIVE}" \
      --selected_config_balanced "${SELECTED_CONFIG_BALANCED}" \
      --fallback_config "${SELECTED_CONFIG_CONSERVATIVE}"
  )"
  if [[ -z "${CONFIG_NAME}" ]]; then
    echo "Failed to resolve CONFIG_NAME from winner metadata."
    exit 2
  fi
fi

if [[ -z "${OUTPUT_JSON}" ]]; then
  OUTPUT_JSON="logs/stage_d/eval/eval_${CONFIG_NAME}_screen_${SLURM_JOB_ID}.json"
fi

export STRICT_ABSTRACTION

EXTRA_ARGS=()
if [[ "${VERIFY_SOLVER_TRAINING,,}" == "true" ]]; then
  EXTRA_ARGS+=(--verify_solver_training)
else
  EXTRA_ARGS+=(--no-verify_solver_training)
fi
if [[ -n "${REQUIRE_BEHAVIOR_EXTENDED}" ]]; then
  if [[ "${REQUIRE_BEHAVIOR_EXTENDED,,}" == "true" ]]; then
    EXTRA_ARGS+=(--require_behavior_extended)
  else
    EXTRA_ARGS+=(--no-require_behavior_extended)
  fi
fi

echo "Running Stage D corrective SCREEN evaluation:"
echo "  checkpoint=${CHECKPOINT_PATH}"
echo "  config=${CONFIG_NAME}"
echo "  episodes_per_seed=${EPISODES_PER_SEED}"
echo "  holdout_seed_base=${HOLDOUT_SEED_BASE}"
echo "  holdout_seed_count=${HOLDOUT_SEED_COUNT}"
echo "  output=${OUTPUT_JSON}"

python3 poker_rl_agent/scripts/evaluate_complete.py \
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
