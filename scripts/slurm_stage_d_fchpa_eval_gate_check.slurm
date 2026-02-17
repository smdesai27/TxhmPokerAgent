#!/bin/bash
#SBATCH --job-name=ah_stage_d_fchpa_gate_check
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
mkdir -p logs/stage_d/slurm

module load python/3.11.11-5e66
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

EVAL_JSON="${EVAL_JSON:-}"
BASELINE_EVAL_JSON="${BASELINE_EVAL_JSON:-logs/stage_d/eval/eval_quadro_stage_d_fchpa_recover_explore_16k_cert_375207.json}"
CI_FLOOR="${CI_FLOOR:-830.0}"
REQUIRE_BEHAVIOR_EXTENDED="${REQUIRE_BEHAVIOR_EXTENDED:-true}"
REQUIRE_SOLVER_VERIFIED="${REQUIRE_SOLVER_VERIFIED:-true}"
REQUIRE_AGGRESSION_GATE="${REQUIRE_AGGRESSION_GATE:-true}"
# Backwards-compatible alias.
REQUIRE_POT_MIX_GATE="${REQUIRE_POT_MIX_GATE:-}"
MAX_ALLIN_FREQ="${MAX_ALLIN_FREQ:-}"
MIN_CALL_CHECK_FREQ="${MIN_CALL_CHECK_FREQ:-}"
MIN_HALF_POT_FREQ="${MIN_HALF_POT_FREQ:-}"
MIN_RAISE_TOTAL_FREQ="${MIN_RAISE_TOTAL_FREQ:-}"
MIN_POT_CHOOSE_GIVEN_LEGAL_FREQ="${MIN_POT_CHOOSE_GIVEN_LEGAL_FREQ:-}"
MIN_FOLD_FREQ="${MIN_FOLD_FREQ:-}"
MAX_FOLD_FREQ="${MAX_FOLD_FREQ:-}"
MIN_ENTROPY_BITS="${MIN_ENTROPY_BITS:-}"
MAX_ENTROPY_BITS="${MAX_ENTROPY_BITS:-}"

if [[ -z "${EVAL_JSON}" ]]; then
  echo "EVAL_JSON must be provided (path to evaluate_complete output)."
  exit 2
fi

EXTRA_ARGS=()
if [[ "${REQUIRE_BEHAVIOR_EXTENDED,,}" == "true" ]]; then
  EXTRA_ARGS+=(--require_behavior_extended)
else
  EXTRA_ARGS+=(--no-require_behavior_extended)
fi
if [[ "${REQUIRE_SOLVER_VERIFIED,,}" == "true" ]]; then
  EXTRA_ARGS+=(--require_solver_verified)
else
  EXTRA_ARGS+=(--no-require_solver_verified)
fi
if [[ -z "${REQUIRE_POT_MIX_GATE}" ]]; then
  if [[ "${REQUIRE_AGGRESSION_GATE,,}" == "true" ]]; then
    EXTRA_ARGS+=(--require_aggression_gate)
  else
    EXTRA_ARGS+=(--no-require_aggression_gate)
  fi
elif [[ "${REQUIRE_POT_MIX_GATE,,}" == "true" ]]; then
  EXTRA_ARGS+=(--require_pot_mix_gate)
else
  EXTRA_ARGS+=(--no-require_pot_mix_gate)
fi
if [[ -n "${MAX_ALLIN_FREQ}" ]]; then
  EXTRA_ARGS+=(--max_allin_freq "${MAX_ALLIN_FREQ}")
fi
if [[ -n "${MIN_CALL_CHECK_FREQ}" ]]; then
  EXTRA_ARGS+=(--min_call_check_freq "${MIN_CALL_CHECK_FREQ}")
fi
if [[ -n "${MIN_HALF_POT_FREQ}" ]]; then
  EXTRA_ARGS+=(--min_half_pot_freq "${MIN_HALF_POT_FREQ}")
fi
if [[ -n "${MIN_RAISE_TOTAL_FREQ}" ]]; then
  EXTRA_ARGS+=(--min_raise_total_freq "${MIN_RAISE_TOTAL_FREQ}")
fi
if [[ -n "${MIN_POT_CHOOSE_GIVEN_LEGAL_FREQ}" ]]; then
  EXTRA_ARGS+=(--min_pot_choose_given_legal_freq "${MIN_POT_CHOOSE_GIVEN_LEGAL_FREQ}")
fi
if [[ -n "${MIN_FOLD_FREQ}" ]]; then
  EXTRA_ARGS+=(--min_fold_freq "${MIN_FOLD_FREQ}")
fi
if [[ -n "${MAX_FOLD_FREQ}" ]]; then
  EXTRA_ARGS+=(--max_fold_freq "${MAX_FOLD_FREQ}")
fi
if [[ -n "${MIN_ENTROPY_BITS}" ]]; then
  EXTRA_ARGS+=(--min_entropy_bits "${MIN_ENTROPY_BITS}")
fi
if [[ -n "${MAX_ENTROPY_BITS}" ]]; then
  EXTRA_ARGS+=(--max_entropy_bits "${MAX_ENTROPY_BITS}")
fi

python3 poker_rl_agent/scripts/check_eval_gates.py \
  --eval_json "${EVAL_JSON}" \
  --baseline_eval_json "${BASELINE_EVAL_JSON}" \
  --ci_floor "${CI_FLOOR}" \
  "${EXTRA_ARGS[@]}"
