#!/bin/bash
# Submit slurm_final_optimization_eval.slurm with optional overrides.
# Usage: ./scripts/submit_final_optimization_eval.sh [options]
#
# Options:
#   --checkpoint PATH      Path to checkpoint (default: checkpoints/latest.pt)
#   --config_name NAME     Config preset name (default: final_optimization)
#   --config_file FILE     YAML config file (default: configs/training_configs.yaml)
#   --episodes N           Episodes per seed (default: 5000)
#   --profile PROFILE      Eval profile (default: cert)
#   --seed N               Base seed (default: 42)
#   --holdout_seed_base N  Holdout seed base (default: 92042)
#   --holdout_seed_count N Number of holdout seeds (default: 5)
#   --no_strict            Disable strict abstraction (default: strict=true)
#   --output_json PATH     Output JSON path (auto-named by default)
#   -h, --help             Show this help

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SLURM_SCRIPT="${SCRIPT_DIR}/slurm_final_optimization_eval.slurm"

# --- Defaults (mirror the SLURM script defaults) ---
CHECKPOINT_PATH="checkpoints/latest.pt"
CONFIG_NAME="final_optimization"
CONFIG_FILE="configs/training_configs.yaml"
STRICT_ABSTRACTION="true"
EPISODES_PER_SEED="5000"
PROFILE="cert"
BASE_SEED="42"
HOLDOUT_SEED_BASE="92042"
HOLDOUT_SEED_COUNT="5"
OUTPUT_JSON=""   # empty → let SLURM script auto-name with job ID

# --- Parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)       CHECKPOINT_PATH="$2"; shift 2 ;;
    --config_name)      CONFIG_NAME="$2";     shift 2 ;;
    --config_file)      CONFIG_FILE="$2";     shift 2 ;;
    --episodes)         EPISODES_PER_SEED="$2"; shift 2 ;;
    --profile)          PROFILE="$2";         shift 2 ;;
    --seed)             BASE_SEED="$2";       shift 2 ;;
    --holdout_seed_base)   HOLDOUT_SEED_BASE="$2";  shift 2 ;;
    --holdout_seed_count)  HOLDOUT_SEED_COUNT="$2"; shift 2 ;;
    --no_strict)        STRICT_ABSTRACTION="false"; shift ;;
    --output_json)      OUTPUT_JSON="$2";     shift 2 ;;
    -h|--help)
      sed -n '2,/^[^#]/{ /^#/!q; s/^# \?//p }' "$0"
      exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1 ;;
  esac
done

# --- Validate ---
if [[ ! -f "${REPO_ROOT}/${CHECKPOINT_PATH}" ]]; then
  echo "ERROR: checkpoint not found: ${REPO_ROOT}/${CHECKPOINT_PATH}" >&2
  exit 1
fi

if [[ ! -f "${REPO_ROOT}/${CONFIG_FILE}" ]]; then
  echo "ERROR: config file not found: ${REPO_ROOT}/${CONFIG_FILE}" >&2
  exit 1
fi

if [[ ! -f "${SLURM_SCRIPT}" ]]; then
  echo "ERROR: SLURM script not found: ${SLURM_SCRIPT}" >&2
  exit 1
fi

# --- Ensure log dirs exist ---
mkdir -p "${REPO_ROOT}/logs/final_opt/slurm" \
         "${REPO_ROOT}/logs/final_opt/eval"

# --- Build sbatch env exports ---
EXPORTS="CHECKPOINT_PATH=${CHECKPOINT_PATH}"
EXPORTS+=",CONFIG_NAME=${CONFIG_NAME}"
EXPORTS+=",CONFIG_FILE=${CONFIG_FILE}"
EXPORTS+=",STRICT_ABSTRACTION=${STRICT_ABSTRACTION}"
EXPORTS+=",EPISODES_PER_SEED=${EPISODES_PER_SEED}"
EXPORTS+=",PROFILE=${PROFILE}"
EXPORTS+=",BASE_SEED=${BASE_SEED}"
EXPORTS+=",HOLDOUT_SEED_BASE=${HOLDOUT_SEED_BASE}"
EXPORTS+=",HOLDOUT_SEED_COUNT=${HOLDOUT_SEED_COUNT}"
if [[ -n "${OUTPUT_JSON}" ]]; then
  EXPORTS+=",OUTPUT_JSON=${OUTPUT_JSON}"
fi

# --- Submit ---
echo "Submitting: ${SLURM_SCRIPT}"
echo "  checkpoint      = ${CHECKPOINT_PATH}"
echo "  config_name     = ${CONFIG_NAME}"
echo "  config_file     = ${CONFIG_FILE}"
echo "  strict          = ${STRICT_ABSTRACTION}"
echo "  episodes        = ${EPISODES_PER_SEED}"
echo "  profile         = ${PROFILE}"
echo "  seed            = ${BASE_SEED}"
echo "  holdout_base    = ${HOLDOUT_SEED_BASE}"
echo "  holdout_count   = ${HOLDOUT_SEED_COUNT}"
[[ -n "${OUTPUT_JSON}" ]] && echo "  output_json     = ${OUTPUT_JSON}"
echo ""

JOB_ID=$(cd "${REPO_ROOT}" && sbatch --export="${EXPORTS}" "${SLURM_SCRIPT}" | awk '{print $NF}')
echo "Submitted job ${JOB_ID}"
echo "Logs: logs/final_opt/slurm/ah_final_optimization_eval_${JOB_ID}.out"
echo "Eval: logs/final_opt/eval/eval_cert_${JOB_ID}.json"
