#!/usr/bin/env bash
#SBATCH --job-name=ah_stage_d_longrun_submit
#SBATCH --output=logs/stage_d/slurm/%x_%j.out
#SBATCH --error=logs/stage_d/slurm/%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
# Stage D guarded long run: train -> screen -> gate -> cert

set -euo pipefail

# Resolve repository root robustly for both:
# 1) direct bash execution, and
# 2) sbatch execution (script runs from Slurm spool path).
if [[ -n "${PROJECT_ROOT:-}" ]]; then
  ROOT_DIR="${PROJECT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/configs/training_configs.yaml" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi

if [[ ! -f "${ROOT_DIR}/configs/training_configs.yaml" ]]; then
  echo "Could not resolve project root containing configs/training_configs.yaml."
  echo "Set PROJECT_ROOT=/absolute/path/to/repo and re-run."
  exit 2
fi
if [[ ! -w "${ROOT_DIR}" ]]; then
  echo "Project root is not writable: ${ROOT_DIR}"
  echo "Re-run from a writable clone or set PROJECT_ROOT to a writable repo path."
  exit 2
fi

cd "${ROOT_DIR}"

mkdir -p logs/stage_d/eval logs/stage_d/slurm logs/stage_d/interview checkpoints/snapshots/stage_d/recovery

STAMP="$(date +%Y%m%d_%H%M%S)"

CONFIG_FILE="${CONFIG_FILE:-configs/training_configs.yaml}"
CONFIG_NAME="${CONFIG_NAME:-quadro_stage_d_fchpa_evfirst_longrun_30k_guarded}"
STRICT_ABSTRACTION="${STRICT_ABSTRACTION:-true}"

SCREEN_EPISODES="${SCREEN_EPISODES:-5000}"
CERT_EPISODES="${CERT_EPISODES:-10000}"
BASE_SEED="${BASE_SEED:-42}"
HOLDOUT_SEED_BASE="${HOLDOUT_SEED_BASE:-100342}"
HOLDOUT_SEED_COUNT="${HOLDOUT_SEED_COUNT:-5}"
CI_FLOOR="${CI_FLOOR:-800.0}"

SNAPSHOT_PATH="${SNAPSHOT_PATH:-checkpoints/snapshots/stage_d/recovery/longrun_30k_${STAMP}.pt}"
SCREEN_JSON="${SCREEN_JSON:-logs/stage_d/eval/eval_longrun_30k_screen_${STAMP}.json}"
CERT_JSON="${CERT_JSON:-logs/stage_d/eval/eval_longrun_30k_cert_${STAMP}.json}"

BASELINE_EVAL_JSON="${BASELINE_EVAL_JSON:-logs/stage_d/eval/eval_selected_21k_auto_cert_20260216_161321.json}"
ANCHOR_CKPT="checkpoints/snapshots/interview_ready/interview_ready_1.pt"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config file not found: ${CONFIG_FILE}"
  exit 2
fi
if [[ ! -f "${ANCHOR_CKPT}" ]]; then
  echo "Missing required checkpoint: ${ANCHOR_CKPT}"
  exit 2
fi

# Hard guard: never allow writes to interview_ready_1
if [[ "${SNAPSHOT_PATH}" == *"checkpoints/snapshots/interview_ready/interview_ready_1.pt" ]]; then
  echo "Refusing to run: SNAPSHOT_PATH points to interview_ready_1 (immutable anchor)."
  exit 3
fi

parse_job_id() {
  local raw="$1"
  echo "${raw%%;*}"
}

echo "Submitting Stage D guarded longrun train..."
TRAIN_RAW="$(
  sbatch --parsable \
    --export=ALL,CONFIG_NAME="${CONFIG_NAME}",CONFIG_FILE="${CONFIG_FILE}",STRICT_ABSTRACTION="${STRICT_ABSTRACTION}",WANDB_RUN_GROUP=stage_d_longrun_30k,WANDB_TAGS=stage_d,fchpa,longrun30k,SNAPSHOT_COPY=true,SNAPSHOT_PATH="${SNAPSHOT_PATH}" \
    scripts/slurm_stage_d_fchpa_corrective_train.slurm
)"
TRAIN_JOB="$(parse_job_id "${TRAIN_RAW}")"

echo "Submitting screen eval..."
SCREEN_RAW="$(
  sbatch --parsable \
    --dependency=afterok:"${TRAIN_JOB}" \
    --export=ALL,CONFIG_NAME="${CONFIG_NAME}",CONFIG_FILE="${CONFIG_FILE}",CHECKPOINT_PATH="${SNAPSHOT_PATH}",STRICT_ABSTRACTION="${STRICT_ABSTRACTION}",EPISODES_PER_SEED="${SCREEN_EPISODES}",PROFILE=standard,BASE_SEED="${BASE_SEED}",HOLDOUT_SEED_BASE="${HOLDOUT_SEED_BASE}",HOLDOUT_SEED_COUNT="${HOLDOUT_SEED_COUNT}",VERIFY_SOLVER_TRAINING=true,REQUIRE_BEHAVIOR_EXTENDED=true,OUTPUT_JSON="${SCREEN_JSON}" \
    scripts/slurm_stage_d_fchpa_corrective_screen_eval.slurm
)"
SCREEN_JOB="$(parse_job_id "${SCREEN_RAW}")"

echo "Submitting gate check..."
GATE_RAW="$(
  sbatch --parsable \
    --dependency=afterok:"${SCREEN_JOB}" \
    --export=ALL,EVAL_JSON="${SCREEN_JSON}",BASELINE_EVAL_JSON="${BASELINE_EVAL_JSON}",CI_FLOOR="${CI_FLOOR}",REQUIRE_BEHAVIOR_EXTENDED=true,REQUIRE_SOLVER_VERIFIED=true,REQUIRE_AGGRESSION_GATE=true,MAX_ALLIN_FREQ=0.05,MIN_CALL_CHECK_FREQ=0.40,MIN_HALF_POT_FREQ=0.01,MIN_RAISE_TOTAL_FREQ=0.08,MIN_FOLD_FREQ=0.35,MAX_FOLD_FREQ=0.55,MIN_ENTROPY_BITS=1.10,MAX_ENTROPY_BITS=1.80 \
    scripts/slurm_stage_d_fchpa_eval_gate_check.slurm
)"
GATE_JOB="$(parse_job_id "${GATE_RAW}")"

echo "Submitting cert eval..."
CERT_RAW="$(
  sbatch --parsable \
    --dependency=afterok:"${GATE_JOB}" \
    --export=ALL,CONFIG_NAME="${CONFIG_NAME}",CONFIG_FILE="${CONFIG_FILE}",CHECKPOINT_PATH="${SNAPSHOT_PATH}",STRICT_ABSTRACTION="${STRICT_ABSTRACTION}",EPISODES_PER_SEED="${CERT_EPISODES}",PROFILE=standard,BASE_SEED="${BASE_SEED}",HOLDOUT_SEED_BASE="$((HOLDOUT_SEED_BASE + 100))",HOLDOUT_SEED_COUNT="${HOLDOUT_SEED_COUNT}",REQUIRE_ROBUST_CI=true,VERIFY_SOLVER_TRAINING=true,REQUIRE_BEHAVIOR_EXTENDED=true,OUTPUT_JSON="${CERT_JSON}" \
    scripts/slurm_stage_d_fchpa_corrective_cert_eval.slurm
)"
CERT_JOB="$(parse_job_id "${CERT_RAW}")"

cat <<EOF
Submitted Stage D guarded longrun chain.
train_job=${TRAIN_JOB}
screen_job=${SCREEN_JOB}
gate_job=${GATE_JOB}
cert_job=${CERT_JOB}

anchor_checkpoint=${ANCHOR_CKPT}
snapshot=${SNAPSHOT_PATH}
screen_eval=${SCREEN_JSON}
cert_eval=${CERT_JSON}

Monitor:
  squeue -u "\$USER"
  tail -f logs/stage_d/slurm/ah_stage_d_fchpa_corrective_train_${TRAIN_JOB}.out
EOF
