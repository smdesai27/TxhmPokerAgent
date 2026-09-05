#!/usr/bin/env bash
# One-command Stage D continuation chain:
# train (21k->24k) -> screen eval -> gate check -> cert eval

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs/stage_d/eval logs/stage_d/slurm logs/stage_d/interview checkpoints/snapshots/stage_d/recovery

STAMP="$(date +%Y%m%d_%H%M%S)"

CONFIG_FILE="${CONFIG_FILE:-configs/training_configs.yaml}"
CONFIG_NAME="${CONFIG_NAME:-quadro_stage_d_fchpa_evfirst_interview_continue_24k}"
STRICT_ABSTRACTION="${STRICT_ABSTRACTION:-true}"
SCREEN_EPISODES="${SCREEN_EPISODES:-5000}"
CERT_EPISODES="${CERT_EPISODES:-10000}"
BASE_SEED="${BASE_SEED:-42}"
HOLDOUT_SEED_BASE="${HOLDOUT_SEED_BASE:-100242}"
HOLDOUT_SEED_COUNT="${HOLDOUT_SEED_COUNT:-5}"
CI_FLOOR="${CI_FLOOR:-800.0}"
SNAPSHOT_PATH="${SNAPSHOT_PATH:-checkpoints/snapshots/stage_d/recovery/interview_continue_24k_${STAMP}.pt}"
SCREEN_JSON="${SCREEN_JSON:-logs/stage_d/eval/eval_interview_continue_24k_screen_${STAMP}.json}"
CERT_JSON="${CERT_JSON:-logs/stage_d/eval/eval_interview_continue_24k_cert_${STAMP}.json}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config file not found: ${CONFIG_FILE}"
  exit 2
fi
if [[ ! -f "checkpoints/snapshots/champion/stage_d_fchpa_21k.pt" ]]; then
  echo "Missing required checkpoint: checkpoints/snapshots/champion/stage_d_fchpa_21k.pt"
  exit 2
fi

parse_job_id() {
  local raw="$1"
  echo "${raw%%;*}"
}

echo "Submitting Stage D interview continuation train..."
TRAIN_RAW="$(
  sbatch --parsable \
    --export=ALL,CONFIG_NAME="${CONFIG_NAME}",CONFIG_FILE="${CONFIG_FILE}",STRICT_ABSTRACTION="${STRICT_ABSTRACTION}",WANDB_RUN_GROUP=stage_d_interview_continue_24k,WANDB_TAGS=stage_d,fchpa,interview,continue24k,SNAPSHOT_COPY=true,SNAPSHOT_PATH="${SNAPSHOT_PATH}" \
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
    --export=ALL,EVAL_JSON="${SCREEN_JSON}",CI_FLOOR="${CI_FLOOR}",REQUIRE_BEHAVIOR_EXTENDED=true,REQUIRE_SOLVER_VERIFIED=true,REQUIRE_AGGRESSION_GATE=true,MAX_ALLIN_FREQ=0.05,MIN_CALL_CHECK_FREQ=0.40,MIN_HALF_POT_FREQ=0.01,MIN_RAISE_TOTAL_FREQ=0.08,MIN_FOLD_FREQ=0.35,MAX_FOLD_FREQ=0.55,MIN_ENTROPY_BITS=1.10,MAX_ENTROPY_BITS=1.80 \
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
Submitted Stage D interview continuation chain.
train_job=${TRAIN_JOB}
screen_job=${SCREEN_JOB}
gate_job=${GATE_JOB}
cert_job=${CERT_JOB}
snapshot=${SNAPSHOT_PATH}
screen_eval=${SCREEN_JSON}
cert_eval=${CERT_JSON}

Monitor:
  squeue -u "\$USER"
  tail -f logs/stage_d/slurm/ah_stage_d_fchpa_corrective_train_${TRAIN_JOB}.out
EOF
