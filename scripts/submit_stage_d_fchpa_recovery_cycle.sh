#!/usr/bin/env bash
# Stage D recovery cycle:
# 1) Probe A train->screen->gate
# 2) Probe B train->screen->gate
# 3) Select best probe with behavior envelope constraints
# 4) Prepare winner checkpoint
# 5) Promote winner branch (+1k) with winner-aware config continuity
# 6) Screen + gate + cert promotion

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs/stage_d/eval logs/stage_d/slurm logs/stage_d/recovery checkpoints/snapshots/stage_d/recovery

STAMP="$(date +%Y%m%d_%H%M%S)"

CONFIG_FILE="${CONFIG_FILE:-configs/training_configs.yaml}"
ANCHOR_CHECKPOINT="${ANCHOR_CHECKPOINT:-checkpoints/snapshots/stage_d/recovery/quadro_stage_d_fchpa_recover_conservative_probe_19k_20260215_155748.pt}"
ANCHOR_TARGET_PATH="${ANCHOR_TARGET_PATH:-checkpoints/snapshots/stage_d/recovery/winner_19k.pt}"
BASELINE_EVAL_JSON="${BASELINE_EVAL_JSON:-logs/stage_d/eval/eval_quadro_stage_d_fchpa_recover_explore_16k_cert_375207.json}"

PROBE_A_CONFIG="${PROBE_A_CONFIG:-quadro_stage_d_fchpa_evfirst_probe_h1_20k}"
PROBE_B_CONFIG="${PROBE_B_CONFIG:-quadro_stage_d_fchpa_evfirst_probe_h2_20k}"
SELECTED_CONFIG="${SELECTED_CONFIG:-auto}"
SELECTED_CONFIG_CONSERVATIVE="${SELECTED_CONFIG_CONSERVATIVE:-quadro_stage_d_fchpa_evfirst_selected_21k_h2}"
SELECTED_CONFIG_BALANCED="${SELECTED_CONFIG_BALANCED:-quadro_stage_d_fchpa_evfirst_selected_21k_h1}"
SELECTED_CONFIG_LABEL="${SELECTED_CONFIG_LABEL:-${SELECTED_CONFIG}}"

CI_FLOOR="${CI_FLOOR:-760.0}"
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

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config file not found: ${CONFIG_FILE}"
  exit 2
fi
if [[ ! -f "${ANCHOR_CHECKPOINT}" ]]; then
  echo "Anchor checkpoint not found: ${ANCHOR_CHECKPOINT}"
  exit 2
fi
if [[ ! -f "${BASELINE_EVAL_JSON}" ]]; then
  echo "Baseline eval JSON not found: ${BASELINE_EVAL_JSON}"
  exit 2
fi

mkdir -p "$(dirname "${ANCHOR_TARGET_PATH}")"
if [[ "${ANCHOR_CHECKPOINT}" != "${ANCHOR_TARGET_PATH}" ]]; then
  cp -f "${ANCHOR_CHECKPOINT}" "${ANCHOR_TARGET_PATH}"
fi
echo "Prepared recovery anchor:"
echo "  source=${ANCHOR_CHECKPOINT}"
echo "  target=${ANCHOR_TARGET_PATH}"

if [[ "${SELECTED_CONFIG_LABEL,,}" == "auto" ]]; then
  SELECTED_CONFIG_LABEL="selected_21k_auto"
fi

parse_job_id() {
  local raw="$1"
  echo "${raw%%;*}"
}

A_SNAPSHOT="checkpoints/snapshots/stage_d/recovery/${PROBE_A_CONFIG}_${STAMP}.pt"
B_SNAPSHOT="checkpoints/snapshots/stage_d/recovery/${PROBE_B_CONFIG}_${STAMP}.pt"
SEL_SNAPSHOT="checkpoints/snapshots/stage_d/recovery/${SELECTED_CONFIG_LABEL}_${STAMP}.pt"

A_SCREEN_JSON="logs/stage_d/eval/eval_${PROBE_A_CONFIG}_screen_${STAMP}.json"
B_SCREEN_JSON="logs/stage_d/eval/eval_${PROBE_B_CONFIG}_screen_${STAMP}.json"
SEL_SCREEN_JSON="logs/stage_d/eval/eval_${SELECTED_CONFIG_LABEL}_screen_${STAMP}.json"
SEL_CERT_JSON="logs/stage_d/eval/eval_${SELECTED_CONFIG_LABEL}_cert_${STAMP}.json"
SELECTION_JSON="logs/stage_d/recovery/selection_${STAMP}.json"
SELECTION_ENV="logs/stage_d/recovery/selection_${STAMP}.env"

echo "Submitting Probe A train (${PROBE_A_CONFIG})..."
A_TRAIN_RAW="$(
  sbatch --parsable \
    --export=ALL,CONFIG_NAME="${PROBE_A_CONFIG}",CONFIG_FILE="${CONFIG_FILE}",STRICT_ABSTRACTION=true,WANDB_RUN_GROUP=stage_d_recovery_probe_a,WANDB_TAGS=stage_d,fchpa,recovery,probe_a,SNAPSHOT_COPY=true,SNAPSHOT_PATH="${A_SNAPSHOT}" \
    scripts/slurm_stage_d_fchpa_corrective_train.slurm
)"
A_TRAIN="$(parse_job_id "${A_TRAIN_RAW}")"

echo "Submitting Probe A screen eval..."
A_SCREEN_RAW="$(
  sbatch --parsable \
    --dependency=afterok:"${A_TRAIN}" \
    --export=ALL,CONFIG_NAME="${PROBE_A_CONFIG}",CONFIG_FILE="${CONFIG_FILE}",CHECKPOINT_PATH="${A_SNAPSHOT}",STRICT_ABSTRACTION=true,EPISODES_PER_SEED=5000,PROFILE=standard,BASE_SEED=42,HOLDOUT_SEED_BASE=97142,HOLDOUT_SEED_COUNT=5,VERIFY_SOLVER_TRAINING=true,OUTPUT_JSON="${A_SCREEN_JSON}" \
    scripts/slurm_stage_d_fchpa_corrective_screen_eval.slurm
)"
A_SCREEN="$(parse_job_id "${A_SCREEN_RAW}")"

echo "Submitting Probe A gate check..."
A_GATE_RAW="$(
  sbatch --parsable \
    --dependency=afterok:"${A_SCREEN}" \
    --export=ALL,EVAL_JSON="${A_SCREEN_JSON}",BASELINE_EVAL_JSON="${BASELINE_EVAL_JSON}",CI_FLOOR="${CI_FLOOR}",REQUIRE_BEHAVIOR_EXTENDED=true,REQUIRE_SOLVER_VERIFIED=true,REQUIRE_AGGRESSION_GATE=true,MAX_ALLIN_FREQ="${MAX_ALLIN_FREQ}",MIN_CALL_CHECK_FREQ="${MIN_CALL_CHECK_FREQ}",MIN_HALF_POT_FREQ="${MIN_HALF_POT_FREQ}",MIN_RAISE_TOTAL_FREQ="${MIN_RAISE_TOTAL_FREQ}",MIN_POT_CHOOSE_GIVEN_LEGAL_FREQ="${MIN_POT_CHOOSE_GIVEN_LEGAL_FREQ}",MIN_FOLD_FREQ="${MIN_FOLD_FREQ}",MAX_FOLD_FREQ="${MAX_FOLD_FREQ}",MIN_ENTROPY_BITS="${MIN_ENTROPY_BITS}",MAX_ENTROPY_BITS="${MAX_ENTROPY_BITS}" \
    scripts/slurm_stage_d_fchpa_eval_gate_check.slurm
)"
A_GATE="$(parse_job_id "${A_GATE_RAW}")"

echo "Submitting Probe B train (${PROBE_B_CONFIG})..."
B_TRAIN_RAW="$(
  sbatch --parsable \
    --dependency=afterany:"${A_TRAIN}" \
    --export=ALL,CONFIG_NAME="${PROBE_B_CONFIG}",CONFIG_FILE="${CONFIG_FILE}",STRICT_ABSTRACTION=true,WANDB_RUN_GROUP=stage_d_recovery_probe_b,WANDB_TAGS=stage_d,fchpa,recovery,probe_b,SNAPSHOT_COPY=true,SNAPSHOT_PATH="${B_SNAPSHOT}" \
    scripts/slurm_stage_d_fchpa_corrective_train.slurm
)"
B_TRAIN="$(parse_job_id "${B_TRAIN_RAW}")"

echo "Submitting Probe B screen eval..."
B_SCREEN_RAW="$(
  sbatch --parsable \
    --dependency=afterok:"${B_TRAIN}" \
    --export=ALL,CONFIG_NAME="${PROBE_B_CONFIG}",CONFIG_FILE="${CONFIG_FILE}",CHECKPOINT_PATH="${B_SNAPSHOT}",STRICT_ABSTRACTION=true,EPISODES_PER_SEED=5000,PROFILE=standard,BASE_SEED=42,HOLDOUT_SEED_BASE=98142,HOLDOUT_SEED_COUNT=5,VERIFY_SOLVER_TRAINING=true,OUTPUT_JSON="${B_SCREEN_JSON}" \
    scripts/slurm_stage_d_fchpa_corrective_screen_eval.slurm
)"
B_SCREEN="$(parse_job_id "${B_SCREEN_RAW}")"

echo "Submitting Probe B gate check..."
B_GATE_RAW="$(
  sbatch --parsable \
    --dependency=afterok:"${B_SCREEN}" \
    --export=ALL,EVAL_JSON="${B_SCREEN_JSON}",BASELINE_EVAL_JSON="${BASELINE_EVAL_JSON}",CI_FLOOR="${CI_FLOOR}",REQUIRE_BEHAVIOR_EXTENDED=true,REQUIRE_SOLVER_VERIFIED=true,REQUIRE_AGGRESSION_GATE=true,MAX_ALLIN_FREQ="${MAX_ALLIN_FREQ}",MIN_CALL_CHECK_FREQ="${MIN_CALL_CHECK_FREQ}",MIN_HALF_POT_FREQ="${MIN_HALF_POT_FREQ}",MIN_RAISE_TOTAL_FREQ="${MIN_RAISE_TOTAL_FREQ}",MIN_POT_CHOOSE_GIVEN_LEGAL_FREQ="${MIN_POT_CHOOSE_GIVEN_LEGAL_FREQ}",MIN_FOLD_FREQ="${MIN_FOLD_FREQ}",MAX_FOLD_FREQ="${MAX_FOLD_FREQ}",MIN_ENTROPY_BITS="${MIN_ENTROPY_BITS}",MAX_ENTROPY_BITS="${MAX_ENTROPY_BITS}" \
    scripts/slurm_stage_d_fchpa_eval_gate_check.slurm
)"
B_GATE="$(parse_job_id "${B_GATE_RAW}")"

echo "Submitting winner selection..."
SELECT_RAW="$(
  sbatch --parsable \
    --dependency=afterany:"${A_GATE}:${B_GATE}" \
    --export=ALL,CANDIDATE_A_NAME=probe_a,CANDIDATE_A_EVAL_JSON="${A_SCREEN_JSON}",CANDIDATE_B_NAME=probe_b,CANDIDATE_B_EVAL_JSON="${B_SCREEN_JSON}",OUTPUT_JSON="${SELECTION_JSON}",OUTPUT_ENV="${SELECTION_ENV}",MIN_CI_FLOOR="${CI_FLOOR}",MAX_ALLIN_FREQ="${MAX_ALLIN_FREQ}",MIN_CALL_CHECK_FREQ="${MIN_CALL_CHECK_FREQ}",MIN_HALF_POT_FREQ="${MIN_HALF_POT_FREQ}",MIN_RAISE_TOTAL_FREQ="${MIN_RAISE_TOTAL_FREQ}",MIN_POT_CHOOSE_GIVEN_LEGAL_FREQ="${MIN_POT_CHOOSE_GIVEN_LEGAL_FREQ}",MIN_FOLD_FREQ="${MIN_FOLD_FREQ}",MAX_FOLD_FREQ="${MAX_FOLD_FREQ}",MIN_ENTROPY_BITS="${MIN_ENTROPY_BITS}",MAX_ENTROPY_BITS="${MAX_ENTROPY_BITS}",EXPLOIT_WEIGHT="${EXPLOIT_WEIGHT}",EXPLOIT_CLIP="${EXPLOIT_CLIP}" \
    scripts/slurm_stage_d_fchpa_recovery_select.slurm
)"
SELECT_JOB="$(parse_job_id "${SELECT_RAW}")"

echo "Submitting winner checkpoint preparation..."
PREP_RAW="$(
  sbatch --parsable \
    --dependency=afterok:"${SELECT_JOB}" \
    --export=ALL,WINNER_ENV_PATH="${SELECTION_ENV}",TARGET_PATH=checkpoints/snapshots/stage_d/recovery/winner_20k.pt \
    scripts/slurm_stage_d_fchpa_recovery_prepare_winner.slurm
)"
PREP_JOB="$(parse_job_id "${PREP_RAW}")"

echo "Submitting selected promotion train (${SELECTED_CONFIG})..."
SEL_TRAIN_RAW="$(
  sbatch --parsable \
    --dependency=afterok:"${PREP_JOB}" \
    --export=ALL,CONFIG_NAME="${SELECTED_CONFIG}",CONFIG_FILE="${CONFIG_FILE}",STRICT_ABSTRACTION=true,WINNER_ENV_PATH="${SELECTION_ENV}",WINNER_JSON_PATH="${SELECTION_JSON}",SELECTED_CONFIG_CONSERVATIVE="${SELECTED_CONFIG_CONSERVATIVE}",SELECTED_CONFIG_BALANCED="${SELECTED_CONFIG_BALANCED}",WANDB_RUN_GROUP=stage_d_recovery_selected_21k,WANDB_TAGS=stage_d,fchpa,recovery,selected_21k,SNAPSHOT_COPY=true,SNAPSHOT_PATH="${SEL_SNAPSHOT}" \
    scripts/slurm_stage_d_fchpa_corrective_train.slurm
)"
SEL_TRAIN="$(parse_job_id "${SEL_TRAIN_RAW}")"

echo "Submitting selected promotion screen eval..."
SEL_SCREEN_RAW="$(
  sbatch --parsable \
    --dependency=afterok:"${SEL_TRAIN}" \
    --export=ALL,CONFIG_NAME="${SELECTED_CONFIG}",CONFIG_FILE="${CONFIG_FILE}",CHECKPOINT_PATH="${SEL_SNAPSHOT}",STRICT_ABSTRACTION=true,WINNER_ENV_PATH="${SELECTION_ENV}",WINNER_JSON_PATH="${SELECTION_JSON}",SELECTED_CONFIG_CONSERVATIVE="${SELECTED_CONFIG_CONSERVATIVE}",SELECTED_CONFIG_BALANCED="${SELECTED_CONFIG_BALANCED}",EPISODES_PER_SEED=5000,PROFILE=standard,BASE_SEED=42,HOLDOUT_SEED_BASE=99142,HOLDOUT_SEED_COUNT=5,VERIFY_SOLVER_TRAINING=true,REQUIRE_BEHAVIOR_EXTENDED=true,OUTPUT_JSON="${SEL_SCREEN_JSON}" \
    scripts/slurm_stage_d_fchpa_corrective_screen_eval.slurm
)"
SEL_SCREEN="$(parse_job_id "${SEL_SCREEN_RAW}")"

echo "Submitting selected promotion gate check..."
SEL_GATE_RAW="$(
  sbatch --parsable \
    --dependency=afterok:"${SEL_SCREEN}" \
    --export=ALL,EVAL_JSON="${SEL_SCREEN_JSON}",BASELINE_EVAL_JSON="${BASELINE_EVAL_JSON}",CI_FLOOR="${CI_FLOOR}",REQUIRE_BEHAVIOR_EXTENDED=true,REQUIRE_SOLVER_VERIFIED=true,REQUIRE_AGGRESSION_GATE=true,MAX_ALLIN_FREQ="${MAX_ALLIN_FREQ}",MIN_CALL_CHECK_FREQ="${MIN_CALL_CHECK_FREQ}",MIN_HALF_POT_FREQ="${MIN_HALF_POT_FREQ}",MIN_RAISE_TOTAL_FREQ="${MIN_RAISE_TOTAL_FREQ}",MIN_POT_CHOOSE_GIVEN_LEGAL_FREQ="${MIN_POT_CHOOSE_GIVEN_LEGAL_FREQ}",MIN_FOLD_FREQ="${MIN_FOLD_FREQ}",MAX_FOLD_FREQ="${MAX_FOLD_FREQ}",MIN_ENTROPY_BITS="${MIN_ENTROPY_BITS}",MAX_ENTROPY_BITS="${MAX_ENTROPY_BITS}" \
    scripts/slurm_stage_d_fchpa_eval_gate_check.slurm
)"
SEL_GATE="$(parse_job_id "${SEL_GATE_RAW}")"

echo "Submitting selected promotion cert eval..."
SEL_CERT_RAW="$(
  sbatch --parsable \
    --dependency=afterok:"${SEL_GATE}" \
    --export=ALL,CONFIG_NAME="${SELECTED_CONFIG}",CONFIG_FILE="${CONFIG_FILE}",CHECKPOINT_PATH="${SEL_SNAPSHOT}",STRICT_ABSTRACTION=true,WINNER_ENV_PATH="${SELECTION_ENV}",WINNER_JSON_PATH="${SELECTION_JSON}",SELECTED_CONFIG_CONSERVATIVE="${SELECTED_CONFIG_CONSERVATIVE}",SELECTED_CONFIG_BALANCED="${SELECTED_CONFIG_BALANCED}",EPISODES_PER_SEED=10000,PROFILE=standard,BASE_SEED=42,HOLDOUT_SEED_BASE=99242,HOLDOUT_SEED_COUNT=5,REQUIRE_ROBUST_CI=true,VERIFY_SOLVER_TRAINING=true,REQUIRE_BEHAVIOR_EXTENDED=true,OUTPUT_JSON="${SEL_CERT_JSON}" \
    scripts/slurm_stage_d_fchpa_corrective_cert_eval.slurm
)"
SEL_CERT="$(parse_job_id "${SEL_CERT_RAW}")"

cat <<EOF
Submission complete.

Anchor checkpoint:
  source=${ANCHOR_CHECKPOINT}
  target=${ANCHOR_TARGET_PATH}

Probe A:
  train=${A_TRAIN}
  screen=${A_SCREEN}
  gate=${A_GATE}
  snapshot=${A_SNAPSHOT}
  eval=${A_SCREEN_JSON}

Probe B:
  train=${B_TRAIN}
  screen=${B_SCREEN}
  gate=${B_GATE}
  snapshot=${B_SNAPSHOT}
  eval=${B_SCREEN_JSON}

Selection:
  select_job=${SELECT_JOB}
  prep_job=${PREP_JOB}
  selection_json=${SELECTION_JSON}
  selection_env=${SELECTION_ENV}

Promotion (selected):
  train=${SEL_TRAIN}
  screen=${SEL_SCREEN}
  gate=${SEL_GATE}
  cert=${SEL_CERT}
  snapshot=${SEL_SNAPSHOT}
  screen_eval=${SEL_SCREEN_JSON}
  cert_eval=${SEL_CERT_JSON}

Status:
  squeue -u "\$USER"
EOF
