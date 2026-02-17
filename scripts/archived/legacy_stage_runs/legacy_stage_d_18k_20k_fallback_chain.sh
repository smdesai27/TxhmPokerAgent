#!/usr/bin/env bash
# Submit Stage D jobs on Oscar:
# 1) Evaluate current 18k checkpoint (screen + gate check + cert if gate passes)
# 2) If 18k gate fails, automatically launch 20k corrective fallback
#    (train -> screen -> gate check -> cert)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p logs/stage_d/eval logs/stage_d/slurm

STAMP="$(date +%Y%m%d_%H%M%S)"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/latest.pt}"
CONFIG_FILE="${CONFIG_FILE:-configs/training_configs.yaml}"
CONFIG_18K="${CONFIG_18K:-quadro_stage_d_fchpa_recover_value_stable_corrective_18k}"
CONFIG_20K="${CONFIG_20K:-quadro_stage_d_fchpa_recover_diverse_corrective_20k_flatexplore}"
BASELINE_EVAL_JSON="${BASELINE_EVAL_JSON:-logs/stage_d/eval/eval_quadro_stage_d_fchpa_recover_explore_16k_cert_375207.json}"
CI_FLOOR="${CI_FLOOR:-830.0}"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}"
  exit 2
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config file not found: ${CONFIG_FILE}"
  exit 2
fi

EVAL_18K_SCREEN_JSON="logs/stage_d/eval/eval_${CONFIG_18K}_screen_${STAMP}.json"
EVAL_18K_CERT_JSON="logs/stage_d/eval/eval_${CONFIG_18K}_cert_${STAMP}.json"
EVAL_20K_SCREEN_JSON="logs/stage_d/eval/eval_${CONFIG_20K}_screen_${STAMP}.json"
EVAL_20K_CERT_JSON="logs/stage_d/eval/eval_${CONFIG_20K}_cert_${STAMP}.json"

parse_job_id() {
  local raw="$1"
  echo "${raw%%;*}"
}

echo "Submitting 18k SCREEN eval (${CONFIG_18K})..."
J18_SCREEN_RAW="$(
  sbatch --parsable \
    --export=ALL,CONFIG_NAME="${CONFIG_18K}",CONFIG_FILE="${CONFIG_FILE}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",STRICT_ABSTRACTION=true,EPISODES_PER_SEED=5000,PROFILE=standard,BASE_SEED=42,HOLDOUT_SEED_BASE=93042,HOLDOUT_SEED_COUNT=5,VERIFY_SOLVER_TRAINING=true,OUTPUT_JSON="${EVAL_18K_SCREEN_JSON}" \
    scripts/slurm_stage_d_fchpa_corrective_screen_eval.slurm
)"
J18_SCREEN="$(parse_job_id "${J18_SCREEN_RAW}")"

echo "Submitting 18k gate check..."
J18_GATE_RAW="$(
  sbatch --parsable \
    --dependency=afterok:"${J18_SCREEN}" \
    --export=ALL,EVAL_JSON="${EVAL_18K_SCREEN_JSON}",BASELINE_EVAL_JSON="${BASELINE_EVAL_JSON}",CI_FLOOR="${CI_FLOOR}",REQUIRE_BEHAVIOR_EXTENDED=true,REQUIRE_SOLVER_VERIFIED=true \
    scripts/slurm_stage_d_fchpa_eval_gate_check.slurm
)"
J18_GATE="$(parse_job_id "${J18_GATE_RAW}")"

echo "Submitting 18k CERT eval (runs only if 18k gate passes)..."
J18_CERT_RAW="$(
  sbatch --parsable \
    --dependency=afterok:"${J18_GATE}" \
    --export=ALL,CONFIG_NAME="${CONFIG_18K}",CONFIG_FILE="${CONFIG_FILE}",CHECKPOINT_PATH="${CHECKPOINT_PATH}",STRICT_ABSTRACTION=true,EPISODES_PER_SEED=10000,PROFILE=standard,BASE_SEED=42,HOLDOUT_SEED_BASE=94042,HOLDOUT_SEED_COUNT=5,REQUIRE_ROBUST_CI=true,VERIFY_SOLVER_TRAINING=true,OUTPUT_JSON="${EVAL_18K_CERT_JSON}" \
    scripts/slurm_stage_d_fchpa_corrective_cert_eval.slurm
)"
J18_CERT="$(parse_job_id "${J18_CERT_RAW}")"

echo "Submitting 20k fallback TRAIN (runs only if 18k gate fails)..."
J20_TRAIN_RAW="$(
  sbatch --parsable \
    --dependency=afternotok:"${J18_GATE}" \
    --export=ALL,CONFIG_NAME="${CONFIG_20K}",CONFIG_FILE="${CONFIG_FILE}",STRICT_ABSTRACTION=true,WANDB_RUN_GROUP=stage_d_fchpa_recover_diverse_20k_fallback,WANDB_TAGS=stage_d,fchpa,20k,fallback,corrective,SNAPSHOT_COPY=true \
    scripts/slurm_stage_d_fchpa_corrective_train.slurm
)"
J20_TRAIN="$(parse_job_id "${J20_TRAIN_RAW}")"

echo "Submitting 20k SCREEN eval (after fallback train)..."
J20_SCREEN_RAW="$(
  sbatch --parsable \
    --dependency=afterok:"${J20_TRAIN}" \
    --export=ALL,CONFIG_NAME="${CONFIG_20K}",CONFIG_FILE="${CONFIG_FILE}",CHECKPOINT_PATH=checkpoints/latest.pt,STRICT_ABSTRACTION=true,EPISODES_PER_SEED=5000,PROFILE=standard,BASE_SEED=42,HOLDOUT_SEED_BASE=95042,HOLDOUT_SEED_COUNT=5,VERIFY_SOLVER_TRAINING=true,OUTPUT_JSON="${EVAL_20K_SCREEN_JSON}" \
    scripts/slurm_stage_d_fchpa_corrective_screen_eval.slurm
)"
J20_SCREEN="$(parse_job_id "${J20_SCREEN_RAW}")"

echo "Submitting 20k gate check..."
J20_GATE_RAW="$(
  sbatch --parsable \
    --dependency=afterok:"${J20_SCREEN}" \
    --export=ALL,EVAL_JSON="${EVAL_20K_SCREEN_JSON}",BASELINE_EVAL_JSON="${BASELINE_EVAL_JSON}",CI_FLOOR="${CI_FLOOR}",REQUIRE_BEHAVIOR_EXTENDED=true,REQUIRE_SOLVER_VERIFIED=true \
    scripts/slurm_stage_d_fchpa_eval_gate_check.slurm
)"
J20_GATE="$(parse_job_id "${J20_GATE_RAW}")"

echo "Submitting 20k CERT eval (runs only if 20k gate passes)..."
J20_CERT_RAW="$(
  sbatch --parsable \
    --dependency=afterok:"${J20_GATE}" \
    --export=ALL,CONFIG_NAME="${CONFIG_20K}",CONFIG_FILE="${CONFIG_FILE}",CHECKPOINT_PATH=checkpoints/latest.pt,STRICT_ABSTRACTION=true,EPISODES_PER_SEED=10000,PROFILE=standard,BASE_SEED=42,HOLDOUT_SEED_BASE=96042,HOLDOUT_SEED_COUNT=5,REQUIRE_ROBUST_CI=true,VERIFY_SOLVER_TRAINING=true,OUTPUT_JSON="${EVAL_20K_CERT_JSON}" \
    scripts/slurm_stage_d_fchpa_corrective_cert_eval.slurm
)"
J20_CERT="$(parse_job_id "${J20_CERT_RAW}")"

cat <<EOF
Submission complete.

18k path:
  SCREEN job: ${J18_SCREEN}
  GATE   job: ${J18_GATE}
  CERT   job: ${J18_CERT} (afterok gate)
  Screen JSON: ${EVAL_18K_SCREEN_JSON}
  Cert JSON:   ${EVAL_18K_CERT_JSON}

20k fallback path (only if 18k gate fails):
  TRAIN  job: ${J20_TRAIN} (afternotok 18k gate)
  SCREEN job: ${J20_SCREEN}
  GATE   job: ${J20_GATE}
  CERT   job: ${J20_CERT} (afterok 20k gate)
  Screen JSON: ${EVAL_20K_SCREEN_JSON}
  Cert JSON:   ${EVAL_20K_CERT_JSON}

Quick status:
  squeue -u "\$USER"
EOF
