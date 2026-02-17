#!/usr/bin/env bash
# Runs a long FCHPA human-vs-bot CLI session and stores transcript for interview artifacts.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-checkpoints/latest.pt}"
CONFIG_FILE="${CONFIG_FILE:-configs/training_configs.yaml}"
CONFIG_NAME="${CONFIG_NAME:-quadro_stage_d_fchpa_recover_selected_20k_conservative}"
HANDS="${HANDS:-200}"
HUMAN_SEAT="${HUMAN_SEAT:-random}"
BOT_POLICY="${BOT_POLICY:-sample}"
POLICY_TEMPERATURE="${POLICY_TEMPERATURE:-1.0}"
SEED="${SEED:-42}"

STAMP="$(date +%Y%m%d_%H%M%S)"
TRANSCRIPT="logs/stage_d/human/human_validation_${STAMP}.log"

mkdir -p logs/stage_d/human

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}"
  exit 2
fi

echo "Writing transcript: ${TRANSCRIPT}"
echo "Play 200 hands manually; action summary is printed at session end."

python3 poker_rl_agent/scripts/play_against_agent.py \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config_file "${CONFIG_FILE}" \
  --config_name "${CONFIG_NAME}" \
  --game_mode fchpa \
  --hands "${HANDS}" \
  --human_seat "${HUMAN_SEAT}" \
  --bot_policy "${BOT_POLICY}" \
  --policy_temperature "${POLICY_TEMPERATURE}" \
  --seed "${SEED}" | tee "${TRANSCRIPT}"
