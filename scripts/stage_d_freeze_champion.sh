#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SOURCE="${1:-checkpoints/latest.pt}"
LABEL="${2:-stage_d_16k_champion}"
ALIAS="${3:-stage_d_16k_champion_latest.pt}"

if [[ ! -f "${SOURCE}" ]]; then
  echo "Source checkpoint not found: ${SOURCE}"
  exit 2
fi

python3 poker_rl_agent/scripts/checkpoint_manager.py snapshot \
  --root checkpoints \
  --source "${SOURCE}" \
  --stage "stage_d/champion" \
  --label "${LABEL}" \
  --active-alias "${ALIAS}" \
  --apply

echo "Champion snapshot frozen from ${SOURCE}."
