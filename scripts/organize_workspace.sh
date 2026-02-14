#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="dry-run"

if [[ "${1:-}" == "--apply" ]]; then
  MODE="apply"
fi

log() {
  echo "[$MODE] $*"
}

relpath() {
  python3 - "$ROOT_DIR" "$1" <<'PY'
import os
import sys
root = os.path.abspath(sys.argv[1])
path = os.path.abspath(sys.argv[2])
print(os.path.relpath(path, root))
PY
}

move_file() {
  local src="$1"
  local dst_dir="$2"
  local dst
  local base
  local stem
  local ext
  local ts

  [[ -f "$src" ]] || return 0
  mkdir -p "$dst_dir"
  dst="$dst_dir/$(basename "$src")"

  if [[ "$src" == "$dst" ]]; then
    return 0
  fi

  if [[ -f "$dst" ]]; then
    if cmp -s "$src" "$dst"; then
      if [[ "$MODE" == "apply" ]]; then
        rm -f "$src"
        log "removed duplicate $(relpath "$src") (same as $(relpath "$dst"))"
      else
        log "would remove duplicate $(relpath "$src") (same as $(relpath "$dst"))"
      fi
      return 0
    fi
    base="$(basename "$dst")"
    stem="${base%.*}"
    ext="${base##*.}"
    if [[ "$stem" == "$base" ]]; then
      ext=""
    else
      ext=".$ext"
    fi
    ts="$(date +%Y%m%d_%H%M%S)"
    dst="$dst_dir/${stem}_dup_${ts}${ext}"
  fi

  if [[ "$MODE" == "apply" ]]; then
    mv -f "$src" "$dst"
    log "moved $(relpath "$src") -> $(relpath "$dst")"
  else
    log "would move $(relpath "$src") -> $(relpath "$dst")"
  fi
}

move_glob() {
  local pattern="$1"
  local dst_dir="$2"
  local f

  shopt -s nullglob
  for f in $pattern; do
    move_file "$f" "$dst_dir"
  done
  shopt -u nullglob
}

cleanup_empty_dirs() {
  local target="$1"
  find "$target" -type d -empty -mindepth 1 -delete 2>/dev/null || true
}

mkdir -p \
  "$ROOT_DIR/logs/stage_d/slurm" \
  "$ROOT_DIR/logs/stage_d/eval" \
  "$ROOT_DIR/logs/stage_d/metrics" \
  "$ROOT_DIR/logs/stage_c/slurm" \
  "$ROOT_DIR/logs/stage_c/eval" \
  "$ROOT_DIR/logs/stage_c/metrics" \
  "$ROOT_DIR/logs/misc" \
  "$ROOT_DIR/checkpoints/snapshots/stage_d"

move_glob "$ROOT_DIR/logs/ah_stage_d_fchpa_*.out" "$ROOT_DIR/logs/stage_d/slurm"
move_glob "$ROOT_DIR/logs/ah_stage_d_fchpa_*.err" "$ROOT_DIR/logs/stage_d/slurm"
move_glob "$ROOT_DIR/logs/ah_stage_d_fchpa_ablate_*.out" "$ROOT_DIR/logs/stage_d/slurm"
move_glob "$ROOT_DIR/logs/ah_stage_d_fchpa_ablate_*.err" "$ROOT_DIR/logs/stage_d/slurm"
move_glob "$ROOT_DIR/logs/ah_stage_d_fchpa_sel16k_*.out" "$ROOT_DIR/logs/stage_d/slurm"
move_glob "$ROOT_DIR/logs/ah_stage_d_fchpa_sel16k_*.err" "$ROOT_DIR/logs/stage_d/slurm"
move_glob "$ROOT_DIR/logs/ah_stage_d_fcpha_*.out" "$ROOT_DIR/logs/stage_d/slurm"
move_glob "$ROOT_DIR/logs/ah_stage_d_fcpha_*.err" "$ROOT_DIR/logs/stage_d/slurm"
move_glob "$ROOT_DIR/logs/eval_stage_d_*.json" "$ROOT_DIR/logs/stage_d/eval"
move_glob "$ROOT_DIR/logs/eval_ablate_*.json" "$ROOT_DIR/logs/stage_d/eval"
move_glob "$ROOT_DIR/logs/eval_quadro_stage_d_*.json" "$ROOT_DIR/logs/stage_d/eval"
move_glob "$ROOT_DIR/logs/metrics_*.jsonl" "$ROOT_DIR/logs/stage_d/metrics"

move_glob "$ROOT_DIR/logs/ah_stage_c_*.out" "$ROOT_DIR/logs/stage_c/slurm"
move_glob "$ROOT_DIR/logs/ah_stage_c_*.err" "$ROOT_DIR/logs/stage_c/slurm"
move_glob "$ROOT_DIR/logs/eval_stage_c_*.json" "$ROOT_DIR/logs/stage_c/eval"
move_glob "$ROOT_DIR/logs/metrics_20260212_*.jsonl" "$ROOT_DIR/logs/stage_c/metrics"

move_glob "$ROOT_DIR/checkpoints/quadro_stage_d_fchpa_*_latest.pt" "$ROOT_DIR/checkpoints/snapshots/stage_d"
move_glob "$ROOT_DIR/checkpoints/ablate_*_latest.pt" "$ROOT_DIR/checkpoints/snapshots/stage_d"

# Final sweep of root-level leftovers into misc to keep logs/ uncluttered.
move_glob "$ROOT_DIR/logs/*.out" "$ROOT_DIR/logs/misc"
move_glob "$ROOT_DIR/logs/*.err" "$ROOT_DIR/logs/misc"
move_glob "$ROOT_DIR/logs/*.json" "$ROOT_DIR/logs/misc"
move_glob "$ROOT_DIR/logs/*.jsonl" "$ROOT_DIR/logs/misc"

cleanup_empty_dirs "$ROOT_DIR/logs"

if [[ "$MODE" == "dry-run" ]]; then
  log "preview complete. Re-run with '--apply' to execute moves."
else
  log "organization complete."
fi
