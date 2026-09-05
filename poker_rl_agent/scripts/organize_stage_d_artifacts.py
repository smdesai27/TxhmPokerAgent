#!/usr/bin/env python3
"""Organize Stage D champion artifacts without mutating source logs/checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[2]

RUN_LABEL_DEFAULT = "20260216_161321_stage_d_evfirst_cycle"
RUN_DIR_DEFAULT = Path("artifacts/runs") / RUN_LABEL_DEFAULT

LEGACY_SCRIPT_MOVES: List[Tuple[Path, Path]] = [
    (
        Path("scripts/slurm_stage_d_fcpha_train.slurm"),
        Path("scripts/archived/legacy_stage_runs/legacy_stage_d_fcpha_train_unsupported_alias.slurm"),
    ),
    (
        Path("scripts/slurm_stage_d_fcpha_eval.slurm"),
        Path("scripts/archived/legacy_stage_runs/legacy_stage_d_fcpha_eval_unsupported_alias.slurm"),
    ),
    (
        Path("scripts/slurm_stage_d_fchpa_train.slurm"),
        Path("scripts/archived/legacy_stage_runs/legacy_stage_d_fchpa_early_train.slurm"),
    ),
    (
        Path("scripts/slurm_stage_d_fchpa_eval.slurm"),
        Path("scripts/archived/legacy_stage_runs/legacy_stage_d_fchpa_early_eval.slurm"),
    ),
    (
        Path("scripts/slurm_stage_d_fchpa_ablation_train.slurm"),
        Path("scripts/archived/legacy_stage_runs/legacy_stage_d_fchpa_ablation_train.slurm"),
    ),
    (
        Path("scripts/slurm_stage_d_fchpa_ablation_eval.slurm"),
        Path("scripts/archived/legacy_stage_runs/legacy_stage_d_fchpa_ablation_eval.slurm"),
    ),
    (
        Path("scripts/slurm_stage_d_fchpa_selected_16k_train.slurm"),
        Path("scripts/archived/legacy_stage_runs/legacy_stage_d_selected16k_train.slurm"),
    ),
    (
        Path("scripts/slurm_stage_d_fchpa_selected_16k_cert_eval.slurm"),
        Path("scripts/archived/legacy_stage_runs/legacy_stage_d_selected16k_cert_eval.slurm"),
    ),
    (
        Path("scripts/slurm_stage_d_fchpa_cert_eval.slurm"),
        Path("scripts/archived/legacy_stage_runs/legacy_stage_d_old_cert_eval.slurm"),
    ),
    (
        Path("scripts/slurm_stage_d_build_style_target.slurm"),
        Path("scripts/archived/legacy_stage_runs/legacy_stage_d_style_target_builder.slurm"),
    ),
    (
        Path("scripts/submit_stage_d_18k_gate_with_20k_fallback.sh"),
        Path("scripts/archived/legacy_stage_runs/legacy_stage_d_18k_20k_fallback_chain.sh"),
    ),
]

ACTIVE_STAGE_D_SCRIPTS = [
    "scripts/slurm_stage_d_fchpa_corrective_train.slurm",
    "scripts/slurm_stage_d_fchpa_corrective_screen_eval.slurm",
    "scripts/slurm_stage_d_fchpa_corrective_cert_eval.slurm",
    "scripts/slurm_stage_d_fchpa_eval_gate_check.slurm",
    "scripts/slurm_stage_d_fchpa_recovery_select.slurm",
    "scripts/slurm_stage_d_fchpa_recovery_prepare_winner.slurm",
    "scripts/submit_stage_d_fchpa_recovery_cycle.sh",
    "scripts/submit_stage_d_fchpa_champion_continue.sh",
]

ALIASES: List[Tuple[str, str]] = [
    ("slurm/probe_h1_train_job391050.out", "logs/stage_d/slurm/ah_stage_d_fchpa_corrective_train_391050.out"),
    ("slurm/probe_h1_screen_job391051.out", "logs/stage_d/slurm/ah_stage_d_fchpa_corrective_screen_391051.out"),
    ("slurm/probe_h1_gate_job391052.out", "logs/stage_d/slurm/ah_stage_d_fchpa_gate_check_391052.out"),
    ("slurm/probe_h2_train_job391053.out", "logs/stage_d/slurm/ah_stage_d_fchpa_corrective_train_391053.out"),
    ("slurm/probe_h2_screen_job391054.out", "logs/stage_d/slurm/ah_stage_d_fchpa_corrective_screen_391054.out"),
    ("slurm/probe_h2_gate_job391055.out", "logs/stage_d/slurm/ah_stage_d_fchpa_gate_check_391055.out"),
    ("slurm/select_job391056.out", "logs/stage_d/slurm/ah_stage_d_fchpa_select_391056.out"),
    ("slurm/prepare_winner_job391057.out", "logs/stage_d/slurm/ah_stage_d_fchpa_prep_winner_391057.out"),
    ("slurm/selected_train_job391058.out", "logs/stage_d/slurm/ah_stage_d_fchpa_corrective_train_391058.out"),
    ("slurm/selected_screen_job391059.out", "logs/stage_d/slurm/ah_stage_d_fchpa_corrective_screen_391059.out"),
    ("slurm/selected_gate_job391060.out", "logs/stage_d/slurm/ah_stage_d_fchpa_gate_check_391060.out"),
    ("slurm/selected_cert_job391061.out", "logs/stage_d/slurm/ah_stage_d_fchpa_corrective_cert_391061.out"),
    ("metrics/probe_h1_train_metrics.jsonl", "logs/metrics_20260216_211335.jsonl"),
    ("metrics/probe_h2_train_metrics.jsonl", "logs/metrics_20260216_225648.jsonl"),
    ("metrics/selected_21k_train_metrics.jsonl", "logs/metrics_20260217_003504.jsonl"),
    ("eval/probe_h1_screen.json", "logs/stage_d/eval/eval_quadro_stage_d_fchpa_evfirst_probe_h1_20k_screen_20260216_161321.json"),
    ("eval/probe_h2_screen.json", "logs/stage_d/eval/eval_quadro_stage_d_fchpa_evfirst_probe_h2_20k_screen_20260216_161321.json"),
    ("eval/selected_21k_screen.json", "logs/stage_d/eval/eval_selected_21k_auto_screen_20260216_161321.json"),
    ("eval/selected_21k_cert.json", "logs/stage_d/eval/eval_selected_21k_auto_cert_20260216_161321.json"),
    ("recovery/selection.json", "logs/stage_d/recovery/selection_20260216_161321.json"),
    ("recovery/selection.env", "logs/stage_d/recovery/selection_20260216_161321.env"),
    ("checkpoints/selected_21k_auto.pt", "checkpoints/snapshots/stage_d/recovery/selected_21k_auto_20260216_161321.pt"),
]

CHAMPION_SOURCE_CKPT = Path("checkpoints/snapshots/stage_d/recovery/selected_21k_auto_20260216_161321.pt")
CHAMPION_DEST_CKPT = Path("checkpoints/snapshots/champion/stage_d_fchpa_21k.pt")
CHAMPION_SIDECAR = Path("checkpoints/snapshots/champion/stage_d_fchpa_21k.json")
CHECKPOINT_ALIAS_REGISTRY = Path("checkpoints/checkpoint_aliases.json")
BEST_EVAL_PATH = Path("logs/stage_d/eval/eval_selected_21k_auto_cert_20260216_161321.json")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _safe_json_load(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _link_target(alias_path: Path, source_path: Path) -> str:
    return os.path.relpath(str(source_path), start=str(alias_path.parent))


def _render_op(kind: str, src: Path | None, dst: Path) -> str:
    if src is None:
        return f"{kind}: {dst}"
    return f"{kind}: {src} -> {dst}"


def _write_json(path: Path, payload: Dict) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _collect_eval_summary(eval_path: Path) -> Dict[str, float | bool]:
    payload = _safe_json_load(eval_path, {})
    def _tier(iterations: int) -> Dict:
        for tier in payload.get("solver_tiers", []):
            if int(tier.get("iterations", -1)) == iterations:
                return tier
        return {}

    default = _tier(5000)
    robust = _tier(10000)
    holdout = payload.get("holdout", {})
    return {
        "pass_all": bool(payload.get("overall/pass_all", False)),
        "pass_default_cfr_gate": bool(payload.get("overall/pass_default_cfr_gate", False)),
        "pass_robustness_gate": bool(payload.get("overall/pass_robustness_gate", False)),
        "pass_holdout_gate": bool(payload.get("overall/pass_holdout_gate", False)),
        "pass_behavior_gate": bool(payload.get("overall/pass_behavior_gate", False)),
        "pass_behavior_gate_extended": bool(payload.get("overall/pass_behavior_gate_extended", False)),
        "pass_aggression_gate": bool(payload.get("overall/pass_aggression_gate", False)),
        "default_mean_bb100": float(default.get("aggregate", {}).get("bb_per_100", {}).get("mean", 0.0)),
        "default_ci95_lower_bb100": float(default.get("ci95_lower_bb100", 0.0)),
        "robust_mean_bb100": float(robust.get("aggregate", {}).get("bb_per_100", {}).get("mean", 0.0)),
        "robust_ci95_lower_bb100": float(robust.get("ci95_lower_bb100", 0.0)),
        "holdout_mean_bb100": float(holdout.get("aggregate", {}).get("bb_per_100", {}).get("mean", 0.0)),
        "holdout_ci95_lower_bb100": float(holdout.get("ci95_lower_bb100", 0.0)),
    }


def _create_or_update_symlink(alias_path: Path, source_path: Path) -> str:
    if not source_path.exists():
        return "missing_source"
    _ensure_parent(alias_path)
    rel_target = _link_target(alias_path, source_path)
    if alias_path.is_symlink():
        if os.readlink(alias_path) == rel_target:
            return "exists_ok"
        alias_path.unlink()
    elif alias_path.exists():
        return "exists_non_symlink"
    alias_path.symlink_to(rel_target)
    return "created"


def _iter_needed_dirs(run_dir: Path) -> Iterable[Path]:
    return [
        run_dir,
        run_dir / "slurm",
        run_dir / "eval",
        run_dir / "metrics",
        run_dir / "recovery",
        run_dir / "checkpoints",
        ROOT / "scripts" / "archived" / "legacy_stage_runs",
        ROOT / "checkpoints" / "snapshots" / "champion",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize Stage D artifacts and freeze champion checkpoint.")
    parser.add_argument("--run-label", default=RUN_LABEL_DEFAULT)
    parser.add_argument("--apply", action="store_true", help="Apply file moves/links/writes.")
    args = parser.parse_args()

    run_dir = ROOT / "artifacts" / "runs" / args.run_label
    manifest_path = run_dir / "manifest.json"
    runs_index_path = ROOT / "artifacts" / "runs" / "index.json"

    ops: List[str] = []

    for directory in _iter_needed_dirs(run_dir):
        if not directory.exists():
            ops.append(_render_op("mkdir", None, directory))

    for src_rel, dst_rel in LEGACY_SCRIPT_MOVES:
        src = ROOT / src_rel
        dst = ROOT / dst_rel
        if src.exists():
            ops.append(_render_op("move", src, dst))

    for alias_rel, source_rel in ALIASES:
        alias = run_dir / alias_rel
        source = ROOT / source_rel
        if not alias.exists():
            ops.append(_render_op("symlink", source, alias))

    if CHAMPION_SOURCE_CKPT.exists():
        ops.append(_render_op("copy", ROOT / CHAMPION_SOURCE_CKPT, ROOT / CHAMPION_DEST_CKPT))
        ops.append(_render_op("write", None, ROOT / CHAMPION_SIDECAR))
        ops.append(_render_op("write", None, ROOT / CHECKPOINT_ALIAS_REGISTRY))
    ops.append(_render_op("write", None, manifest_path))
    ops.append(_render_op("write", None, runs_index_path))

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] Stage D organization plan ({len(ops)} operations)")
    for op in ops:
        print(f" - {op}")

    if not args.apply:
        return

    for directory in _iter_needed_dirs(run_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for src_rel, dst_rel in LEGACY_SCRIPT_MOVES:
        src = ROOT / src_rel
        dst = ROOT / dst_rel
        if not src.exists():
            continue
        _ensure_parent(dst)
        if dst.exists():
            # Keep the first archived copy; if source still exists, remove source.
            src.unlink()
            continue
        shutil.move(str(src), str(dst))

    alias_rows: Dict[str, str] = {}
    source_rows: Dict[str, str] = {}
    for alias_rel, source_rel in ALIASES:
        alias_path = run_dir / alias_rel
        source_path = ROOT / source_rel
        status = _create_or_update_symlink(alias_path, source_path)
        alias_rows[alias_rel] = str(alias_path.relative_to(ROOT))
        source_rows[alias_rel] = source_rel
        if status == "missing_source":
            print(f"warning: missing source for alias '{alias_rel}': {source_path}")
        elif status == "exists_non_symlink":
            print(f"warning: alias path exists as regular file, skipped: {alias_path}")

    if not CHAMPION_SOURCE_CKPT.exists():
        raise FileNotFoundError(f"missing champion source checkpoint: {ROOT / CHAMPION_SOURCE_CKPT}")
    _ensure_parent(ROOT / CHAMPION_DEST_CKPT)
    shutil.copy2(ROOT / CHAMPION_SOURCE_CKPT, ROOT / CHAMPION_DEST_CKPT)
    ckpt_sha = _sha256(ROOT / CHAMPION_DEST_CKPT)

    cert_summary = _collect_eval_summary(ROOT / BEST_EVAL_PATH)
    sidecar = {
        "alias": "stage_d_fchpa_21k",
        "source_checkpoint": str(CHAMPION_SOURCE_CKPT),
        "source_eval_cert": str(BEST_EVAL_PATH),
        "sha256": ckpt_sha,
        "created_at_utc": _utc_now(),
        "config_name": "quadro_stage_d_fchpa_evfirst_selected_21k_h1",
        "step": 21000,
        "gate_summary": cert_summary,
    }
    _write_json(ROOT / CHAMPION_SIDECAR, sidecar)

    alias_registry = _safe_json_load(ROOT / CHECKPOINT_ALIAS_REGISTRY, {})
    if not isinstance(alias_registry, dict):
        alias_registry = {}
    alias_registry["stage_d_fchpa_21k"] = {
        "path": str(CHAMPION_DEST_CKPT),
        "source_checkpoint": str(CHAMPION_SOURCE_CKPT),
        "source_eval_cert": str(BEST_EVAL_PATH),
        "sha256": ckpt_sha,
        "created_at_utc": sidecar["created_at_utc"],
        "step": 21000,
        "config_name": "quadro_stage_d_fchpa_evfirst_selected_21k_h1",
    }
    _write_json(ROOT / CHECKPOINT_ALIAS_REGISTRY, alias_registry)

    manifest = {
        "run_label": args.run_label,
        "created_at_utc": _utc_now(),
        "source_paths": source_rows,
        "alias_paths": alias_rows,
        "best_eval_path": str(BEST_EVAL_PATH),
        "best_checkpoint_path": str(CHAMPION_SOURCE_CKPT),
        "summary_metrics": cert_summary,
    }
    _write_json(manifest_path, manifest)

    runs_index = _safe_json_load(runs_index_path, {"runs": []})
    if not isinstance(runs_index, dict):
        runs_index = {"runs": []}
    runs = runs_index.get("runs", [])
    if not isinstance(runs, list):
        runs = []
    new_row = {
        "run_label": args.run_label,
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "best_eval_path": str(BEST_EVAL_PATH),
        "best_checkpoint_path": str(CHAMPION_SOURCE_CKPT),
        "created_at_utc": manifest["created_at_utc"],
    }
    runs = [row for row in runs if isinstance(row, dict) and row.get("run_label") != args.run_label]
    runs.append(new_row)
    runs_index["runs"] = sorted(runs, key=lambda row: row.get("run_label", ""))
    _write_json(runs_index_path, runs_index)

    print(f"[APPLY] Completed. Manifest: {manifest_path}")
    print(f"[APPLY] stage_d_fchpa_21k: {ROOT / CHAMPION_DEST_CKPT}")


if __name__ == "__main__":
    main()
