#!/usr/bin/env python3
"""Freeze Stage D interview alignment artifacts.

This script codifies the operational decision:
- keep `interview_ready_1` as canonical champion
- mark the latest 34k continuation as non-promoted exploratory branch
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RUN_LABEL = "20260220_012455_stage_d_longrun_rebalance_34k_nonpromoted"
DEFAULT_CHAMPION_CHECKPOINT = Path("checkpoints/snapshots/interview_ready/interview_ready_1.pt")
DEFAULT_CHAMPION_CERT_EVAL = Path("logs/stage_d/eval/eval_selected_21k_auto_cert_20260216_161321.json")
DEFAULT_BRANCH_CHECKPOINT = Path("checkpoints/snapshots/stage_d/recovery/longrun_rebalance_34k_20260220_012455.pt")
DEFAULT_BRANCH_SCREEN_EVAL = Path("logs/stage_d/eval/eval_longrun_rebalance_34k_screen_20260220_012455.json")
DEFAULT_BRANCH_GATE_LOG = Path("logs/stage_d/slurm/ah_stage_d_fchpa_gate_check_432657.out")
DEFAULT_BRANCH_METRICS = Path("logs/metrics_20260220_062514.jsonl")

CHECKPOINT_ALIASES_PATH = Path("checkpoints/checkpoint_aliases.json")
RUNS_INDEX_PATH = Path("artifacts/runs/index.json")


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


def _safe_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_required_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _float_or_nan(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return False


def _tier_stats(payload: Dict, tier_name: str) -> Dict[str, float]:
    tier = {}
    for row in payload.get("solver_tiers", []):
        if str(row.get("name")) == tier_name:
            tier = row
            break

    aggregate = tier.get("aggregate", {}).get("bb_per_100", {})
    mean = _float_or_nan(aggregate.get("mean"))
    ci95 = _float_or_nan(aggregate.get("ci95"))
    lower = _float_or_nan(tier.get("ci95_lower_bb100"))
    if not math.isfinite(lower) and math.isfinite(mean) and math.isfinite(ci95):
        lower = mean - ci95
    return {
        "mean_bb100": mean,
        "ci95_bb100": ci95,
        "ci95_lower_bb100": lower,
    }


def _extract_eval_summary(path: Path) -> Dict[str, float | bool]:
    payload = _read_required_json(path)
    default = _tier_stats(payload, "default_cfr")
    robust = _tier_stats(payload, "robustness")
    holdout_agg = payload.get("holdout", {}).get("aggregate", {}).get("bb_per_100", {})
    holdout_mean = _float_or_nan(holdout_agg.get("mean"))
    holdout_ci = _float_or_nan(holdout_agg.get("ci95"))
    holdout_lower = _float_or_nan(payload.get("holdout", {}).get("ci95_lower_bb100"))
    if not math.isfinite(holdout_lower) and math.isfinite(holdout_mean) and math.isfinite(holdout_ci):
        holdout_lower = holdout_mean - holdout_ci

    default_diag = (
        payload.get("diagnostics/summary", {})
        .get("solver_tiers", {})
        .get("default_cfr", {})
    )

    return {
        "overall/pass_all": _to_bool(payload.get("overall/pass_all", False)),
        "overall/pass_default_cfr_gate": _to_bool(payload.get("overall/pass_default_cfr_gate", False)),
        "overall/pass_robustness_gate": _to_bool(payload.get("overall/pass_robustness_gate", False)),
        "overall/pass_holdout_gate": _to_bool(payload.get("overall/pass_holdout_gate", False)),
        "overall/pass_behavior_gate": _to_bool(payload.get("overall/pass_behavior_gate", False)),
        "overall/pass_behavior_gate_extended": _to_bool(payload.get("overall/pass_behavior_gate_extended", False)),
        "overall/pass_aggression_gate": _to_bool(payload.get("overall/pass_aggression_gate", False)),
        "overall/pass_behavior_envelope": _to_bool(payload.get("overall/pass_behavior_envelope", False)),
        "overall/require_behavior_extended": _to_bool(payload.get("overall/require_behavior_extended", False)),
        "default/mean_bb100": default["mean_bb100"],
        "default/ci95_lower_bb100": default["ci95_lower_bb100"],
        "robust/mean_bb100": robust["mean_bb100"],
        "robust/ci95_lower_bb100": robust["ci95_lower_bb100"],
        "holdout/mean_bb100": holdout_mean,
        "holdout/ci95_lower_bb100": holdout_lower,
        "diag/preflop/fold": _float_or_nan(default_diag.get("diagnostics/action_freq_preflop/fold")),
        "diag/preflop/call_check": _float_or_nan(default_diag.get("diagnostics/action_freq_preflop/call_check")),
        "diag/preflop/half_pot": _float_or_nan(default_diag.get("diagnostics/action_freq_preflop/half_pot")),
        "diag/preflop/pot_raise": _float_or_nan(default_diag.get("diagnostics/action_freq_preflop/pot_raise")),
        "diag/preflop/allin": _float_or_nan(default_diag.get("diagnostics/action_freq_preflop/allin")),
        "diag/preflop/entropy_bits": _float_or_nan(default_diag.get("diagnostics/preflop_action_entropy_bits")),
        "diag/preflop/raise_total": (
            _float_or_nan(default_diag.get("diagnostics/action_freq_preflop/half_pot"))
            + _float_or_nan(default_diag.get("diagnostics/action_freq_preflop/pot_raise"))
            + _float_or_nan(default_diag.get("diagnostics/action_freq_preflop/allin"))
        ),
    }


def _extract_gate_summary(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"missing gate log: {path}")
    raw = path.read_text(encoding="utf-8")
    rows = [line.strip() for line in raw.splitlines() if line.strip()]
    out: Dict[str, object] = {}
    for row in rows:
        if "=" not in row:
            continue
        if " " in row and row.count("=") > 1 and not row.startswith("default_observed "):
            for token in row.split():
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key.startswith("pass_") or key == "promotion_pass":
                    out[key] = _to_bool(value)
                else:
                    maybe_float = _float_or_nan(value)
                    out[key] = value if not math.isfinite(maybe_float) else maybe_float
            continue
        if row.startswith("default_observed "):
            # default_observed fold=... call=... half=...
            tokens = row.split()
            for token in tokens[1:]:
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                out[f"default_observed/{key}"] = _float_or_nan(value)
            continue
        if row.startswith("delta_vs_baseline_default_mean="):
            # Two key-value pairs in one row.
            for token in row.split():
                if "=" not in token:
                    continue
                key, value = token.split("=", 1)
                out[key] = _float_or_nan(value)
            continue
        key, value = row.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key.startswith("pass_") or key == "promotion_pass":
            out[key] = _to_bool(value)
        else:
            maybe_float = _float_or_nan(value)
            out[key] = value if not math.isfinite(maybe_float) else maybe_float
    return out


def _extract_metrics_summary(path: Path) -> Dict[str, float | int]:
    if not path.exists():
        raise FileNotFoundError(f"missing metrics jsonl: {path}")
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"metrics file is empty: {path}")

    def _mean_last(key: str, n: int) -> float:
        sample = rows[-n:] if len(rows) > n else rows
        vals = [_float_or_nan(row.get(key)) for row in sample]
        vals = [v for v in vals if math.isfinite(v)]
        if not vals:
            return float("nan")
        return float(sum(vals) / len(vals))

    return {
        "rows": len(rows),
        "iteration_start": int(rows[0].get("iteration", 0)),
        "iteration_end": int(rows[-1].get("iteration", 0)),
        "nonfinite_grad_skips_max": int(max(_float_or_nan(r.get("train/nonfinite_grad_skips")) for r in rows)),
        "nonfinite_loss_batches_max": int(max(_float_or_nan(r.get("train/nonfinite_loss_batches")) for r in rows)),
        "rollout_mean_bb_last100": _mean_last("rollout/mean_final_return_bb", 100),
        "preflop_fold_last100": _mean_last("train/preflop_fold_freq", 100),
        "preflop_call_last100": _mean_last("train/preflop_call_check_freq", 100),
        "preflop_half_last100": _mean_last("train/preflop_half_pot_freq", 100),
        "preflop_pot_last100": _mean_last("train/preflop_pot_raise_freq", 100),
        "preflop_allin_last100": _mean_last("train/preflop_allin_freq", 100),
        "preflop_entropy_last100": _mean_last("train/preflop_action_entropy_bits", 100),
    }


def _relative_symlink(alias: Path, source: Path) -> str:
    return os.path.relpath(str(source), start=str(alias.parent))


def _link(alias: Path, source: Path) -> None:
    alias.parent.mkdir(parents=True, exist_ok=True)
    target = _relative_symlink(alias, source)
    if alias.is_symlink():
        if os.readlink(alias) == target:
            return
        alias.unlink()
    elif alias.exists():
        raise FileExistsError(f"alias exists as non-symlink: {alias}")
    alias.symlink_to(target)


def _ops_for_dirs(paths: Iterable[Path]) -> List[str]:
    ops = []
    for path in paths:
        if not path.exists():
            ops.append(f"mkdir: {path}")
    return ops


def _ops_for_links(link_map: Dict[Path, Path]) -> List[str]:
    ops = []
    for alias, source in link_map.items():
        if not source.exists():
            ops.append(f"missing_source: {source}")
            continue
        if alias.is_symlink():
            target = _relative_symlink(alias, source)
            if os.readlink(alias) == target:
                continue
        if alias.exists() and not alias.is_symlink():
            ops.append(f"exists_non_symlink: {alias}")
            continue
        ops.append(f"symlink: {source} -> {alias}")
    return ops


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Stage D interview goal-alignment pack.")
    parser.add_argument("--run_label", default=DEFAULT_RUN_LABEL)
    parser.add_argument("--champion_checkpoint", default=str(DEFAULT_CHAMPION_CHECKPOINT))
    parser.add_argument("--champion_eval", default=str(DEFAULT_CHAMPION_CERT_EVAL))
    parser.add_argument("--branch_checkpoint", default=str(DEFAULT_BRANCH_CHECKPOINT))
    parser.add_argument("--branch_screen_eval", default=str(DEFAULT_BRANCH_SCREEN_EVAL))
    parser.add_argument("--branch_gate_log", default=str(DEFAULT_BRANCH_GATE_LOG))
    parser.add_argument("--branch_metrics", default=str(DEFAULT_BRANCH_METRICS))
    parser.add_argument(
        "--branch_alias",
        default="stage_d_longrun_rebalance_34k_nonpromoted_20260220_012455",
        help="Alias key written into checkpoints/checkpoint_aliases.json",
    )
    parser.add_argument("--apply", action="store_true", help="Apply writes. Default is dry-run.")
    args = parser.parse_args()

    run_dir = ROOT / "artifacts" / "runs" / args.run_label
    champion_checkpoint = ROOT / args.champion_checkpoint
    champion_eval = ROOT / args.champion_eval
    branch_checkpoint = ROOT / args.branch_checkpoint
    branch_screen_eval = ROOT / args.branch_screen_eval
    branch_gate_log = ROOT / args.branch_gate_log
    branch_metrics = ROOT / args.branch_metrics

    required_paths = [
        champion_checkpoint,
        champion_eval,
        branch_checkpoint,
        branch_screen_eval,
        branch_gate_log,
        branch_metrics,
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(str(path) for path in missing))

    champion_sha = _sha256(champion_checkpoint)
    branch_sha = _sha256(branch_checkpoint)
    champion_eval_summary = _extract_eval_summary(champion_eval)
    branch_eval_summary = _extract_eval_summary(branch_screen_eval)
    branch_gate_summary = _extract_gate_summary(branch_gate_log)
    branch_metrics_summary = _extract_metrics_summary(branch_metrics)

    ci_lower_delta = (
        branch_eval_summary["default/ci95_lower_bb100"]
        - champion_eval_summary["default/ci95_lower_bb100"]
    )
    promoted = bool(branch_gate_summary.get("promotion_pass", False))

    link_map: Dict[Path, Path] = {
        run_dir / "eval" / "champion_cert_21k.json": champion_eval,
        run_dir / "eval" / "screen_34k.json": branch_screen_eval,
        run_dir / "slurm" / "gate_34k.out": branch_gate_log,
        run_dir / "metrics" / "train_34k_continuation.jsonl": branch_metrics,
        run_dir / "checkpoints" / "champion_interview_ready_1.pt": champion_checkpoint,
        run_dir / "checkpoints" / "nonpromoted_34k_snapshot.pt": branch_checkpoint,
    }
    dirs = [
        run_dir,
        run_dir / "eval",
        run_dir / "slurm",
        run_dir / "metrics",
        run_dir / "checkpoints",
        run_dir / "report",
    ]

    manifest = {
        "run_label": args.run_label,
        "created_at_utc": _utc_now(),
        "status": "non_promoted_exploratory",
        "decision": "freeze_champion_do_not_promote_34k",
        "canonical_champion": {
            "checkpoint_path": str(DEFAULT_CHAMPION_CHECKPOINT),
            "checkpoint_sha256": champion_sha,
            "eval_cert_path": str(DEFAULT_CHAMPION_CERT_EVAL),
        },
        "exploratory_branch": {
            "checkpoint_path": str(Path(args.branch_checkpoint)),
            "checkpoint_sha256": branch_sha,
            "eval_screen_path": str(Path(args.branch_screen_eval)),
            "gate_log_path": str(Path(args.branch_gate_log)),
            "metrics_path": str(Path(args.branch_metrics)),
            "promotion_pass": promoted,
            "failure_reason": (
                "ci_floor_not_met" if not _to_bool(branch_gate_summary.get("pass_ci_floor", False)) else "other"
            ),
        },
        "comparison": {
            "champion_default_ci95_lower_bb100": champion_eval_summary["default/ci95_lower_bb100"],
            "branch_default_ci95_lower_bb100": branch_eval_summary["default/ci95_lower_bb100"],
            "delta_branch_minus_champion_ci95_lower_bb100": ci_lower_delta,
        },
        "summary_metrics": {
            "champion_eval": champion_eval_summary,
            "branch_eval": branch_eval_summary,
            "branch_gate": branch_gate_summary,
            "branch_training": branch_metrics_summary,
        },
        "alias_paths": {str(alias.relative_to(ROOT)): str(source.relative_to(ROOT)) for alias, source in link_map.items()},
    }

    report_json = {
        "title": "Stage D Goal Alignment: Freeze Champion, Do Not Promote 34k",
        "created_at_utc": _utc_now(),
        "goal": "interview_presentable_performance",
        "decision": manifest["decision"],
        "champion": manifest["canonical_champion"],
        "branch": manifest["exploratory_branch"],
        "headline": {
            "branch_stable": (
                branch_metrics_summary["nonfinite_grad_skips_max"] == 0
                and branch_metrics_summary["nonfinite_loss_batches_max"] == 0
            ),
            "branch_overall_pass_all": branch_eval_summary["overall/pass_all"],
            "branch_promotion_pass": promoted,
            "branch_ci_floor_pass": _to_bool(branch_gate_summary.get("pass_ci_floor", False)),
            "ci95_lower_delta_bb100_branch_minus_champion": ci_lower_delta,
        },
        "interview_narrative": [
            "Champion remains interview_ready_1 based on stronger certified CI-lower.",
            "34k branch is retained as reproducible, stable, non-promoted evidence.",
            "Promotion rule is strict: statistical gates plus CI-floor must pass.",
        ],
    }

    report_md = "\n".join(
        [
            "# Stage D Goal Alignment",
            "",
            "- Decision: freeze champion, do not promote 34k branch.",
            f"- Champion checkpoint: `{DEFAULT_CHAMPION_CHECKPOINT}`",
            f"- Champion cert eval: `{DEFAULT_CHAMPION_CERT_EVAL}`",
            f"- Branch checkpoint: `{Path(args.branch_checkpoint)}`",
            f"- Branch screen eval: `{Path(args.branch_screen_eval)}`",
            f"- Branch gate log: `{Path(args.branch_gate_log)}`",
            "",
            "## Why Not Promote 34k",
            "",
            f"- Branch default CI95 lower: `{branch_eval_summary['default/ci95_lower_bb100']:.3f}`",
            f"- Champion default CI95 lower: `{champion_eval_summary['default/ci95_lower_bb100']:.3f}`",
            f"- Delta (branch - champion): `{ci_lower_delta:.3f}`",
            f"- Gate promotion pass: `{promoted}`",
            f"- CI-floor pass: `{_to_bool(branch_gate_summary.get('pass_ci_floor', False))}`",
            "",
            "## Stability Snapshot (34k Branch)",
            "",
            f"- nonfinite_grad_skips_max: `{branch_metrics_summary['nonfinite_grad_skips_max']}`",
            f"- nonfinite_loss_batches_max: `{branch_metrics_summary['nonfinite_loss_batches_max']}`",
            f"- preflop_fold_last100: `{branch_metrics_summary['preflop_fold_last100']:.4f}`",
            f"- preflop_call_last100: `{branch_metrics_summary['preflop_call_last100']:.4f}`",
            f"- preflop_half_last100: `{branch_metrics_summary['preflop_half_last100']:.4f}`",
            f"- preflop_pot_last100: `{branch_metrics_summary['preflop_pot_last100']:.4f}`",
            f"- preflop_allin_last100: `{branch_metrics_summary['preflop_allin_last100']:.4f}`",
            f"- preflop_entropy_last100: `{branch_metrics_summary['preflop_entropy_last100']:.4f}`",
            "",
            "## Interview Framing",
            "",
            "- We promote only when statistical gates and CI-floor pass.",
            "- This keeps claims conservative, reproducible, and technically defensible.",
        ]
    ) + "\n"

    outputs = {
        "manifest": run_dir / "manifest.json",
        "report_json": run_dir / "report" / "goal_alignment_summary.json",
        "report_md": run_dir / "report" / "goal_alignment_summary.md",
        "runs_index": ROOT / RUNS_INDEX_PATH,
        "checkpoint_aliases": ROOT / CHECKPOINT_ALIASES_PATH,
    }

    ops: List[str] = []
    ops.extend(_ops_for_dirs(dirs))
    ops.extend(_ops_for_links(link_map))
    ops.extend([f"write: {path}" for path in outputs.values()])

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] Stage D goal-alignment packaging ({len(ops)} operations)")
    for op in ops:
        print(f" - {op}")

    if not args.apply:
        return

    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)
    for alias, source in link_map.items():
        _link(alias, source)

    _write_json(outputs["manifest"], manifest)
    _write_json(outputs["report_json"], report_json)
    outputs["report_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report_md"].write_text(report_md, encoding="utf-8")

    runs_index = _safe_json(outputs["runs_index"], {"runs": []})
    if not isinstance(runs_index, dict):
        runs_index = {"runs": []}
    runs = runs_index.get("runs", [])
    if not isinstance(runs, list):
        runs = []
    row = {
        "run_label": args.run_label,
        "manifest_path": str(outputs["manifest"].relative_to(ROOT)),
        "best_checkpoint_path": str(DEFAULT_CHAMPION_CHECKPOINT),
        "best_eval_path": str(DEFAULT_CHAMPION_CERT_EVAL),
        "exploratory_checkpoint_path": str(Path(args.branch_checkpoint)),
        "exploratory_eval_path": str(Path(args.branch_screen_eval)),
        "status": "non_promoted_exploratory",
        "created_at_utc": _utc_now(),
    }
    runs = [item for item in runs if isinstance(item, dict) and item.get("run_label") != args.run_label]
    runs.append(row)
    runs_index["runs"] = sorted(runs, key=lambda item: item.get("run_label", ""))
    _write_json(outputs["runs_index"], runs_index)

    aliases = _safe_json(outputs["checkpoint_aliases"], {})
    if not isinstance(aliases, dict):
        aliases = {}
    champion_entry = aliases.get("interview_ready_1", {})
    if not isinstance(champion_entry, dict):
        champion_entry = {}
    champion_entry.update(
        {
            "path": str(DEFAULT_CHAMPION_CHECKPOINT),
            "sha256": champion_sha,
            "status": "canonical_champion",
            "frozen_for_interview": True,
            "source_eval_cert": str(DEFAULT_CHAMPION_CERT_EVAL),
        }
    )
    aliases["interview_ready_1"] = champion_entry
    aliases[args.branch_alias] = {
        "path": str(Path(args.branch_checkpoint)),
        "sha256": branch_sha,
        "status": "non_promoted_exploratory",
        "promotion_pass": promoted,
        "source_eval_screen": str(Path(args.branch_screen_eval)),
        "source_gate_log": str(Path(args.branch_gate_log)),
        "created_at_utc": _utc_now(),
        "reason": "ci_floor_not_met",
    }
    _write_json(outputs["checkpoint_aliases"], aliases)

    print(f"[APPLY] Completed packaging for run_label={args.run_label}")
    print(f"[APPLY] Champion preserved: {DEFAULT_CHAMPION_CHECKPOINT} sha256={champion_sha}")
    print(f"[APPLY] Exploratory branch recorded: {Path(args.branch_checkpoint)} sha256={branch_sha}")


if __name__ == "__main__":
    main()
