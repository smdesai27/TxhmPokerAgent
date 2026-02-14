#!/usr/bin/env python3
"""Summarize Stage D ablation eval reports and recommend a winner."""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional


def _get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    return d.get(key, default)


def _extract_tier(report: Dict[str, Any], iterations: int) -> Optional[Dict[str, Any]]:
    for tier in report.get("solver_tiers", []):
        if int(tier.get("iterations", -1)) == iterations:
            return tier
    return None


def _tier_metric(tier: Optional[Dict[str, Any]], key: str, default: float = float("nan")) -> float:
    if not tier:
        return default
    if key == "mean_bb100":
        return float(tier.get("aggregate", {}).get("bb_per_100", {}).get("mean", default))
    if key == "ci95_lower_bb100":
        return float(tier.get("ci95_lower_bb100", default))
    return float(tier.get(key, default))


def _diag(tier: Optional[Dict[str, Any]], key: str, default: float = float("nan")) -> float:
    if not tier:
        return default
    return float(tier.get(f"diagnostics/{key}", default))


def _pass(report: Dict[str, Any], key: str) -> bool:
    return bool(_get(report, key, False))


def _load_reports(paths: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for path in sorted(paths):
        with open(path, "r", encoding="utf-8") as f:
            report = json.load(f)
        default_tier = _extract_tier(report, 5000)
        robust_tier = _extract_tier(report, 10000)
        rows.append(
            {
                "path": path,
                "name": os.path.basename(path),
                "pass_all": _pass(report, "overall/pass_all"),
                "pass_default": _pass(report, "overall/pass_default_cfr_gate"),
                "pass_robust": _pass(report, "overall/pass_robustness_gate"),
                "pass_holdout": _pass(report, "overall/pass_holdout_gate"),
                "pass_behavior": _pass(report, "overall/pass_behavior_gate"),
                "default_mean": _tier_metric(default_tier, "mean_bb100"),
                "default_ci_low": _tier_metric(default_tier, "ci95_lower_bb100"),
                "robust_mean": _tier_metric(robust_tier, "mean_bb100"),
                "robust_ci_low": _tier_metric(robust_tier, "ci95_lower_bb100"),
                "fold_freq": _diag(default_tier, "action_freq_preflop/fold"),
                "call_check_freq": _diag(default_tier, "action_freq_preflop/call_check"),
                "half_pot_freq": _diag(default_tier, "action_freq_preflop/half_pot"),
                "allin_freq": _diag(default_tier, "action_freq_preflop/allin"),
                "entropy_bits": _diag(default_tier, "preflop_action_entropy_bits"),
            }
        )
    return rows


def _score(row: Dict[str, Any]) -> float:
    # Primary sort key for promotion: default-tier CI lower bound.
    return float(row["default_ci_low"])


def _print_rows(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No ablation eval reports found.")
        return
    print("Stage D Ablation Summary")
    print("-" * 100)
    for row in rows:
        print(
            f"{row['name']}: pass_all={row['pass_all']} "
            f"default_ci_low={row['default_ci_low']:.3f} default_mean={row['default_mean']:.3f} "
            f"fold={row['fold_freq']:.4f} call/check={row['call_check_freq']:.4f} "
            f"half_pot={row['half_pot_freq']:.4f} allin={row['allin_freq']:.4f} "
            f"entropy={row['entropy_bits']:.4f}"
        )


def _recommend(rows: List[Dict[str, Any]]) -> None:
    passing = [r for r in rows if r["pass_default"] and r["pass_robust"] and r["pass_holdout"] and r["pass_behavior"]]
    if not passing:
        print("\nRecommendation: no ablation passed all gates. Do not promote to 16k yet.")
        print("Next action: run Ablation B/C evals or retune behavior settings before promotion.")
        return

    ranked = sorted(
        passing,
        key=lambda r: (
            _score(r),
            -float(r["fold_freq"]),  # lower fold is better
            float(r["call_check_freq"]),
            float(r["half_pot_freq"]),
        ),
        reverse=True,
    )
    winner = ranked[0]
    print("\nRecommended winner")
    print("-" * 100)
    print(
        f"{winner['name']} | default_ci_low={winner['default_ci_low']:.3f} "
        f"default_mean={winner['default_mean']:.3f} fold={winner['fold_freq']:.4f} "
        f"call/check={winner['call_check_freq']:.4f} half_pot={winner['half_pot_freq']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Stage D ablation eval JSON reports.")
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        default="logs/stage_d/eval/eval_*ablate*.json",
        help="Glob for ablation eval files.",
    )
    args = parser.parse_args()

    paths = glob.glob(args.glob_pattern)
    rows = _load_reports(paths)
    rows = sorted(rows, key=lambda r: _score(r), reverse=True)
    _print_rows(rows)
    _recommend(rows)


if __name__ == "__main__":
    main()
