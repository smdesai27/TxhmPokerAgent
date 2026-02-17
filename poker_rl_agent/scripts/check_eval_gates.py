#!/usr/bin/env python3
"""Check Stage D evaluation gates and print promotion decision."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Tuple


def _get_tier(report: Dict[str, object], iterations: int) -> Dict[str, object] | None:
    for tier in report.get("solver_tiers", []):
        if int(tier.get("iterations", -1)) == iterations:
            return tier
    return None


def _tier_mean_ci(tier: Dict[str, object] | None) -> Tuple[float, float]:
    if not tier:
        return float("nan"), float("nan")
    mean = float(tier.get("aggregate", {}).get("bb_per_100", {}).get("mean", float("nan")))
    ci_low = float(tier.get("ci95_lower_bb100", float("nan")))
    return mean, ci_low


def _to_bool(report: Dict[str, object], key: str) -> bool:
    return bool(report.get(key, False))


def check_report(
    report: Dict[str, object],
    ci_floor: float,
    require_behavior_extended: bool,
    require_solver_verified: bool,
    require_aggression_gate: bool = True,
    max_allin_freq: float | None = None,
    min_call_check_freq: float | None = None,
    min_half_pot_freq: float | None = None,
    min_raise_total_freq: float | None = None,
    min_pot_choose_given_legal_freq: float | None = None,
    min_fold_freq: float | None = None,
    max_fold_freq: float | None = None,
    min_entropy_bits: float | None = None,
    max_entropy_bits: float | None = None,
) -> Tuple[bool, Dict[str, object]]:
    default_tier = _get_tier(report, 5000)
    robust_tier = _get_tier(report, 10000)
    default_mean, default_ci_low = _tier_mean_ci(default_tier)
    robust_mean, robust_ci_low = _tier_mean_ci(robust_tier)

    pass_default = _to_bool(report, "overall/pass_default_cfr_gate")
    pass_robust = _to_bool(report, "overall/pass_robustness_gate")
    pass_holdout = _to_bool(report, "overall/pass_holdout_gate")
    pass_behavior = _to_bool(report, "overall/pass_behavior_gate")
    pass_behavior_ext = _to_bool(report, "overall/pass_behavior_gate_extended")
    pass_aggression = bool(
        report.get("overall/pass_aggression_gate", report.get("overall/pass_pot_mix_gate", True))
    )
    pass_solver_verified = bool(report.get("verification/solver_training_verified", False))
    default_fold = float(default_tier.get("diagnostics/action_freq_preflop/fold", float("nan"))) if default_tier else float("nan")
    default_call = float(default_tier.get("diagnostics/action_freq_preflop/call_check", float("nan"))) if default_tier else float("nan")
    default_half = float(default_tier.get("diagnostics/action_freq_preflop/half_pot", float("nan"))) if default_tier else float("nan")
    default_pot = float(default_tier.get("diagnostics/action_freq_preflop/pot_raise", float("nan"))) if default_tier else float("nan")
    default_allin = float(default_tier.get("diagnostics/action_freq_preflop/allin", float("nan"))) if default_tier else float("nan")
    default_entropy = float(default_tier.get("diagnostics/preflop_action_entropy_bits", float("nan"))) if default_tier else float("nan")
    default_pot_choose_legal = (
        float(default_tier.get("diagnostics/preflop_choose_given_legal/pot_raise", float("nan")))
        if default_tier
        else float("nan")
    )
    default_raise_total = (
        default_half + default_pot + default_allin
        if all(math.isfinite(v) for v in (default_half, default_pot, default_allin))
        else float("nan")
    )

    envelope_checks = []
    if max_allin_freq is not None:
        envelope_checks.append(bool(math.isfinite(default_allin) and default_allin <= float(max_allin_freq)))
    if min_call_check_freq is not None:
        envelope_checks.append(bool(math.isfinite(default_call) and default_call >= float(min_call_check_freq)))
    if min_half_pot_freq is not None:
        envelope_checks.append(bool(math.isfinite(default_half) and default_half >= float(min_half_pot_freq)))
    if min_raise_total_freq is not None:
        envelope_checks.append(bool(math.isfinite(default_raise_total) and default_raise_total >= float(min_raise_total_freq)))
    if min_pot_choose_given_legal_freq is not None:
        envelope_checks.append(
            bool(
                math.isfinite(default_pot_choose_legal)
                and default_pot_choose_legal >= float(min_pot_choose_given_legal_freq)
            )
        )
    if min_fold_freq is not None:
        envelope_checks.append(bool(math.isfinite(default_fold) and default_fold >= float(min_fold_freq)))
    if max_fold_freq is not None:
        envelope_checks.append(bool(math.isfinite(default_fold) and default_fold <= float(max_fold_freq)))
    if min_entropy_bits is not None:
        envelope_checks.append(bool(math.isfinite(default_entropy) and default_entropy >= float(min_entropy_bits)))
    if max_entropy_bits is not None:
        envelope_checks.append(bool(math.isfinite(default_entropy) and default_entropy <= float(max_entropy_bits)))
    pass_behavior_envelope = all(envelope_checks) if envelope_checks else True

    checks = {
        "pass_default_gate": pass_default,
        "pass_robustness_gate": pass_robust,
        "pass_holdout_gate": pass_holdout,
        "pass_behavior_gate": pass_behavior,
        "pass_behavior_extended_gate": pass_behavior_ext or (not require_behavior_extended),
        "pass_aggression_gate": pass_aggression or (not require_aggression_gate),
        "pass_behavior_envelope": pass_behavior_envelope,
        "pass_solver_verified": pass_solver_verified or (not require_solver_verified),
        "pass_ci_floor": bool(math.isfinite(default_ci_low) and default_ci_low >= ci_floor),
    }

    passed = all(checks.values())
    summary = {
        "checks": checks,
        "default_mean_bb100": default_mean,
        "default_ci95_lower_bb100": default_ci_low,
        "robust_mean_bb100": robust_mean,
        "robust_ci95_lower_bb100": robust_ci_low,
        "overall_pass_all": _to_bool(report, "overall/pass_all"),
        "behavior_gate": report.get("behavior_gate", {}),
        "behavior_gate_extended": report.get("behavior_gate_extended", {}),
        "default_tier_observed": {
            "fold_freq": default_fold,
            "call_check_freq": default_call,
            "half_pot_freq": default_half,
            "pot_raise_freq": default_pot,
            "raise_total_freq": default_raise_total,
            "pot_choose_given_legal_freq": default_pot_choose_legal,
            "allin_freq": default_allin,
            "preflop_entropy_bits": default_entropy,
        },
        "aggression_gate": report.get("aggression_gate", report.get("pot_mix_gate", {})),
    }
    return passed, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Stage D eval report against promotion gates.")
    parser.add_argument("--eval_json", required=True, help="Path to evaluate_complete JSON report.")
    parser.add_argument("--ci_floor", type=float, default=830.0, help="Minimum default-tier CI95 lower bound.")
    parser.add_argument(
        "--require_behavior_extended",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require overall/pass_behavior_gate_extended=true.",
    )
    parser.add_argument(
        "--require_solver_verified",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require verification/solver_training_verified=true.",
    )
    parser.add_argument(
        "--require_aggression_gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require overall/pass_aggression_gate=true.",
    )
    parser.add_argument(
        "--require_pot_mix_gate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Deprecated alias for --require_aggression_gate.",
    )
    parser.add_argument(
        "--baseline_eval_json",
        default="",
        help="Optional baseline eval JSON for delta diagnostics (for example 16k cert).",
    )
    parser.add_argument("--max_allin_freq", type=float, default=None)
    parser.add_argument("--min_call_check_freq", type=float, default=None)
    parser.add_argument("--min_half_pot_freq", type=float, default=None)
    parser.add_argument("--min_raise_total_freq", type=float, default=None)
    parser.add_argument("--min_pot_choose_given_legal_freq", type=float, default=None)
    parser.add_argument("--min_fold_freq", type=float, default=None)
    parser.add_argument("--max_fold_freq", type=float, default=None)
    parser.add_argument("--min_entropy_bits", type=float, default=None)
    parser.add_argument("--max_entropy_bits", type=float, default=None)
    args = parser.parse_args()

    eval_path = Path(args.eval_json)
    if not eval_path.exists():
        print(f"ERROR: eval JSON not found: {eval_path}", file=sys.stderr)
        raise SystemExit(2)

    report = json.loads(eval_path.read_text(encoding="utf-8"))
    if args.require_pot_mix_gate is None:
        require_aggression_gate = bool(args.require_aggression_gate)
    else:
        require_aggression_gate = bool(args.require_pot_mix_gate)

    passed, summary = check_report(
        report,
        ci_floor=args.ci_floor,
        require_behavior_extended=bool(args.require_behavior_extended),
        require_solver_verified=bool(args.require_solver_verified),
        require_aggression_gate=require_aggression_gate,
        max_allin_freq=args.max_allin_freq,
        min_call_check_freq=args.min_call_check_freq,
        min_half_pot_freq=args.min_half_pot_freq,
        min_raise_total_freq=args.min_raise_total_freq,
        min_pot_choose_given_legal_freq=args.min_pot_choose_given_legal_freq,
        min_fold_freq=args.min_fold_freq,
        max_fold_freq=args.max_fold_freq,
        min_entropy_bits=args.min_entropy_bits,
        max_entropy_bits=args.max_entropy_bits,
    )

    print("Stage D Gate Check")
    print(f"eval_json={eval_path}")
    print(
        "default_mean={:.3f} default_ci95_lower={:.3f} robust_mean={:.3f} robust_ci95_lower={:.3f}".format(
            summary["default_mean_bb100"],
            summary["default_ci95_lower_bb100"],
            summary["robust_mean_bb100"],
            summary["robust_ci95_lower_bb100"],
        )
    )
    for key, ok in summary["checks"].items():
        print(f"{key}={ok}")
    observed = summary["default_tier_observed"]
    print(
        "default_observed "
        "fold={:.4f} call={:.4f} half={:.4f} pot={:.4f} raise_total={:.4f} pot|legal={:.4f} allin={:.4f} entropy={:.4f}".format(
            observed["fold_freq"],
            observed["call_check_freq"],
            observed["half_pot_freq"],
            observed["pot_raise_freq"],
            observed["raise_total_freq"],
            observed["pot_choose_given_legal_freq"],
            observed["allin_freq"],
            observed["preflop_entropy_bits"],
        )
    )

    if args.baseline_eval_json:
        baseline_path = Path(args.baseline_eval_json)
        if baseline_path.exists():
            baseline_report = json.loads(baseline_path.read_text(encoding="utf-8"))
            _, baseline_summary = check_report(
                baseline_report,
                ci_floor=args.ci_floor,
                require_behavior_extended=False,
                require_solver_verified=False,
            )
            print(
                "delta_vs_baseline_default_mean={:.3f} delta_vs_baseline_default_ci95_lower={:.3f}".format(
                    summary["default_mean_bb100"] - baseline_summary["default_mean_bb100"],
                    summary["default_ci95_lower_bb100"] - baseline_summary["default_ci95_lower_bb100"],
                )
            )
        else:
            print(f"baseline_eval_json_not_found={baseline_path}")

    print(f"promotion_pass={passed}")
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
