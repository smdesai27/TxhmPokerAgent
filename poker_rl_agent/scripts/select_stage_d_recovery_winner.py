#!/usr/bin/env python3
"""Selects the best Stage D recovery probe based on gate + behavior constraints."""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass
class CandidateScore:
    name: str
    eval_json: str
    config_name: str
    checkpoint: str
    default_mean: float
    default_ci95_lower: float
    robust_mean: float
    robust_ci95_lower: float
    fold: float
    call_check: float
    half_pot: float
    pot_raise: float
    raise_total: float
    allin: float
    entropy: float
    pot_choose_given_legal: float
    exploit_weighted_bb100: float
    exploit_clipped_bb100: float
    winner_score: float
    pass_default: bool
    pass_robust: bool
    pass_holdout: bool
    pass_behavior: bool
    pass_behavior_extended: bool
    pass_aggression: bool
    pass_solver_verified: bool
    pass_envelope: bool
    eligible: bool
    reasons: List[str]


def _get_tier(report: Dict[str, object], iterations: int) -> Dict[str, object] | None:
    for tier in report.get("solver_tiers", []):
        if int(tier.get("iterations", -1)) == iterations:
            return tier
    return None


def _to_bool(report: Dict[str, object], key: str) -> bool:
    return bool(report.get(key, False))


def _finite(v: float) -> bool:
    return math.isfinite(float(v))


def _escape_single_quotes(value: str) -> str:
    return value.replace("'", "'\"'\"'")


def _score_candidate(
    name: str,
    report_path: Path,
    min_ci_floor: float,
    max_allin_freq: float,
    min_call_check_freq: float,
    min_half_pot_freq: float,
    min_raise_total_freq: float,
    min_pot_choose_given_legal_freq: float | None,
    min_fold_freq: float,
    max_fold_freq: float,
    min_entropy_bits: float,
    max_entropy_bits: float,
    exploit_weight: float,
    exploit_clip: float,
) -> CandidateScore:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    default_tier = _get_tier(report, 5000)
    robust_tier = _get_tier(report, 10000)
    if default_tier is None:
        raise ValueError(f"{name}: missing 5000-iteration default tier in {report_path}")
    if robust_tier is None:
        raise ValueError(f"{name}: missing 10000-iteration robustness tier in {report_path}")

    default_mean = float(default_tier.get("aggregate", {}).get("bb_per_100", {}).get("mean", float("nan")))
    default_ci95_lower = float(default_tier.get("ci95_lower_bb100", float("nan")))
    robust_mean = float(robust_tier.get("aggregate", {}).get("bb_per_100", {}).get("mean", float("nan")))
    robust_ci95_lower = float(robust_tier.get("ci95_lower_bb100", float("nan")))

    fold = float(default_tier.get("diagnostics/action_freq_preflop/fold", float("nan")))
    call_check = float(default_tier.get("diagnostics/action_freq_preflop/call_check", float("nan")))
    half_pot = float(default_tier.get("diagnostics/action_freq_preflop/half_pot", float("nan")))
    pot_raise = float(default_tier.get("diagnostics/action_freq_preflop/pot_raise", float("nan")))
    pot_choose_given_legal = float(
        default_tier.get("diagnostics/preflop_choose_given_legal/pot_raise", float("nan"))
    )
    allin = float(default_tier.get("diagnostics/action_freq_preflop/allin", float("nan")))
    raise_total = (
        half_pot + pot_raise + allin
        if _finite(half_pot) and _finite(pot_raise) and _finite(allin)
        else float("nan")
    )
    entropy = float(default_tier.get("diagnostics/preflop_action_entropy_bits", float("nan")))

    pass_default = _to_bool(report, "overall/pass_default_cfr_gate")
    pass_robust = _to_bool(report, "overall/pass_robustness_gate")
    pass_holdout = _to_bool(report, "overall/pass_holdout_gate")
    pass_behavior = _to_bool(report, "overall/pass_behavior_gate")
    pass_behavior_extended = _to_bool(report, "overall/pass_behavior_gate_extended")
    pass_aggression = bool(
        _to_bool(report, "overall/pass_aggression_gate")
        or (
            ("overall/pass_aggression_gate" not in report)
            and (
                ("overall/pass_pot_mix_gate" not in report)
                or _to_bool(report, "overall/pass_pot_mix_gate")
            )
        )
    )
    pass_solver_verified = bool(report.get("verification/solver_training_verified", False))
    exploit_weighted_bb100 = float(report.get("overall/exploit_weighted_bb100", 0.0))
    exploit_clipped_bb100 = float(np.clip(exploit_weighted_bb100, -abs(exploit_clip), abs(exploit_clip)))
    winner_score = float(default_ci95_lower + (exploit_weight * exploit_clipped_bb100))

    envelope_checks = {
        "allin<=max": _finite(allin) and allin <= max_allin_freq,
        "call>=min": _finite(call_check) and call_check >= min_call_check_freq,
        "half>=min": _finite(half_pot) and half_pot >= min_half_pot_freq,
        "raise_total>=min": _finite(raise_total) and raise_total >= min_raise_total_freq,
        "fold>=min": _finite(fold) and fold >= min_fold_freq,
        "fold<=max": _finite(fold) and fold <= max_fold_freq,
        "entropy>=min": _finite(entropy) and entropy >= min_entropy_bits,
        "entropy<=max": _finite(entropy) and entropy <= max_entropy_bits,
        "default_ci>=floor": _finite(default_ci95_lower) and default_ci95_lower >= min_ci_floor,
    }
    if min_pot_choose_given_legal_freq is not None:
        envelope_checks["pot|legal>=min"] = _finite(pot_choose_given_legal) and (
            pot_choose_given_legal >= min_pot_choose_given_legal_freq
        )
    pass_envelope = all(envelope_checks.values())

    reasons = []
    if not pass_default:
        reasons.append("default_gate_fail")
    if not pass_robust:
        reasons.append("robustness_gate_fail")
    if not pass_holdout:
        reasons.append("holdout_gate_fail")
    if not pass_behavior:
        reasons.append("behavior_gate_fail")
    if not pass_behavior_extended:
        reasons.append("behavior_extended_gate_fail")
    if not pass_aggression:
        reasons.append("aggression_gate_fail")
    if not pass_solver_verified:
        reasons.append("solver_verification_fail")
    for key, ok in envelope_checks.items():
        if not ok:
            reasons.append(f"envelope:{key}")

    checkpoint = str(report.get("checkpoint", ""))
    config_name = str(report.get("config_name", ""))
    eligible = len(reasons) == 0

    return CandidateScore(
        name=name,
        eval_json=str(report_path),
        config_name=config_name,
        checkpoint=checkpoint,
        default_mean=default_mean,
        default_ci95_lower=default_ci95_lower,
        robust_mean=robust_mean,
        robust_ci95_lower=robust_ci95_lower,
        fold=fold,
        call_check=call_check,
        half_pot=half_pot,
        pot_raise=pot_raise,
        raise_total=raise_total,
        allin=allin,
        entropy=entropy,
        pot_choose_given_legal=pot_choose_given_legal,
        exploit_weighted_bb100=exploit_weighted_bb100,
        exploit_clipped_bb100=exploit_clipped_bb100,
        winner_score=winner_score,
        pass_default=pass_default,
        pass_robust=pass_robust,
        pass_holdout=pass_holdout,
        pass_behavior=pass_behavior,
        pass_behavior_extended=pass_behavior_extended,
        pass_aggression=pass_aggression,
        pass_solver_verified=pass_solver_verified,
        pass_envelope=pass_envelope,
        eligible=eligible,
        reasons=reasons,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the best Stage D recovery probe winner.")
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate in the form NAME=PATH_TO_EVAL_JSON. Pass at least two.",
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_env", required=True)
    parser.add_argument("--min_ci_floor", type=float, default=760.0)
    parser.add_argument("--max_allin_freq", type=float, default=0.05)
    parser.add_argument("--min_call_check_freq", type=float, default=0.40)
    parser.add_argument("--min_half_pot_freq", type=float, default=0.01)
    parser.add_argument("--min_raise_total_freq", type=float, default=0.08)
    parser.add_argument("--min_pot_choose_given_legal_freq", type=float, default=None)
    parser.add_argument("--min_fold_freq", type=float, default=0.35)
    parser.add_argument("--max_fold_freq", type=float, default=0.55)
    parser.add_argument("--min_entropy_bits", type=float, default=1.1)
    parser.add_argument("--max_entropy_bits", type=float, default=1.8)
    parser.add_argument("--exploit_weight", type=float, default=0.20)
    parser.add_argument("--exploit_clip", type=float, default=500.0)
    args = parser.parse_args()

    if len(args.candidate) < 2:
        raise SystemExit("At least two --candidate entries are required.")

    scored: List[CandidateScore] = []
    for raw in args.candidate:
        if "=" not in raw:
            raise SystemExit(f"Invalid --candidate '{raw}', expected NAME=PATH.")
        name, path_str = raw.split("=", 1)
        name = name.strip()
        path = Path(path_str.strip())
        if not path.exists():
            raise SystemExit(f"Candidate eval JSON not found: {path}")
        scored.append(
            _score_candidate(
                name=name,
                report_path=path,
                min_ci_floor=float(args.min_ci_floor),
                max_allin_freq=float(args.max_allin_freq),
                min_call_check_freq=float(args.min_call_check_freq),
                min_half_pot_freq=float(args.min_half_pot_freq),
                min_raise_total_freq=float(args.min_raise_total_freq),
                min_pot_choose_given_legal_freq=(
                    None
                    if args.min_pot_choose_given_legal_freq is None
                    else float(args.min_pot_choose_given_legal_freq)
                ),
                min_fold_freq=float(args.min_fold_freq),
                max_fold_freq=float(args.max_fold_freq),
                min_entropy_bits=float(args.min_entropy_bits),
                max_entropy_bits=float(args.max_entropy_bits),
                exploit_weight=float(args.exploit_weight),
                exploit_clip=float(args.exploit_clip),
            )
        )

    eligible = [row for row in scored if row.eligible]
    selected = None
    if eligible:
        eligible.sort(
            key=lambda row: (row.winner_score, row.default_ci95_lower, row.default_mean),
            reverse=True,
        )
        selected = eligible[0]

    payload = {
        "selected": None if selected is None else {
            "name": selected.name,
            "config_name": selected.config_name,
            "checkpoint": selected.checkpoint,
            "eval_json": selected.eval_json,
            "default_mean_bb100": selected.default_mean,
            "default_ci95_lower_bb100": selected.default_ci95_lower,
            "robust_mean_bb100": selected.robust_mean,
            "robust_ci95_lower_bb100": selected.robust_ci95_lower,
            "fold_freq": selected.fold,
            "call_check_freq": selected.call_check,
            "half_pot_freq": selected.half_pot,
            "pot_raise_freq": selected.pot_raise,
            "raise_total_freq": selected.raise_total,
            "pot_choose_given_legal_freq": selected.pot_choose_given_legal,
            "allin_freq": selected.allin,
            "preflop_entropy_bits": selected.entropy,
            "exploit_weighted_bb100": selected.exploit_weighted_bb100,
            "exploit_clipped_bb100": selected.exploit_clipped_bb100,
            "winner_score": selected.winner_score,
        },
        "candidates": [
            {
                "name": row.name,
                "config_name": row.config_name,
                "checkpoint": row.checkpoint,
                "eval_json": row.eval_json,
                "default_mean_bb100": row.default_mean,
                "default_ci95_lower_bb100": row.default_ci95_lower,
                "robust_mean_bb100": row.robust_mean,
                "robust_ci95_lower_bb100": row.robust_ci95_lower,
                "fold_freq": row.fold,
                "call_check_freq": row.call_check,
                "half_pot_freq": row.half_pot,
                "pot_raise_freq": row.pot_raise,
                "raise_total_freq": row.raise_total,
                "pot_choose_given_legal_freq": row.pot_choose_given_legal,
                "allin_freq": row.allin,
                "preflop_entropy_bits": row.entropy,
                "exploit_weighted_bb100": row.exploit_weighted_bb100,
                "exploit_clipped_bb100": row.exploit_clipped_bb100,
                "winner_score": row.winner_score,
                "eligible": row.eligible,
                "reasons": row.reasons,
            }
            for row in scored
        ],
        "thresholds": {
            "min_ci_floor": float(args.min_ci_floor),
            "max_allin_freq": float(args.max_allin_freq),
            "min_call_check_freq": float(args.min_call_check_freq),
            "min_half_pot_freq": float(args.min_half_pot_freq),
            "min_raise_total_freq": float(args.min_raise_total_freq),
            "min_pot_choose_given_legal_freq": (
                None
                if args.min_pot_choose_given_legal_freq is None
                else float(args.min_pot_choose_given_legal_freq)
            ),
            "min_fold_freq": float(args.min_fold_freq),
            "max_fold_freq": float(args.max_fold_freq),
            "min_entropy_bits": float(args.min_entropy_bits),
            "max_entropy_bits": float(args.max_entropy_bits),
            "exploit_weight": float(args.exploit_weight),
            "exploit_clip": float(args.exploit_clip),
        },
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out_env = Path(args.output_env)
    out_env.parent.mkdir(parents=True, exist_ok=True)
    if selected is not None:
        if not selected.checkpoint:
            raise SystemExit(f"Selected candidate '{selected.name}' missing checkpoint in eval report.")
        lines = [
            f"WINNER_NAME='{_escape_single_quotes(selected.name)}'",
            f"WINNER_CONFIG_NAME='{_escape_single_quotes(selected.config_name)}'",
            f"WINNER_EVAL_JSON='{_escape_single_quotes(selected.eval_json)}'",
            f"WINNER_CHECKPOINT='{_escape_single_quotes(selected.checkpoint)}'",
        ]
        out_env.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        out_env.write_text("", encoding="utf-8")

    print(f"Wrote selection JSON: {out_json}")
    print(f"Wrote winner env: {out_env}")
    if selected is None:
        print("No eligible winner found.")
        raise SystemExit(3)
    print(
        "Selected winner: "
        f"{selected.name} (default_ci95_lower={selected.default_ci95_lower:.3f}, "
        f"default_mean={selected.default_mean:.3f})"
    )


if __name__ == "__main__":
    main()
