import argparse
import gc
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from poker_rl_agent.environment.openspiel_wrapper import PokerEnv
from poker_rl_agent.evaluation.baseline_agents import AlwaysCallAgent, PotPressureAgent, RandomAgent
from poker_rl_agent.evaluation.evaluator import Evaluator
from poker_rl_agent.models.alpha_holdem_net import AlphaHoldemNetwork
from poker_rl_agent.training.checkpointing import load_checkpoint
from poker_rl_agent.utils.config import Config


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(config_file: str, config_name: str) -> Config:
    config = Config()
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        merged = {}
        if "default" in data and isinstance(data["default"], dict):
            merged.update(data["default"])
        if config_name != "default" and config_name in data and isinstance(data[config_name], dict):
            merged.update(data[config_name])

        for key, value in merged.items():
            if hasattr(config, key):
                setattr(config, key, value)

    if hasattr(config, "sync_legacy_fields"):
        config.sync_legacy_fields()
    return config


from poker_rl_agent.evaluation.stats import t95_multiplier


def _aggregate(values):
    arr = np.array(values, dtype=np.float64)
    n = int(arr.size)
    mean = float(arr.mean()) if n else 0.0
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    stderr = float(std / np.sqrt(n)) if n > 1 else 0.0
    # Student-t for the small seed-count samples (n typically 3-5); 1.96 for df > 30.
    ci95 = float(t95_multiplier(n) * stderr)
    return {"mean": mean, "std": std, "stderr": stderr, "ci95": ci95, "n": n}


def build_profile_tiers(profile: str, episodes_per_seed: int, betting_abstraction: str) -> List[Dict[str, int]]:
    profile = str(profile).lower()
    abstraction = str(betting_abstraction).lower()
    is_fullgame = abstraction == "fullgame"

    if profile == "quick":
        quick_episodes = min(int(episodes_per_seed), 800 if is_fullgame else 1000)
        if is_fullgame:
            return [
                {"name": "default_cfr", "iterations": 128, "seeds": 2, "episodes_per_seed": quick_episodes},
                {"name": "robustness", "iterations": 256, "seeds": 2, "episodes_per_seed": quick_episodes},
            ]
        return [
            {"name": "default_cfr", "iterations": 256, "seeds": 2, "episodes_per_seed": quick_episodes},
            {"name": "robustness", "iterations": 512, "seeds": 2, "episodes_per_seed": quick_episodes},
        ]
    if profile == "exhaustive":
        eps = int(episodes_per_seed)
        if is_fullgame:
            return [
                {"name": "default_cfr", "iterations": 2000, "seeds": 5, "episodes_per_seed": eps},
                {"name": "robustness", "iterations": 5000, "seeds": 3, "episodes_per_seed": eps},
                {"name": "strong_10k", "iterations": 10000, "seeds": 3, "episodes_per_seed": eps},
                {"name": "strong_20k", "iterations": 20000, "seeds": 3, "episodes_per_seed": eps},
            ]
        return [
            {"name": "default_cfr", "iterations": 5000, "seeds": 5, "episodes_per_seed": eps},
            {"name": "robustness", "iterations": 10000, "seeds": 3, "episodes_per_seed": eps},
            {"name": "strong_20k", "iterations": 20000, "seeds": 3, "episodes_per_seed": eps},
            {"name": "strong_50k", "iterations": 50000, "seeds": 3, "episodes_per_seed": eps},
        ]

    if profile == "cert":
        eps = int(episodes_per_seed)
        return [
            {"name": "weak_1k", "iterations": 1000, "seeds": 3, "episodes_per_seed": eps},
            {"name": "default_cfr", "iterations": 5000, "seeds": 5, "episodes_per_seed": eps},
            {"name": "competent_10k", "iterations": 10000, "seeds": 5, "episodes_per_seed": eps},
            {"name": "near_equilibrium_50k", "iterations": 50000, "seeds": 3, "episodes_per_seed": eps},
            {"name": "near_nash_100k", "iterations": 100000, "seeds": 3, "episodes_per_seed": eps},
        ]

    # Standard profile (default).
    eps = int(episodes_per_seed)
    if is_fullgame:
        return [
            {"name": "default_cfr", "iterations": 2000, "seeds": 5, "episodes_per_seed": eps},
            {"name": "robustness", "iterations": 5000, "seeds": 3, "episodes_per_seed": eps},
        ]
    return [
        {"name": "default_cfr", "iterations": 5000, "seeds": 5, "episodes_per_seed": eps},
        {"name": "robustness", "iterations": 10000, "seeds": 3, "episodes_per_seed": eps},
    ]


def _diag_from_metrics(row: Dict[str, float]) -> Dict[str, float]:
    fold = float(row.get("diagnostics/action_freq_preflop/fold", 0.0))
    call_check = float(row.get("diagnostics/action_freq_preflop/call_check", 0.0))
    half_pot = float(row.get("diagnostics/action_freq_preflop/half_pot", 0.0))
    pot_raise = float(row.get("diagnostics/action_freq_preflop/pot_raise", 0.0))
    allin = float(row.get("diagnostics/action_freq_preflop/allin", 0.0))
    freqs = [max(0.0, fold), max(0.0, call_check), max(0.0, half_pot), max(0.0, pot_raise), max(0.0, allin)]
    freq_sum = float(sum(freqs))
    if freq_sum > 0.0:
        probs = [f / freq_sum for f in freqs if f > 0.0]
        preflop_entropy_bits = float(-sum(p * np.log2(max(p, 1e-12)) for p in probs))
        preflop_dominant_action_freq = float(max(freqs) / freq_sum)
    else:
        preflop_entropy_bits = 0.0
        preflop_dominant_action_freq = 0.0
    return {
        "diagnostics/action_freq_preflop/fold": fold,
        "diagnostics/action_freq_preflop/call_check": call_check,
        "diagnostics/action_freq_preflop/half_pot": half_pot,
        "diagnostics/action_freq_preflop/pot_raise": pot_raise,
        "diagnostics/action_freq_preflop/allin": allin,
        "diagnostics/preflop_action_entropy_bits": preflop_entropy_bits,
        "diagnostics/preflop_dominant_action_freq": preflop_dominant_action_freq,
        "diagnostics/preflop_legal_rate/pot_raise": float(
            row.get("diagnostics/preflop_legal_rate/pot_raise", 0.0)
        ),
        "diagnostics/preflop_legal_rate/half_pot": float(
            row.get("diagnostics/preflop_legal_rate/half_pot", 0.0)
        ),
        "diagnostics/preflop_choose_given_legal/pot_raise": float(
            row.get("diagnostics/preflop_choose_given_legal/pot_raise", 0.0)
        ),
        "diagnostics/preflop_choose_given_legal/half_pot": float(
            row.get("diagnostics/preflop_choose_given_legal/half_pot", 0.0)
        ),
        "diagnostics/showdown_rate": float(row.get("diagnostics/showdown_rate", 0.0)),
        "diagnostics/avg_pot_size_bb": float(row.get("diagnostics/avg_pot_size_bb", 0.0)),
    }


def _aggregate_diagnostics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    fold = int(sum(int(r.get("diagnostics/_preflop_count_fold", 0)) for r in rows))
    call_check = int(sum(int(r.get("diagnostics/_preflop_count_call_check", 0)) for r in rows))
    half_pot = int(sum(int(r.get("diagnostics/_preflop_count_half_pot", 0)) for r in rows))
    pot_raise = int(sum(int(r.get("diagnostics/_preflop_count_pot_raise", 0)) for r in rows))
    allin = int(sum(int(r.get("diagnostics/_preflop_count_allin", 0)) for r in rows))
    preflop_total = int(sum(int(r.get("diagnostics/_preflop_total_count", 0)) for r in rows))
    preflop_legal_pot = int(sum(int(r.get("diagnostics/_preflop_legal_count_pot_raise", 0)) for r in rows))
    preflop_legal_half = int(sum(int(r.get("diagnostics/_preflop_legal_count_half_pot", 0)) for r in rows))
    preflop_choose_pot_legal = int(
        sum(int(r.get("diagnostics/_preflop_choose_legal_count_pot_raise", 0)) for r in rows)
    )
    preflop_choose_half_legal = int(
        sum(int(r.get("diagnostics/_preflop_choose_legal_count_half_pot", 0)) for r in rows)
    )
    showdown_hands = int(sum(int(r.get("diagnostics/_showdown_hands", 0)) for r in rows))
    total_hands = int(sum(int(r.get("diagnostics/_total_hands", 0)) for r in rows))
    total_terminal_pot_bb = float(sum(float(r.get("diagnostics/_total_terminal_pot_bb", 0.0)) for r in rows))

    preflop_denom = max(1, preflop_total)
    hand_denom = max(1, total_hands)
    return {
        "diagnostics/action_freq_preflop/fold": float(fold) / float(preflop_denom),
        "diagnostics/action_freq_preflop/call_check": float(call_check) / float(preflop_denom),
        "diagnostics/action_freq_preflop/half_pot": float(half_pot) / float(preflop_denom),
        "diagnostics/action_freq_preflop/pot_raise": float(pot_raise) / float(preflop_denom),
        "diagnostics/action_freq_preflop/allin": float(allin) / float(preflop_denom),
        "diagnostics/preflop_action_entropy_bits": _diag_from_metrics(
            {
                "diagnostics/action_freq_preflop/fold": float(fold) / float(preflop_denom),
                "diagnostics/action_freq_preflop/call_check": float(call_check) / float(preflop_denom),
                "diagnostics/action_freq_preflop/half_pot": float(half_pot) / float(preflop_denom),
                "diagnostics/action_freq_preflop/pot_raise": float(pot_raise) / float(preflop_denom),
                "diagnostics/action_freq_preflop/allin": float(allin) / float(preflop_denom),
            }
        )["diagnostics/preflop_action_entropy_bits"],
        "diagnostics/preflop_dominant_action_freq": _diag_from_metrics(
            {
                "diagnostics/action_freq_preflop/fold": float(fold) / float(preflop_denom),
                "diagnostics/action_freq_preflop/call_check": float(call_check) / float(preflop_denom),
                "diagnostics/action_freq_preflop/half_pot": float(half_pot) / float(preflop_denom),
                "diagnostics/action_freq_preflop/pot_raise": float(pot_raise) / float(preflop_denom),
                "diagnostics/action_freq_preflop/allin": float(allin) / float(preflop_denom),
            }
        )["diagnostics/preflop_dominant_action_freq"],
        "diagnostics/preflop_legal_rate/pot_raise": float(preflop_legal_pot) / float(preflop_denom),
        "diagnostics/preflop_legal_rate/half_pot": float(preflop_legal_half) / float(preflop_denom),
        "diagnostics/preflop_choose_given_legal/pot_raise": (
            float(preflop_choose_pot_legal) / float(max(1, preflop_legal_pot))
        ),
        "diagnostics/preflop_choose_given_legal/half_pot": (
            float(preflop_choose_half_legal) / float(max(1, preflop_legal_half))
        ),
        "diagnostics/showdown_rate": float(showdown_hands) / float(hand_denom),
        "diagnostics/avg_pot_size_bb": float(total_terminal_pot_bb) / float(hand_denom),
        "diagnostics/_preflop_total_count": int(preflop_total),
        "diagnostics/_preflop_legal_count_pot_raise": int(preflop_legal_pot),
        "diagnostics/_preflop_legal_count_half_pot": int(preflop_legal_half),
        "diagnostics/_preflop_choose_legal_count_pot_raise": int(preflop_choose_pot_legal),
        "diagnostics/_preflop_choose_legal_count_half_pot": int(preflop_choose_half_legal),
        "diagnostics/_total_hands": int(total_hands),
    }


def _compute_solver_verification(
    solver_tiers: List[Dict[str, object]],
    control_bb100: float,
) -> Tuple[bool, Dict[str, float], List[str]]:
    errors = []
    control_delta = {}

    for tier in solver_tiers:
        tier_name = str(tier.get("name", "tier"))
        iterations = int(tier.get("iterations", -1))
        if iterations <= 0:
            errors.append(f"{tier_name}:iterations<=0")
        for row in tier.get("per_seed", []):
            if float(row.get("baseline/build_seconds", 0.0)) <= 0.0:
                errors.append(f"{tier_name}:build_seconds<=0")
                break
        mean_bb = float(tier.get("aggregate", {}).get("bb_per_100", {}).get("mean", 0.0))
        delta = mean_bb - float(control_bb100)
        control_delta[tier_name] = float(delta)
        if not np.isfinite(delta):
            errors.append(f"{tier_name}:nonfinite_control_delta")

    return (len(errors) == 0), control_delta, errors


def _primary_eval_seed_set(base_seed: int, tiers: List[Dict[str, int]]) -> set:
    seeds = set()
    for tier in tiers:
        tier_seeds = int(tier.get("seeds", 0))
        for offset in range(max(0, tier_seeds)):
            seeds.add(int(base_seed) + offset)
    return seeds


def _passes_robustness_gate(robustness_tier: Dict[str, object], require_robust_ci: bool) -> bool:
    if robustness_tier is None:
        return False
    if require_robust_ci:
        return bool(float(robustness_tier.get("ci95_lower_bb100", 0.0)) > 0.0)
    mean_bb100 = float(robustness_tier.get("aggregate", {}).get("bb_per_100", {}).get("mean", 0.0))
    return bool(mean_bb100 > 0.0)


def _passes_behavior_gate(config, metrics_row: Dict[str, object]) -> bool:
    enabled = bool(getattr(config, "BEHAVIOR_GATE_ENABLE", True))
    if not enabled:
        return True

    fold_freq = float(metrics_row.get("diagnostics/action_freq_preflop/fold", 0.0))
    allin_freq = float(metrics_row.get("diagnostics/action_freq_preflop/allin", 0.0))
    entropy_bits = float(metrics_row.get("diagnostics/preflop_action_entropy_bits", 0.0))

    max_fold = float(getattr(config, "BEHAVIOR_GATE_MAX_FOLD_FREQ", 0.78))
    max_allin = float(getattr(config, "BEHAVIOR_GATE_MAX_ALLIN_FREQ", 0.06))
    min_entropy = float(getattr(config, "BEHAVIOR_GATE_MIN_PRE_FLOP_ENTROPY_BITS", 0.75))
    return bool(
        fold_freq <= max_fold
        and allin_freq <= max_allin
        and entropy_bits >= min_entropy
    )


def _passes_behavior_gate_extended(config, metrics_row: Dict[str, object]) -> bool:
    enabled = bool(getattr(config, "BEHAVIOR_GATE_ENABLE", True))
    if not enabled:
        return True

    call_check_freq = float(metrics_row.get("diagnostics/action_freq_preflop/call_check", 0.0))
    half_pot_freq = float(metrics_row.get("diagnostics/action_freq_preflop/half_pot", 0.0))

    min_call_check = float(getattr(config, "BEHAVIOR_GATE_MIN_CALL_CHECK_FREQ", 0.50))
    min_half_pot = float(getattr(config, "BEHAVIOR_GATE_MIN_HALF_POT_FREQ", 0.01))
    return bool(call_check_freq >= min_call_check and half_pot_freq >= min_half_pot)


def _passes_aggression_gate(config, metrics_row: Dict[str, object]) -> tuple[bool, Dict[str, float]]:
    enabled = bool(getattr(config, "EVAL_AGGRESSION_GATE_ENABLE", True))
    half_pot_freq = float(metrics_row.get("diagnostics/action_freq_preflop/half_pot", 0.0))
    pot_raise_freq = float(metrics_row.get("diagnostics/action_freq_preflop/pot_raise", 0.0))
    allin_freq = float(metrics_row.get("diagnostics/action_freq_preflop/allin", 0.0))
    raise_total_freq = half_pot_freq + pot_raise_freq + allin_freq
    min_raise_total = float(getattr(config, "EVAL_MIN_PRE_FLOP_RAISE_TOTAL_FREQ", 0.08))
    passed = bool((not enabled) or raise_total_freq >= min_raise_total)
    return passed, {
        "enabled": bool(enabled),
        "min_preflop_raise_total_freq": float(min_raise_total),
        "observed_preflop_raise_total_freq": float(raise_total_freq),
        "observed_half_pot_freq": float(half_pot_freq),
        "observed_pot_raise_freq": float(pot_raise_freq),
        "observed_allin_freq": float(allin_freq),
    }


def _passes_behavior_envelope(config, metrics_row: Dict[str, object]) -> tuple[bool, Dict[str, float]]:
    enabled = bool(getattr(config, "EVAL_ENVELOPE_ENABLE", False))
    fold_freq = float(metrics_row.get("diagnostics/action_freq_preflop/fold", 0.0))
    call_check_freq = float(metrics_row.get("diagnostics/action_freq_preflop/call_check", 0.0))
    allin_freq = float(metrics_row.get("diagnostics/action_freq_preflop/allin", 0.0))
    entropy_bits = float(metrics_row.get("diagnostics/preflop_action_entropy_bits", 0.0))

    min_fold = float(getattr(config, "EVAL_MIN_PRE_FLOP_FOLD_FREQ", 0.35))
    max_fold = float(getattr(config, "EVAL_MAX_PRE_FLOP_FOLD_FREQ", 0.55))
    min_call = float(getattr(config, "EVAL_MIN_PRE_FLOP_CALL_CHECK_FREQ", 0.40))
    max_allin = float(getattr(config, "EVAL_MAX_PRE_FLOP_ALLIN_FREQ", 0.05))
    min_entropy = float(getattr(config, "EVAL_MIN_PRE_FLOP_ENTROPY_BITS", 1.10))
    max_entropy = float(getattr(config, "EVAL_MAX_PRE_FLOP_ENTROPY_BITS", 1.80))

    if not enabled:
        passed = True
    else:
        passed = bool(
            min_fold <= fold_freq <= max_fold
            and call_check_freq >= min_call
            and allin_freq <= max_allin
            and min_entropy <= entropy_bits <= max_entropy
        )

    return passed, {
        "enabled": bool(enabled),
        "min_preflop_fold_freq": float(min_fold),
        "max_preflop_fold_freq": float(max_fold),
        "min_preflop_call_check_freq": float(min_call),
        "max_preflop_allin_freq": float(max_allin),
        "min_preflop_entropy_bits": float(min_entropy),
        "max_preflop_entropy_bits": float(max_entropy),
        "observed_preflop_fold_freq": float(fold_freq),
        "observed_preflop_call_check_freq": float(call_check_freq),
        "observed_preflop_allin_freq": float(allin_freq),
        "observed_preflop_entropy_bits": float(entropy_bits),
    }


def _load_solver_style_target(style_profile_path: str) -> Dict[str, float]:
    if not style_profile_path:
        return {}
    if not os.path.exists(style_profile_path):
        return {}
    try:
        payload = json.loads(open(style_profile_path, "r", encoding="utf-8").read())
    except Exception:
        return {}

    target = payload.get("target", {})
    if not isinstance(target, dict):
        target = {}

    return {
        "target_pot_given_legal": float(
            payload.get(
                "target/preflop_choose_given_legal/pot_raise",
                target.get("preflop_choose_given_legal/pot_raise", 0.0),
            )
        ),
        "target_half_given_legal": float(
            payload.get(
                "target/preflop_choose_given_legal/half_pot",
                target.get("preflop_choose_given_legal/half_pot", 0.0),
            )
        ),
        "target_fold": float(target.get("fold", 0.0)),
        "target_call_check": float(target.get("call_check", 0.0)),
        "target_half_pot": float(target.get("half_pot", 0.0)),
        "target_pot_raise": float(target.get("pot_raise", 0.0)),
        "target_allin": float(target.get("allin", 0.0)),
    }


def _passes_pot_mix_gate(config, metrics_row: Dict[str, object], solver_target: Dict[str, float]) -> tuple[bool, Dict[str, float]]:
    abstraction = str(getattr(config, "BETTING_ABSTRACTION", "fcpa")).lower()
    if abstraction not in {"fchpa", "fcpha"}:
        return True, {
            "enabled": False,
            "skipped_non_fchpa": True,
            "required_choose_given_legal_pot_raise": 0.0,
            "observed_choose_given_legal_pot_raise": float(
                metrics_row.get("diagnostics/preflop_choose_given_legal/pot_raise", 0.0)
            ),
            "solver_target_choose_given_legal_pot_raise": 0.0,
        }

    enabled = bool(getattr(config, "EVAL_POT_MIX_GATE_ENABLE", True))
    observed = float(metrics_row.get("diagnostics/preflop_choose_given_legal/pot_raise", 0.0))
    target_pot_given_legal = float(solver_target.get("target_pot_given_legal", 0.0))
    min_abs = float(getattr(config, "EVAL_POT_MIX_MIN_ABS", 0.01))
    target_fraction = float(getattr(config, "EVAL_POT_MIX_TARGET_FRACTION", 0.60))
    required = max(min_abs, target_fraction * target_pot_given_legal)
    passed = bool((not enabled) or observed >= required)
    return passed, {
        "enabled": bool(enabled),
        "required_choose_given_legal_pot_raise": float(required),
        "observed_choose_given_legal_pot_raise": float(observed),
        "solver_target_choose_given_legal_pot_raise": float(target_pot_given_legal),
    }


def _compute_exploit_weighted_score(random_stats: Dict[str, object], call_stats: Dict[str, object], pot_stats: Dict[str, object]):
    random_bb = float(random_stats.get("bb_per_100", 0.0))
    call_bb = float(call_stats.get("bb_per_100", 0.0))
    pot_bb = float(pot_stats.get("bb_per_100", 0.0))
    weighted = (0.25 * random_bb) + (0.35 * call_bb) + (0.40 * pot_bb)
    return {
        "random_bb100": random_bb,
        "always_call_bb100": call_bb,
        "pot_pressure_bb100": pot_bb,
        "weighted_bb100": float(weighted),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config_file", type=str, default="configs/training_configs.yaml")
    parser.add_argument("--config_name", type=str, default="quadro_stage_b")
    parser.add_argument("--episodes_per_seed", type=int, default=5000)
    parser.add_argument("--profile", type=str, choices=["quick", "standard", "exhaustive", "cert"], default="standard")
    parser.add_argument("--holdout_seed_base", type=int, default=10042)
    parser.add_argument("--holdout_seed_count", type=int, default=5)
    parser.add_argument("--require_robust_ci", action="store_true")
    parser.add_argument(
        "--require_behavior_extended",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Require behavior-extended gate in overall/pass_all. Defaults to config EVAL_REQUIRE_BEHAVIOR_EXTENDED.",
    )
    parser.add_argument(
        "--verify_solver_training",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found at '{args.checkpoint}'. "
            "Pass --checkpoint with a valid .pt file."
        )

    config = _load_config(args.config_file, args.config_name)
    strict_abstraction_env = os.environ.get("STRICT_ABSTRACTION")
    if strict_abstraction_env is not None:
        config.STRICT_ABSTRACTION = str(strict_abstraction_env).strip().lower() in {"1", "true", "yes", "on"}
    if args.seed is not None:
        config.SEED = int(args.seed)
    if hasattr(config, "sync_legacy_fields"):
        config.sync_legacy_fields()

    base_seed = int(config.SEED)
    _set_seed(base_seed)

    env = PokerEnv(
        game_name=config.GAME_NAME,
        env_preset=config.ENV_PRESET,
        betting_abstraction=config.BETTING_ABSTRACTION,
        strict_abstraction=bool(getattr(config, "STRICT_ABSTRACTION", True)),
    )
    requested_abstraction = env.get_requested_betting_abstraction()
    effective_abstraction = env.get_effective_betting_abstraction()
    strict_abstraction = bool(getattr(config, "STRICT_ABSTRACTION", True))
    if strict_abstraction and requested_abstraction != effective_abstraction:
        raise RuntimeError(
            "Strict abstraction mode detected mismatch: "
            f"requested={requested_abstraction}, effective={effective_abstraction}."
        )
    config.BETTING_ABSTRACTION = effective_abstraction
    num_actions = env.num_actions()

    target_device = torch.device(config.DEVICE)
    model = AlphaHoldemNetwork(num_actions, config).to(target_device)
    load_checkpoint(args.checkpoint, model, map_location=target_device)
    model.eval()

    evaluator = Evaluator(model, config=config, device=str(target_device))
    tiers = build_profile_tiers(args.profile, args.episodes_per_seed, config.BETTING_ABSTRACTION)

    print(f"Evaluator device: {target_device}")
    print(
        f"Running complete eval profile='{args.profile}' with "
        f"{len(tiers)} solver tier(s), seed base={base_seed}"
    )

    random_stats = evaluator.evaluate(
        RandomAgent(),
        num_episodes=int(args.episodes_per_seed),
        seed=base_seed,
        collect_diagnostics=True,
    )
    call_stats = evaluator.evaluate(
        AlwaysCallAgent(),
        num_episodes=int(args.episodes_per_seed),
        seed=base_seed + 1,
        collect_diagnostics=True,
    )
    pot_pressure_stats = evaluator.evaluate(
        PotPressureAgent(),
        num_episodes=int(args.episodes_per_seed),
        seed=base_seed + 2,
        collect_diagnostics=True,
    )
    print(
        f"Random BB/100={random_stats['bb_per_100']:.3f} | "
        f"AlwaysCall BB/100={call_stats['bb_per_100']:.3f} | "
        f"PotPressure BB/100={pot_pressure_stats['bb_per_100']:.3f}"
    )

    exploit_summary = _compute_exploit_weighted_score(random_stats, call_stats, pot_pressure_stats)

    control_episodes = min(int(args.episodes_per_seed), 2000)
    control_zero_iter = evaluator.evaluate_vs_solver_baseline(
        baseline_algo=config.EVAL_BASELINE_ALGO,
        iterations=0,
        num_episodes=control_episodes,
        seed=base_seed,
        collect_diagnostics=True,
    )
    print(
        f"Control baseline (0-iter MCCFR-ES): bb/100={control_zero_iter['bb_per_100']:.3f}, "
        f"build_s={control_zero_iter['baseline/build_seconds']:.3f}"
    )

    solver_tiers = []
    for tier in tiers:
        per_seed = []
        iterations = int(tier["iterations"])
        seeds = int(tier["seeds"])
        tier_eps = int(tier["episodes_per_seed"])

        print(
            f"Tier '{tier['name']}': baseline={config.EVAL_BASELINE_ALGO}, "
            f"iters={iterations}, seeds={seeds}, episodes/seed={tier_eps}"
        )

        for offset in range(seeds):
            eval_seed = base_seed + offset
            stats = evaluator.evaluate_vs_solver_baseline(
                baseline_algo=config.EVAL_BASELINE_ALGO,
                iterations=iterations,
                num_episodes=tier_eps,
                seed=eval_seed,
                collect_diagnostics=True,
            )
            per_seed.append(stats)
            print(
                f"  seed={eval_seed}: bb/100={stats['bb_per_100']:.3f}, "
                f"stderr={stats['std_err']:.3f}, build_s={stats['baseline/build_seconds']:.3f}"
            )

        bb_values = [row["bb_per_100"] for row in per_seed]
        agg = _aggregate(bb_values)
        ci95_lower = float(agg["mean"] - agg["ci95"])
        pass_gate = bool(ci95_lower > 0.0)
        diag = _aggregate_diagnostics(per_seed)
        tier_row = {
            "name": tier["name"],
            "iterations": iterations,
            "seeds": seeds,
            "episodes_per_seed": tier_eps,
            "per_seed": per_seed,
            "aggregate": {"bb_per_100": agg},
            "ci95_lower_bb100": ci95_lower,
            "pass_gate": pass_gate,
            **diag,
        }
        solver_tiers.append(tier_row)
        print(
            f"  aggregate: mean={agg['mean']:.3f}, stderr={agg['stderr']:.3f}, "
            f"ci95=+/-{agg['ci95']:.3f}, lower={ci95_lower:.3f}, pass={pass_gate}"
        )

        # Free solver policy memory before building next (larger) tier.
        evaluator._baseline_cache.clear()
        gc.collect()

    solver_training_verified, control_delta_bb100, verification_errors = _compute_solver_verification(
        solver_tiers=solver_tiers,
        control_bb100=float(control_zero_iter["bb_per_100"]),
    )

    default_tier = solver_tiers[0] if solver_tiers else None
    robustness_tier = solver_tiers[1] if len(solver_tiers) > 1 else default_tier
    pass_default_cfr_gate = bool(default_tier is not None and default_tier["ci95_lower_bb100"] > 0.0)
    pass_robustness_gate = _passes_robustness_gate(
        robustness_tier=robustness_tier,
        require_robust_ci=bool(args.require_robust_ci),
    )

    holdout_iterations = int(default_tier["iterations"]) if default_tier is not None else int(config.EVAL_BASELINE_ITERS)
    holdout_eps = int(default_tier["episodes_per_seed"]) if default_tier is not None else int(args.episodes_per_seed)
    holdout_seed_base = int(args.holdout_seed_base)
    holdout_seed_count = max(1, int(args.holdout_seed_count))
    primary_seeds = _primary_eval_seed_set(base_seed=base_seed, tiers=tiers)
    holdout_seeds = {holdout_seed_base + i for i in range(holdout_seed_count)}
    if primary_seeds.intersection(holdout_seeds):
        raise ValueError(
            "Holdout seeds overlap with primary evaluation seeds. "
            f"primary={sorted(primary_seeds)} holdout={sorted(holdout_seeds)}. "
            "Use a different --holdout_seed_base to keep the holdout gate independent."
        )

    holdout_rows = []
    for offset in range(holdout_seed_count):
        seed = holdout_seed_base + offset
        stats = evaluator.evaluate_vs_solver_baseline(
            baseline_algo=config.EVAL_BASELINE_ALGO,
            iterations=holdout_iterations,
            num_episodes=holdout_eps,
            seed=seed,
            collect_diagnostics=True,
        )
        holdout_rows.append(stats)

    holdout_agg = _aggregate([row["bb_per_100"] for row in holdout_rows])
    holdout_ci95_lower = float(holdout_agg["mean"] - holdout_agg["ci95"])
    holdout_diag = _aggregate_diagnostics(holdout_rows)
    pass_holdout_gate = bool(holdout_ci95_lower > 0.0)
    behavior_row = default_tier if default_tier is not None else holdout_diag
    pass_behavior_gate = _passes_behavior_gate(config, behavior_row)
    pass_behavior_gate_extended = _passes_behavior_gate_extended(config, behavior_row)
    pass_aggression_gate, aggression_gate_row = _passes_aggression_gate(config, behavior_row)
    pass_behavior_envelope, behavior_envelope_row = _passes_behavior_envelope(config, behavior_row)
    if args.require_behavior_extended is None:
        require_behavior_extended = bool(getattr(config, "EVAL_REQUIRE_BEHAVIOR_EXTENDED", True))
    else:
        require_behavior_extended = bool(args.require_behavior_extended)

    verification_ok = bool((not args.verify_solver_training) or solver_training_verified)
    exploit_weight = float(getattr(config, "EVAL_EXPLOIT_WEIGHT", 0.20))
    exploit_clip = float(getattr(config, "EVAL_EXPLOIT_CLIP_BB100", 500.0))
    clipped_exploit = float(np.clip(exploit_summary["weighted_bb100"], -abs(exploit_clip), abs(exploit_clip)))
    winner_score = float(default_tier["ci95_lower_bb100"] + (exploit_weight * clipped_exploit)) if default_tier else 0.0
    pass_all = bool(
        pass_default_cfr_gate
        and pass_robustness_gate
        and pass_holdout_gate
        and pass_behavior_gate
        and ((not require_behavior_extended) or pass_behavior_gate_extended)
        and pass_aggression_gate
        and pass_behavior_envelope
        and verification_ok
    )

    print(
        f"Overall gates: default_cfr={pass_default_cfr_gate}, "
        f"robustness={pass_robustness_gate}, holdout={pass_holdout_gate}, "
        f"behavior={pass_behavior_gate}, behavior_extended={pass_behavior_gate_extended}, "
        f"aggression={pass_aggression_gate}, envelope={pass_behavior_envelope}, "
        f"require_behavior_extended={require_behavior_extended}, "
        f"verified={solver_training_verified}, pass_all={pass_all}"
    )

    report = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "checkpoint": args.checkpoint,
        "device": str(target_device),
        "config_name": args.config_name,
        "profile": args.profile,
        "seed_base": base_seed,
        "meta/requested_betting_abstraction": str(requested_abstraction),
        "meta/effective_betting_abstraction": str(effective_abstraction),
        "meta/strict_abstraction": bool(strict_abstraction),
        "summary/random": random_stats,
        "summary/always_call": call_stats,
        "summary/pot_pressure": pot_pressure_stats,
        "summary/exploit_weighted": exploit_summary,
        "solver_tiers": solver_tiers,
        "verification/solver_training_verified": bool(solver_training_verified),
        "verification/control_zero_iter": control_zero_iter,
        "verification/control_delta_bb100": control_delta_bb100,
        "verification/errors": verification_errors,
        "holdout": {
            "seed_base": holdout_seed_base,
            "seeds": holdout_seed_count,
            "iterations": holdout_iterations,
            "episodes_per_seed": holdout_eps,
            "per_seed": holdout_rows,
            "aggregate": {"bb_per_100": holdout_agg},
            "ci95_lower_bb100": holdout_ci95_lower,
            **holdout_diag,
        },
        "diagnostics/summary": {
            "random": _diag_from_metrics(random_stats),
            "always_call": _diag_from_metrics(call_stats),
            "solver_tiers": {
                row["name"]: _diag_from_metrics(row) for row in solver_tiers
            },
            "holdout": _diag_from_metrics({**holdout_diag}),
        },
        "overall/pass_default_cfr_gate": pass_default_cfr_gate,
        "overall/pass_robustness_gate": pass_robustness_gate,
        "overall/pass_holdout_gate": pass_holdout_gate,
        "overall/pass_behavior_gate": pass_behavior_gate,
        "overall/pass_behavior_gate_extended": pass_behavior_gate_extended,
        "overall/pass_aggression_gate": pass_aggression_gate,
        "overall/pass_behavior_envelope": pass_behavior_envelope,
        # Backwards-compatible alias retained for older scripts.
        "overall/pass_pot_mix_gate": pass_aggression_gate,
        "overall/require_behavior_extended": bool(require_behavior_extended),
        "overall/exploit_weighted_bb100": float(exploit_summary["weighted_bb100"]),
        "overall/winner_score": float(winner_score),
        "overall/pass_all": pass_all,
        "behavior_gate": {
            "enabled": bool(getattr(config, "BEHAVIOR_GATE_ENABLE", True)),
            "max_fold_freq": float(getattr(config, "BEHAVIOR_GATE_MAX_FOLD_FREQ", 0.78)),
            "max_allin_freq": float(getattr(config, "BEHAVIOR_GATE_MAX_ALLIN_FREQ", 0.06)),
            "min_preflop_entropy_bits": float(getattr(config, "BEHAVIOR_GATE_MIN_PRE_FLOP_ENTROPY_BITS", 0.75)),
            "observed": {
                "fold_freq": float(behavior_row.get("diagnostics/action_freq_preflop/fold", 0.0)),
                "allin_freq": float(behavior_row.get("diagnostics/action_freq_preflop/allin", 0.0)),
                "preflop_entropy_bits": float(behavior_row.get("diagnostics/preflop_action_entropy_bits", 0.0)),
            },
        },
        "behavior_gate_extended": {
            "enabled": bool(getattr(config, "BEHAVIOR_GATE_ENABLE", True)),
            "min_call_check_freq": float(getattr(config, "BEHAVIOR_GATE_MIN_CALL_CHECK_FREQ", 0.50)),
            "min_half_pot_freq": float(getattr(config, "BEHAVIOR_GATE_MIN_HALF_POT_FREQ", 0.01)),
            "observed_call_check_freq": float(
                behavior_row.get("diagnostics/action_freq_preflop/call_check", 0.0)
            ),
            "observed_half_pot_freq": float(
                behavior_row.get("diagnostics/action_freq_preflop/half_pot", 0.0)
            ),
            "pass": bool(pass_behavior_gate_extended),
        },
        "aggression_gate": aggression_gate_row,
        "behavior_envelope": behavior_envelope_row,
        # Backwards-compatible alias retained for older scripts.
        "pot_mix_gate": aggression_gate_row,
        "selection_score": {
            "default_ci95_lower_bb100": float(default_tier["ci95_lower_bb100"]) if default_tier else 0.0,
            "exploit_weight": float(exploit_weight),
            "exploit_clip_bb100": float(exploit_clip),
            "clipped_exploit_bb100": float(clipped_exploit),
            "winner_score": float(winner_score),
        },
    }

    # -- Hand-strength correlation analysis --
    try:
        from poker_rl_agent.evaluation.baseline_agents import RandomAgent as _HSRandom
        hs_opponent = _HSRandom()
        hs_result = evaluator.evaluate_hand_strength_correlation(
            opponent=hs_opponent,
            num_episodes=2000,
            seed=base_seed + 100,
            num_bins=5,
        )
        report["hand_strength_correlation"] = hs_result
        print("\n--- Hand-Strength Correlation (preflop) ---")
        print(f"{'Bin':>12s} {'Count':>7s} {'Fold':>7s} {'Call':>7s} {'HalfP':>7s} {'PotR':>7s} {'AllIn':>7s}")
        for row in hs_result["hand_strength_table"]:
            print(
                f"{row['equity_bin']:>12s} {row['count']:>7d} "
                f"{row['freq_fold']:>7.3f} {row['freq_call_check']:>7.3f} "
                f"{row['freq_half_pot']:>7.3f} {row['freq_pot_raise']:>7.3f} "
                f"{row['freq_allin']:>7.3f}"
            )
    except Exception as exc:
        print(f"Hand-strength correlation skipped: {exc}")

    # -- W&B cert curve logging --
    if args.profile == "cert" and solver_tiers:
        try:
            import wandb as _wandb
            wandb_mode = os.environ.get("WANDB_MODE", "disabled")
            if wandb_mode != "disabled":
                _wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "alpha-holdem-poker"),
                    tags=["eval", "cert"],
                    reinit=True,
                )
                columns = ["mccfr_iterations", "bb_per_100_mean", "bb_per_100_stderr", "ci95_lower"]
                table_data = []
                for tier_row in solver_tiers:
                    tier_agg = tier_row.get("aggregate", {}).get("bb_per_100", {})
                    iters = int(tier_row.get("iterations", 0))
                    mean_bb = float(tier_agg.get("mean", 0.0))
                    stderr_bb = float(tier_agg.get("stderr", 0.0))
                    ci_lower = float(tier_row.get("ci95_lower_bb100", 0.0))
                    table_data.append([iters, mean_bb, stderr_bb, ci_lower])
                    _wandb.log({f"eval/bb100_mccfr_{iters}": mean_bb})
                wt = _wandb.Table(columns=columns, data=table_data)
                _wandb.log({"eval/winrate_vs_cfr_curve": wt})

                # Log hand-strength table if available
                if "hand_strength_correlation" in report:
                    hs_cols = ["equity_bin", "count", "freq_fold", "freq_call_check", "freq_half_pot", "freq_pot_raise", "freq_allin"]
                    hs_data = [
                        [r.get(c, 0) for c in hs_cols]
                        for r in report["hand_strength_correlation"]["hand_strength_table"]
                    ]
                    hs_table = _wandb.Table(columns=hs_cols, data=hs_data)
                    _wandb.log({"eval/hand_strength_correlation": hs_table})

                _wandb.finish()
                print("W&B cert curve logged.")
        except Exception as exc:
            print(f"W&B cert logging skipped: {exc}")

    output_json = args.output_json
    if not output_json:
        run_id = getattr(config, "RUN_ID", "") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_json = os.path.join("logs", f"eval_complete_{run_id}.json")
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Wrote complete evaluation report: {output_json}")


if __name__ == "__main__":
    main()
