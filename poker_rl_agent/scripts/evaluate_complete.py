import argparse
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
from poker_rl_agent.evaluation.baseline_agents import AlwaysCallAgent, RandomAgent
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


def _aggregate(values):
    arr = np.array(values, dtype=np.float64)
    mean = float(arr.mean()) if arr.size else 0.0
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    stderr = float(std / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    ci95 = float(1.96 * stderr)
    return {"mean": mean, "std": std, "stderr": stderr, "ci95": ci95, "n": int(arr.size)}


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
        "diagnostics/showdown_rate": float(showdown_hands) / float(hand_denom),
        "diagnostics/avg_pot_size_bb": float(total_terminal_pot_bb) / float(hand_denom),
        "diagnostics/_preflop_total_count": int(preflop_total),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config_file", type=str, default="configs/training_configs.yaml")
    parser.add_argument("--config_name", type=str, default="quadro_stage_b")
    parser.add_argument("--episodes_per_seed", type=int, default=5000)
    parser.add_argument("--profile", type=str, choices=["quick", "standard", "exhaustive"], default="standard")
    parser.add_argument("--holdout_seed_base", type=int, default=10042)
    parser.add_argument("--holdout_seed_count", type=int, default=5)
    parser.add_argument("--require_robust_ci", action="store_true")
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
    print(
        f"Random BB/100={random_stats['bb_per_100']:.3f} | "
        f"AlwaysCall BB/100={call_stats['bb_per_100']:.3f}"
    )

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

    verification_ok = bool((not args.verify_solver_training) or solver_training_verified)
    pass_all = bool(
        pass_default_cfr_gate
        and pass_robustness_gate
        and pass_holdout_gate
        and pass_behavior_gate
        and verification_ok
    )

    print(
        f"Overall gates: default_cfr={pass_default_cfr_gate}, "
        f"robustness={pass_robustness_gate}, holdout={pass_holdout_gate}, "
        f"behavior={pass_behavior_gate}, behavior_extended={pass_behavior_gate_extended}, "
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
    }

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
