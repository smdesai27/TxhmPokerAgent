import argparse
import json
import os
import random
import sys
from datetime import datetime

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=100, help="Backwards-compatible alias for episodes per seed")
    parser.add_argument("--episodes_per_seed", type=int, default=None)
    parser.add_argument("--baseline_algo", type=str, default=None)
    parser.add_argument("--baseline_iters", type=int, default=None)
    parser.add_argument("--baseline_seeds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--config_file", type=str, default="configs/training_configs.yaml")
    parser.add_argument("--config_name", type=str, default="default")
    args = parser.parse_args()

    config = _load_config(args.config_file, args.config_name)
    strict_abstraction_env = os.environ.get("STRICT_ABSTRACTION")
    if strict_abstraction_env is not None:
        config.STRICT_ABSTRACTION = str(strict_abstraction_env).strip().lower() in {"1", "true", "yes", "on"}
    if args.baseline_algo is not None:
        config.EVAL_BASELINE_ALGO = args.baseline_algo
    if args.baseline_iters is not None:
        config.EVAL_BASELINE_ITERS = int(args.baseline_iters)
    if args.baseline_seeds is not None:
        config.EVAL_BASELINE_SEEDS = int(args.baseline_seeds)
    if args.seed is not None:
        config.SEED = int(args.seed)
    if hasattr(config, "sync_legacy_fields"):
        config.sync_legacy_fields()

    episodes_per_seed = int(args.episodes_per_seed or args.episodes)
    baseline_seed_count = max(1, int(config.EVAL_BASELINE_SEEDS))
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

    print(f"Evaluator device: {target_device}")
    evaluator = Evaluator(model, config=config, device=str(target_device))

    print(f"Evaluating against Random Agent for {episodes_per_seed} episodes...")
    random_stats = evaluator.evaluate(RandomAgent(), num_episodes=episodes_per_seed, seed=base_seed)
    print(f"Average Return vs Random: {random_stats['avg_return']:.3f}")
    print(f"BB/100 vs Random: {random_stats['bb_per_100']:.3f}")

    print(f"Evaluating against Always Call Agent for {episodes_per_seed} episodes...")
    call_stats = evaluator.evaluate(AlwaysCallAgent(), num_episodes=episodes_per_seed, seed=base_seed + 1)
    print(f"Average Return vs AlwaysCall: {call_stats['avg_return']:.3f}")
    print(f"BB/100 vs AlwaysCall: {call_stats['bb_per_100']:.3f}")

    print(
        f"Evaluating against {config.EVAL_BASELINE_LABEL} solver baseline "
        f"({config.EVAL_BASELINE_ALGO}, {config.EVAL_BASELINE_ITERS} iterations) "
        f"for {baseline_seed_count} seed(s)..."
    )

    baseline_per_seed = []
    for offset in range(baseline_seed_count):
        eval_seed = base_seed + offset
        stats = evaluator.evaluate_vs_solver_baseline(
            baseline_algo=config.EVAL_BASELINE_ALGO,
            iterations=config.EVAL_BASELINE_ITERS,
            num_episodes=episodes_per_seed,
            seed=eval_seed,
        )
        baseline_per_seed.append(stats)
        print(
            f"Seed {eval_seed}: BB/100={stats['bb_per_100']:.3f} "
            f"(stderr {stats['std_err']:.3f}), build_s={stats['baseline/build_seconds']:.3f}, "
            f"cache_hit={stats['baseline/cache_hit']}"
        )

    baseline_bb = [row["bb_per_100"] for row in baseline_per_seed]
    baseline_return = [row["avg_return"] for row in baseline_per_seed]
    baseline_summary = {
        "bb_per_100": _aggregate(baseline_bb),
        "avg_return": _aggregate(baseline_return),
    }
    ci95_lower_bb100 = baseline_summary["bb_per_100"]["mean"] - baseline_summary["bb_per_100"]["ci95"]
    ci95_upper_bb100 = baseline_summary["bb_per_100"]["mean"] + baseline_summary["bb_per_100"]["ci95"]
    pass_primary_gate = bool(ci95_lower_bb100 > 0.0)
    print(
        f"Aggregate vs {config.EVAL_BASELINE_LABEL}: "
        f"BB/100 mean={baseline_summary['bb_per_100']['mean']:.3f}, "
        f"stderr={baseline_summary['bb_per_100']['stderr']:.3f}, "
        f"95% CI +/- {baseline_summary['bb_per_100']['ci95']:.3f}, "
        f"lower={ci95_lower_bb100:.3f}, pass={pass_primary_gate}"
    )

    report = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "checkpoint": args.checkpoint,
        "device": str(target_device),
        "config_name": args.config_name,
        "seed_base": base_seed,
        "meta/requested_betting_abstraction": str(requested_abstraction),
        "meta/effective_betting_abstraction": str(effective_abstraction),
        "meta/strict_abstraction": bool(strict_abstraction),
        "episodes_per_seed": episodes_per_seed,
        "random": random_stats,
        "always_call": call_stats,
        "solver_baseline": {
            "label": config.EVAL_BASELINE_LABEL,
            "algo": config.EVAL_BASELINE_ALGO,
            "iterations": int(config.EVAL_BASELINE_ITERS),
            "num_seeds": baseline_seed_count,
            "per_seed": baseline_per_seed,
            "aggregate": baseline_summary,
        },
        "acceptance/ci95_lower_bb100": float(ci95_lower_bb100),
        "acceptance/ci95_upper_bb100": float(ci95_upper_bb100),
        "acceptance/pass_primary_gate": bool(pass_primary_gate),
    }

    output_json = args.output_json
    if not output_json:
        run_id = getattr(config, "RUN_ID", "") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_json = os.path.join("logs", f"eval_{run_id}.json")
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Wrote evaluation summary: {output_json}")


if __name__ == "__main__":
    main()
