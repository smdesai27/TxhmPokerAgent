#!/usr/bin/env python3
"""Builds a solver-derived preflop style target profile JSON."""

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
from poker_rl_agent.evaluation.evaluator import Evaluator
from poker_rl_agent.models.alpha_holdem_net import AlphaHoldemNetwork
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/training_configs.yaml")
    parser.add_argument("--config_name", type=str, default="quadro_stage_d_fchpa_recover_conservative_probe_19k")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--episodes_per_seed", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", type=str, default="")
    args = parser.parse_args()

    config = _load_config(args.config_file, args.config_name)
    config.SEED = int(args.seed)
    _set_seed(config.SEED)

    env = PokerEnv(
        game_name=config.GAME_NAME,
        env_preset=config.ENV_PRESET,
        betting_abstraction=config.BETTING_ABSTRACTION,
        strict_abstraction=bool(getattr(config, "STRICT_ABSTRACTION", True)),
    )
    config.BETTING_ABSTRACTION = env.get_effective_betting_abstraction()
    num_actions = env.num_actions()

    model = AlphaHoldemNetwork(num_actions, config).to(torch.device("cpu"))
    model.eval()
    evaluator = Evaluator(model, config=config, device="cpu")

    profile = evaluator.build_solver_style_profile(
        iterations=int(args.iterations),
        seeds=int(args.seeds),
        episodes_per_seed=int(args.episodes_per_seed),
        baseline_algo=getattr(config, "EVAL_BASELINE_ALGO", "mccfr_external_sampling"),
        seed_base=int(config.SEED),
    )

    out_path = args.output_json
    if not out_path:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join("logs", "stage_d", "targets", f"solver_style_target_{stamp}.json")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "config_name": args.config_name,
        "config_file": args.config_file,
        "meta/requested_betting_abstraction": env.get_requested_betting_abstraction(),
        "meta/effective_betting_abstraction": env.get_effective_betting_abstraction(),
        **profile,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    target = payload.get("target", {})
    print(f"Wrote solver style target profile: {out_path}")
    print(
        "Target preflop mix: "
        f"fold={float(target.get('fold', 0.0)):.4f}, "
        f"call={float(target.get('call_check', 0.0)):.4f}, "
        f"half={float(target.get('half_pot', 0.0)):.4f}, "
        f"pot={float(target.get('pot_raise', 0.0)):.4f}, "
        f"allin={float(target.get('allin', 0.0)):.4f}"
    )
    print(
        "Legal-conditioned targets: "
        f"pot|legal={float(payload.get('target/preflop_choose_given_legal/pot_raise', 0.0)):.4f}, "
        f"half|legal={float(payload.get('target/preflop_choose_given_legal/half_pot', 0.0)):.4f}"
    )


if __name__ == "__main__":
    main()
