"""Stage 0: own-checkpoint ladder evaluation (checkpoint-vs-checkpoint).

WHY THIS EXISTS
---------------
The headline "+897 bb/100 vs MCCFR-ES" is NOT a strength signal: the MCCFR-ES
baseline at the iteration counts used is effectively near-random, so beating it
by a huge margin says little about how strong the agent actually is. This script
produces the REAL strength signal: a confidence-interval-backed,
checkpoint-vs-checkpoint result on identical betting abstractions, reusing the
already-validated variance-reduced ``Evaluator.evaluate_duplicate`` (mirrored
duplicate hands + Student-t CI).

Checkpoint A is loaded as the Evaluator's agent model (it samples its policy
exactly as in live eval). Checkpoint B is loaded as a *deterministic*
``ModelPolicyAgent`` opponent (argmax over masked logits) so the duplicate-hand
card-luck cancellation actually holds. We then run ``evaluate_duplicate`` and
report A's bb/100 vs B with a 95% CI and the sample size (pairs).

EVERY reported number carries a CI and a sample size. We never label the
MCCFR-ES "+897" number as "strength".

CORRECTNESS ORACLE (self-play sanity check)
-------------------------------------------
Running with ``--checkpoint_a == --checkpoint_b`` (the same file) MUST yield
``bb_per_100 ~= 0`` with a 95% CI that straddles 0. Heads-up poker is zero-sum,
so a policy playing against a copy of itself has zero expected value; the
duplicate-hand pairing drives the empirical mean tightly toward 0. A run where
A==B but the CI excludes 0 by a wide margin indicates a bug (e.g. seat-dependent
encoding, nondeterministic opponent, or a stale model handle). State this in any
review of the output.

HARD GUARD
----------
Both checkpoints must share the same action count / betting abstraction. FCPA
(4 actions) vs FCHPA (5 actions) would shape-mismatch the legal-action mask
inside ``masked_logits`` and silently corrupt results, so we fail fast with a
clear error before any hands are played. The action count is derived from the
*config* used to construct each model (the abstraction baked into the
checkpoint), not guessed.

USAGE
-----
    python -m poker_rl_agent.scripts.evaluate_own_ladder \
        --checkpoint_a A.pt --checkpoint_b B.pt \
        --config_file configs/training_configs.yaml \
        --config_name <preset> \
        --num_pairs 5000 --seed 42 --output_json out.json

``--config_name`` describes the abstraction/game for BOTH checkpoints (they must
match to be comparable). If you need different configs per checkpoint, pass
``--config_name_b``; the guard still requires the resulting action counts to be
equal.
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from poker_rl_agent.environment.openspiel_wrapper import PokerEnv
from poker_rl_agent.evaluation.evaluator import Evaluator
from poker_rl_agent.evaluation.model_policy_adapter import ModelPolicyAgent
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
    """Same resolution order as evaluate_complete.py: defaults -> yaml default ->
    named preset. Kept identical so the abstraction matches what trained the ckpt."""
    config = Config()
    if config_file and os.path.exists(config_file):
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


def _resolve_abstraction(config):
    """Construct a PokerEnv to resolve the effective abstraction + action count,
    failing fast under STRICT_ABSTRACTION if requested != effective (same policy
    as evaluate_complete.py)."""
    env = PokerEnv(
        game_name=config.GAME_NAME,
        env_preset=config.ENV_PRESET,
        betting_abstraction=config.BETTING_ABSTRACTION,
        strict_abstraction=bool(getattr(config, "STRICT_ABSTRACTION", True)),
    )
    requested = env.get_requested_betting_abstraction()
    effective = env.get_effective_betting_abstraction()
    strict = bool(getattr(config, "STRICT_ABSTRACTION", True))
    if strict and requested != effective:
        raise RuntimeError(
            "Strict abstraction mode detected mismatch: "
            f"requested={requested}, effective={effective}."
        )
    config.BETTING_ABSTRACTION = effective
    return env, effective, int(env.num_actions())


def _load_model(checkpoint_path, num_actions, config, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. Pass a valid .pt file."
        )
    model = AlphaHoldemNetwork(num_actions, config).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Own-checkpoint ladder: A (agent) vs B (deterministic opponent) bb/100 with CI."
    )
    parser.add_argument("--checkpoint_a", type=str, required=True, help="Agent checkpoint (.pt).")
    parser.add_argument("--checkpoint_b", type=str, required=True, help="Opponent checkpoint (.pt).")
    parser.add_argument("--config_file", type=str, default="configs/training_configs.yaml")
    parser.add_argument("--config_name", type=str, required=True, help="Preset describing the abstraction for BOTH checkpoints.")
    parser.add_argument(
        "--config_name_b",
        type=str,
        default=None,
        help="Optional separate preset for checkpoint_b. Action counts must still match.",
    )
    parser.add_argument("--num_pairs", type=int, default=5000, help="Duplicate-hand pairs (each = 2 mirrored hands).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    _set_seed(int(args.seed))

    # Load config(s). STRICT_ABSTRACTION may be overridden via env, matching the
    # rest of the eval tooling for experiment integrity.
    config_a = _load_config(args.config_file, args.config_name)
    config_b = _load_config(args.config_file, args.config_name_b or args.config_name)
    strict_env = os.environ.get("STRICT_ABSTRACTION")
    if strict_env is not None:
        flag = str(strict_env).strip().lower() in {"1", "true", "yes", "on"}
        config_a.STRICT_ABSTRACTION = flag
        config_b.STRICT_ABSTRACTION = flag
    config_a.SEED = int(args.seed)
    config_b.SEED = int(args.seed)
    if hasattr(config_a, "sync_legacy_fields"):
        config_a.sync_legacy_fields()
    if hasattr(config_b, "sync_legacy_fields"):
        config_b.sync_legacy_fields()

    device = torch.device(config_a.DEVICE)

    # Resolve abstractions + action counts for both sides.
    _, abstraction_a, num_actions_a = _resolve_abstraction(config_a)
    _, abstraction_b, num_actions_b = _resolve_abstraction(config_b)

    # --- HARD GUARD: identical action count / betting abstraction. ---
    # FCPA(4) vs FCHPA(5) would shape-mismatch masked_logits against the legal
    # mask and silently corrupt the comparison. Fail fast, loudly.
    if num_actions_a != num_actions_b:
        raise ValueError(
            "Checkpoint abstraction mismatch -- refusing to compare incomparable agents.\n"
            f"  checkpoint_a: config='{args.config_name}', abstraction='{abstraction_a}', num_actions={num_actions_a}\n"
            f"  checkpoint_b: config='{args.config_name_b or args.config_name}', abstraction='{abstraction_b}', num_actions={num_actions_b}\n"
            "Both checkpoints must use the same betting abstraction (e.g. both FCHPA=5 or both FCPA=4). "
            "Pass matching --config_name / --config_name_b."
        )

    num_actions = num_actions_a

    print(f"Device: {device}")
    print(
        f"Abstraction (both): '{abstraction_a}' | num_actions={num_actions} | "
        f"num_pairs={args.num_pairs} ({2 * int(args.num_pairs)} hands) | seed={args.seed}"
    )

    # Load A as the agent, B as the deterministic opponent.
    model_a = _load_model(args.checkpoint_a, num_actions, config_a, device)
    model_b = _load_model(args.checkpoint_b, num_actions, config_b, device)

    evaluator = Evaluator(model_a, config=config_a, device=str(device))
    opponent = ModelPolicyAgent(
        model_b,
        num_actions=num_actions,
        device=device,
        max_action_history=int(getattr(config_b, "MAX_ACTION_HISTORY", 64)),
    )

    same_checkpoint = os.path.abspath(args.checkpoint_a) == os.path.abspath(args.checkpoint_b)
    if same_checkpoint:
        print(
            "ORACLE CHECK: checkpoint_a == checkpoint_b (self-play). "
            "Expect bb_per_100 ~= 0 with a CI straddling 0 (zero-sum)."
        )

    result = evaluator.evaluate_duplicate(opponent, num_pairs=int(args.num_pairs), seed=int(args.seed))

    bb_per_100 = float(result["bb_per_100"])
    ci95_lower = float(result["bb_per_100_ci95_lower"])
    ci95_upper = float(result["bb_per_100_ci95_upper"])
    pairs = int(result["pairs"])

    # Winner determination requires CI separation from 0; otherwise it's a tie
    # within noise. We NEVER call a within-noise result a win.
    if ci95_lower > 0.0:
        winner = "checkpoint_a"
    elif ci95_upper < 0.0:
        winner = "checkpoint_b"
    else:
        winner = "tie_within_ci"

    report = {
        "checkpoint_a": os.path.abspath(args.checkpoint_a),
        "checkpoint_b": os.path.abspath(args.checkpoint_b),
        "config_name": args.config_name,
        "config_name_b": args.config_name_b or args.config_name,
        "betting_abstraction": abstraction_a,
        "num_actions": num_actions,
        "seed": int(args.seed),
        "self_play_oracle": bool(same_checkpoint),
        "bb_per_100": bb_per_100,
        "ci95_lower": ci95_lower,
        "ci95_upper": ci95_upper,
        "bb_per_100_stderr": float(result["bb_per_100_stderr"]),
        "winner": winner,
        "pairs": pairs,
        "episodes": int(result["episodes"]),
        "method": str(result["method"]),
        "ci_method": str(result["ci_method"]),
        # Honesty note travels with the artifact so downstream readers can't be misled.
        "note": (
            "Real strength signal: checkpoint-vs-checkpoint, variance-reduced via "
            "mirrored duplicate hands, Student-t 95% CI over `pairs` per-pair means. "
            "NOT the MCCFR-ES '+897' number (that baseline is near-random and is not strength)."
        ),
    }

    print(
        f"A vs B: bb/100={bb_per_100:.3f}  CI95=[{ci95_lower:.3f}, {ci95_upper:.3f}]  "
        f"pairs={pairs}  winner={winner}"
    )
    if same_checkpoint:
        straddles_zero = ci95_lower <= 0.0 <= ci95_upper
        print(
            f"ORACLE: self-play CI {'STRADDLES 0 (PASS)' if straddles_zero else 'EXCLUDES 0 (INVESTIGATE)'}"
        )

    output_json = args.output_json
    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"Wrote ladder result: {output_json}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
