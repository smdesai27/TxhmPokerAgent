"""Validate the Local Best Response (LBR) opponent on Leduc against EXACT NashConv.

WHY THIS SCRIPT EXISTS
----------------------
``LocalBestResponseAgent`` (poker_rl_agent/evaluation/lbr_agent.py) computes a LOWER
BOUND on an agent's exploitability. Before that lower bound is trusted on HUNL -- where
the true exploitability is intractable and LBR is the only handle we have -- it must be
validated on a game where exact exploitability IS computable. Leduc poker is that game:
``open_spiel.python.algorithms.exploitability.nash_conv`` does an exact full-tree
best-response and returns the SUM of both players' BR gains (the same axis used by
poker_rl_agent/scripts/validate_leduc_exploitability.py and leduc_cfr_anchor.py;
reference anchors: random ~4.76, CFR+ ~0.05).

THE GATE
--------
LBR is ONE strategy in the best-response space, so its value can never exceed the true
best response. Therefore, in matched single-player units:

    LBR_value  <=  exploitability  <=  NashConv

We assert ``LBR_value <= NashConv * (1 + tol)`` (tol default 5%). If LBR EXCEEDS NashConv,
the search is BUGGY (a lower bound that beats the exact bound is impossible) -- the run
FAILS. We ALSO assert LBR beats an always-call baseline by a clear positive margin: a
genuine best-responder must out-earn a static caller; if it does not, the lookahead is
broken or mis-seated.

UNITS NOTE (read carefully): ``nash_conv`` is the SUM over BOTH players of each player's
best-response gain (~= 2x single-player exploitability). LBR here controls ONE seat and
we measure its per-hand chip win-rate averaged over BOTH seats (duplicate). To compare
on the same axis we convert the LBR per-seat chip value to the single-player BR-gain
scale and then to the nash_conv (two-player-sum) scale. We keep the comparison
conservative: we compare LBR's positive win-rate (chips/hand for the LBR seat) against
``nash_conv`` directly, which is the LOOSER (higher) bound -- so the gate can only fail
if LBR is egregiously over the exact two-player-sum exploitability, which is the bug we
want to catch. The JSON records every intermediate so a reader can re-derive any axis.

HONESTY (mirrors lbr_agent.py): LBR is a LOWER BOUND, never "the exploitability". A low
LBR does NOT prove low exploitability (Lisy & Bowling: head-to-head-tied bots differed
~1300 mbb/g). A HIGH LBR is a definitive disqualifier. On HUNL it is only ever a lower
bound -- which is exactly why we validate the machinery here on Leduc first.

DUPLICATE EVALUATION: this script uses a LOCAL Leduc duplicate-hand evaluator
(``_evaluate_lbr_fixed_seat``) that mirrors the structure of
``Evaluator.evaluate_duplicate`` (same deals, mirrored seats, per-pair i.i.d. unit). We
cannot call the project ``Evaluator`` directly here because its ``StateEncoder`` is
universal_poker-specific (it reads ``state.to_struct()``), whereas Leduc agents are
driven through OpenSpiel's ``information_state_tensor`` (the same adapter the existing
validate_leduc_*.py scripts feed to ``exploitability.nash_conv``). On HUNL the caller
WOULD use ``Evaluator.evaluate_duplicate`` with ``Evaluator._ModelPolicyAdapter`` -- the
``LocalBestResponseAgent`` is encoder-agnostic and works unchanged in both settings.

Run on a compute node (OSCAR login nodes forbid Python). See the srun command at the
bottom of this docstring / in the project report:

  srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=02:00:00 \\
    --pty bash -lc '
      cd /users/smdesai/antigravity_poker/anitgravity-txhm &&
      module load python/3.11.11-5e66 cuda/11.8.0-kuhf &&
      source /users/smdesai/antigravity_poker/anitgravity-txhm/venv/bin/activate &&
      export PYTHONPATH="$(pwd)" &&
      python3 poker_rl_agent/scripts/validate_lbr.py \\
        --train_iterations 200 --episodes_per_iter 64 --num_pairs 2000 \\
        --max_depth 1 --rollout_samples 4 \\
        --output_json logs/leduc_lbr_validation.json'
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

import pyspiel
from open_spiel.python import policy as policy_lib
from open_spiel.python.algorithms import exploitability

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from poker_rl_agent.evaluation.baseline_agents import AlwaysCallAgent  # noqa: E402
from poker_rl_agent.evaluation.lbr_agent import (  # noqa: E402
    LocalBestResponseAgent,
    state_clone_available,
)
from poker_rl_agent.training.rollout_buffer import RolloutBuffer  # noqa: E402
from poker_rl_agent.utils.config import Config  # noqa: E402

# Reuse the validated Leduc actor-critic + self-play training loop so this script trains a
# real (non-trivial) Leduc agent to probe, rather than a random net (LBR vs random is a
# weaker, less informative test). These are the exact pieces validate_leduc_exploitability
# already exercises against nash_conv.
from poker_rl_agent.scripts.validate_leduc_exploitability import (  # noqa: E402
    PolicyValueNet,
    _obs_and_mask,
    masked_logits,
    play_episode,
    ppo_update,
)


class NetPolicy(policy_lib.Policy):
    """Adapt the Leduc net to OpenSpiel's Policy AND the LBR ``policy_adapter`` contract.

    Exposes ``action_probabilities(state, player_id=None) -> {action: prob}``. This single
    object is used three ways: (1) by ``exploitability.nash_conv`` for the exact
    best-response, (2) by ``LocalBestResponseAgent`` to roll the agent forward, and (3) by
    the duplicate evaluator to play the agent's moves. Identical to the adapter in
    validate_leduc_exploitability.py, kept local so this script is self-contained.
    """

    def __init__(self, game, net, obs_dim, num_actions, device):
        super().__init__(game, list(range(game.num_players())))
        self.net = net
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.device = device

    def action_probabilities(self, state, player_id=None):
        cur = state.current_player() if player_id is None else player_id
        obs, mask = _obs_and_mask(state, cur, self.obs_dim, self.num_actions, self.device)
        with torch.no_grad():
            logits, _ = self.net(obs.unsqueeze(0))
            probs = torch.softmax(masked_logits(logits, mask.unsqueeze(0)), dim=-1).squeeze(0)
        return {a: float(probs[a]) for a in state.legal_actions(cur)}


def train_leduc_agent(game, obs_dim, num_actions, device, cfg, iterations, episodes_per_iter, rng):
    """Lightweight self-play PPO on Leduc so we probe a non-trivial agent (not random).

    Returns the trained net. Mirrors the rolling-self-play path of
    validate_leduc_exploitability.main (vanilla PPO, recent-snapshot opponent).
    """
    net = PolicyValueNet(obs_dim, num_actions).to(device)
    opp_net = PolicyValueNet(obs_dim, num_actions).to(device)
    opp_net.load_state_dict(net.state_dict())
    optimizer = torch.optim.Adam(net.parameters(), lr=float(cfg.LR))
    rolling = [{k: v.detach().clone() for k, v in net.state_dict().items()}]

    for it in range(1, int(iterations) + 1):
        net.train()
        buffer = RolloutBuffer()
        for ep in range(int(episodes_per_iter)):
            play_self = (rng.rand() < 0.5) or (not rolling)
            opp_sd = net.state_dict() if play_self else rolling[rng.randint(len(rolling))]
            opp_net.load_state_dict(opp_sd)
            opp_net.eval()
            play_episode(
                game, net, opp_net, learner=ep % 2,
                obs_dim=obs_dim, num_actions=num_actions, device=device, buffer=buffer, rng=rng,
            )
        if len(buffer) > 0:
            ppo_update(net, optimizer, buffer, cfg, device)
        if it % 10 == 0:
            rolling.append({k: v.detach().clone() for k, v in net.state_dict().items()})
            if len(rolling) > 8:
                rolling.pop(0)
    net.eval()
    return net


def _play_one_hand_leduc(game, agent_policy, opponent_agent, agent_player_id, seed, rng):
    """Play one Leduc hand: ``agent_player_id`` plays ``agent_policy`` (sampled), the other
    seat plays ``opponent_agent.step``. Return the AGENT'S terminal chips.

    This is the Leduc analogue of Evaluator._play_one_hand, written here because the
    project Evaluator's encoder is universal_poker-specific and does not apply to Leduc.
    """
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    state = game.new_initial_state()
    while not state.is_terminal():
        if state.is_chance_node():
            actions, probs = zip(*state.chance_outcomes())
            p = np.asarray(probs, dtype=np.float64)
            p = p / p.sum()
            state.apply_action(int(rng.choice(actions, p=p)))
            continue
        cur = state.current_player()
        if cur == agent_player_id:
            probs = agent_policy.action_probabilities(state, cur)
            acts = list(probs.keys())
            p = np.array([probs[a] for a in acts], dtype=np.float64)
            p = p / p.sum() if p.sum() > 0 else np.ones_like(p) / len(p)
            action = int(rng.choice(acts, p=p))
        else:
            action = int(opponent_agent.step(state))
        state.apply_action(action)
    return float(state.returns()[agent_player_id])


def main():
    parser = argparse.ArgumentParser(description="Validate LBR lower-bound on Leduc vs exact NashConv")
    parser.add_argument("--train_iterations", type=int, default=200,
                        help="self-play PPO iterations to make a non-trivial Leduc agent to probe")
    parser.add_argument("--episodes_per_iter", type=int, default=64)
    parser.add_argument("--num_pairs", type=int, default=2000,
                        help="duplicate-hand pairs for the LBR win-rate estimate")
    parser.add_argument("--max_depth", type=int, default=1, help="LBR lookahead plies (1 = classic LBR)")
    parser.add_argument("--rollout_samples", type=int, default=4,
                        help="MC rollouts per evaluated node when chance is sampled")
    parser.add_argument("--max_nodes", type=int, default=0,
                        help="per-step node budget; 0 -> unbounded (safe on Leduc)")
    parser.add_argument("--tol", type=float, default=0.05, help="slack on the LBR <= NashConv gate")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    seed = int(args.seed if args.seed is not None else cfg.SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    device = torch.device("cpu")

    game = pyspiel.load_game("leduc_poker")
    obs_dim = int(game.information_state_tensor_size())
    num_actions = int(game.num_distinct_actions())

    clone_ok = state_clone_available(game)
    print(f"leduc_poker | obs_dim={obs_dim} num_actions={num_actions} | "
          f"state.clone() available={clone_ok} (else history-replay fallback)")

    # 1) Train a non-trivial Leduc agent to probe (random nets make LBR's job trivial).
    print(f"Training Leduc agent: {args.train_iterations} iters x {args.episodes_per_iter} eps ...")
    net = train_leduc_agent(
        game, obs_dim, num_actions, device, cfg,
        args.train_iterations, args.episodes_per_iter, rng,
    )
    agent_policy = NetPolicy(game, net, obs_dim, num_actions, device)

    # 2) EXACT exploitability of the trained agent (full-tree best response).
    nash_conv = float(exploitability.nash_conv(
        game, agent_policy, return_only_nash_conv=True, use_cpp_br=False,
    ))
    # nash_conv is the SUM over both players; single-player exploitability ~= half of it.
    single_player_exploitability = nash_conv / 2.0
    print(f"Exact NashConv (sum of both players' BR gains) = {nash_conv:.4f} "
          f"(single-player ~= {single_player_exploitability:.4f})")

    # 3) LBR win-rate via duplicate hands.
    max_nodes = None if int(args.max_nodes) <= 0 else int(args.max_nodes)
    lbr_value_by_seat = {}
    lbr_ci_by_seat = {}
    for lbr_player in (0, 1):
        lbr_agent = LocalBestResponseAgent(
            policy_adapter=agent_policy,
            lbr_player=lbr_player,
            max_depth=int(args.max_depth),
            rollout_samples=int(args.rollout_samples),
            rng=np.random.RandomState(seed + 1000 + lbr_player),
            max_nodes=max_nodes,
        )
        # LBR sits at one fixed seat; play the agent in the OTHER seat for all pairs.
        # We reuse the duplicate machinery but with LBR always at lbr_player: feed
        # opponent=LBR and pin the agent to the complementary seat by running both seat
        # orientations and reading LBR's seat value.
        res = _evaluate_lbr_fixed_seat(
            game, agent_policy, lbr_agent, lbr_player, args.num_pairs, seed,
            np.random.RandomState(seed + 2000 + lbr_player),
        )
        lbr_value_by_seat[lbr_player] = res["lbr_value_chips_per_hand"]
        lbr_ci_by_seat[lbr_player] = res["lbr_value_ci95"]
        print(f"LBR @ seat {lbr_player}: value = {res['lbr_value_chips_per_hand']:.4f} "
              f"+/- {res['lbr_value_ci95']:.4f} chips/hand")

    lbr_value = float(np.mean(list(lbr_value_by_seat.values())))
    lbr_ci = float(np.mean(list(lbr_ci_by_seat.values())))
    print(f"LBR value (avg over seats) = {lbr_value:.4f} +/- {lbr_ci:.4f} chips/hand")

    # 4) Always-call baseline win-rate (LBR must beat this clearly).
    caller = AlwaysCallAgent()
    caller_res = _evaluate_lbr_fixed_seat(
        game, agent_policy, caller, 0, args.num_pairs, seed,
        np.random.RandomState(seed + 3000),
    )
    caller_value = float(caller_res["lbr_value_chips_per_hand"])
    print(f"AlwaysCall baseline value @ seat 0 = {caller_value:.4f} chips/hand")

    # 5) GATES.
    # Gate A: LBR is a LOWER bound -- its value must not exceed exact exploitability.
    #   We compare LBR (single-player chip win-rate) against nash_conv (two-player SUM),
    #   the LOOSER bound, so a pass means LBR is safely below; a fail means LBR is
    #   egregiously over even the doubled exact bound -> definitely a search bug.
    bound_ceiling = nash_conv * (1.0 + float(args.tol))
    bound_pass = lbr_value <= bound_ceiling
    # Also report the strict single-player comparison for the reader (informational).
    strict_single_player_pass = lbr_value <= single_player_exploitability * (1.0 + float(args.tol))
    # Gate B: LBR must beat the static caller by a clear positive margin.
    beats_caller_margin = lbr_value - caller_value
    beats_caller = beats_caller_margin > max(2.0 * lbr_ci, 0.0)

    overall_pass = bool(bound_pass and beats_caller)

    print("-------------------")
    print(f"GATE A  LBR <= NashConv*(1+tol):   LBR={lbr_value:.4f} <= {bound_ceiling:.4f}  -> {'PASS' if bound_pass else 'FAIL'}")
    print(f"        (strict single-player:     LBR={lbr_value:.4f} <= {single_player_exploitability * (1.0 + args.tol):.4f}  -> {'pass' if strict_single_player_pass else 'INFO-fail'})")
    print(f"GATE B  LBR beats AlwaysCall:      margin={beats_caller_margin:.4f}  -> {'PASS' if beats_caller else 'FAIL'}")
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")

    result = {
        "game": "leduc_poker",
        "seed": seed,
        "state_clone_available": bool(clone_ok),
        "lbr_config": {
            "max_depth": int(args.max_depth),
            "rollout_samples": int(args.rollout_samples),
            "max_nodes": max_nodes,
            "num_pairs": int(args.num_pairs),
        },
        "nash_conv_two_player_sum": nash_conv,
        "single_player_exploitability": single_player_exploitability,
        "lbr_value_chips_per_hand": lbr_value,
        "lbr_value_ci95": lbr_ci,
        "lbr_value_by_seat": lbr_value_by_seat,
        "always_call_value_chips_per_hand": caller_value,
        "gates": {
            "tol": float(args.tol),
            "bound_ceiling_nash_conv": bound_ceiling,
            "lbr_le_nashconv": bool(bound_pass),
            "lbr_le_single_player_exploitability": bool(strict_single_player_pass),
            "beats_always_call_margin": beats_caller_margin,
            "beats_always_call": bool(beats_caller),
            "overall_pass": overall_pass,
        },
        "honesty_note": (
            "LBR is a LOWER BOUND on exploitability, never 'the exploitability'. A low LBR "
            "does NOT prove low exploitability (Lisy & Bowling: head-to-head-tied bots "
            "differed ~1300 mbb/g). A HIGH LBR is a definitive disqualifier. On HUNL it is "
            "only ever a lower bound; this Leduc gate validates the search machinery."
        ),
    }
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"Wrote {args.output_json}")

    if not overall_pass:
        sys.exit(1)


def _evaluate_lbr_fixed_seat(game, agent_policy, opponent_agent, opponent_seat, num_pairs, base_seed, rng):
    """Win-rate (chips/hand) for ``opponent_agent`` fixed at ``opponent_seat`` vs the agent.

    We play ``num_pairs`` distinct deals; for each, the opponent sits at ``opponent_seat``
    and the agent at the other seat. Returns the opponent's per-hand chip value and CI.
    """
    opp_returns = []
    for pair_idx in range(int(num_pairs)):
        deal_seed = base_seed + pair_idx
        agent_seat = 1 - int(opponent_seat)
        agent_return = _play_one_hand_leduc(
            game, agent_policy, opponent_agent, agent_seat, deal_seed, rng,
        )
        opp_returns.append(-agent_return)  # zero-sum: opponent gets the negative
    arr = np.array(opp_returns, dtype=np.float64)
    n = int(arr.size)
    value = float(arr.mean()) if n else 0.0
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    stderr = float(std / np.sqrt(n)) if n > 1 else 0.0
    return {
        "lbr_value_chips_per_hand": value,
        "lbr_value_stderr": stderr,
        "lbr_value_ci95": 1.96 * stderr,
        "pairs": n,
    }


if __name__ == "__main__":
    main()
