"""Leduc NFSP-lite: PPO best-responder + supervised average-policy network (fictitious play).

The R1-R4 sweep (see docs/leduc_sweep_log.md) showed PPO self-play reduces Leduc exploitability ~7x
(to ~0.67) but the iterate CYCLES, and averaging snapshot NETWORKS does not reach Nash. The
principled fix (Heinrich & Silver 2016, NFSP) is to average the STRATEGY: a separate average-policy
network pi_bar is trained by supervised classification on a RESERVOIR of the RL best-responder's
(state, action) pairs over ALL of training. The best-responder plays AGAINST pi_bar (fictitious
play), so pi_bar converges toward Nash. We report exploitability (NashConv) of pi_bar.

Reuses the shared net / masking / PPO / exploitability-adapter from validate_leduc_exploitability.

Run on a compute node (OSCAR login nodes forbid Python):
  python poker_rl_agent/scripts/validate_leduc_nfsp.py --iterations 2000 --output_json logs/nfsp.json
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

import pyspiel
from open_spiel.python.algorithms import exploitability

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from poker_rl_agent.training.rollout_buffer import RolloutBuffer  # noqa: E402
from poker_rl_agent.utils.config import Config  # noqa: E402
from poker_rl_agent.scripts.validate_leduc_exploitability import (  # noqa: E402
    PolicyValueNet, masked_logits, _obs_and_mask, NetPolicy, ppo_update,
)


class Reservoir:
    """Reservoir buffer (uniform sample over the full history) of (obs, legal, action)."""

    def __init__(self, capacity, obs_dim, num_actions, rng):
        self.cap = int(capacity)
        self.rng = rng
        self.seen = 0
        self.size = 0
        self.obs = np.zeros((self.cap, obs_dim), dtype=np.float32)
        self.legal = np.zeros((self.cap, num_actions), dtype=np.float32)
        self.act = np.zeros((self.cap,), dtype=np.int64)

    def add(self, obs, legal, action):
        self.seen += 1
        if self.size < self.cap:
            i = self.size
            self.size += 1
        else:
            j = self.rng.randint(self.seen)
            if j >= self.cap:
                return
            i = j
        self.obs[i] = obs
        self.legal[i] = legal
        self.act[i] = int(action)

    def sample(self, batch):
        n = min(int(batch), self.size)
        idx = self.rng.randint(self.size, size=n)
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.legal[idx]),
            torch.from_numpy(self.act[idx]),
        )


def play_episode_nfsp(game, br_net, avg_net, learner, obs_dim, num_actions, device, buffer, reservoir, rng):
    """Best-responder (learner) plays vs the average policy pi_bar (avg_net). Records the
    best-responder's (obs, legal, action) into both the PPO buffer and the reservoir."""
    state = game.new_initial_state()
    start = len(buffer)
    while not state.is_terminal():
        if state.is_chance_node():
            actions, probs = zip(*state.chance_outcomes())
            state.apply_action(int(rng.choice(actions, p=np.asarray(probs, dtype=np.float64))))
            continue
        cur = state.current_player()
        obs, mask = _obs_and_mask(state, cur, obs_dim, num_actions, device)
        with torch.no_grad():
            if cur == learner:
                logits, value = br_net(obs.unsqueeze(0))
                dist = Categorical(logits=masked_logits(logits, mask.unsqueeze(0)))
                action = dist.sample()
                a = int(action.item())
                buffer.add(
                    state={"obs": obs.detach().cpu(), "legal": mask.detach().cpu()},
                    action=a, reward=0.0, done=False,
                    log_prob=float(dist.log_prob(action).item()), value=float(value.item()),
                )
                reservoir.add(obs.cpu().numpy(), mask.cpu().numpy(), a)
            else:
                logits, _ = avg_net(obs.unsqueeze(0))
                dist = Categorical(logits=masked_logits(logits, mask.unsqueeze(0)))
                a = int(dist.sample().item())
        state.apply_action(a)
    r = float(state.returns()[learner])
    if len(buffer) > start:
        buffer.transitions[-1].reward = r
        buffer.transitions[-1].done = 1.0
    return r


def sl_update(avg_net, sl_opt, reservoir, batch, n_updates, device):
    """Train pi_bar by cross-entropy to imitate the reservoir of best-responder actions."""
    if reservoir.size < batch:
        return None
    last = None
    for _ in range(int(n_updates)):
        obs, legal, act = reservoir.sample(batch)
        obs, legal, act = obs.to(device), legal.to(device), act.to(device)
        logits, _ = avg_net(obs)
        loss = F.cross_entropy(masked_logits(logits, legal), act)
        sl_opt.zero_grad(set_to_none=True)
        loss.backward()
        sl_opt.step()
        last = float(loss.item())
    return last


def main():
    parser = argparse.ArgumentParser(description="Leduc NFSP-lite (fictitious play) exploitability")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--episodes_per_iter", type=int, default=128)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--br_entropy", type=float, default=0.05)
    parser.add_argument("--ppo_epochs", type=int, default=0, help="0 -> Config default; higher = stronger best-responder")
    parser.add_argument("--sl_lr", type=float, default=0.001)
    parser.add_argument("--sl_updates", type=int, default=4)
    parser.add_argument("--sl_batch", type=int, default=256)
    parser.add_argument("--reservoir", type=int, default=400000)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    cfg = Config()
    cfg.PPO_ENTROPY_COEF = float(args.br_entropy)
    if args.ppo_epochs > 0:
        cfg.PPO_EPOCHS = int(args.ppo_epochs)
    seed = int(args.seed if args.seed is not None else cfg.SEED)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    device = torch.device("cpu")

    game = pyspiel.load_game("leduc_poker")
    obs_dim = int(game.information_state_tensor_size())
    num_actions = int(game.num_distinct_actions())

    br_net = PolicyValueNet(obs_dim, num_actions, hidden=args.hidden).to(device)   # PPO best-responder
    avg_net = PolicyValueNet(obs_dim, num_actions, hidden=args.hidden).to(device)  # pi_bar (SL average)
    ppo_opt = torch.optim.Adam(br_net.parameters(), lr=float(cfg.LR))
    sl_opt = torch.optim.Adam(avg_net.parameters(), lr=float(args.sl_lr))
    reservoir = Reservoir(args.reservoir, obs_dim, num_actions, rng)

    def eval_pi_bar():
        avg_net.eval()
        return float(exploitability.nash_conv(
            game, NetPolicy(game, avg_net, obs_dim, num_actions, device),
            return_only_nash_conv=True, use_cpp_br=False,
        ))

    print(f"leduc_poker NFSP-lite | obs_dim={obs_dim} num_actions={num_actions} | "
          f"br_entropy={args.br_entropy} sl_lr={args.sl_lr} sl_updates={args.sl_updates} "
          f"episodes/iter={args.episodes_per_iter} iters={args.iterations}")

    curve = []
    init_nc = eval_pi_bar()
    best = init_nc
    curve.append({"iteration": 0, "pi_bar_nash_conv": init_nc})
    print(f"iter    0 | pi_bar NashConv = {init_nc:.4f}")

    for it in range(1, int(args.iterations) + 1):
        br_net.train()
        avg_net.train()
        buffer = RolloutBuffer()
        for ep in range(int(args.episodes_per_iter)):
            play_episode_nfsp(game, br_net, avg_net, learner=ep % 2,
                              obs_dim=obs_dim, num_actions=num_actions, device=device,
                              buffer=buffer, reservoir=reservoir, rng=rng)
        if len(buffer) > 0:
            ppo_update(br_net, ppo_opt, buffer, cfg, device, entropy_coef=args.br_entropy)
        sl_update(avg_net, sl_opt, reservoir, args.sl_batch, args.sl_updates, device)

        if it % args.eval_every == 0 or it == int(args.iterations):
            nc = eval_pi_bar()
            best = min(best, nc)
            curve.append({"iteration": it, "pi_bar_nash_conv": nc})
            print(f"iter {it:4d} | pi_bar NashConv = {nc:.4f}  (best {best:.4f})")

    final_nc = curve[-1]["pi_bar_nash_conv"]
    result = {
        "game": "leduc_poker",
        "method": "nfsp_lite (PPO best-responder + SL average policy)",
        "obs_dim": obs_dim, "num_actions": num_actions,
        "iterations": int(args.iterations), "episodes_per_iter": int(args.episodes_per_iter),
        "br_entropy": args.br_entropy, "sl_lr": args.sl_lr, "sl_updates": args.sl_updates,
        "sl_batch": args.sl_batch, "reservoir": args.reservoir, "seed": seed,
        "initial_nash_conv": init_nc, "final_nash_conv": final_nc, "best_nash_conv": best,
        "improvement_factor": (init_nc / best) if best > 0 else None,
        "curve": curve,
        "note": "pi_bar = supervised average over the reservoir of best-responder actions; this is "
                "the convergent (low-exploitability) estimate, unlike the cycling current iterate.",
    }
    imp = result["improvement_factor"]
    print(f"\nNFSP-lite pi_bar NashConv: {init_nc:.4f} (init) -> best {best:.4f} -> {final_nc:.4f} (final)"
          + (f"  [{imp:.2f}x below init]" if imp else ""))
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
