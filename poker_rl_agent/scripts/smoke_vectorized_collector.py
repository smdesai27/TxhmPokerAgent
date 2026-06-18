"""Parity + throughput smoke for VectorizedSelfPlayCollector vs the serial SelfPlayWorker.

Runs N self-play episodes (opponent = the policy net, so no league needed) through BOTH the serial
worker and the vectorized collector with a fresh untrained net, and checks they agree STATISTICALLY
(not bit-exact — RNG consumption order differs): similar transitions/episode, similar near-zero mean
return (self-play is zero-sum over random seat assignment), similar action distribution, all-finite.
Also reports wall-clock speedup (on a GPU node this should be large; that is the whole point of S1).

Run on a compute node (login nodes forbid Python), e.g.:
  srun -p gpu-debug --gres=gpu:1 --cpus-per-task=8 --mem=16G --time=00:30:00 \
    python poker_rl_agent/scripts/smoke_vectorized_collector.py --episodes 300 --batch_games 64
"""
import argparse
import time

import numpy as np
import torch

from poker_rl_agent.algorithms.self_play import SelfPlayWorker
from poker_rl_agent.algorithms.vectorized_self_play import VectorizedSelfPlayCollector
from poker_rl_agent.environment.openspiel_wrapper import PokerEnv
from poker_rl_agent.models.alpha_holdem_net import AlphaHoldemNetwork
from poker_rl_agent.utils.config import Config


def stats(episodes, num_actions):
    n_tr = sum(len(e[0]) for e in episodes)
    rets = [e[1] for e in episodes]
    act = np.zeros(num_actions, dtype=np.float64)
    finite = True
    for ep in episodes:
        trans, fr = ep[0], ep[1]
        if not np.isfinite(fr):
            finite = False
        for t in trans:
            a = int(t["action"])
            if 0 <= a < num_actions:
                act[a] += 1
            if not (np.isfinite(t["log_prob"]) and np.isfinite(t["value"])):
                finite = False
    return {
        "episodes": len(episodes),
        "transitions": int(n_tr),
        "tr_per_ep": n_tr / max(1, len(episodes)),
        "mean_return": float(np.mean(rets)) if rets else 0.0,
        "action_dist": (act / max(1.0, act.sum())).round(3).tolist(),
        "all_finite": bool(finite),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--batch_games", type=int, default=64)
    ap.add_argument("--abstraction", type=str, default="fchpa")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = Config()
    cfg.BETTING_ABSTRACTION = args.abstraction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} abstraction={args.abstraction} episodes={args.episodes} B={args.batch_games}")

    env = PokerEnv(game_name=cfg.GAME_NAME, env_preset=cfg.ENV_PRESET,
                   betting_abstraction=args.abstraction, strict_abstraction=False)
    num_actions = env.num_actions()
    model = AlphaHoldemNetwork(num_actions, cfg).to(device)
    model.eval()

    def spec():  # self-play: opponent is the policy net (both model and agent None)
        return None, None, None

    # --- serial ---
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    worker = SelfPlayWorker(env, device=device, max_action_history=cfg.MAX_ACTION_HISTORY)
    t0 = time.time()
    ser = [worker.generate_episode(policy_model=model, bb_size=cfg.BB_SIZE)
           for _ in range(args.episodes)]
    t_ser = time.time() - t0

    # --- vectorized ---
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    coll = VectorizedSelfPlayCollector(env, device=device, num_games=args.batch_games,
                                       max_action_history=cfg.MAX_ACTION_HISTORY)
    t0 = time.time()
    vec = coll.collect(model, spec, args.episodes, bb_size=cfg.BB_SIZE)
    t_vec = time.time() - t0

    s_ser, s_vec = stats(ser, num_actions), stats(vec, num_actions)
    print("SERIAL     ", s_ser, f"  {t_ser:.1f}s")
    print("VECTORIZED ", s_vec, f"  {t_vec:.1f}s  speedup={t_ser / max(t_vec, 1e-6):.1f}x")

    l1 = float(np.abs(np.array(s_vec["action_dist"]) - np.array(s_ser["action_dist"])).sum())
    print(f"action_dist L1 distance = {l1:.3f}")

    ok = True
    ok &= s_vec["transitions"] > 0 and s_ser["transitions"] > 0
    ok &= s_vec["all_finite"] and s_ser["all_finite"]
    ok &= abs(s_vec["tr_per_ep"] - s_ser["tr_per_ep"]) < 0.5 * max(s_ser["tr_per_ep"], 1.0)
    ok &= abs(s_vec["mean_return"] - s_ser["mean_return"]) < 0.75
    ok &= l1 < 0.25
    print("PARITY_SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
