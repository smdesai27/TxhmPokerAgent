"""Tabular CFR / CFR+ anchor line for the Leduc exploitability figures.

This is NOT an RL method and does NOT advance the project's neural-self-play thesis. It exists purely
as an HONEST, calibrated reference: tabular CFR/CFR+ solve Leduc to ~0 exploitability in seconds, so
plotting their NashConv on the SAME axis as the neural runs lets a reader judge whether the neural
numbers (random ~4.76, NFSP-lite ~0.56, MMD/NashPG ~0.05-0.15) are good. Reported on OpenSpiel's
exploitability.nash_conv axis = SUM of both players' best-response gains (~2x single-player
exploitability), identical to the neural validation scripts.

Run on a compute node (OSCAR login nodes forbid Python):
  python poker_rl_agent/scripts/leduc_cfr_anchor.py --iterations 300 --output_json logs/leduc_cfr.json
"""
import argparse
import json
import os

import pyspiel
from open_spiel.python.algorithms import cfr, exploitability


def run_solver(game, solver, iterations, eval_every, label):
    curve = []
    best = None
    for i in range(1, int(iterations) + 1):
        solver.evaluate_and_update_policy()
        if i % eval_every == 0 or i == int(iterations):
            nc = float(exploitability.nash_conv(game, solver.average_policy()))
            best = nc if best is None else min(best, nc)
            curve.append({"iteration": i, "nash_conv": nc})
            print(f"  [{label}] iter {i:4d} | NashConv = {nc:.6f}  (best {best:.6f})")
    return {"label": label, "iterations": int(iterations), "best_nash_conv": best,
            "final_nash_conv": curve[-1]["nash_conv"], "curve": curve}


def main():
    parser = argparse.ArgumentParser(description="Tabular CFR/CFR+ anchor line for Leduc exploitability")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--eval_every", type=int, default=25)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    game = pyspiel.load_game("leduc_poker")
    print(f"leduc_poker tabular solvers | NashConv axis = sum of both players' BR gains "
          f"(random ~4.76, Nash = 0)")

    results = {}
    print("CFR (vanilla):")
    results["cfr"] = run_solver(game, cfr.CFRSolver(game), args.iterations, args.eval_every, "CFR")
    print("CFR+:")
    results["cfr_plus"] = run_solver(game, cfr.CFRPlusSolver(game), args.iterations, args.eval_every, "CFR+")

    out = {
        "game": "leduc_poker",
        "method": "tabular CFR / CFR+ (OpenSpiel) — honest anchor line, NOT an RL method",
        "iterations": int(args.iterations),
        "random_policy_nash_conv_reference": 4.76,
        "results": results,
        "note": "Tabular solvers reach ~0 NashConv in seconds; included only as a calibrated reference "
                "line (different paradigm: no network, no PPO). NashConv = sum of both players' "
                "best-response gains (~2x single-player exploitability).",
    }
    print(f"\nCFR best NashConv  = {results['cfr']['best_nash_conv']:.6f}")
    print(f"CFR+ best NashConv = {results['cfr_plus']['best_nash_conv']:.6f}")
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
