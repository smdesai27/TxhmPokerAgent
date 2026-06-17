# Leduc exploitability — auto-research loop log (standing GOAL)

> **Standing goal:** iterate hyperparameter sweeps on the Leduc validation harness until we reach
> a RESPECTABLE, reportable exploitability result, then write it up. The loop is driven by OSCAR
> SLURM sweeps — each round's job completion re-invokes the agent to analyze + launch the next
> round. A fresh agent: read this file + HANDOFF.md, check "Best so far", launch the next round.

## Success criterion ("good performance worth reporting")
Metric = **averaged-policy NashConv (exploitability)** on `leduc_poker` — opponent-independent,
exact, ungameable. Reference points: random policy ≈ **4.76**; converged solver (CFR) → **0**;
NFSP reaches very low exploitability. (NashConv = sum of both players' best-response gains, chips.)
- **Primary target:** avg-policy NashConv **≤ 0.45 (≥10× below random)** with a clean curve.
- **Stretch:** ≤ 0.2 (NFSP-competitive).
- **Stop:** hit target, OR a clear plateau (2 consecutive refinement rounds fail to improve the
  best by >10%) → report best + plateau analysis honestly.

## Method
`poker_rl_agent/scripts/validate_leduc_exploitability.py` with sweepable flags:
`--entropy_coef --entropy_final` (linear anneal) `--lr --episodes_per_iter --self_fraction`
`--avg_window` (fictitious-play averaging window) `--clip_eps --gae_lambda --ppo_epochs`
`--policy_loss {vanilla,trinal_clip} --selfplay {rolling,kbest_elo}`. Sweeps run as parallel
SLURM jobs on OSCAR scratch `/oscar/scratch/smdesai/leduc_verify`; runners are `/tmp/leduc_sweep*.sh`.
Pull `sw*_*.json` artifacts into `docs/` as rounds finish.

Hypotheses for lowering exploitability: more exploration (higher/annealed entropy), lower LR,
bigger batch (lower-variance gradients), and especially a LONGER fictitious-play averaging window
(`--avg_window`) since the time-average — not the cycling current iterate — is what converges.

## Rounds
### Round 1 (RUNNING) — one-factor scan vs baseline
400 iters, seed 42, vanilla + rolling. Configs: `baseline`, `ent03/ent06/ent10`,
`entanneal`(0.10→0.005), `lr1e4`, `batch192`, `avgwin30`, `self08`, `self02`,
`combo`(ent06+batch128+avgwin30). **Results: _pending_** (fill from `sw1_*.log/json`).

## Best so far
| round | config | avg-policy NashConv (best) | reduction vs random | notes |
|---|---|---|---|---|
| pre | vanilla + rolling, 500 iters | 1.50 | 3.2× | iterate cycles; average converges |

## Prior ablation (context — not part of this sweep)
Trinal-Clip ≡ vanilla on Leduc (clips never fire on-policy / at this scale); simplified Elo-kBSP
underperformed rolling. See HANDOFF.md §5.
