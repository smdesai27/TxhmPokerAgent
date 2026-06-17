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
### Round 1 (DONE) — one-factor scan (400 iters, seed 42, vanilla+rolling); artifacts docs/leduc_sweep1/
| config | avg-best | best→final (stability) | note |
|---|---|---|---|
| ent10 (entropy 0.10) | **1.21** | 1.21→1.68 (cycles) | highest entropy → lowest floor |
| entanneal 0.10→0.005 | 1.24 | 1.24→1.41 | anneal aids stability |
| batch192 | 1.28 | **1.28→1.28 (no cycling)** | bigger batch eliminates cycling |
| baseline | 1.28 | 1.28→2.68 (cycles hard) | reference |
| ent06 | 1.29 | 1.29→2.13 | |
| self02 | 1.33 | 1.33→1.33 | more snapshot play = stable, not lower |
| avgwin30 | 1.37 | flat | longer averaging HURT (averages early-junk in) |
| combo | 1.38 | | |
| self08 | 1.42 | | more self-play = worse |
| lr1e4 | 1.91 | flat | too slow at 400 iters |

**Throughlines:** ↑entropy lowers the floor; ↑batch removes cycling; anneal stabilizes.
**Anti-findings:** low LR too slow; longer averaging window hurts (keep avg recent/small); more
self-play hurts. **Plateau:** everything ~1.2 at 400 iters → likely a vanilla-PPO-self-play floor;
Round 2 tests longer training + entropy×big-batch×anneal, else escalate to NFSP-style averaging.

### Round 2 (RUNNING) — combine winners + train longer + 2nd-seed confirm
1200–1600 iters, eval_every 50. Configs: `combo42`/`combo7` (ent 0.10→0.003, batch 256, 1200 it,
seeds 42/7), `bb384` (ent 0.08→0.003, batch 384), `ent15` (ent 0.15→0.002, batch 192), `long`
(ent 0.05→0.005, batch 192, 1600 it), `combo_avg4` (combo + avg_window 4). **Results: _pending_.**

## Best so far
| round | config | avg-policy NashConv (best) | reduction vs random | notes |
|---|---|---|---|---|
| pre | vanilla + rolling, 500 iters | 1.50 | 3.2× | iterate cycles; average converges |
| R1 | ent10 (entropy 0.10) | **1.21** | 3.9× | lowest floor; still cycles |
| R1 | batch192 | 1.28 | 3.7× | most STABLE (no cycling) |

## Prior ablation (context — not part of this sweep)
Trinal-Clip ≡ vanilla on Leduc (clips never fire on-policy / at this scale); simplified Elo-kBSP
underperformed rolling. See HANDOFF.md §5.
