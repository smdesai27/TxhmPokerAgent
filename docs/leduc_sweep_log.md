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

### Round 2 (DONE) — combine winners + train longer + 2nd seed (artifacts docs/leduc_sweep2/)
| config | avg-best | avg-final | note |
|---|---|---|---|
| combo7 (seed 7, ent0.10→0.003, batch256, 1200it) | **0.81** | 2.03 | best so far (~5.9× below random) |
| combo_avg4 (combo + avg_window 4) | 0.88 | 3.33 | smaller recent avg helps the best |
| ent15 (0.15→0.002, batch192) | 0.96 | 1.92 | |
| combo42 (same as combo7, seed 42) | 1.00 | 3.36 | ~0.2 seed variance vs combo7 |
| long (batch192, 1600it) | 1.04 | 1.67 | |
| bb384 (batch384) | 1.16 | 3.29 | biggest batch not best |

**Finding:** longer training + high/annealed entropy pushed the best below 1.0 (→0.81), but EVERY
run **cycled hard at the end** (finals 1.7–3.4) — the average reaches its low mid-run then the
diverging iterate poisons the recent average. **Instability is now the blocker** (not the floor).
Seed variance ~0.2. Smaller avg_window (4) helped the best.

### Round 3 (DONE) — LR annealing (artifacts docs/leduc_sweep3/)
| config | avg-best (recent window=4) | avg-final | note |
|---|---|---|---|
| lrdecay42 (lr 3e-4→1e-5) | **0.67** | 2.33 | new best (7.1×); still cycles |
| lrdecay13 (seed 13) | 0.68 | 1.35 | |
| bb384decay (batch 384) | 0.69 | 1.95 | |
| lrdecay3e5 (→3e-5) | 0.83 | 2.29 | |
| lrdecay7 (seed 7) | 0.93 | 2.62 | seed outlier (~0.25 variance) |
| entlow (ent→0) | 1.27 | 2.53 | entropy→0 hurts |

**Finding:** LR annealing lowered the floor (0.81→0.67) but did NOT stop the cycling (finals 1.3–2.6).
Smaller LR just slows the rotation — in imperfect-info self-play the iterates CYCLE AROUND the
equilibrium, they don't converge to it. The `avg_window=4` recent average tracks the cycle. **Real
fix:** average over the FULL post-warmup trajectory (fictitious-play / NFSP-style average), which
averages OUT the rotation. (R1's avgwin30 looked bad only because at 400 iters the policy was still
improving; over a long cycling run a full post-warmup average should be low AND stable.)

### Round 4 (DONE) — full post-warmup snapshot FP-average (artifacts docs/leduc_sweep4/)
| config | FP snapshot-average NashConv | recent-window best |
|---|---|---|
| fp_bb384 | 0.84 | 0.69 |
| fp13 | 0.92 | 0.68 |
| fp_warm50 (skip 50%) | 0.96 | 0.67 |
| fp42 | 1.01 | 0.67 |
| fp7 | 1.27 | 0.93 |
| fp_snap5 (finer snaps) | 1.28 | 0.73 |

**Finding (hypothesis REJECTED):** averaging snapshot NETWORKS over the full post-warmup trajectory
is WORSE (0.84–1.28) than the best recent point (0.67). Reason: the self-play iterates make wide
excursions (0.67↔2.6), not a tight orbit around Nash; averaging network params/outputs over wide
excursions gives a muddy mixture, not the equilibrium. Proper fictitious play averages the STRATEGY
via supervised learning on a reservoir of the agent's actions — i.e. NFSP. Snapshot-averaging ≠ NFSP.

### Round 5 (DONE) — NFSP-lite (artifacts docs/leduc_sweep5/)
| config | π̄ NashConv (best) | note |
|---|---|---|
| nfsp_ent10 (BR entropy 0.10) | 1.21 | best NFSP variant |
| nfsp_2k42 / nfsp_2k7 | 1.49 / 1.49 | 2000 iters |
| nfsp_sl8 | 1.51 | more SL updates |
| nfsp_3k | 1.53 | 3000 iters |

**Finding:** NFSP-lite π̄ only reached ~1.2–1.5 — WORSE than the PPO best-checkpoint (0.67). NFSP is
slow (literature uses millions of steps) and, more importantly, my best-responder is weak (one
on-policy PPO batch/iter), so π̄ averages weak responses. Higher BR entropy helped (1.21). Fix:
strengthen the best-responder (more PPO epochs/iter so it actually best-responds) + run much longer.

### Round 6 (RUNNING via sbatch) — NFSP-lite with a STRONG best-responder + long run
Strengthen the BR (ppo_epochs ↑, episodes/iter ↑, BR entropy 0.10) and run ~5–6k iters; submit via
`sbatch` (detached, survives SSH drops — R5's controlling SSH was killed). 2 seeds. **Results: _pending_.**

## Best so far
| round | config | avg-policy NashConv (best) | reduction vs random | notes |
|---|---|---|---|---|
| pre | vanilla + rolling, 500 iters | 1.50 | 3.2× | iterate cycles; average converges |
| R1 | ent10 (entropy 0.10) | 1.21 | 3.9× | lowest floor; still cycles |
| R2 | combo7 (ent0.10→0.003, batch256) | 0.81 | 5.9× | below 1.0; cycles |
| R3 | lrdecay42 (lr-anneal + ent + batch256) | **0.67** | **7.1×** | BEST so far; cycling best-checkpoint |
| R4 | snapshot FP-average | 0.84 | 5.7× | network-averaging worse |
| R5 | NFSP-lite (ent10) | 1.21 | 3.9× | π̄ undertrained / weak BR → R6 strengthens BR |

## Best so far
| round | config | avg-policy NashConv (best) | reduction vs random | notes |
|---|---|---|---|---|
| pre | vanilla + rolling, 500 iters | 1.50 | 3.2× | iterate cycles; average converges |
| R1 | ent10 (entropy 0.10) | 1.21 | 3.9× | lowest floor; still cycles |
| R2 | combo7 (ent0.10→0.003, batch256) | 0.81 | 5.9× | below 1.0; cycles |
| R3 | lrdecay42 (lr-anneal + ent + batch256) | **0.67** | **7.1×** | best so far; recent-avg, cycles |
| R4 | FP snapshot-average | 0.84 (best FP) | 5.7× | snapshot-averaging worse than R3; → NFSP |

## Best so far
| round | config | avg-policy NashConv (best) | reduction vs random | notes |
|---|---|---|---|---|
| pre | vanilla + rolling, 500 iters | 1.50 | 3.2× | iterate cycles; average converges |
| R1 | ent10 (entropy 0.10) | 1.21 | 3.9× | lowest floor; still cycles |
| R2 | combo7 (ent0.10→0.003, batch256) | 0.81 | 5.9× | below 1.0; cycles |
| **R3** | **lrdecay42 (lr-anneal + ent + batch256)** | **0.67** | **7.1×** | best; recent-avg still cycles → R4 full FP-avg |

## Best so far
| round | config | avg-policy NashConv (best) | reduction vs random | notes |
|---|---|---|---|---|
| pre | vanilla + rolling, 500 iters | 1.50 | 3.2× | iterate cycles; average converges |
| R1 | ent10 (entropy 0.10) | 1.21 | 3.9× | lowest floor; still cycles |
| R1 | batch192 | 1.28 | 3.7× | most STABLE (no cycling) |
| **R2** | **combo7 (ent0.10→0.003, batch256, 1200it, seed7)** | **0.81** | **5.9×** | best; but cycles to 2.0 at end |

## Prior ablation (context — not part of this sweep)
Trinal-Clip ≡ vanilla on Leduc (clips never fire on-policy / at this scale); simplified Elo-kBSP
underperformed rolling. See HANDOFF.md §5.
