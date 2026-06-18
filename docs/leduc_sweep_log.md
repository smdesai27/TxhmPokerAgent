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

### Round 6 (DONE via sbatch) — strong-BR (ppo_epochs 10) long NFSP, 5000 iters
| config | π̄ NashConv best → final | note |
|---|---|---|
| nfsp6_7 (seed 7) | **0.78 → 0.81** | STABLE (final ≈ best); 6.2× below random |
| nfsp6_42 (seed 42) | 0.83 → 0.94 | stable ~0.8–0.9 |

**Finding:** a strong best-responder + 5000 iters made NFSP-lite π̄ converge to a STABLE ~0.78–0.83
(vs R5's undertrained 1.21). Unlike the PPO iterate (which cycles), this is a genuinely converged
exploitability estimate — a cleaner headline than the cycling 0.67 best-checkpoint. Still above the
0.45 target, and the curve was still drifting down at 5k iters → NFSP just needs more iterations.

### Round 7 (DONE via sbatch) — long NFSP-lite (~20k iters), seeds 7/42; artifacts docs/leduc_sweep7/
Strong-BR NFSP-lite, 20000 iters, more SL (sl_updates 20, batch 1024, reservoir 1M), br_entropy 0.10.
| seed | π̄ NashConv: init → final (lowest observed) | vs random 4.76 | curve shape |
|---|---|---|---|
| 7  | 4.83 → **0.538** (lowest 0.538) | 8.85× | steep drop to ~0.66 by 5k; noisy band 0.61–0.70 through 16k; dips to 0.538 (with an UP-step at 19k) |
| 42 | 4.54 → 0.588 (lowest 0.568 @17k) | 8.10× | drop to ~0.74 by 6k; flattens 0.57–0.59 from 13k; ticks UP 0.568→0.588 at the very end |

**Finding:** 4× more iterations than R6 pushed π̄ to **final 0.538 / 0.588 (mean 0.56)** — the lowest and the
only genuinely **non-divergent** (vs the cycling PPO iterate) estimate of the campaign. The curve is **broadly
decreasing then plateauing with local fluctuations** — it stops cycling and settles into a noisy band; it does
**NOT** monotonically descend and does **NOT** converge to Nash. **0.56 is ~8× below random (4.76) but still
~9× ABOVE tuned NFSP (~0.06) and far from Nash (0).**

## DECISION: loop concluded — report as METHODOLOGY, not a headline number
Verified by a 4-lens adversarial review (game-theory / stats-rigor / skeptic / recruiter; unanimous verdict
`reframe_as_methodology_not_result`). The number 0.54–0.59 on Leduc is **not impressive on its own**: Leduc is
a solved 936-state game, tabular CFR hits ~0 in seconds, tuned NFSP reaches ~0.06. **The credibility is the
METHOD, not the magnitude.** Stop point reached (NFSP-lite is the principled fix and it stabilized; chasing
≤0.45 would not change the story and this is an off-track project).

**Verified headline (use verbatim — survives all 4 lenses):**
> On Leduc poker — a 2-player game small enough that exact, ungameable exploitability (NashConv) is computable
> — a from-scratch PyTorch self-play stack reproduces the textbook failure-and-fix: the raw PPO self-play
> iterate **cycles** (best checkpoint 0.67, final iterate diverges to ~1.3–2.6) and averaging network
> **parameters fails** (0.84–1.28), while only **NFSP-style strategy-averaging** stops cycling, settling to a
> stable, non-divergent NashConv of **~0.54–0.59** across two seeds (random ≈4.76, tuned NFSP ≈0.06, Nash = 0).
> Reported as a methodology + honest-ablation result, not a competitive solver number.

**Where the real credibility is:** (1) choosing an EXACT, deterministic, opponent-independent, ungameable
metric on a game where it's tractable — precisely because HUNL exploitability is intractable; (2) a
correctly-reasoned 3-way ablation over ~35 configs isolating WHY strategy-averaging works; (3) honest negatives
reported not buried (snapshot net-averaging rejected; NFSP-lite initially WORSE than the PPO checkpoint;
Trinal-Clip ≡ vanilla at Leduc scale; Elo-kBSP underperformed rolling); (4) engineering fidelity — the Leduc
harness reuses the project's own RolloutBuffer/GAE, post-forward masked logits, and the SAME PPO
hyperparameters as the HUNL configs, so it exercises the real recipe.

**DO NOT SAY (writeup landmines — verified by review):**
- ❌ "converges" → ✅ "stabilizes / plateaus / stops cycling and settles" (data plateau ~0.55, ~9× above Nash).
- ❌ "monotonically descending" → ✅ "broadly decreasing then plateauing with local fluctuations" (seed7 up-steps at 19k; seed42 rises at the end — monotonicity is FALSE).
- ❌ "8.98× / x below init" (per-seed random-init denominator = double cherry-pick) → ✅ only "~8× below random 4.76".
- ❌ headline the best 0.538 → ✅ report BOTH finals 0.538/0.588 (mean 0.56); pre-commit to FINAL for both seeds.
- ❌ a multiplier without both anchors → ✅ always "~8× below random AND ~9× above tuned NFSP (0.06)".
- ❌ "respectable / impressive" in the prose (self-grading) → ✅ state what was done, let the reader judge.
- ❌ "reproduces the RL-vs-CFR gap" (overstates scope) → ✅ "empirically illustrates, on a game where exploitability is exactly computable, why naive self-play needs the averaging CFR/NFSP were designed around".
- ❌ attach 0.54 to the HUNL/AlphaHoldem agent → ✅ it's a LEDUC rigor probe for the self-play DYNAMICS; HUNL exploitability is intractable and was never measured.
- ❌ bare "NFSP" → ✅ "NFSP-lite / NFSP-style" (single on-policy PPO BR, no target net / anticipatory dynamics; minimal demonstrator, so the 0.06 gap is apples-to-oranges by design).
- Note: the PPO best-checkpoint (0.67) is competitive in MAGNITUDE at equal compute; NFSP-lite's win is STABILITY/non-divergence, not a lower number.

## Best so far (FINAL — authoritative; supersedes the per-round tables above)
| round | config | avg/π̄ NashConv | vs random 4.76 | stability | notes |
|---|---|---|---|---|---|
| pre | vanilla + rolling, 500 it | 1.50 | 3.2× | cycles | current iterate |
| R1 | ent10 (entropy 0.10) | 1.21 | 3.9× | cycles | lowest floor at 400 it |
| R2 | combo7 (ent anneal + batch256) | 0.81 | 5.9× | cycles | finals 1.7–3.4 |
| R3 | lrdecay42 (lr-anneal) | 0.67 | 7.1× | cycles | lowest PPO best-checkpoint; final diverges 1.3–2.6 |
| R4 | snapshot net FP-average | 0.84 | 5.7× | — | network-averaging FAILS (worse than R3) |
| R5 | NFSP-lite (weak BR) | 1.21 | 3.9× | — | π̄ undertrained |
| R6 | NFSP-lite (strong BR, 5k it) | 0.78 | 6.1× | stable | first non-divergent estimate |
| **R7** | **NFSP-lite (strong BR, 20k it)** | **0.538 / 0.588 (mean 0.56)** | **~8×** | **stable, non-divergent** | lowest + only non-cycling; still ~9× above tuned NFSP (0.06) |

## Round 8 (RUNNING via sbatch) — MMD/NashPG: make the ITERATE converge (last-iterate)
Re-opened the loop at the user's request to get a LOWER number, via the brainstorm's #1 pick
(docs/leduc_improvement_ideas.md): Magnetic Mirror Descent / NashPG = "PPO + reverse-KL to a
periodically-refreshed magnet rho" (Sokota et al. 2022; lineage MMD → R-NaD/DeepNash → NashPG).
Implemented in `validate_leduc_exploitability.py` as `--mmd_kl_coef`(alpha) + `--mmd_refresh_every`(K)
(commit a1c133d). The HEADLINE metric flips to the **CURRENT ITERATE** (MMD makes it converge — no
averaging needed), unlike R1–R7 where only the average converged.

Also added the honest anchor line: `leduc_cfr_anchor.py` (tabular CFR/CFR+, OpenSpiel) — smoke-tested,
**CFR+ → 0.050 NashConv at 60 iters → ~0 with more** (the reference 0-line; NOT an RL method).

**Smoke (alpha=0.1, K=10, 150 iters):** current-iterate NashConv descended MONOTONICALLY 5.05 → 1.55
(no cycling, no NaN) — mechanism validated. **Sweep (jobs 3313390–97, seed 7, 4000 iters):** alpha ∈
{0.05,0.1,0.2,0.3} × K ∈ {1,10,50,100,500}, plus a strong-BR variant (ppo_epochs 10, entropy 0.02).
Artifacts: docs/leduc_mmd_sweep/.

**RESULT (honest negative — first attempt did NOT beat existing results):** the MMD iterate crashes
5.05→~1–2 in the first 200 iters but then **WANDERS in a noisy 0.7–3.1 band** for the rest of training —
it does NOT converge to a low fixed point. Best transient current-iterate **0.72** (α=0.2,K=500) but
final 1.68; best average 0.80; all 8 configs UNSTABLE (final ≫ best). vs references: R3 PPO best-ckpt
0.67, R7 NFSP avg **0.56**, CFR+ ~0.05. So MMD **damps the catastrophic divergence** (plain PPO blew up to
4.88; MMD stays bounded ~1–2) but does **not** give clean last-iterate convergence at these settings.

**Diagnosis:** dominated by gradient/advantage VARIANCE. Two likely causes: (1) I rely on PPO's *clip* as
the mirror-descent KL-to-previous-iterate proximal term — but ran 4–10 PPO epochs/iter, so the policy
moves too far each step (true MMD ≈ a small/single mirror step with an explicit KL-to-π_old). (2) The
*sampled* GAE advantage is far noisier than the all-actions counterfactual values tabular MMD/NeuRD use.
**Next:** (a) add an explicit `(1/η)·KL(π‖π_old)` proximal term + drop to 1–2 epochs/iter (cheap, ~15 LOC);
if it still wanders, (b) all-actions advantage (bigger lift) or pivot to faithful-NFSP (rank 2) for a safe
lower number. **Units note:** OpenSpiel NashConv = sum of both players' BR gains ≈ 2× single-player
exploitability; published NFSP ~0.06 ≈ ~0.12 on this axis; CFR+ ~0.004–0.02.

## Round 9 (RUNNING via sbatch) — MMD proximal fix + faithful NFSP (both, user-requested)
Fixes for the two methods, run in parallel (commit 6c8d0a8). All seed 7, detached sbatch (jobs
3316357–66). Artifacts → docs/leduc_r9_*/ when done.

**(A) MMD + proximal (`--mmd_prox_coef`):** R8's magnet-only MMD wandered because it used PPO's clip
(over 4–10 epochs) as the trust region. Added an explicit `(1/η)·KL(π‖π_old)` mirror-descent proximal
term + few epochs. **Smoke confirmed the wandering is GONE** — the current iterate now descends smoothly
& monotonically (5.05→3.86, final==best) — but stable⇒slow, so these run 10k iters. Sweep spans the
speed/stability tradeoff: prox ∈ {0.1,0.3}, epochs ∈ {1,3}, α ∈ {0.05,0.1,0.2}, K=100, lr ∈ {3e-4,1e-3}.

**(B) faithful NFSP (`--anticipatory_eta`):** R5–R7 ran at effective η=1 (always BR), which H&S show
plateaus. Added anticipatory η (learner plays BR w.p. η, feeding PPO+reservoir; else π̄, recording
nothing = reservoir hygiene). Smoke: π̄ descends monotone. Sweep: η ∈ {0.1,0.2}, episodes 256–512,
strong BR (ppo_epochs 10), reservoir 2M, 8k–14k iters.

**Looking for:** MMD — a LOW, STABLE current iterate (final≈best, the on-thesis last-iterate win);
NFSP — π̄ well below R7's 0.56. **Results: _pending_.**

## Prior ablation (context — not part of this sweep)
Trinal-Clip ≡ vanilla on Leduc (clips never fire on-policy / at this scale); simplified Elo-kBSP
underperformed rolling. See HANDOFF.md §5.
