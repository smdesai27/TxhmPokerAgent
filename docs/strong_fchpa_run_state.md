# "Strong head-to-head" FCHPA GPU run — live state

## ⬛ RESULT (2026-06-20): the scaled run REGRESSED the champion — honest negative
Run 3330857 completed all 90k iters cleanly (1d14h47m, finite losses, rollout_mean_bb +3..+12 vs its
league throughout). **Stage 0 ladder verdict (job 3356563, logs/ladder/new_vs_champ.json):**
- Oracle new-vs-new: +14.6 bb/100, CI95 [−11, +40] → straddles 0 (eval sound).
- **NEW (90k) vs CHAMPION (21k): −511.1 bb/100, CI95 [−582, −440] → the new model LOSES decisively.**
- Confirmed REAL (not a load bug): `load_checkpoint` uses strict `load_state_dict` and did not raise, so
  the trained weights loaded correctly. The new model genuinely regressed.

**Behavior probe (job 3357321, logs/ladder/behavior_probe.json) — diagnosis = OVER-AGGRESSION DRIFT, not collapse:**
| matchup | NEW (90k) | CHAMPION (21k) |
|---|---|---|
| vs random | **+1029 bb/100** | +948 |
| vs always-call (calling station) | **−612 bb/100** | +539 |
The new model is NOT collapsed (it crushes random even harder than the champion) — it drifted into a
bluff-happy "maniac" that beats random/folding opponents but is trivially exploited by anything that just
calls down (always-call −612; the balanced champion −511). ROOT CAUSE (in the run config): the opponent mix
was 90% co-evolving league snapshots + 10% random with `EXPLOIT_OPPONENT_PROB=0.0` / `RANDOM_OPPONENT_PROB=0.0`
— NO calling opponents ever punished over-aggression, and the league co-evolved to fold to it (feedback loop).

**Concrete fix for a retry:** (1) turn ON exploit opponents (`EXPLOIT_OPPONENT_PROB`>0, always_call/sticky_call)
so over-aggression is punished during training; (2) ANCHOR the champion permanently in the league as a fixed
balanced opponent; (3) evaluate-vs-champion periodically + keep-best (never promote a regression); (4) lower
entropy / shorter run to limit drift.

**Mechanism (writeup gold):** classic self-play DRIFT / catastrophic forgetting of a strong warm-start. The
training metric (rollout_mean_bb vs the *co-evolving* league) stayed POSITIVE the whole run — it kept
"winning" locally — while the absolute strategy wandered away from the champion's certified strategy. The
rigorous Stage 0 ladder (fixed strong reference) caught the −511 regression the training metric masked. Same
lesson as AlphaHoldem (league win-rate ≠ strength) and the Leduc cycling finding: **measure vs a fixed strong
reference, not a co-evolving league.** The eval discipline did its job.

**Champion `stage_d_fchpa_21k.pt` (21k) remains the strongest model.** To actually get stronger, the fix is to
ANCHOR the champion: keep it permanently in the league as a fixed opponent + evaluate-vs-champion every N
iters and keep-best + lower LR + little/no entropy injection + shorter run. (Optional retry; off-track.)

---

# "Strong head-to-head" FCHPA GPU run — original plan / state

> The S7 multi-day run is LAUNCHED. This doc is the single source of truth for its state + what to do
> when it finishes (the run is ~30h; context may compact). Companion: docs/strong_hunl_roadmap.md.

## RUNNING NOW
- **SLURM job 3330857** (`ah_strong_fchpa_gpu`), detached `sbatch`, OSCAR `gpu` partition, 48h walltime.
- Preset `strong_fchpa_gpu` (configs/training_configs.yaml): warm-starts the champion
  `checkpoints/snapshots/champion/stage_d_fchpa_21k.pt` (resumes at **step 21000**),
  `CFR_ITERATIONS=111000` (ABSOLUTE target step → **90,000 real iters**, ~9e7 hands).
- **Full stack active:** S1 vectorized collector (`USE_VECTORIZED_COLLECTOR`, `ROLLOUT_BATCH_GAMES=512`)
  + S3 corrected PFSP (`PFSP_MODE=loss`) + S4 Trinal-Clip (`PPO_DUAL_CLIP=true`, delta1=3; value-clip off).
  ROLLOUT_EPISODES=1024, PPO_MINIBATCH_SIZE=2048, PPO_EPOCHS=4, LR=7.5e-5, K_BEST=16, entropy 0.006→0.003.
- WANDB offline. Output: `logs/strong_fchpa/slurm/ah_strong_fchpa_gpu_3330857.out` (incl. an nvidia-smi util
  sampler). Checkpoints every 200 iters → `checkpoints/latest.pt`; end snapshot →
  `checkpoints/snapshots/strong_fchpa/strong_fchpa_gpu_3330857.pt`; best → `checkpoints/best_eval_strong_fchpa_gpu.pt`.

## VALIDATED before launch (all on live OSCAR torch/openspiel)
- Canary (job 3330805, 30 real iters in 50s): finite losses (policy ~±0.001, value ~1.0-1.6 scaled),
  rollout_mean_bb +3..+12, ~1.1-1.3 s/iter, checkpoint written, clean exit. (GPU util ~5% is EXPECTED —
  a 1.5M-param net can't saturate an L40S; the bottleneck is CPU game-stepping on 12 cores. Throughput,
  not GPU util, is the metric.)
- S3 PFSP unit test 3/3; S6 LBR Leduc gate PASS (LBR 1.10 ≤ NashConv 1.42, beats always-call); S5 Stage 0
  self-play oracle straddles 0 (job 3320324).

## QOS / OSCAR notes (norm-gpu)
- Per-user hard limits: **cpu=12, gres/gpu=2, mem=192G**. (16 CPUs caused QOSMaxCpuPerUserLimit — fixed to 12.)
- `gpu` partition MaxTime=UNLIMITED. Login nodes: `module load git` (PATH varies by node; run scripts via `bash -l`).
- OSCAR repo `/users/smdesai/antigravity_poker/anitgravity-txhm` is on branch `claude/sharp-swartz-266392`
  (autoresearch-run1 wip is in `git stash@{0}`, recoverable). `git pull` to get new commits.

## WHEN TRAINING FINISHES (or hits walltime) — the payoff measurement
1. **Stage 0 ladder (the headline strength signal):** new model vs the champion and prior checkpoints,
   with MANY pairs for tight CIs (canary oracle used 1500 → CI ±35; use **5k–20k pairs**).
   `python -m poker_rl_agent.scripts.evaluate_own_ladder --checkpoint_a <new> --checkpoint_b
   checkpoints/snapshots/champion/stage_d_fchpa_21k.pt --config_file configs/training_configs.yaml
   --config_name strong_fchpa_gpu --num_pairs 10000 --seed 42 --output_json logs/ladder_new_vs_champ.json`
   (and vs stage_d_12k_baseline.pt, the collapsed latest.pt). Run the self-play-vs-itself oracle first.
2. **LBR on the new model** (`validate_lbr.py` pattern adapted to HUNL via Evaluator + LBR agent), reported
   as a LOWER BOUND. REFINEMENT TODO: report LBR_value MINUS the self-play baseline value at that seat for a
   rigorous exploitability lower bound (the strict single-player INFO-check flagged the raw value isn't directly
   comparable to single-player exploitability without baseline subtraction).
3. Optional Stage 2 (Slumbot bridge) if time. Then **S8 writeup** (docs/strong_hunl_roadmap.md has the
   honest-ceiling framing + landmine list; never claim near-Nash / beats-Slumbot without a CI).

## IF WALLTIME CUTS IT OFF before 90k iters
Resume: submit a continuation that sets `RESUME_FROM: checkpoints/latest.pt` (clone the preset →
`strong_fchpa_gpu_resume`) and re-`sbatch`. Checkpoints every 200 iters bound the loss to ≤200 iters.
