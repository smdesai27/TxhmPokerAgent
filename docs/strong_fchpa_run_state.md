# "Strong head-to-head" FCHPA GPU run — live state

> The S7 multi-day run is LAUNCHED. This doc is the single source of truth for its state + what to do
> when it finishes (the run is ~30h; context may compact). Companion: docs/strong_hunl_roadmap.md.

## RUNNING NOW
- **SLURM job 3330857** (`ah_strong_fchpa_gpu`), detached `sbatch`, OSCAR `gpu` partition, 48h walltime.
- Preset `strong_fchpa_gpu` (configs/training_configs.yaml): warm-starts the champion
  `checkpoints/snapshots/interview_ready/interview_ready_1.pt` (resumes at **step 21000**),
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
   checkpoints/snapshots/interview_ready/interview_ready_1.pt --config_file configs/training_configs.yaml
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
