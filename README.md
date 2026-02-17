# AlphaHoldEm Poker RL Agent

This repository contains a PyTorch implementation of a Poker Reinforcement Learning agent inspired by the **AlphaHoldEm** architecture. It uses a **Pseudo-Siamese Neural Network** trained with **PPO + K-best self-play league** within the **OpenSpiel** environment.

Repository: [smdesai27/anitgravity-txhm](https://github.com/smdesai27/anitgravity-txhm)

## 📌 Features
- **Pseudo-Siamese Architecture**: Separate towers for Card embeddings and Action history processing.
- **Actor-Critic Training**: PPO objective with clipped policy updates and GAE advantages.
- **K-Best League Self-Play**: Opponents are sampled from a rolling top-K snapshot pool (PFSP weighting).
- **Heads-Up No-Limit Hold'em**: Designed for HUNL (Universal Poker), extensible to K players.
- **Self-Play**: Parallel data collection via self-play trajectories.
- **Evaluation**: Benchmarking against Random/heuristic baselines and a solver-generated MCCFR-ES baseline.

## 🛠️ Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: OpenSpiel (`open_spiel`) is required. If pip install fails, refer to [DeepMind OpenSpiel](https://github.com/deepmind/open_spiel) installation guide.*

2. **Configuration**
   Adjust hyperparameters in `poker_rl_agent/utils/config.py`:
   - `GAME_NAME`: "universal_poker" (default) or "leduc_poker" for testing.
   - `BETTING_ABSTRACTION`: `fcpa`, `fchpa` (legacy alias `fcpha`), or `fullgame`.
   - `STRICT_ABSTRACTION`: `true` for fail-fast experiment integrity (recommended).
   - `CFR_ITERATIONS`: Training duration.
   - `DEVICE`: 'cuda' or 'cpu'.

3. **Weights & Biases (Online Graphs)**
   ```bash
   export WANDB_API_KEY=...
   export WANDB_PROJECT=alpha-holdem-poker
   export WANDB_ENTITY=<your_wandb_entity>
   # Optional:
   wandb login
   ```

## 🚀 Usage

### Training
Train the agent using PPO + K-best self-play:
```bash
python poker_rl_agent/scripts/train.py
```
Checkpoints will be saved to `checkpoints/`.

### Evaluation
Evaluate a trained checkpoint against baselines:
```bash
python poker_rl_agent/scripts/evaluate.py --checkpoint checkpoints/point_100.pt --episodes_per_seed 1000 --baseline_seeds 3
```

Run a complete multi-tier benchmark report (default CFR gate + robustness gate):
```bash
python poker_rl_agent/scripts/evaluate_complete.py \
  --checkpoint checkpoints/latest.pt \
  --config_file configs/training_configs.yaml \
  --config_name quadro_stage_b \
  --episodes_per_seed 5000 \
  --profile standard \
  --verify_solver_training \
  --holdout_seed_base 10042 \
  --holdout_seed_count 5 \
  --output_json logs/eval_complete_stage_b.json
```

Stage C certification-style eval (strict robustness CI):
```bash
python poker_rl_agent/scripts/evaluate_complete.py \
  --checkpoint checkpoints/latest.pt \
  --config_file configs/training_configs.yaml \
  --config_name quadro_stage_c_fcpa \
  --episodes_per_seed 5000 \
  --profile exhaustive \
  --verify_solver_training \
  --require_robust_ci \
  --holdout_seed_base 12042 \
  --holdout_seed_count 5 \
  --output_json logs/eval_complete_stage_c_cert.json
```

Import an existing local metrics JSONL into W&B (for already completed runs):
```bash
python poker_rl_agent/scripts/wandb_import_jsonl.py \
  --metrics_jsonl logs/metrics_20260212_185050.jsonl \
  --project alpha-holdem-poker \
  --entity <your_wandb_entity> \
  --run_name stage_c_fcpa_reimport \
  --step_key _step
```

### Interactive Play
Play heads-up against the agent in terminal (classic Texas Hold'em fullgame by default):
```bash
python poker_rl_agent/scripts/play_against_agent.py \
  --checkpoint checkpoints/latest.pt \
  --config_file configs/training_configs.yaml \
  --config_name quadro_stage_b \
  --game_mode fullgame \
  --human_seat 0 \
  --hands 20 \
  --bot_policy sample \
  --policy_temperature 1.0
```
If your checkpoint is FCPA (4 actions) and you play `--game_mode fullgame`, adapter mode is enabled by default and the script prints:
`Adapter mode active: qualitative play only, not benchmark-comparable.`
Use `--disable_adapter` to fail fast instead of adapting action spaces.

## 🧠 Architecture Details

### Neural Network
- **Card Tower**: Embeds Hole Cards and Community Cards.
- **Action Tower**: LSTM-based processing of betting history.
- **Heads**:
  - Policy logits for legal action selection
  - State-value estimate for PPO critic target

### Algorithm
- **On-Policy PPO**: Uses rollout trajectories, clipped policy loss, value loss, and entropy regularization.
- **League Self-Play**: Samples opponents from stored top-K snapshots and updates Elo tracking.

## 📂 Structure
- `poker_rl_agent/`
  - `models/`: PyTorch model definitions.
  - `algorithms/`: CFR logic and self-play.
  - `environment/`: OpenSpiel wrappers.
  - `training/`: Training loop and buffer.
  - `scripts/`: Executable scripts.

## ⚠️ Notes
- Training generally requires `open_spiel`'s `universal_poker` game with an ACPC definition for full HUNL. By default, this repo falls back to `leduc_poker` or simplified config if HUNL isn't fully configured in your OpenSpiel install.
- Long training runs (3-7 days) are recommended for strong play.
- The default environment is strict HUNL FCPA (`pyspiel.hunl_game_string("fcpa")`).
- W&B logging supports `online`, `offline`, and `disabled` modes via `WANDB_MODE`.

## 🔧 Advanced Configuration & Troubleshooting
If you encounter instability (diverging loss, NaN values), use the robust configuration system.

### Running with Presets
We provide a `scripts/debug_training.py` that supports YAML-based configs:
```bash
# Run with 'debug' preset (small net, short run)
python scripts/debug_training.py --config_name debug

# Run with Quadro medium preset
python scripts/debug_training.py --config_name quadro_medium

# Run with Quadro long preset
python scripts/debug_training.py --config_name quadro_long

# Run Stage C FCPA preset (10k iters)
python scripts/debug_training.py --config_name quadro_stage_c_fcpa

# Run Stage D FCHPA bridge preset (5-action abstraction)
python scripts/debug_training.py --config_name quadro_stage_d_fchpa

# Continue Stage D FCHPA from latest checkpoint with anti-collapse controls
python scripts/debug_training.py --config_name quadro_stage_d_fchpa_continue

# Continue Stage D FCHPA tuned run to 12k
python scripts/debug_training.py --config_name quadro_stage_d_fchpa_continue_12k_tuned

# Stage D ablation presets (1500-step continuation from latest checkpoint)
python scripts/debug_training.py --config_name quadro_stage_d_fchpa_recover_explore_ablate
python scripts/debug_training.py --config_name quadro_stage_d_fchpa_recover_diverse_ablate
python scripts/debug_training.py --config_name quadro_stage_d_fchpa_recover_value_stable_ablate

# Stage D selected continuation to 16k (choose winning ablation family)
python scripts/debug_training.py --config_name quadro_stage_d_fchpa_recover_explore_16k

# Run Stage E fullgame progressive preset (generalization track)
python scripts/debug_training.py --config_name quadro_stage_e_fullgame_progressive
```

### SLURM-Compatible Eval Command Pattern
You can run the complete benchmark via `sbatch --wrap` on HPC:
```bash
sbatch --job-name=ah_eval_complete \
  --output=logs/%x_%j.out \
  --error=logs/%x_%j.err \
  --time=12:00:00 \
  --cpus-per-task=8 \
  --mem=32G \
  --gres=gpu:1 \
  --wrap="cd /oscar/home/smdesai/antigravity_poker/anitgravity-txhm && source venv/bin/activate && python poker_rl_agent/scripts/evaluate_complete.py --checkpoint checkpoints/latest.pt --config_file configs/training_configs.yaml --config_name quadro_stage_b --episodes_per_seed 5000 --profile standard --output_json logs/eval_complete_stage_b.json"
```

Or use the provided Stage C batch scripts directly:
```bash
sbatch scripts/slurm_stage_c_train.slurm
sbatch scripts/slurm_stage_c_eval.slurm
```

Stage D FCHPA bridge scripts:
```bash
sbatch scripts/slurm_stage_d_fchpa_train.slurm
sbatch scripts/slurm_stage_d_fchpa_eval.slurm
```

Stage D continuation and ablation scripts:
```bash
# Certify current Stage D checkpoint
sbatch scripts/slurm_stage_d_fchpa_cert_eval.slurm

# Ablation loop (run with each *_ablate config and unique HOLDOUT_SEED_BASE values)
sbatch --export=ALL,CONFIG_NAME=quadro_stage_d_fchpa_recover_explore_ablate,WANDB_MODE=online scripts/slurm_stage_d_fchpa_ablation_train.slurm
sbatch --export=ALL,CONFIG_NAME=quadro_stage_d_fchpa_recover_explore_ablate,HOLDOUT_SEED_BASE=61042,WANDB_MODE=online scripts/slurm_stage_d_fchpa_ablation_eval.slurm

# Promote selected winner to 16k and certify
sbatch --export=ALL,CONFIG_NAME=quadro_stage_d_fchpa_recover_explore_16k,WANDB_MODE=online scripts/slurm_stage_d_fchpa_selected_16k_train.slurm
sbatch --export=ALL,CONFIG_NAME=quadro_stage_d_fchpa_recover_explore_16k,WANDB_MODE=online scripts/slurm_stage_d_fchpa_selected_16k_cert_eval.slurm

# Optional corrective +1k if cert passes but fold > 0.45
sbatch --export=ALL,CONFIG_NAME=quadro_stage_d_fchpa_recover_explore_corrective_17k,WANDB_MODE=online scripts/slurm_stage_d_fchpa_selected_16k_train.slurm
sbatch --export=ALL,CONFIG_NAME=quadro_stage_d_fchpa_recover_explore_corrective_17k,WANDB_MODE=online scripts/slurm_stage_d_fchpa_selected_16k_cert_eval.slurm
```

Stage D 16k -> corrective 17k -> long 20k flow (balanced EV + behavior):
```bash
# 1) Freeze current 16k as champion snapshot
scripts/stage_d_freeze_champion.sh checkpoints/latest.pt stage_d_16k_champion stage_d_16k_champion_latest.pt

# 2) Corrective continuation to 17k (diverse corrective family)
sbatch --export=ALL,CONFIG_NAME=quadro_stage_d_fchpa_recover_diverse_corrective_17k,WANDB_MODE=online,WANDB_RUN_GROUP=stage_d_corrective_17k scripts/slurm_stage_d_fchpa_corrective_train.slurm

# 3) Screening eval after 17k
sbatch --export=ALL,CONFIG_NAME=quadro_stage_d_fchpa_recover_diverse_corrective_17k,EPISODES_PER_SEED=5000 scripts/slurm_stage_d_fchpa_corrective_screen_eval.slurm

# 4) Gate check (requires behavior_extended and CI floor >= 830)
sbatch --export=ALL,EVAL_JSON=logs/stage_d/eval/eval_quadro_stage_d_fchpa_recover_diverse_corrective_17k_screen_<jobid>.json,BASELINE_EVAL_JSON=logs/stage_d/eval/eval_quadro_stage_d_fchpa_recover_explore_16k_cert_375207.json,CI_FLOOR=830 scripts/slurm_stage_d_fchpa_eval_gate_check.slurm

# 5) If behavior_extended fails but EV stays strong: 17k -> 18k value-stable corrective
sbatch --export=ALL,CONFIG_NAME=quadro_stage_d_fchpa_recover_value_stable_corrective_18k,WANDB_MODE=online,WANDB_RUN_GROUP=stage_d_corrective_18k scripts/slurm_stage_d_fchpa_corrective_train.slurm
sbatch --export=ALL,CONFIG_NAME=quadro_stage_d_fchpa_recover_value_stable_corrective_18k,EPISODES_PER_SEED=5000 scripts/slurm_stage_d_fchpa_corrective_screen_eval.slurm

# 6) If gates pass: long continuation to 20k (winning family) + full cert eval
sbatch --export=ALL,CONFIG_NAME=quadro_stage_d_fchpa_recover_diverse_corrective_20k,WANDB_MODE=online,WANDB_RUN_GROUP=stage_d_long_20k scripts/slurm_stage_d_fchpa_corrective_train.slurm
sbatch --export=ALL,CONFIG_NAME=quadro_stage_d_fchpa_recover_diverse_corrective_20k,EPISODES_PER_SEED=10000,REQUIRE_ROBUST_CI=true scripts/slurm_stage_d_fchpa_corrective_cert_eval.slurm
```

Stage D recovery cycle (automated probes + winner selection + promotion):
```bash
# Runs:
# Probe A (balanced 19k) -> screen -> gate
# Probe B (conservative 19k) -> screen -> gate
# Winner selection by CI + behavior envelope
# Winner promotion to selected 20k -> screen -> gate -> cert
bash scripts/submit_stage_d_fchpa_recovery_cycle.sh
```

Human validation transcript (200-hand FCHPA CLI session):
```bash
bash scripts/run_stage_d_human_validation.sh
```

Stage D artifact locations:
- `logs/stage_d/slurm/`
- `logs/stage_d/eval/`
- `logs/stage_d/metrics/`
- `checkpoints/snapshots/stage_d/`

Cleanup and checkpoint indexing:
```bash
scripts/organize_workspace.sh --apply
python poker_rl_agent/scripts/checkpoint_manager.py organize --root checkpoints --apply
python poker_rl_agent/scripts/checkpoint_manager.py index --root checkpoints --apply
```

Manual gate check from terminal (without SLURM wrapper):
```bash
python poker_rl_agent/scripts/check_eval_gates.py \
  --eval_json logs/stage_d/eval/eval_quadro_stage_d_fchpa_recover_diverse_corrective_17k_screen_<jobid>.json \
  --baseline_eval_json logs/stage_d/eval/eval_quadro_stage_d_fchpa_recover_explore_16k_cert_375207.json \
  --ci_floor 830 \
  --require_behavior_extended \
  --require_solver_verified
```

Create a named snapshot from current latest:
```bash
python poker_rl_agent/scripts/checkpoint_manager.py snapshot \
  --root checkpoints \
  --source checkpoints/latest.pt \
  --stage stage_d \
  --label candidate_16k \
  --active-alias candidate_16k_latest.pt \
  --apply
```

Stage E fullgame progressive scripts:
```bash
sbatch scripts/slurm_stage_e_fullgame_train.slurm
sbatch scripts/slurm_stage_e_fullgame_eval.slurm
```

If running online W&B logging on SLURM:
```bash
export WANDB_API_KEY=...
export WANDB_PROJECT=alpha-holdem-poker
export WANDB_ENTITY=<your_wandb_entity>
sbatch --export=ALL,WANDB_MODE=online,WANDB_RUN_GROUP=stage_c_fcpa scripts/slurm_stage_c_train.slurm
```

### Snapshot Archiving
Create an immutable run snapshot (model + eval + metrics + logs + manifest):
```bash
python poker_rl_agent/scripts/archive_run.py \
  --run_name stage_c_fcpa_pass_all \
  --stage stage_c_fcpa \
  --checkpoint checkpoints/latest.pt \
  --metrics logs/metrics_20260212_185050.jsonl \
  --eval_json logs/eval_stage_c_366833.json \
  --slurm_out logs/ah_stage_c_train_357399.out logs/ah_stage_c_eval_366833.out \
  --slurm_err logs/ah_stage_c_train_357399.err logs/ah_stage_c_eval_366833.err \
  --config_file configs/training_configs.yaml \
  --config_name quadro_stage_c_fcpa \
  --archive_root artifacts/runs
```

### Ablation Studies
Modify `configs/training_configs.yaml` to create new presets. Available debugging features:
- **Gradient Clipping**: Prevents exploding gradients (`GRAD_CLIP`).
- **AMP Support**: Optional mixed precision on CUDA (`AMP: true`).
- **Detailed Logging**: Logs weight norms and gradients to WandB.
- **Checkpointing**: Limits disk usage by keeping only the latest N checkpoints (`MAX_CHECKPOINTS: 5`).

### Troubleshooting Guide
- **Loss Increases**: Try lowering LR (`LR: 1e-4`) or enabling `USE_TARGET_NET`.
- **NaN Values**: Reduce `LR`, ensure `GRAD_CLIP` is on, or check `Regret Matching` code for zero division.
- **Disk Full**: Reduce `MAX_CHECKPOINTS` in config.
