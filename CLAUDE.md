# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaHoldEm Poker RL Agent — a PyTorch + OpenSpiel implementation of a Heads-Up No-Limit Hold'em (HUNL) agent using a Pseudo-Siamese Neural Network trained with PPO + K-best self-play league.

**Python 3.10+ required.** Core deps: PyTorch, OpenSpiel, NumPy (<2.0), PyYAML, W&B.

## Common Commands

### Setup
```bash
pip install -r requirements.txt
```

### Training
```bash
# With a YAML preset (preferred)
python scripts/debug_training.py --config_name <preset>
# Direct training (uses Config defaults)
python poker_rl_agent/scripts/train.py
```
Available presets: `debug`, `quadro_medium`, `quadro_long`, `quadro_stage_c_fcpa`, `quadro_stage_d_fchpa`, `quadro_stage_e_fullgame_progressive`, and various continuation/ablation configs — see `configs/training_configs.yaml`.

### Evaluation
```bash
# Full benchmark report
python poker_rl_agent/scripts/evaluate_complete.py \
  --checkpoint checkpoints/latest.pt \
  --config_file configs/training_configs.yaml \
  --config_name <preset> \
  --episodes_per_seed 5000

# Gate check on eval JSON
python poker_rl_agent/scripts/check_eval_gates.py \
  --eval_json <path> --ci_floor 830 --require_behavior_extended
```

### Interactive Play
```bash
python poker_rl_agent/scripts/play_against_agent.py \
  --checkpoint checkpoints/latest.pt \
  --config_name quadro_stage_b \
  --game_mode fullgame --hands 20
```

### Serving (FastAPI)
```bash
uvicorn poker_rl_agent.serving.app:app --reload
```

### Tests
```bash
pytest tests/
```

### HPC (SLURM)
SLURM batch scripts live in `scripts/*.slurm`. Always activate venv (`source venv/bin/activate`) and request an interactive session (`interact`) before running Python on HPC.

## Architecture

### Training Pipeline
```
train.py → Trainer
  ├─ SelfPlayWorker.generate_episode()    # on-policy trajectories
  │    └─ opponent from: league (PFSP) / random / exploit agents
  ├─ RolloutBuffer.compute_advantages()   # GAE(γ=0.995, λ=0.95)
  ├─ PPO update (clipped loss + value loss + entropy)
  ├─ LeagueManager (K-best snapshots, Elo tracking)
  ├─ Evaluator (periodic: winrate, behavior gates, exploitation)
  └─ Checkpointing (v2 format)
```

### Neural Network: `AlphaHoldemNetwork`
Pseudo-Siamese architecture with two towers merged into a shared trunk:

- **CardTower** — 53-token embedding (52 cards + unknown), MLP with LayerNorm. Input: 7 card indices (2 hole + 5 community).
- **ActionTower** — Action embedding → 2-layer LSTM. Processes betting history sequences (padded to `max_action_history=64`).
- **Trunk** — Concatenates card features + action features + 11D scalar features (stacks, pot, phase). Dense layers with LayerNorm and optional residual blocks.
- **Policy head** → action logits (masked to legal actions).
- **Value head** → scalar state value.

### State Representation (`state_representation.py`)
Card indices (0-51, 52=unknown), action history as padded integer sequence, 11D scalar features normalized by starting stacks, and a legal action binary mask.

### Betting Abstractions
- `fcpa` (4 actions): fold, call, pot-raise, all-in
- `fchpa` (5 actions): fold, call, half-pot, pot-raise, all-in
- `fullgame`: unrestricted raise sizes (with optional curriculum via `action_curriculum.py`)

Set via `BETTING_ABSTRACTION` in config. `STRICT_ABSTRACTION=true` enforces fail-fast on mismatches.

### Multi-Stage Development
The project progresses through stages with increasing complexity:
- **Stage C** — FCPA (4-action) training, 10k iterations
- **Stage D** — FCHPA (5-action) bridge, ablation studies, corrective continuations
- **Stage E** — Fullgame progressive (unrestricted actions with curriculum)

Each stage has dedicated SLURM scripts, eval profiles, and checkpoint management workflows.

## Key Files

| Component | Path |
|-----------|------|
| Config dataclass (170+ hyperparameters) | `poker_rl_agent/utils/config.py` |
| Training presets (YAML) | `configs/training_configs.yaml` |
| Main training loop | `poker_rl_agent/training/trainer.py` |
| Self-play trajectory generation | `poker_rl_agent/algorithms/self_play.py` |
| League management (K-best + PFSP) | `poker_rl_agent/training/league_manager.py` |
| Rollout buffer + GAE | `poker_rl_agent/training/rollout_buffer.py` |
| Network definition | `poker_rl_agent/models/alpha_holdem_net.py` |
| Card tower | `poker_rl_agent/models/card_tower.py` |
| Action tower (LSTM) | `poker_rl_agent/models/action_tower.py` |
| OpenSpiel wrapper | `poker_rl_agent/environment/openspiel_wrapper.py` |
| State encoding | `poker_rl_agent/environment/state_representation.py` |
| Evaluator + behavior gates | `poker_rl_agent/evaluation/evaluator.py` |
| Baseline agents | `poker_rl_agent/evaluation/baseline_agents.py` |
| Checkpoint manager | `poker_rl_agent/scripts/checkpoint_manager.py` |
| Fullgame action curriculum | `poker_rl_agent/training/action_curriculum.py` |

## Stability Notes

Gradient explosion has been a recurring issue. Current mitigations (V2 fixes):
- Input normalization (divide by 10000), target regret normalization (std)
- LayerNorm in all towers, Xavier initialization, depth=2
- Constant LR=5e-5, batch=64, gradient clipping=0.5
- NaN checks active, regret clipping [-10k, 10k]
- Policy/value head weights initialized uniform(-0.01, 0.01)
