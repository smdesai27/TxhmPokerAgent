# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaHoldEm Poker RL Agent: a PyTorch reinforcement learning agent for Heads-Up No-Limit Texas Hold'em (HUNL). Uses a pseudo-siamese neural network trained with PPO + K-best self-play league via OpenSpiel.

## Commands

### Install
```bash
pip install -r requirements.txt
# or with optional deps:
pip install -e ".[dev,serving]"
```

### Run Tests
```bash
pytest tests/ -xvs
# Single test:
pytest tests/test_stability.py::test_ppo_training_step -xvs
```

### Train (local debug)
```bash
python scripts/debug_training.py --config_name debug
```

### Train (with YAML preset)
```bash
python scripts/debug_training.py --config_file configs/training_configs.yaml --config_name <preset>
```

### Evaluate
```bash
python poker_rl_agent/scripts/evaluate.py --checkpoint checkpoints/latest.pt --episodes_per_seed 1000
```

### Serve (FastAPI)
```bash
python -m poker_rl_agent.serving.app --checkpoint checkpoints/latest.pt --config_file configs/training_configs.yaml --config_name <preset>
```

### Interactive Play
```bash
python poker_rl_agent/scripts/play_against_agent.py --checkpoint checkpoints/latest.pt
```

## Architecture

### Neural Network (pseudo-siamese)
`poker_rl_agent/models/alpha_holdem_net.py` — `AlphaHoldemNetwork`:
- **Card Tower** (`card_tower.py`): Embeds 7 cards (2 hole + 5 community) via shared 53-token embedding (52 cards + unknown), 3 FC layers with LayerNorm.
- **Action Tower** (`action_tower.py`): LSTM over betting history with packed sequences. Embeds `num_actions + 1` tokens (extra = padding).
- **Shared Trunk**: Concatenates card features + action features + 11 scalar features → Linear → 2 ResidualBlocks (with LayerNorm, not BatchNorm).
- **Heads**: Policy logits + scalar value. Both initialized near-zero for stability.
- Masking is applied *post-forward* via `masked_logits()` (not inside the network).

### Training Pipeline
`poker_rl_agent/training/trainer.py` — main loop:
1. Sample opponent from K-best league (`league_manager.py`) with PFSP weighting
2. Collect on-policy trajectories via `SelfPlayWorker` (`algorithms/self_play.py`)
3. Compute GAE advantages in `RolloutBuffer` (`training/rollout_buffer.py`)
4. PPO update with clipped policy loss, Huber value loss, entropy regularization
5. Periodically snapshot to league, evaluate vs MCCFR-ES baseline, checkpoint

### Environment
`poker_rl_agent/environment/openspiel_wrapper.py` — `PokerEnv`: wraps OpenSpiel's universal_poker. Three betting abstractions: `fcpa` (4 actions), `fchpa` (5 actions, alias `fcpha`), `fullgame`.

`poker_rl_agent/environment/state_representation.py` — `StateEncoder`: encodes OpenSpiel state into tensors (hole_cards, community_cards, action_history, legal_action_mask, scalars).

### Configuration
`poker_rl_agent/utils/config.py` — `Config` dataclass with 150+ fields. Resolution order: defaults → YAML `defaults` section → named preset → env var overrides. Presets live in `configs/training_configs.yaml`.

### Evaluation
`poker_rl_agent/evaluation/evaluator.py` — multi-baseline evaluation with behavior gates (fold frequency, aggression, entropy envelopes, pot-mix distribution). Eval scripts in `poker_rl_agent/scripts/evaluate_complete.py` and `check_eval_gates.py`.

### Multi-Stage Training Pipeline
Training progresses through stages with increasing action space complexity:
- **Stage C**: FCPA (4 actions), 10k iterations, strict behavior gates
- **Stage D**: FCHPA (5 actions), 12k+ iterations, continuation from Stage C with corrective fine-tuning and ablation studies
- **Stage E**: Fullgame with progressive curriculum (phase 1: max 1x pot, phase 2: max 2.5x pot, phase 3: full)

### Checkpointing
`poker_rl_agent/training/checkpointing.py` — format v2 `.pt` files containing model/optimizer/scheduler state, league snapshots, config hash. Resume via `RESUME_FROM` config field.

### Key Dependencies
- Python 3.10+, PyTorch 2.1+, OpenSpiel 1.2+, numpy <2.0
- W&B for experiment tracking (offline mode by default)
- FastAPI + uvicorn for serving

## Conventions
- Betting abstraction aliases: `fcpha` is accepted as alias for `fchpa`
- `STRICT_ABSTRACTION=true` for experiment integrity (fail-fast on action mismatch)
- LayerNorm everywhere (not BatchNorm) due to PPO batch instability
- Config field `CFR_ITERATIONS` is the main training duration parameter (name is legacy)
- HPC jobs use SLURM scripts in `scripts/slurm_*.slurm`
