# Project Memory & Context

## Operational Constraints (HPC)
> [!IMPORTANT]
> **HPC Interactive Session Required**
> Before running any Python scripts or training commands in the shell, you must request an interactive session.
> Command: `interact`
>
> **Virtual Environment**
> Always activate the virtual environment:
> `source venv/bin/activate`

## Project Overview
**Poker RL Agent**: A counterfactual regret minimization (CFR+) agent using a Pseudo-Siamese Neural Network (AlphaHoldem style).

### Key Components
- **Framework**: PyTorch + OpenSpiel
- **Algorithm**: Deep CFR / CFR+ with External Sampling
- **Network**: `AlphaHoldemNetwork` (Card Tower + Action Tower)

### Key File Map
- **Config**: `poker_rl_agent/utils/config.py` - Hyperparameters and architecture settings.
- **Trainer**: `poker_rl_agent/training/trainer.py` - Main training loop, buffer management, loss computation.
- **Worker**: `poker_rl_agent/algorithms/self_play.py` - Trajectory generation, MCCFR logic, Regret computation.
- **Model**: `poker_rl_agent/models/alpha_holdem_net.py` - Neural network definition.
- **State**: `poker_rl_agent/environment/state_representation.py` - Feature encoding and normalization.

## Emergency Status (V2 Fixes - Jan 2026)
Addressed critical gradient explosion and stability issues.
- **Normalization**: Input scaling (/10000), Target Regret normalization (std).
- **Architecture**: LayerNorm everywhere, Xavier Init, Depth=2.
- **Optimization**: Constant LR=5e-5, Batch=64, GradClip=0.5.
- **Safety**: NaN checks active, Regret Clipping [-10k, 10k].
