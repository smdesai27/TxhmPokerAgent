# AlphaHoldEm Poker RL Agent

This repository contains a PyTorch implementation of a Poker Reinforcement Learning agent inspired by the **AlphaHoldEm** architecture. It uses a **Pseudo-Siamese Neural Network** trained with **PPO + K-best self-play league** within the **OpenSpiel** environment.

Repository: [smdesai27/anitgravity-txhm](https://github.com/smdesai27/anitgravity-txhm)

## 📌 Features
- **Pseudo-Siamese Architecture**: Separate towers for Card embeddings and Action history processing.
- **Actor-Critic Training**: PPO objective with clipped policy updates and GAE advantages.
- **K-Best League Self-Play**: Opponents are sampled from a rolling top-K snapshot pool (PFSP weighting).
- **Heads-Up No-Limit Hold'em**: Designed for HUNL (Universal Poker), extensible to K players.
- **Self-Play**: Parallel data collection via self-play trajectories.
- **Evaluation**: Benchmarking against Random and Rule-based baselines.

## 🛠️ Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: OpenSpiel (`open_spiel`) is required. If pip install fails, refer to [DeepMind OpenSpiel](https://github.com/deepmind/open_spiel) installation guide.*

2. **Configuration**
   Adjust hyperparameters in `poker_rl_agent/utils/config.py`:
   - `GAME_NAME`: "universal_poker" (default) or "leduc_poker" for testing.
   - `CFR_ITERATIONS`: Training duration.
   - `DEVICE`: 'cuda' or 'cpu'.

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
python poker_rl_agent/scripts/evaluate.py --checkpoint checkpoints/point_100.pt --episodes 1000
```

### Interactive Play
Play against the agent in terminal:
```bash
python poker_rl_agent/scripts/play_against_agent.py
```

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
- WandB logging is enabled by default. Set `WANDB_mode=offline` if needed.

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
