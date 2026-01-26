# AlphaHoldEm Poker RL Agent

This repository contains a PyTorch implementation of a Poker Reinforcement Learning agent inspired by the **AlphaHoldEm** architecture. It uses a **Pseudo-Siamese Neural Network** trained with **Deep CFR+** (Counterfactual Regret Minimization) within the **OpenSpiel** environment.

Repository: [smdesai27/anitgravity-txhm](https://github.com/smdesai27/anitgravity-txhm)

## 📌 Features
- **Pseudo-Siamese Architecture**: Separate towers for Card embeddings and Action history processing.
- **Deep CFR+ Training**: Replaces tabular CFR with neural network function approximation.
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
Train the agent using Self-Play and Deep CFR:
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
- **Head**: Predicts cumulative regrets for each legal action.

### Algorithm
- **External Sampling MCCFR**: Traverses the game tree for the current player, updating the Regret Network based on counterfactual values.
- **Regret Matching+**: Converts predicted regrets into a strategy for exploring the game tree.

## 📂 Structure
- `poker_rl_agent/`
  - `models/`: PyTorch model definitions.
  - `algorithms/`: CFR logic and self-play.
  - `environment/`: OpenSpiel wrappers.
  - `training/`: Training loop and buffer.
  - `scripts/`: Executable scripts.

## ⚠️ Notes
- Training generally requires `open_spiel`'s `universal_poker` game with an ACPC definition for full HUNL. By default, this repo falls back to `leduc_poker` or simplified config if HUNL isn't fully configured in your OpenSpiel install.
- Deep CFR requires significant compute.
