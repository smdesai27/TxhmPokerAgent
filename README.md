# Heads-Up No-Limit Hold'em — an end-to-end self-play RL agent

A from-scratch PyTorch reinforcement-learning system for **Heads-Up No-Limit Texas Hold'em (HUNL)**,
**inspired by** the AlphaHoldem architecture ([Zhao et al., AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20394)).
It trains an actor-critic with **PPO + GAE**, a **K-best self-play league**, and a multi-stage betting-abstraction
curriculum, and ships with an **automated evaluation harness** and a **live play-against-the-agent web demo**.

- **Repo:** [smdesai27/TxhmPokerAgent](https://github.com/smdesai27/TxhmPokerAgent)
- **Live demo:** [txhm-poker-api.onrender.com](https://txhm-poker-api.onrender.com) — play HUNL against the agent (free instance; first hand may take ~30s to wake)
- **Writeup & slides:** [docs/WRITEUP.md](docs/WRITEUP.md) · [docs/AlphaHoldem_RL_deck.pdf](docs/AlphaHoldem_RL_deck.pdf)
- **Best agent:** Stage D FCHPA (5-action) checkpoint at 21k steps — `checkpoints/snapshots/interview_ready/interview_ready_1.pt`

> **Scope note (please read).** This is an *AlphaHoldem-inspired* system, **not a faithful reproduction** of the
> paper. It keeps the two-tower pseudo-siamese actor-critic silhouette and search-free single-forward-pass
> inference, but deliberately substitutes an embedding+LSTM front-end, vanilla+regularized PPO, and a discrete
> betting abstraction so the whole thing trains on a single GPU. See [Differences from the paper](#differences-from-the-paper).

---

## What's actually here

- **Pseudo-siamese network** ([`models/alpha_holdem_net.py`](poker_rl_agent/models/alpha_holdem_net.py)): an
  untied **card tower** (53-token embedding → MLP) and **action tower** (LSTM over betting history), late-fused
  with an 11-d scalar feature stream into a residual trunk, then split into **policy + value** heads. LayerNorm
  (not BatchNorm) for small-batch PPO stability; legal-action masking applied post-forward in a dtype-safe way.
- **PPO + GAE training** ([`training/trainer.py`](poker_rl_agent/training/trainer.py)): clipped policy loss,
  Huber value loss, entropy regularization, gradient clipping, non-finite skip-guards, and an optional adaptive-KL
  LR controller.
- **K-best self-play league** ([`training/league_manager.py`](poker_rl_agent/training/league_manager.py)): a
  rolling pool of past snapshots with rating-weighted opponent sampling, plus scripted/exploit opponents.
- **Multi-stage curriculum**: FCPA (4 actions) → FCHPA (5 actions), with a full-game stage scaffolded.
- **Evaluation harness** ([`evaluation/`](poker_rl_agent/evaluation)): multi-seed win-rate with confidence
  intervals, holdout-seed independence checks, a zero-iteration control, and **behavior gates** (fold frequency,
  aggression, entropy, pot-mix) that act as automated regression checks against policy collapse.
- **Serving + demo** ([`serving/`](poker_rl_agent/serving), [`public/`](public)): a FastAPI backend (one masked
  forward pass per bot action) and a browser frontend to play HUNL against the agent.

## Results — honestly

The Stage D FCHPA **21k champion** passes the full gate suite (solver-tier win-rate floors, holdout, and all
behavior/aggression gates) and learned a **non-degenerate, balanced** strategy:

| Metric (champion, 21k FCHPA) | Value |
|---|---|
| Preflop entropy | 1.44 bits |
| Fold / call / raise mix | 46% / 43% / 11% |
| Behavior + aggression gates | all pass |

**Important caveats about the win-rate numbers:**

- Reported win-rates are **vs scripted baselines and an MCCFR external-sampling opponent built in-house** — *not*
  vs Slumbot/DeepStack and *not* vs a tuned solver. On the full 200-BB game the MCCFR opponent is near-uniform on
  most unseen information states, so large bb/100 margins mostly reflect a weak opponent. Treat them as a sanity
  signal, not a strength claim, and **do not compare them to AlphaHoldem's published mbb/h figures.**
- An opponent-independent metric (**exploitability / NashConv on the FCPA abstraction**, where it is tractable) is
  the right yardstick and is on the roadmap — the code path exists in
  [`evaluation/evaluator.py`](poker_rl_agent/evaluation/evaluator.py) but is disabled by default.
- The most recent long run (`checkpoints/latest.pt`, ~49k steps) **failed its behavior gates** (entropy collapse →
  fold-heavy play) and is intentionally **not** the deployed model. The behavior gates catching that is a feature,
  not a footnote.

## Differences from the paper

| Axis | AlphaHoldem (paper) | This repo |
|---|---|---|
| Card / action input | multi-channel 4×13 / 4×9 binary tensors → ConvNets | 53-token embedding → MLP / LSTM over betting history |
| Loss | **Trinal-Clip PPO** (ratio + negative-advantage + dynamic value-target clips) | vanilla clipped PPO + Huber value + optional behavior regularizer |
| Self-play | top-K **by Elo** | top-K by rolling rollout score; rating-temperature opponent sampling |
| Game scope | abstraction-free full no-limit game | FCPA (4) / FCHPA (5) discrete betting abstraction |
| Inference | search-free, single forward pass | search-free, single forward pass ✔ |
| Params | ~8.6M (ConvNet + FC) | ~1.5M (LSTM-heavy) |

What is genuinely **reproduced**: the two-tower pseudo-siamese actor-critic shape, search-free single-pass
inference, and a self-play-league training loop. What is a **deliberate deviation**: the input representation
(embedding/LSTM vs ConvNet tensors), the loss (vanilla vs trinal-clip), and the betting abstraction (FCPA/FCHPA
vs full game). What is an **extension**: the multi-stage curriculum and the behavior-gate evaluation harness.

---

## Setup

```bash
pip install -r requirements.txt
```

> **OpenSpiel note.** `open_spiel` publishes Linux (manylinux) wheels only. On macOS/Windows you must build it
> from source (see the [OpenSpiel install guide](https://github.com/deepmind/open_spiel)). The Docker/Render
> deploy path is Linux and works out of the box.

## Usage

```bash
# Train (small debug preset)
python scripts/debug_training.py --config_name debug

# Train a real preset (see configs/training_configs.yaml for the full list)
python scripts/debug_training.py --config_file configs/training_configs.yaml --config_name quadro_stage_c_fcpa

# Evaluate a checkpoint (multi-seed, with behavior gates)
python poker_rl_agent/scripts/evaluate_complete.py \
  --checkpoint checkpoints/snapshots/interview_ready/interview_ready_1.pt \
  --config_file configs/training_configs.yaml --config_name quadro_stage_d_fchpa \
  --episodes_per_seed 5000 --profile standard --output_json logs/eval_champion.json

# Serve the demo backend (defaults to the champion checkpoint)
python -m poker_rl_agent.serving.app --host 0.0.0.0 --port 8000

# Play against the agent in the terminal
python poker_rl_agent/scripts/play_against_agent.py \
  --checkpoint checkpoints/snapshots/interview_ready/interview_ready_1.pt \
  --config_name quadro_stage_d_fchpa --game_mode fchpa --hands 20
```

Training presets (debug, Stage A–E, the Stage D recovery/ablation families) live in
[`configs/training_configs.yaml`](configs/training_configs.yaml). SLURM job scripts for the Brown OSCAR cluster are
under [`scripts/`](scripts) (active) and [`scripts/archived/`](scripts/archived) (historical).

## Configuration (selected)

Set in [`poker_rl_agent/utils/config.py`](poker_rl_agent/utils/config.py) or a YAML preset:

- `BETTING_ABSTRACTION`: `fcpa` (4), `fchpa` (5, alias `fcpha`), or `fullgame`.
- `STRICT_ABSTRACTION`: `true` to fail fast on any abstraction mismatch (recommended).
- `CFR_ITERATIONS`: training duration (legacy name — it is the PPO iteration count, not CFR).
- W&B logging via `WANDB_MODE` (`online` / `offline` / `disabled`) and `WANDB_API_KEY`/`WANDB_PROJECT`/`WANDB_ENTITY`.

## Tests

```bash
pytest tests/ -xvs
```

Covers the state encoder, the dual-head network + masking, the OpenSpiel wrapper and abstraction guards, the
baseline/MCCFR utilities, league sampling, Trainer construction + PPO stability metrics, and the serving layer.
(Requires a working `open_spiel` install, i.e. Linux or a source build.)

## Repository layout

```
poker_rl_agent/
  models/        pseudo-siamese network (card tower, action tower, trunk, heads)
  algorithms/    self-play workers + baseline/solver utilities
  environment/   OpenSpiel wrapper + state encoding
  training/      PPO trainer, rollout buffer, league manager, checkpointing
  evaluation/    multi-seed evaluator, behavior gates, baseline agents
  serving/       FastAPI app + game manager
  scripts/       train / evaluate / play entry points
configs/         training presets (YAML)
public/          web demo frontend
```

## License

MIT — see [LICENSE](LICENSE).
