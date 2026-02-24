# AlphaHoldEm Poker RL Agent — Interview Prep Guide

## Table of Contents

1. [Project Summary (Elevator Pitch)](#1-project-summary)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Neural Network Architecture — Deep Dive](#3-neural-network-architecture)
4. [Environment & State Representation](#4-environment--state-representation)
5. [Training Algorithm — PPO with K-Best Self-Play](#5-training-algorithm)
6. [League System & Opponent Scheduling](#6-league-system--opponent-scheduling)
7. [Evaluation Framework & Behavior Gates](#7-evaluation-framework--behavior-gates)
8. [Training Progression — Stage-by-Stage](#8-training-progression)
9. [Hyperparameter Tuning Story](#9-hyperparameter-tuning-story)
10. [Stability Engineering](#10-stability-engineering)
11. [Comparison with AlphaHoldem Paper](#11-comparison-with-alphaholdem-paper)
12. [Serving & Deployment](#12-serving--deployment)
13. [Testing Strategy](#13-testing-strategy)
14. [File-by-File Reference](#14-file-by-file-reference)
15. [Anticipated Interview Questions](#15-anticipated-interview-questions)

---

## 1. Project Summary

**What is this?** A reinforcement learning agent that plays Heads-Up No-Limit Texas Hold'em (HUNL) poker, trained via PPO (Proximal Policy Optimization) with a K-best self-play league.

**Inspired by:** The AlphaHoldem paper (Zhao et al., AAAI 2022), but with significant divergences in architecture, algorithm, and training methodology.

**Key tech stack:** Python 3.10+, PyTorch, OpenSpiel (Google's game framework), NumPy, W&B, FastAPI, SLURM (HPC).

**What makes it interesting:**
- Pseudo-siamese neural network with separate card and action processing towers
- Multi-stage training curriculum (FCPA → FCHPA → Fullgame) with increasing action complexity
- Extensive behavior-gated evaluation against MCCFR-ES solver baselines
- Automated hyperparameter ablation studies tracked across 20+ named config presets
- Champion model achieves bb/100 = 898 against solver baseline (CI95 lower = 815)

---

## 2. High-Level Architecture

```
                    ┌──────────────────────────────────────┐
                    │         scripts/debug_training.py     │
                    │  (entry point — loads YAML config)    │
                    └───────────────┬──────────────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────────────┐
                    │           Trainer (trainer.py)         │
                    │  Orchestrates the entire training loop │
                    └───┬───────┬───────┬──────┬───────┬───┘
                        │       │       │      │       │
         ┌──────────────▼─┐   ┌─▼──────┐   ┌──▼────┐  │  ┌──▼───────────┐
         │ SelfPlayWorker  │   │Rollout │   │ PPO   │  │  │  Evaluator   │
         │ (self_play.py)  │   │Buffer  │   │Update │  │  │(evaluator.py)│
         │ Generates       │   │(GAE)   │   │       │  │  │  vs solver,  │
         │ episodes        │   │        │   │       │  │  │  baselines   │
         └───────┬─────────┘   └────────┘   └───────┘  │  └──────────────┘
                 │                                      │
     ┌───────────▼────────────┐              ┌──────────▼──────────┐
     │   AlphaHoldemNetwork   │              │   LeagueManager     │
     │  (alpha_holdem_net.py) │              │ (league_manager.py) │
     │  CardTower + ActionTower│              │ K-best + Elo + PFSP │
     │  → Trunk → Policy/Value │              └─────────────────────┘
     └───────────┬────────────┘
                 │
     ┌───────────▼────────────┐
     │   PokerEnv (OpenSpiel) │
     │  (openspiel_wrapper.py)│
     │  + StateEncoder        │
     └────────────────────────┘
```

**Data flow per training iteration:**
1. `_collect_rollouts()` — SelfPlayWorker plays games (model vs league/random/exploit opponent), collects (state, action, reward, log_prob, value) tuples
2. `RolloutBuffer.compute_advantages()` — Computes GAE (Generalized Advantage Estimation) with gamma=0.995, lambda=0.95
3. `_ppo_update()` — Runs multiple epochs of minibatch PPO updates with clipped surrogate loss
4. Periodically: `Evaluator.evaluate()` runs assessment against solver, `LeagueManager` stores snapshots

---

## 3. Neural Network Architecture

### AlphaHoldemNetwork (`poker_rl_agent/models/alpha_holdem_net.py`)

**Design:** Pseudo-siamese architecture — two separate input towers that process different modalities, then merge into a shared trunk.

```
  Hole Cards (2 ints) ──► CardTower ──────────┐
  Community Cards (5 ints) ──┘                  │
                                                ├──► Concatenate ──► Trunk ──► Policy Head (logits)
  Action History (var len) ──► ActionTower ────┘                           └──► Value Head (scalar)
  Scalar Features (11D) ─────────────────────────┘
```

#### CardTower (`poker_rl_agent/models/card_tower.py`)
- **Input:** 7 card indices (0–51, 52=unknown for unrevealed community cards)
- **Embedding:** `nn.Embedding(53, EMBEDDING_DIM=64)` — 52 cards + 1 unknown token
- **Processing:** Flatten 7 embeddings (7 * 64 = 448D) → MLP with `NUM_LAYERS_CARD=3` dense layers, each: Linear → LayerNorm → ReLU → Dropout(0.1)
- **Output:** 256D feature vector

**Why this design (vs. AlphaHoldem paper):** The paper uses a 4x13x6 binary tensor processed by CNN. This project uses integer-indexed embeddings because:
- Simpler implementation, fewer parameters
- Embeddings learn card-specific representations end-to-end
- The "unknown" token (index 52) naturally handles unrevealed community cards without masking

#### ActionTower (`poker_rl_agent/models/action_tower.py`)
- **Input:** Variable-length sequence of action indices, padded to `MAX_ACTION_HISTORY=64`
- **Embedding:** `nn.Embedding(num_actions+1, EMBEDDING_DIM, padding_idx=num_actions)` — uses the extra index as padding
- **Processing:** 2-layer LSTM with `pack_padded_sequence` for efficiency
- **Output:** Final hidden state hx[-1] → 256D feature vector

**Why LSTM (not Transformer):** Betting histories are relatively short (typically <20 actions per hand). LSTM handles variable-length sequences with pack_padded_sequence, is simpler, and has lower overhead for this sequence length.

#### Trunk
- Takes concatenated [card_features, action_features, scalar_features] = 256 + 256 + 11 = 523D
- Dense layers with LayerNorm, ReLU, Dropout(0.1)
- Optional `ResidualBlock` layers (when `USE_RESIDUAL=True`): each block has LayerNorm → Linear → ReLU → Dropout → Linear → residual connection

#### Output Heads
- **Policy head:** Linear(hidden_dim, num_actions) → logits (masked to legal actions via `masked_logits`)
- **Value head:** Linear(hidden_dim, 1) → scalar state value estimate

#### Weight Initialization
- All Linear layers: Xavier uniform (`init_weights` in `model_utils.py`)
- LSTM hidden-to-hidden weights: Orthogonal initialization
- **Policy/value heads:** Small uniform(-0.01, 0.01) — prevents initially overconfident predictions
- This was a critical stability fix (see Section 10)

#### masked_logits (`poker_rl_agent/models/model_utils.py`)
```python
def masked_logits(logits, legal_mask):
    # Sets illegal action logits to dtype-safe minimum
    # Handles fp16/bf16 by using finfo(dtype).min
    floor = torch.finfo(logits.dtype).min
    return logits.masked_fill(~legal_mask.bool(), floor)
```
This ensures the softmax over logits assigns zero probability to illegal actions.

---

## 4. Environment & State Representation

### PokerEnv (`poker_rl_agent/environment/openspiel_wrapper.py`)

Wraps OpenSpiel's `universal_poker` game with HUNL-specific configuration:
- **Stack size:** 20,000 chips (200 big blinds)
- **Blinds:** 50/100
- **Players:** 2 (heads-up)

**Betting Abstractions** — control the discrete action space:
| Abstraction | Actions | Description |
|---|---|---|
| `fcpa` (4) | fold, call, pot-raise, all-in | Simplest, fastest to train |
| `fchpa` (5) | fold, call, half-pot, pot-raise, all-in | Adds a medium bet size |
| `fullgame` | All legal raise sizes | Unrestricted (with optional curriculum) |

**Strict Abstraction Mode (`STRICT_ABSTRACTION=True`):** If the environment returns actions outside the expected set, raises an error immediately. This is a fail-fast mechanism — critical for catching configuration mismatches between training and evaluation.

**Alias handling:** The code normalizes "fcpha" → "fchpa" (a common typo) in `_normalize_abbreviation()`.

### StateEncoder (`poker_rl_agent/environment/state_representation.py`)

Transforms raw OpenSpiel game state into tensors:

| Feature | Shape | Description |
|---|---|---|
| `hole_cards` | (2,) int | Player's private cards, 0-51 |
| `community_cards` | (5,) int | Board cards, 52 for unknown |
| `action_history` | (N,) int | Variable-length sequence of action IDs |
| `action_history_length` | scalar | Actual length (for pack_padded_sequence) |
| `scalars` | (11,) float | Normalized game features |
| `legal_action_mask` | (num_actions,) bool | Which actions are legal |

**11D Scalar Features:**
1. pot_size / starting_stack (normalized)
2. hero_stack / starting_stack
3. opponent_stack / starting_stack
4. hero_contribution / starting_stack
5. opponent_contribution / starting_stack
6-9. Street one-hot encoding [preflop, flop, turn, river]
10. seat_flag (0 or 1 — which seat the hero is in)
11. acting_flag (1.0 if it's hero's turn)

**Normalization by starting_stack (20000):** All monetary values are divided by 20000, keeping inputs in [0,1] range. This was a V2 stability fix — raw chip counts caused gradient explosion.

---

## 5. Training Algorithm

### PPO with K-Best Self-Play (`Trainer` in `poker_rl_agent/training/trainer.py`)

**Algorithm choice: PPO, not CFR or IMPALA**
- The AlphaHoldem paper uses IMPALA (distributed off-policy actor-critic with V-trace)
- This project uses PPO (on-policy, clipped surrogate) because:
  - Simpler to implement and debug
  - Mature PyTorch ecosystem support
  - Better suited for single-GPU training (no distributed actor infrastructure needed)
  - On-policy is more sample-efficient for self-play where the data distribution shifts constantly

### Training Loop Flow (per iteration)

```python
for iteration in range(start_step, total_iterations):
    # 1. Compute scheduled values
    progress = effective_schedule_progress(iteration)
    temperature = policy_temperature(progress)    # 1.20 → 1.00
    entropy_coef = entropy_coef(progress)         # 0.010 → 0.003
    style_coef = style_reg_coef(progress)         # 0.020 → 0.005

    # 2. Collect rollouts
    _collect_rollouts(progress, temperature)

    # 3. Compute advantages (GAE)
    rollout_buffer.compute_advantages(gamma=0.995, lam=0.95)

    # 4. PPO update
    _ppo_update(entropy_coef, style_coef)

    # 5. Periodic evaluation
    if iteration % EVAL_FREQ == 0:
        evaluator.evaluate()

    # 6. League snapshot
    if iteration % SNAPSHOT_INTERVAL == 0:
        league.add_snapshot(model, step=iteration)

    # 7. Checkpoint
    if iteration % CHECKPOINT_FREQ == 0:
        save_checkpoint(...)
```

### PPO Update (`_ppo_update`)

The PPO update iterates over `PPO_EPOCHS=2` epochs, each time shuffling the buffer into minibatches of size `PPO_MINIBATCH_SIZE=256`:

**Loss computation:**
```
total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus + style_coef * style_loss
```

Where:
- **Policy loss (clipped surrogate):**
  ```
  ratio = exp(new_log_prob - old_log_prob)
  surr1 = ratio * advantages
  surr2 = clamp(ratio, 1-eps, 1+eps) * advantages
  policy_loss = -min(surr1, surr2).mean()
  ```
  With `PPO_CLIP_EPS=0.2` (later tightened to 0.15 in final optimization)

- **Value loss (Huber):**
  ```
  scaled_returns = returns / VALUE_TARGET_SCALE  # divide by 20.0
  value_loss = huber_loss(value_pred, scaled_returns, delta=10.0)
  ```
  Returns are scaled down by 20x to keep gradients manageable (chip returns can be ±20000)

- **Entropy bonus:** `entropy = -sum(p * log(p))` over the policy distribution — encourages exploration

- **Style regularization (optional):** KL divergence between the agent's preflop action distribution and a solver-derived target profile + hinge loss on pot-raise frequency

**Early stopping on KL divergence:**
If the mean KL divergence between old and new policies exceeds `PPO_TARGET_KL`, the remaining minibatches in that epoch are skipped. This prevents catastrophic policy updates.

**Gradient clipping:** `torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP=0.5)`

**Non-finite gradient skipping:** If any gradient is NaN/Inf, the entire update step is skipped (`SKIP_NONFINITE_GRAD=True`)

### Rollout Buffer (`poker_rl_agent/training/rollout_buffer.py`)

Stores transitions on CPU (to save GPU memory). Key features:
- **GAE computation** (`compute_advantages`): Reverse sweep with `gamma=0.995, lambda=0.95`
  ```python
  for t in reversed(range(len(buffer))):
      delta = reward[t] + gamma * next_value - value[t]
      advantage = delta + gamma * lambda * next_advantage
  ```
- **Minibatch iteration** (`iterate_minibatches`): Shuffles indices, yields batches. Uses `_collate_states` with `pad_sequence` for variable-length action histories.

### Self-Play Worker (`poker_rl_agent/algorithms/self_play.py`)

`SelfPlayWorker.generate_episode()`:
1. Resets environment, gets initial state
2. Alternates between players until terminal
3. For the training player: sample action from model policy (with temperature)
4. For the opponent: use opponent_model or opponent_agent
5. Collects transitions only for train_player
6. Assigns the game's final return as the reward for the last transition

**Curriculum masking** (for fullgame): If a `FullgameActionCurriculum` is active, applies `mask_for_state()` to restrict raise sizes based on training progress.

---

## 6. League System & Opponent Scheduling

### LeagueManager (`poker_rl_agent/training/league_manager.py`)

Maintains a pool of the **K-best historical policy snapshots** (K=16 in later stages).

**Mechanism:**
1. Every `SNAPSHOT_INTERVAL` iterations, the current model is deep-copied and scored
2. The league keeps only the K-best entries, sorted by (score, rating)
3. When selecting an opponent, uses **PFSP (Prioritized Fictitious Self-Play)**:
   ```python
   weights = softmax(ratings * pfsp_beta)
   opponent = weighted_random_choice(league_entries, weights)
   ```
   Higher-rated opponents are more likely to be chosen (controlled by `PFSP_BETA`)

**Elo tracking:**
After each episode against a league opponent, Elo ratings are updated:
```python
expected = 1 / (1 + 10^((opponent_rating - current_rating) / 400))
new_rating = old_rating + K * (result - expected)
```
With `K_FACTOR=24.0` and `INIT_RATING=1200.0`

### Opponent Mix

Each training episode selects an opponent type:
- **League opponents** (60-85%): Past strong versions of the agent — prevents catastrophic forgetting
- **Exploit opponents** (0-25%): Deterministic baseline agents (AlwaysCall, PotPressure, StickyCall, PassiveCaller) — ensures the agent can exploit weak strategies
- **Random opponents** (10-30%): Ensures broad state space coverage

The mix evolved through training stages:
- Early stages: 80% league, 20% random
- Later stages: 65-75% league, 15-25% exploit, 10% random

---

## 7. Evaluation Framework & Behavior Gates

### Evaluator (`poker_rl_agent/evaluation/evaluator.py`)

Multi-tier evaluation system:

**1. Winrate vs. MCCFR-ES Solver:**
- Uses OpenSpiel's `external_sampling_mccfr_solver` with 5000-10000 iterations
- Plays 200-300 episodes per seed, with 3 random seeds
- Reports mean bb/100 (big blinds per 100 hands) and 95% confidence interval
- **Solver is cached** per-process for speed

**2. Behavior Gates (Binary pass/fail checks):**

The agent must pass ALL gates to qualify as a valid checkpoint:

| Gate | Threshold | Purpose |
|---|---|---|
| max_fold_freq | < 0.55-0.78 | Prevent over-folding |
| max_allin_freq | < 0.03-0.05 | Prevent gambling/all-in spam |
| min_entropy_bits | > 0.75-1.10 | Ensure strategy diversity |
| min_call_check_freq | > 0.40-0.47 | Ensure calling range exists |
| min_half_pot_freq | > 0.01 | Use the half-pot option |
| min_raise_total_freq | > 0.08 | Show aggression |
| envelope (fold range) | 0.35-0.55 | Realistic fold frequency band |

**Why gates matter:** Without gates, PPO can collapse to degenerate strategies (e.g., always fold, always all-in) that still win against specific opponents but play terribly in general.

**3. Exploitation Score:**
```
winner_score = CI95_lower + exploit_weight * min(exploit_bb100, clip)
```
Where `exploit_weight=0.20` and `clip=500.0` — this rewards both raw performance and ability to exploit weak opponents.

**4. Hand-Strength Correlation:**
Bins preflop decisions by coarse hand strength, checks that the agent's fold/call/raise frequencies make strategic sense (e.g., folding more with weak hands).

### evaluate_complete.py Script

The standalone evaluation script runs:
1. Multi-tier MCCFR-ES evaluation (default, robustness, holdout seeds)
2. All behavior gates (basic + extended)
3. Exploitation analysis vs. baseline agents
4. Hand-strength correlation
5. Outputs a comprehensive JSON report
6. Has configurable profiles: `quick`, `standard`, `exhaustive`, `interview`

---

## 8. Training Progression

### Stage A: Initial Training (FCPA, ~1500 iterations)
- **Config:** `quadro_stage_a`
- **Setup:** 4-action abstraction, 128 rollout episodes, hidden_dim=192
- **LR:** 1e-4, gradient clip 0.5
- **Result:** bb/100 ≈ 640 vs random baseline
- **Key observation:** Model learns basic fold/call/raise patterns quickly but policy entropy is low

### Stage B: Extended Training (FCPA, 9000 iterations)
- **Config:** `quadro_stage_b`
- **Setup:** Enabled `PPO_ADAPTIVE_KL` for the first time, K_BEST=12, 256 rollout episodes
- **LR:** 1e-4 with adaptive KL controller
- **Result:** bb/100 ≈ 617
- **Key lesson:** Adaptive KL prevents catastrophic policy updates during long runs

### Stage C: FCPA Continuation (~10000 additional iterations)
- **Config:** `quadro_stage_c_fcpa`
- **Setup:** Resumed from Stage B checkpoint, 384 rollout episodes, LR reduced to 7.5e-5
- **K_BEST:** Increased to 16, PFSP_BETA reduced to 1.5
- **Target KL:** Tightened to 0.01 (from 0.015)
- **Result:** bb/100 ≈ 367 vs strong solver (but still winning comfortably vs baselines)
- **Key observation:** Performance against solver is harder to improve; behavior became more realistic

### Stage D: FCHPA Bridge (5-action, 3000–21000+ iterations)

This stage introduced the **half-pot bet** (5th action). It had the most complex development history with multiple ablation studies and recovery branches.

#### Initial D (3000 iterations)
- **Config:** `quadro_stage_d_fchpa`
- Fresh start with 5-action abstraction to learn the new action
- Behavior gates initially failed due to low half-pot usage

#### D Continue (to 9000 iterations)
- **Config:** `quadro_stage_d_fchpa_continue`
- Added **exploration schedules**: temperature 1.20→1.00, entropy 0.010→0.003
- Introduced `LEAGUE_RANDOM_OPPONENT_PROB=0.20`
- Behavior gates began passing

#### D Ablation Studies (13500 iterations)
Three parallel ablation branches from the 9000-checkpoint, each testing different hypotheses:
1. **Explore ablate** (`recover_explore_ablate`): Higher temperature (1.12), higher entropy start (0.008)
2. **Diverse ablate** (`recover_diverse_ablate`): Higher random opponent prob (0.25), PFSP_BETA=0.9
3. **Value-stable ablate** (`recover_value_stable_ablate`): Higher value_coef (0.35), higher MIN_ROLLOUT_TRANSITIONS (1024)

#### D Recovery Rounds (16k → 17k → 18k → 19k → 20k → 21k)
Each round continued from the best checkpoint of the previous round:
- **18k anchor:** Established as recovery baseline with value_stable parameters
- **19k probes:** Balanced vs. Conservative — tested tight vs. relaxed exploration
- **20k:** Multiple competing configs (conservative, balanced, c1_strict_conservative, c2_halfpot_nudge, potmix variants)
- **21k champion:** bb/100 = 898, CI95 lower = 815 → **promoted as `interview_ready_1.pt`**

#### D Post-Champion (24k and beyond)
- **Config:** `quadro_stage_d_fchpa_evfirst_interview_continue_24k`
- Continued from `interview_ready_1.pt`
- Tightened LR to 6e-5, added exploit opponents (25% weight)
- Added envelope gates (preflop fold freq in [0.35, 0.55])
- **34k evaluation:** bb/100 dropped, CI95 lower = 748 (below 800 floor) → rejected

#### Final Optimization (34k → 39k → 49k)
- **Config:** `final_optimization`, `final_optimization_low_temp`
- PPO_CLIP_EPS tightened to 0.15 (from 0.2)
- K_BEST increased to 20, PFSP_BETA=0.8
- Increased EVAL_BASELINE_ITERS to 10000 for stronger solver
- Low-temp variant: temperature 0.90→0.85, entropy 0.002→0.001

### Stage E: Fullgame Progressive (planned)
- **Config:** `quadro_stage_e_fullgame_progressive`
- Full unrestricted action space with curriculum:
  - Phase 1 (0-30%): max raise = pot * 1.0x
  - Phase 2 (30-70%): max raise = pot * 2.5x
  - Phase 3 (70-100%): unrestricted

---

## 9. Hyperparameter Tuning Story

This is the story of how hyperparameters were adjusted through training, and why:

### Learning Rate: 1e-4 → 7.5e-5 → 6.5e-5 → 6e-5 → 5e-5
- **Rationale:** Lower LR as training progresses to avoid destabilizing well-learned policies
- Stage A/B: 1e-4 — fast initial learning
- Stage C: 7.5e-5 — slower, more stable updates
- Stage D recovery: 6.5e-5 — even more conservative for fine-tuning
- Final optimization: 5e-5 — minimal disturbance to champion policy

### Gradient Clipping: 0.5 (constant)
- Set to 0.5 throughout, after gradient explosion was identified as a recurring issue
- **History:** Default was 1.0, reduced to 0.5 after V2 stability fixes

### PPO Target KL: 0.015 → 0.01 → 0.008
- **Rationale:** Tighter KL constraint prevents catastrophic forgetting in later stages
- Combined with adaptive KL controller in later stages

### Rollout Episodes: 128 → 256 → 384 → 512
- **Rationale:** More episodes per iteration = lower variance in advantage estimates
- Trade-off: more compute per iteration but more stable learning signal

### MIN_ROLLOUT_TRANSITIONS: 192 → 512 → 768 → 1024 → 1152
- Ensures each PPO update has enough data, even if some episodes are very short

### K_BEST: 8 → 10 → 12 → 16 → 20
- **Rationale:** Larger league provides more diverse opponents
- Prevents the agent from overfitting to a single opponent style

### PFSP_BETA: 2.0 → 1.5 → 1.1 → 1.0 → 0.8
- **Lower PFSP_BETA** = more uniform sampling from the league (less focus on highest-rated)
- **Rationale:** As the league matures, want the agent to face a wider range of difficulty

### Temperature Schedule: 1.20→1.00 (decaying)
- **High temperature early:** More exploration, helps discover diverse strategies
- **Low temperature late:** Exploitation, sharpens the policy
- Different stages used different schedules (e.g., 1.15→1.00, 1.05→1.00, 1.02→1.00)
- **Final optimization:** Actually went sub-1.0 (0.90→0.85) for maximum exploitation

### Entropy Coefficient: 0.010→0.003 (decaying)
- **High early:** Forces exploration, prevents premature policy collapse
- **Low late:** Allows the policy to be more confident
- Different stages had different schedules based on observed behavior entropy

### PPO_VALUE_COEF: 0.25 → 0.30 → 0.35
- **Rationale:** Higher value coefficient emphasizes value function accuracy
- The value-stable ablation variant (VALUE_COEF=0.35) won multiple recovery rounds

### PPO_CLIP_EPS: 0.2 → 0.15
- **Final optimization only:** Tighter clipping prevents large policy swings

### PPO_ADV_CLIP: 5.0 → 4.0
- Clips extreme advantage values to prevent gradient spikes

---

## 10. Stability Engineering

Gradient explosion and training instability were recurring challenges. The fixes, collectively called "V2 stability":

### Problem: Gradient Explosion
Raw chip values (up to ±20,000) caused extremely large loss values and gradient norms, leading to NaN weights.

### Solutions Implemented:

1. **Input normalization:** All monetary features divided by starting_stack (20,000), keeping scalars in [0,1]

2. **Value target scaling:** `VALUE_TARGET_SCALE=20.0` — returns divided by 20 before computing value loss. This keeps the value head's outputs in a reasonable range.

3. **LayerNorm everywhere:** `USE_LAYERNORM=True` in CardTower, ActionTower, and trunk. Normalizes activations at each layer to prevent signal magnification.

4. **Xavier initialization:** All Linear layers use Xavier uniform init, which accounts for layer fan-in/fan-out to set appropriate initial weight magnitudes.

5. **Small head initialization:** Policy and value heads initialized with uniform(-0.01, 0.01) instead of Xavier. This prevents the model from being initially overconfident.

6. **Gradient clipping:** `GRAD_CLIP=0.5` — hard cap on gradient norm.

7. **NaN detection + skip:** `SKIP_NONFINITE_GRAD=True` — if any gradient is NaN or Inf, the entire optimizer step is skipped.

8. **Advantage clipping:** `PPO_ADV_CLIP=5.0` — extreme advantage values are clamped to [-5, 5].

9. **Huber loss for value function:** Instead of MSE (which amplifies large errors quadratically), Huber loss with `delta=10.0` transitions to linear loss for large errors.

10. **Explained variance monitoring:** Tracks how well the value function predicts returns. If variance of returns is below `EXPLAINED_VAR_VAR_FLOOR=1e-4`, the metric is marked invalid (prevents division-by-zero artifacts).

---

## 11. Comparison with AlphaHoldem Paper

### What's the Same

| Aspect | Paper | This Project |
|---|---|---|
| Game | HUNL poker | HUNL poker |
| Pseudo-siamese network | Yes | Yes |
| Two towers (card + action) | Yes | Yes |
| Self-play league | Yes | Yes |
| Policy + Value heads | Yes | Yes |
| OpenSpiel for game engine | Yes | Yes |

### Key Divergences

| Aspect | AlphaHoldem Paper | This Project | Rationale |
|---|---|---|---|
| **RL Algorithm** | IMPALA + V-trace (off-policy, distributed) | PPO (on-policy, single machine) | Simpler, no distributed infra needed, better for single-GPU |
| **Card Representation** | 4x13x6 binary tensor (rank x suit x hand_round) | Integer indices (0-51) with learned embeddings | More compact, learns card-specific features end-to-end |
| **Card Processing** | CNN (convolutional layers) | Embedding + MLP | No spatial structure to exploit — cards don't have "adjacency" like pixels |
| **Action Processing** | "Action encoder" (details sparse in paper) | LSTM with pack_padded_sequence | Explicit temporal modeling of betting sequence |
| **Training Infrastructure** | Distributed actors + central learner (IMPALA) | Single-machine self-play loop | Practical for academic/HPC setting |
| **League Mechanism** | Payoff-based priority, frozen policies | K-best with Elo tracking + PFSP sampling | More principled opponent selection |
| **Evaluation** | Against existing poker bots (Slumbot, etc.) | Against MCCFR-ES solver + behavior gates | More rigorous; solver provides game-theoretic baseline |
| **Behavior Gates** | None mentioned | Extensive (entropy, fold freq, allin freq, etc.) | Prevents degenerate strategies that might still win locally |
| **Action Abstraction** | Not explicitly discussed | Multi-stage FCPA → FCHPA → Fullgame curriculum | Progressive complexity increase |
| **Style Regularization** | Not mentioned | KL divergence to solver-derived preflop profile | Nudges policy toward game-theoretically sound play |
| **Value Loss** | MSE (implied) | Huber loss with scaled returns | Better gradient properties for high-variance poker returns |
| **Exploration** | Not detailed | Temperature schedules, entropy annealing, opponent mix scheduling | Systematic exploration→exploitation transition |

### Key Design Decisions to Discuss

**1. Why PPO instead of IMPALA?**
- IMPALA requires distributed actor infrastructure (Ray, NCCL, etc.)
- The off-policy correction (V-trace) adds complexity
- PPO is simpler to implement, debug, and reason about
- For a research/learning project, on-policy is more interpretable
- The trade-off is sample efficiency, but with a K-best league, stale experiences are avoided anyway

**2. Why embeddings instead of binary tensors?**
- A 4x13x6 tensor is sparse (mostly zeros) — wasteful
- Embeddings are a compact, dense representation that the network learns to organize
- Unknown community cards are naturally represented by a dedicated token
- The paper's CNN approach assumes spatial structure in cards that doesn't really exist in poker

**3. Why behavior gates?**
- This is an innovation not present in the paper
- PPO can converge to degenerate Nash equilibria in self-play (e.g., both players always fold)
- Gates act as regularization, ensuring the agent plays "real poker"
- Inspired by hand-crafted poker metrics used in professional player analysis

**4. Why Huber loss instead of MSE?**
- Poker returns are high-variance (you can win or lose 20000 chips)
- MSE amplifies large errors quadratically → gradient explosion
- Huber loss is linear for large errors, preventing gradient spikes
- Combined with VALUE_TARGET_SCALE=20.0, keeps gradients manageable

---

## 12. Serving & Deployment

### FastAPI Web Server (`poker_rl_agent/serving/`)

**Architecture:**
- `app.py` — FastAPI application with lifespan context manager
- `game_manager.py` — Thread-safe session management
- `schemas.py` — Pydantic models for request/response

**Endpoints:**
```
GET  /                      — Static HTML UI
GET  /api/v1/health         — Health check
POST /api/v1/session        — Create game session
GET  /api/v1/session/{id}   — Get current state
POST /api/v1/session/{id}/action   — Submit human action
POST /api/v1/session/{id}/new_hand — Start new hand
```

**Bot action selection:** The server loads a checkpoint, creates a model, and for each bot turn:
1. Encode the game state using StateEncoder
2. Forward pass through the model
3. Apply `masked_logits` for legal actions
4. Either `argmax` or `Categorical(logits=logits).sample()` based on `bot_policy`
5. Temperature scaling: `logits / temperature`

**Session management:** Thread-safe with per-session locks, TTL-based cleanup (1 hour), background cleanup coroutine.

### CLI Play (`poker_rl_agent/scripts/play_against_agent.py`)
- Interactive terminal-based play against the agent
- Supports FCPA-to-fullgame adapter mode (using 4-action checkpoint in fullgame environment)

---

## 13. Testing Strategy

### test_stability.py (~1670 lines)

Comprehensive test suite covering:

1. **State Encoder tests:** Verify tensor shapes, dtypes, ranges, legal masks
2. **Model output tests:** Verify forward pass produces valid logits/values, masked logits work
3. **Solver baseline tests:** Cache creation, reuse, evaluation flow
4. **Diagnostics tests:** Preflop action frequency tracking, behavior gate computation
5. **Curriculum tests:** Phase transitions, mask application
6. **Baseline agent tests:** Each agent produces valid actions
7. **League tests:** Snapshot management, PFSP sampling, Elo updates
8. **Training smoke tests:** End-to-end mini training run (few iterations)
9. **Schedule tests:** Linear schedule interpolation
10. **Gate tests:** Behavior gate pass/fail logic
11. **Checkpointing tests:** Save/load round-trip, version compatibility

---

## 14. File-by-File Reference

### Core Model
| File | Lines | Purpose |
|---|---|---|
| `models/alpha_holdem_net.py` | ~180 | Network architecture — pseudo-siamese trunk, residual blocks, policy/value heads |
| `models/card_tower.py` | ~50 | Card embedding + MLP |
| `models/action_tower.py` | ~60 | Action embedding + LSTM |
| `models/model_utils.py` | ~40 | Weight init, masked_logits |

### Training
| File | Lines | Purpose |
|---|---|---|
| `training/trainer.py` | ~1130 | Main training loop — rollout collection, PPO update, evaluation, scheduling |
| `training/rollout_buffer.py` | ~160 | Experience storage, GAE computation, minibatch iteration |
| `training/league_manager.py` | ~120 | K-best league, Elo tracking, PFSP sampling |
| `training/action_curriculum.py` | ~90 | Fullgame progressive raise limits |
| `training/checkpointing.py` | ~130 | V2 checkpoint format, save/load |

### Algorithms
| File | Lines | Purpose |
|---|---|---|
| `algorithms/self_play.py` | ~130 | Episode generation for self-play |
| `algorithms/regret_matching.py` | ~30 | Regret matching+ (utility function, not used in main loop) |

### Environment
| File | Lines | Purpose |
|---|---|---|
| `environment/openspiel_wrapper.py` | ~200 | OpenSpiel HUNL configuration, betting abstraction |
| `environment/state_representation.py` | ~180 | State → tensor encoding |

### Evaluation
| File | Lines | Purpose |
|---|---|---|
| `evaluation/evaluator.py` | ~450 | Multi-tier evaluation (solver, baselines, behavior, hand-strength) |
| `evaluation/baseline_agents.py` | ~120 | RandomAgent, AlwaysCallAgent, PotPressureAgent, etc. |

### Configuration
| File | Lines | Purpose |
|---|---|---|
| `utils/config.py` | ~200 | 170+ hyperparameter dataclass |
| `configs/training_configs.yaml` | ~1950 | 30+ named training presets |

### Serving
| File | Lines | Purpose |
|---|---|---|
| `serving/app.py` | ~240 | FastAPI web server |
| `serving/game_manager.py` | ~320 | Session management, bot action selection |
| `serving/schemas.py` | ~100 | Pydantic request/response models |

### Scripts
| File | Purpose |
|---|---|
| `scripts/debug_training.py` | Training entry point (loads YAML config) |
| `scripts/evaluate_complete.py` | Standalone comprehensive evaluation |
| `scripts/play_against_agent.py` | Interactive CLI play |
| `scripts/check_eval_gates.py` | Gate-check on evaluation JSON |

### Tests
| File | Lines | Purpose |
|---|---|---|
| `tests/test_stability.py` | ~1670 | Comprehensive test suite |

---

## 15. Anticipated Interview Questions

### Architecture Questions

**Q: Why did you choose a pseudo-siamese architecture?**
A: Cards and actions are fundamentally different modalities. Cards are a fixed-size set (7 indices), while actions are a variable-length sequence. Processing them separately allows each tower to use the most appropriate architecture (MLP for cards, LSTM for actions). They merge in the trunk where cross-modal reasoning happens (e.g., "given my strong hand and the opponent's betting pattern, I should raise").

**Q: Why not use a Transformer for the action tower?**
A: Betting histories are short (typically <20 actions). LSTM is simpler, has lower overhead, and pack_padded_sequence handles variable lengths efficiently. Transformer's advantage in long-range dependencies isn't needed here. However, for fullgame (where action sequences can be longer), a Transformer could be beneficial — it's a potential future improvement.

**Q: Why LayerNorm instead of BatchNorm?**
A: BatchNorm depends on batch statistics, which are noisy in RL where batch composition changes every iteration (different game states, different opponents). LayerNorm normalizes per-sample, making it more stable for RL training.

**Q: How does the legal action masking work?**
A: After the policy head produces raw logits for all actions, `masked_logits` sets illegal action logits to the minimum representable float for the dtype (e.g., -65504 for fp16). When softmax is applied, these become effectively zero probability. This is differentiable and gradient-safe.

### Algorithm Questions

**Q: Why PPO over other RL algorithms?**
A: PPO is on-policy, which means the data distribution matches the current policy — important in self-play where the game dynamics change as the opponent improves. PPO's clipped surrogate objective prevents catastrophic policy updates. Compared to IMPALA (used in the paper), PPO is simpler, doesn't need V-trace correction, and works well on a single machine. The trade-off is sample efficiency, but the K-best league ensures diverse training data.

**Q: How do you handle the credit assignment problem in poker?**
A: Poker has delayed rewards (you only know the outcome at showdown). I use GAE (Generalized Advantage Estimation) with gamma=0.995 and lambda=0.95. The high gamma discounts very little, appropriate because a single poker hand is short (~5-15 actions). The value function learns to estimate the expected return from any state, providing the intermediate signal that GAE needs.

**Q: Why gamma=0.995 and not 1.0?**
A: A discount factor slightly less than 1.0 provides numerical stability and introduces a slight preference for earlier rewards. In practice, for the short episodes in poker, the difference is minimal, but gamma=1.0 can cause instability in the value function fitting.

**Q: What is the K-best self-play league and why?**
A: Without a league, self-play can cycle: policy A beats B, B learns to beat A, C beats B, but C loses to A. The K-best league maintains K=16 historical snapshots of the model, selected by performance. When training, 60-85% of opponents are sampled from this league via PFSP (stronger opponents sampled more often). This provides: (1) stability against cycling, (2) diverse opponents, (3) gradual difficulty increase.

**Q: What are behavior gates and why are they necessary?**
A: Behavior gates are hard constraints on the agent's action distribution. For example, fold frequency must be below 78%, all-in frequency below 5%, entropy above 1.1 bits. Without gates, PPO can find degenerate equilibria that technically "win" in self-play but play unrealistically (e.g., always fold preflop — if both players do this, it's an equilibrium). Gates force the policy to maintain a realistic action distribution.

### Training Questions

**Q: Walk me through how a single training iteration works.**
A: (See Section 5 — the training loop flow)

**Q: How did you debug gradient explosion?**
A: I tracked gradient norms, loss values, and weight statistics per iteration. The root cause was that raw chip values (±20000) created very large loss values. Fixes: normalize inputs by starting_stack, scale value targets by 20x, use Huber loss instead of MSE, apply gradient clipping at 0.5, skip NaN gradients, initialize heads with small weights, and use LayerNorm everywhere.

**Q: Why did you use multiple training stages?**
A: Progressive complexity: FCPA (4 actions) → FCHPA (5 actions) → Fullgame (unrestricted). Starting with fewer actions makes the learning problem easier (smaller action space, simpler dynamics). Once the agent learns strong fundamentals with 4 actions, adding the half-pot bet (5th action) is an incremental change. The fullgame stage uses a curriculum to gradually allow larger raise sizes.

**Q: How did you decide which hyperparameters to change between stages?**
A: Empirically, through ablation studies. For example, Stage D ran three parallel experiments varying: exploration temperature, opponent diversity, and value coefficient. I tracked behavior gates and solver performance, selected the best branch, and continued. The YAML config file has 30+ named presets documenting this progression.

### Poker-Specific Questions

**Q: How do you represent the state of a poker game?**
A: 11D scalar features (pot, stacks, contributions — all normalized), 7 card indices (2 hole + 5 board), variable-length action history, and a legal action mask. The scalars are normalized by starting_stack (20000) to keep values in [0,1]. Unknown community cards use index 52.

**Q: What is MCCFR-ES and why do you evaluate against it?**
A: Monte Carlo Counterfactual Regret Minimization with External Sampling. It's a game-theoretic algorithm that converges to a Nash equilibrium. It provides a principled baseline — if the agent can win against a solver, it's exploiting solver weaknesses while maintaining a strong strategy. It's much more meaningful than evaluating against rule-based bots.

**Q: What are betting abstractions and why do they matter?**
A: In HUNL, there are theoretically infinitely many raise sizes. Betting abstractions discretize this to a manageable set: FCPA has 4 actions (fold/call/pot/allin), FCHPA has 5 (adds half-pot). This makes the game tractable for neural network training. The trade-off is that the agent can only bet at these discrete sizes, which limits expressiveness but dramatically reduces the action space complexity.

### Software Engineering Questions

**Q: How is the config system designed?**
A: A Python dataclass (`Config`) with 170+ fields and default values. Training presets are YAML files that override specific fields. The `debug_training.py` script merges the "default" section with a named config section. `sync_legacy_fields` keeps backward-compatible aliases consistent. This allows running different experiments by just changing `--config_name`.

**Q: How do you handle checkpointing?**
A: V2 format checkpoint includes: model state dict, optimizer state, scheduler state, league state, config hash, and metadata (step, timestamp, version). Supports single-file mode (rolling latest.pt) and multi-file mode (point_N.pt with automatic pruning to MAX_CHECKPOINTS). `load_checkpoint` handles V1 compatibility.

**Q: How is the evaluation system designed?**
A: The Evaluator class is composable: it can evaluate against any agent (model, solver, or baseline). The solver is cached per-process for efficiency. Results include both raw winrates and behavior diagnostics. The `evaluate_complete.py` script orchestrates multi-tier evaluation with different solver strengths and reports everything in a JSON file.

**Q: How did you use the HPC cluster?**
A: SLURM batch scripts in `scripts/*.slurm` for each training stage. Each stage gets its own named job, requests GPU resources, activates the virtual environment, and runs `debug_training.py` with the appropriate `--config_name`. W&B tracks experiments in online mode.

### Design Decision Questions

**Q: What would you do differently if starting over?**
A:
- Consider using a Transformer for the action tower — it could handle variable lengths more elegantly
- Implement distributed self-play (like IMPALA) for faster training
- Use a continuous action head for bet sizing instead of discrete abstractions
- Build a more systematic hyperparameter search (Optuna/Ray Tune) instead of manual ablation
- Add position-aware features (button vs. big blind) more explicitly

**Q: What were the biggest challenges?**
A:
1. **Gradient explosion** — took multiple iterations to diagnose and fix (see Section 10)
2. **Degenerate strategies** — agent would collapse to always-fold or always-allin without behavior gates
3. **Evaluation reliability** — MCCFR-ES solver has its own variance; needed multiple seeds and confidence intervals
4. **Action space transition** — going from 4 to 5 actions required careful fine-tuning to learn the new half-pot action
5. **Hyperparameter sensitivity** — small changes in entropy coefficient or temperature could cause catastrophic performance drops

**Q: What are the limitations of your approach?**
A:
- Action abstractions limit expressiveness (can only bet at discrete sizes)
- Single-machine training limits scale
- No opponent modeling (doesn't adapt to specific opponent tendencies)
- Evaluation against MCCFR-ES (which itself isn't perfectly converged) limits the upper bound on measured performance
- No game-theoretic guarantees (PPO finds good policies, but not provably Nash equilibria)
