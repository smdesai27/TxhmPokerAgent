import os
import random
from datetime import datetime
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from ..algorithms.self_play import SelfPlayWorker
from ..environment.openspiel_wrapper import PokerEnv
from ..evaluation.baseline_agents import RandomAgent
from ..evaluation.evaluator import Evaluator
from ..models.alpha_holdem_net import AlphaHoldemNetwork
from ..models.model_utils import masked_logits
from ..training.checkpointing import load_checkpoint, save_checkpoint
from ..training.league_manager import LeagueManager
from ..training.rollout_buffer import RolloutBuffer
from ..utils.config import Config
from ..utils.logging_utils import init_wandb, log_metrics, log_metrics_local


class Trainer:
    def __init__(self, config=None):
        self.config = config if config else Config()
        if hasattr(self.config, "sync_legacy_fields"):
            self.config.sync_legacy_fields()

        if not getattr(self.config, "RUN_ID", ""):
            self.config.RUN_ID = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.run_id = self.config.RUN_ID
        self.device = self.config.DEVICE
        self._set_seed(self.config.SEED)

        self.env = PokerEnv(
            game_name=self.config.GAME_NAME,
            env_preset=self.config.ENV_PRESET,
            betting_abstraction=self.config.BETTING_ABSTRACTION,
        )
        self.num_actions = self.env.num_actions()

        self.model = AlphaHoldemNetwork(self.num_actions, self.config).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.LR,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        self.scheduler = None
        if self.config.LR_SCHEDULER == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=10,
            )

        self.amp_enabled = self.config.AMP and str(self.device).startswith("cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        worker_env = PokerEnv(
            game_name=self.config.GAME_NAME,
            env_preset=self.config.ENV_PRESET,
            betting_abstraction=self.config.BETTING_ABSTRACTION,
        )
        self.worker = SelfPlayWorker(
            worker_env,
            device=self.device,
            max_action_history=self.config.MAX_ACTION_HISTORY,
        )
        self.rollout_buffer = RolloutBuffer()

        self.league = LeagueManager(
            k_best=self.config.K_BEST,
            init_rating=self.config.LEAGUE_INIT_RATING,
            k_factor=self.config.LEAGUE_K_FACTOR,
            pfsp_beta=self.config.PFSP_BETA,
        )
        self.league.add_snapshot(self.model, step=0, score=0.0)
        self._opponent_cache = {}

        self.evaluator = Evaluator(self.model, self.config, device=self.device)
        self.global_step = 0

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_model_clone(self):
        model = AlphaHoldemNetwork(self.num_actions, self.config).to(self.device)
        model.eval()
        return model

    def _get_opponent_model(self, entry):
        cached = self._opponent_cache.get(entry.entry_id)
        if cached and cached[0] == entry.step:
            return cached[1]

        model = self._build_model_clone()
        model.load_state_dict(entry.state_dict)
        self._opponent_cache[entry.entry_id] = (entry.step, model)
        return model

    def _collect_rollouts(self):
        self.rollout_buffer.clear()
        returns = []

        min_transitions = max(1, int(getattr(self.config, "MIN_ROLLOUT_TRANSITIONS", 1)))
        target_episodes = max(1, int(self.config.ROLLOUT_EPISODES))
        max_episodes = max(target_episodes, min_transitions * 4)
        episodes_collected = 0

        self.model.eval()
        while episodes_collected < target_episodes or (
            len(self.rollout_buffer) < min_transitions and episodes_collected < max_episodes
        ):
            opponent_entry = self.league.sample_opponent()
            opponent_model = self.model if opponent_entry is None else self._get_opponent_model(opponent_entry)

            episode_transitions, final_return, _ = self.worker.generate_episode(
                policy_model=self.model,
                opponent_model=opponent_model,
                train_player=None,
                bb_size=self.config.BB_SIZE,
            )

            for transition in episode_transitions:
                self.rollout_buffer.add(
                    state=transition["state"],
                    action=transition["action"],
                    reward=transition["reward"],
                    done=bool(transition["done"]),
                    log_prob=transition["log_prob"],
                    value=transition["value"],
                )

            returns.append(final_return)

            if opponent_entry is not None:
                if final_return > 0:
                    result = 1.0
                elif final_return < 0:
                    result = 0.0
                else:
                    result = 0.5
                self.league.update_elo(opponent_entry.entry_id, result)

            episodes_collected += 1

        mean_return = float(np.mean(returns)) if returns else 0.0
        return {
            "rollout/episodes": int(episodes_collected),
            "rollout/mean_final_return_bb": mean_return,
            "rollout/transitions": len(self.rollout_buffer),
            "rollout/min_target_transitions": min_transitions,
            "league/current_elo": self.league.current_rating,
        }

    def _to_device(self, batch_states: Dict[str, torch.Tensor]):
        return {k: v.to(self.device) for k, v in batch_states.items()}

    def _ppo_update(self):
        if len(self.rollout_buffer) == 0:
            return {"train/updates": 0, "train/nonfinite_grad_skips": 0}

        self.model.train()
        advantages, returns = self.rollout_buffer.compute_advantages(
            gamma=self.config.PPO_GAMMA,
            gae_lambda=self.config.PPO_GAE_LAMBDA,
        )

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        value_scale = max(float(getattr(self.config, "VALUE_TARGET_SCALE", 1.0)), 1e-6)
        value_loss_type = str(getattr(self.config, "VALUE_LOSS_TYPE", "huber")).lower()
        huber_delta = max(float(getattr(self.config, "VALUE_HUBER_DELTA", 1.0)) / value_scale, 1e-6)
        skip_nonfinite_grad = bool(getattr(self.config, "SKIP_NONFINITE_GRAD", True))

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_clipfrac = 0.0
        total_explained_var = 0.0
        total_grad_norm = 0.0
        minibatch_count = 0
        optimizer_updates = 0
        nonfinite_grad_skips = 0
        nonfinite_loss_batches = 0

        self.optimizer.zero_grad(set_to_none=True)
        grad_accum = max(1, int(self.config.GRAD_ACCUM_STEPS))
        accum = 0
        early_stop = False

        for _ in range(self.config.PPO_EPOCHS):
            for batch in self.rollout_buffer.iterate_minibatches(
                minibatch_size=self.config.PPO_MINIBATCH_SIZE,
                advantages=advantages,
                returns=returns,
                shuffle=True,
            ):
                states = self._to_device(batch["states"])
                actions = batch["actions"].to(self.device)
                old_log_probs = batch["old_log_probs"].to(self.device)
                adv = batch["advantages"].to(self.device)
                target_returns = batch["returns"].to(self.device)
                scaled_returns = target_returns / value_scale

                with torch.amp.autocast(device_type="cuda", enabled=self.amp_enabled):
                    outputs = self.model(states)
                    logits = masked_logits(outputs["policy_logits"], states["legal_action_mask"])
                    dist = Categorical(logits=logits)

                    new_log_probs = dist.log_prob(actions)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(
                        ratio,
                        1.0 - self.config.PPO_CLIP_EPS,
                        1.0 + self.config.PPO_CLIP_EPS,
                    ) * adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    clipfrac = ((ratio - 1.0).abs() > self.config.PPO_CLIP_EPS).float().mean()

                    value_pred = outputs["state_value"] / value_scale
                    if value_loss_type == "mse":
                        value_loss = F.mse_loss(value_pred, scaled_returns)
                    else:
                        value_loss = F.huber_loss(value_pred, scaled_returns, delta=huber_delta)

                    loss = (
                        policy_loss
                        + self.config.PPO_VALUE_COEF * value_loss
                        - self.config.PPO_ENTROPY_COEF * entropy
                    ) / grad_accum

                if not torch.isfinite(loss.detach()).item():
                    nonfinite_loss_batches += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    accum = 0
                    continue

                approx_kl = (old_log_probs - new_log_probs).mean().item()
                with torch.no_grad():
                    var_y = torch.var(scaled_returns, unbiased=False)
                    if var_y.item() <= 1e-12:
                        explained_var = 0.0
                    else:
                        residual_var = torch.var(scaled_returns - value_pred.detach(), unbiased=False)
                        explained_var = float((1.0 - (residual_var / (var_y + 1e-8))).item())

                if self.amp_enabled:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum += 1
                minibatch_count += 1
                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                total_kl += float(approx_kl)
                total_clipfrac += float(clipfrac.item())
                total_explained_var += explained_var

                if accum % grad_accum == 0:
                    if self.amp_enabled:
                        self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.GRAD_CLIP,
                    )
                    grad_norm_value = float(grad_norm.item())
                    grad_is_nonfinite = not np.isfinite(grad_norm_value)

                    if grad_is_nonfinite and skip_nonfinite_grad:
                        nonfinite_grad_skips += 1
                        self.optimizer.zero_grad(set_to_none=True)
                        if self.amp_enabled:
                            self.scaler.update()
                        accum = 0
                        continue

                    if self.amp_enabled:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    optimizer_updates += 1
                    total_grad_norm += grad_norm_value

                if approx_kl > self.config.PPO_TARGET_KL:
                    early_stop = True
                    break

            if early_stop:
                break

        if accum % grad_accum != 0:
            if self.amp_enabled:
                self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.GRAD_CLIP,
            )
            grad_norm_value = float(grad_norm.item())
            grad_is_nonfinite = not np.isfinite(grad_norm_value)
            if grad_is_nonfinite and skip_nonfinite_grad:
                nonfinite_grad_skips += 1
                self.optimizer.zero_grad(set_to_none=True)
                if self.amp_enabled:
                    self.scaler.update()
            else:
                if self.amp_enabled:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                optimizer_updates += 1
                total_grad_norm += grad_norm_value

        if self.scheduler is not None and minibatch_count > 0:
            # Scheduler tracks performance (higher is better), use negative value loss proxy.
            self.scheduler.step(-(total_value_loss / minibatch_count))

        denom = max(1, minibatch_count)
        return {
            "train/updates": optimizer_updates,
            "train/policy_loss": total_policy_loss / denom,
            "train/value_loss": total_value_loss / denom,
            "train/entropy": total_entropy / denom,
            "train/approx_kl": total_kl / denom,
            "train/clipfrac": total_clipfrac / denom,
            "train/explained_var": total_explained_var / denom,
            "train/grad_norm": total_grad_norm / max(1, optimizer_updates),
            "train/nonfinite_grad_skips": int(nonfinite_grad_skips),
            "train/nonfinite_loss_batches": int(nonfinite_loss_batches),
        }

    def _evaluate(self):
        self.model.eval()
        metrics = {}

        if self.config.EVAL_ENABLE_RANDOM:
            random_stats = self.evaluator.evaluate(
                RandomAgent(),
                num_episodes=max(20, self.config.EVAL_EPISODES // 2),
            )
            metrics.update(
                {
                    "eval/random_bb100": random_stats["bb_per_100"],
                    "eval/random_avg_return": random_stats["avg_return"],
                }
            )

        if self.config.EVAL_ENABLE_CFR:
            baseline_stats = self.evaluator.evaluate_vs_solver_baseline(
                baseline_algo=self.config.EVAL_BASELINE_ALGO,
                iterations=self.config.EVAL_BASELINE_ITERS,
                num_episodes=self.config.EVAL_EPISODES,
                seed=self.config.SEED,
            )
            metrics.update(
                {
                    "eval/mccfr_es_bb100": baseline_stats["bb_per_100"],
                    "eval/mccfr_es_avg_return": baseline_stats["avg_return"],
                    "eval/mccfr_es_stderr": baseline_stats["std_err"],
                    "eval/mccfr_es_algo": baseline_stats["baseline/algo"],
                    "eval/mccfr_es_iters": baseline_stats["baseline/iters"],
                    "eval/mccfr_es_seed": baseline_stats["baseline/seed"],
                    "eval/mccfr_es_baseline_build_s": baseline_stats["baseline/build_seconds"],
                    "eval/mccfr_es_baseline_cache_hit": float(baseline_stats["baseline/cache_hit"]),
                    # Backwards-compatible aliases.
                    "eval/cfr_bb100": baseline_stats["bb_per_100"],
                    "eval/cfr_avg_return": baseline_stats["avg_return"],
                    "eval/cfr_stderr": baseline_stats["std_err"],
                }
            )

        if self.config.EVAL_ENABLE_NASH_CONV:
            try:
                metrics["eval/nash_conv"] = self.evaluator.evaluate_nash_conv()
            except Exception:
                metrics["eval/nash_conv"] = float("nan")

        return metrics

    def _save_checkpoint(self, step: int, extra_metrics: Dict[str, float]):
        if self.config.CHECKPOINT_SINGLE_FILE:
            path = os.path.join("checkpoints", "latest.pt")
        else:
            path = os.path.join("checkpoints", f"point_{step}.pt")
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=step,
            path=path,
            max_keep=self.config.MAX_CHECKPOINTS,
            single_file=self.config.CHECKPOINT_SINGLE_FILE,
            save_optimizer=self.config.CHECKPOINT_SAVE_OPTIMIZER,
            save_scheduler=self.config.CHECKPOINT_SAVE_SCHEDULER,
            save_league=self.config.CHECKPOINT_SAVE_LEAGUE,
            save_extra=self.config.CHECKPOINT_SAVE_EXTRA,
            league_state=self.league.state_dict() if self.config.CHECKPOINT_SAVE_LEAGUE else {},
            config=self.config,
            extra=extra_metrics if self.config.CHECKPOINT_SAVE_EXTRA else {},
        )

    def _maybe_resume(self):
        checkpoint_path = self.config.RESUME_FROM
        if not checkpoint_path:
            return 0

        payload = load_checkpoint(
            path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            map_location=self.device,
        )
        league_state = payload.get("league_state", {})
        if league_state:
            self.league.load_state_dict(league_state)
            self._opponent_cache.clear()
        return int(payload.get("step", 0))

    def train(self):
        init_wandb(config=self.config)
        start_step = self._maybe_resume()
        print(
            f"Starting training: start_step={start_step}, "
            f"target_iterations={self.config.CFR_ITERATIONS}, "
            f"device={self.device}, "
            f"run_id={self.run_id}",
            flush=True,
        )

        for iteration in range(start_step, self.config.CFR_ITERATIONS):
            metrics = {
                "iteration": iteration + 1,
                "lr": self.optimizer.param_groups[0]["lr"],
                "run_id": self.run_id,
            }

            rollout_metrics = self._collect_rollouts()
            metrics.update(rollout_metrics)

            update_metrics = self._ppo_update()
            metrics.update(update_metrics)

            if (iteration + 1) % self.config.SNAPSHOT_INTERVAL == 0:
                snapshot_score = rollout_metrics.get("rollout/mean_final_return_bb", 0.0)
                self.league.add_snapshot(self.model, step=iteration + 1, score=snapshot_score)
                self._opponent_cache.clear()
                metrics["league/size"] = len(self.league.entries)

            if self.config.EVAL_FREQ > 0 and (iteration + 1) % self.config.EVAL_FREQ == 0:
                metrics.update(self._evaluate())

            if (iteration + 1) % self.config.CHECKPOINT_FREQ == 0:
                self._save_checkpoint(iteration + 1, metrics)

            if (iteration + 1) % self.config.LOG_INTERVAL == 0:
                log_metrics(metrics, step=iteration + 1)
                if self.config.OFFLINE_LOGGING:
                    log_metrics_local(metrics, step=iteration + 1, out_dir="logs")
                print(
                    f"[iter={iteration + 1}] "
                    f"rollout_mean_bb={metrics.get('rollout/mean_final_return_bb', float('nan')):.4f} "
                    f"policy_loss={metrics.get('train/policy_loss', float('nan')):.6f} "
                    f"value_loss={metrics.get('train/value_loss', float('nan')):.6f}",
                    flush=True,
                )

            self.global_step = iteration + 1

        # Guarantee a final checkpoint exists even if the last step does not hit CHECKPOINT_FREQ.
        if self.global_step > 0 and (self.global_step % self.config.CHECKPOINT_FREQ != 0):
            self._save_checkpoint(self.global_step, {"iteration": self.global_step})
            print(
                f"Saved final checkpoint at step {self.global_step}",
                flush=True,
            )

        print(
            f"Training completed successfully at step {self.global_step}",
            flush=True,
        )


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
