import json
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
from ..evaluation.baseline_agents import (
    AlwaysCallAgent,
    PassiveCallerAgent,
    PotPressureAgent,
    RandomAgent,
    StickyCallAgent,
)
from ..evaluation.evaluator import Evaluator
from ..models.alpha_holdem_net import AlphaHoldemNetwork
from ..models.model_utils import masked_logits
from .proto_fullgame_action_curriculum import FullgameActionCurriculum
from ..training.checkpointing import load_checkpoint, save_checkpoint
from ..training.league_manager import LeagueManager
from ..training.rollout_buffer import RolloutBuffer
from ..utils.config import Config
from ..utils.logging_utils import init_wandb, log_metrics, log_metrics_local


class Trainer:
    def __init__(self, config=None):
        self.config = config if config else Config()
        #for legaacy support 
        if hasattr(self.config, "sync_legacy_fields"):
            self.config.sync_legacy_fields()

        # add metadata field for run ID if not already set, used for logging and checkpointing
        if not getattr(self.config, "RUN_ID", ""):
            self.config.RUN_ID = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.run_id = self.config.RUN_ID
        self.device = self.config.DEVICE
        self._set_seed(self.config.SEED)
        strict_abstraction = bool(getattr(self.config, "STRICT_ABSTRACTION", True))

        # initialize environment first to resolve betting abstraction and action space
        self.env = PokerEnv(
            game_name=self.config.GAME_NAME,
            env_preset=self.config.ENV_PRESET,
            betting_abstraction=self.config.BETTING_ABSTRACTION,
            strict_abstraction=strict_abstraction,
        )
        #both abs saved for PokerEnv
        self.requested_betting_abstraction = self.env.get_requested_betting_abstraction()
        self.effective_betting_abstraction = self.env.get_effective_betting_abstraction()
        self.config.BETTING_ABSTRACTION = self.env.get_effective_betting_abstraction()
        # pull num actions from the enviorment for consistancy 
        self.num_actions = self.env.num_actions()

        # build modle and optimizer
        self.model = AlphaHoldemNetwork(self.num_actions, self.config).to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LR,
            # to prevent overfitting
            weight_decay=self.config.WEIGHT_DECAY,
        )
        self._base_lr = float(self.config.LR)
        self._lr_multiplier = 1.0
        self._kl_low_streak = 0

        self.scheduler = None

        # run bp on float32 on cuda
        self.amp_enabled = self.config.AMP and str(self.device).startswith("cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        # initialize a diff environment for the worker
        worker_env = PokerEnv(
            game_name=self.config.GAME_NAME,
            env_preset=self.config.ENV_PRESET,
            betting_abstraction=self.config.BETTING_ABSTRACTION,
            strict_abstraction=strict_abstraction,
        )
        self.worker = SelfPlayWorker(
            worker_env,
            device=self.device,
            max_action_history=self.config.MAX_ACTION_HISTORY,
        )
        self.curriculum = None
        # for future fullgame support 
        if (
            str(self.config.BETTING_ABSTRACTION).lower() == "fullgame"
            and bool(getattr(self.config, "FULLGAME_CURRICULUM_ENABLE", True))
        ):
            self.curriculum = FullgameActionCurriculum(self.config)

        self.rollout_buffer = RolloutBuffer()

        self.league = LeagueManager(
            k_best=self.config.K_BEST,
            init_rating=self.config.LEAGUE_INIT_RATING,
            k_factor=self.config.LEAGUE_K_FACTOR,
            pfsp_beta=self.config.PFSP_BETA,
            pfsp_mode=getattr(self.config, "PFSP_MODE", "loss"),
        )
        #add inttial model to make sure league is never empty
        self.league.add_snapshot(self.model, step=0, score=0.0)
        # cache for opponent models to avoid redundant loading 
        self._opponent_cache = {}
        self._exploit_agents = self._build_exploit_agents()
        self._style_target = self._load_style_target_profile()

        self.evaluator = Evaluator(self.model, self.config, device=self.device)
        self.global_step = 0
        #for checkpointing best model
        self.best_eval_score = float("-inf")
        self.best_eval_step = 0

    @staticmethod
    # set all seeds together as good practice 
    # python for sampling agents, np for adv calcs and torch for model
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _linear_schedule(start: float, end: float, progress: float, decay_frac: float) -> float:
        decay = max(float(decay_frac), 1e-8)
        clamped_progress = min(max(float(progress), 0.0), 1.0)
        ratio = min(clamped_progress / decay, 1.0)
        return float(start + (end - start) * ratio)

    def _effective_schedule_progress(self, iteration: int, start_step: int) -> float:
        """
        By default this is relative to the continuation window (resume -> target),
        so resumed runs can still apply start/end schedules meaningfully.
        """
        #for seperate resumption logic
        if bool(getattr(self.config, "SCHEDULE_RELATIVE_TO_RESUME", True)):
            total_span = max(1, int(self.config.CFR_ITERATIONS) - int(start_step))
            return float(iteration + 1 - int(start_step)) / float(total_span)
        return float(iteration + 1) / float(max(1, int(self.config.CFR_ITERATIONS)))

    def _policy_temperature(self, progress: float) -> float:
        start = float(getattr(self.config, "SELF_PLAY_TEMPERATURE_START", 1.0))
        end = float(getattr(self.config, "SELF_PLAY_TEMPERATURE_END", 1.0))
        decay = float(getattr(self.config, "SELF_PLAY_TEMPERATURE_DECAY_FRAC", 1.0))
        #calc temp with linear decay
        return max(1e-3, self._linear_schedule(start, end, progress, decay))

    def _entropy_coef(self, progress: float) -> float:
        start = float(getattr(self.config, "PPO_ENTROPY_COEF_START", self.config.PPO_ENTROPY_COEF))
        end = float(getattr(self.config, "PPO_ENTROPY_COEF_END", self.config.PPO_ENTROPY_COEF))
        decay = float(getattr(self.config, "PPO_ENTROPY_DECAY_FRAC", 1.0))
        return max(0.0, self._linear_schedule(start, end, progress, decay))

    def _style_reg_coef(self, progress: float) -> float:
        if not bool(getattr(self.config, "STYLE_REG_ENABLE", False)):
            return 0.0
        start = float(getattr(self.config, "STYLE_REG_COEF_START", 0.0))
        end = float(getattr(self.config, "STYLE_REG_COEF_END", 0.0))
        decay = float(getattr(self.config, "STYLE_REG_DECAY_FRAC", 1.0))
        return max(0.0, self._linear_schedule(start, end, progress, decay))

    def _resolve_opponent_mix(self):
        """Resolves league/random/exploit ratios with backwards-compatible fallback."""
        league_prob = float(getattr(self.config, "LEAGUE_OPPONENT_PROB", 0.0))
        random_prob = float(getattr(self.config, "RANDOM_OPPONENT_PROB", 0.0))
        exploit_prob = float(getattr(self.config, "EXPLOIT_OPPONENT_PROB", 0.0))

        # defensive defaulting to avoid silent misconfg that lead to no league training
        if league_prob <= 0.0 and random_prob <= 0.0 and exploit_prob <= 0.0:
            random_prob = float(getattr(self.config, "LEAGUE_RANDOM_OPPONENT_PROB", 0.0))
            random_prob = min(max(random_prob, 0.0), 1.0)
            league_prob = 1.0 - random_prob
            exploit_prob = 0.0

        probs = np.array([league_prob, exploit_prob, random_prob], dtype=np.float64)
        probs = np.clip(probs, 0.0, None)
        total = float(probs.sum())
        if total <= 0.0:
            probs = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            total = 1.0
        probs /= total
        return {
            "league": float(probs[0]),
            "exploit": float(probs[1]),
            "random": float(probs[2]),
        }

    @staticmethod
    def _agent_from_name(name: str):
        normalized = str(name).strip().lower()
        if normalized in {"always_call", "alwayscall"}:
            return AlwaysCallAgent()
        if normalized in {"pot_pressure", "potpressure"}:
            return PotPressureAgent()
        if normalized in {"sticky_call", "stickycall"}:
            return StickyCallAgent()
        if normalized in {"passive_caller", "passivecaller"}:
            return PassiveCallerAgent()
        if normalized in {"random", "rand"}:
            return RandomAgent()
        return None

    def _build_exploit_agents(self):
        raw = str(getattr(self.config, "EXPLOIT_OPPONENT_SET", "always_call"))
        names = [tok.strip() for tok in raw.split(",") if tok.strip()]
        agents = []
        for name in names:
            agent = self._agent_from_name(name)
            if agent is not None:
                agents.append(agent)
        if not agents:
            # defensive defualt to make sure pool is never empty
            agents.append(AlwaysCallAgent())
        return agents

    # Style regularizer: optional behavior-shaping term, disabled by default.
    def _load_style_target_profile(self):
        if not bool(getattr(self.config, "STYLE_REG_ENABLE", False)):
            return None
        if not bool(getattr(self.config, "STYLE_TARGET_USE_SOLVER_PROFILE", True)):
            return None
        path = str(getattr(self.config, "STYLE_TARGET_PROFILE_PATH", "")).strip()
        if not path:
            return None
        if not os.path.exists(path):
            print(f"Style target profile not found, disabling style regularizer: {path}", flush=True)
            return None

        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception as exc:
            print(f"Failed loading style target profile '{path}': {exc}", flush=True)
            return None

        target = payload.get("target", {})
        if not isinstance(target, dict):
            target = {}
        keys = ["fold", "call_check", "half_pot", "pot_raise", "allin"]
        values = np.array([float(target.get(k, 0.0)) for k in keys], dtype=np.float64)
        values = np.clip(values, 0.0, None)
        if values.sum() <= 0.0:
            print(f"Style target profile has empty target distribution, disabling: {path}", flush=True)
            return None
        values = values / values.sum()

        target_pot_given_legal = float(
            payload.get(
                "target/preflop_choose_given_legal/pot_raise",
                payload.get("diagnostics/preflop_choose_given_legal/pot_raise", 0.0),
            )
        )
        target_half_given_legal = float(
            payload.get(
                "target/preflop_choose_given_legal/half_pot",
                payload.get("diagnostics/preflop_choose_given_legal/half_pot", 0.0),
            )
        )

        return {
            "path": path,
            "target_dist": values.astype(np.float32),
            "target_pot_given_legal": max(0.0, target_pot_given_legal),
            "target_half_given_legal": max(0.0, target_half_given_legal),
        }

    def _preflop_bucket_indices(self):
        abstraction = str(getattr(self.config, "BETTING_ABSTRACTION", "fcpa")).lower()
        if abstraction in {"fchpa", "fcpha"}:
            return {
                "fold": 0,
                "call_check": 1,
                "half_pot": 2,
                "pot_raise": 3,
                "allin": 4,
            }
        if abstraction == "fcpa":
            return {
                "fold": 0,
                "call_check": 1,
                "half_pot": None,
                "pot_raise": 2,
                "allin": 3,
            }
        return {
            "fold": None,
            "call_check": None,
            "half_pot": None,
            "pot_raise": None,
            "allin": None,
        }

    def _apply_lr_multiplier(self):
        lr = self._base_lr * self._lr_multiplier
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _maybe_adjust_adaptive_lr(self, avg_kl: float):
        metrics = {
            "train/lr_multiplier": float(self._lr_multiplier),
            "train/adaptive_lr_event": 0.0,
            "train/kl_low_streak": float(self._kl_low_streak),
        }
        if not bool(getattr(self.config, "PPO_ADAPTIVE_KL_ENABLE", False)):
            return metrics

        target_kl = max(float(self.config.PPO_TARGET_KL), 1e-8)
        high_mult = max(float(getattr(self.config, "PPO_ADAPTIVE_KL_HIGH", 2.0)), 1.0)
        low_mult = max(float(getattr(self.config, "PPO_ADAPTIVE_KL_LOW", 0.5)), 0.0)
        decay = min(max(float(getattr(self.config, "PPO_ADAPTIVE_LR_DECAY", 0.5)), 1e-6), 1.0)
        growth = max(float(getattr(self.config, "PPO_ADAPTIVE_LR_GROWTH", 1.05)), 1.0)
        high_threshold = target_kl * high_mult
        low_threshold = target_kl * low_mult

        #for logging and metrics, event = 1 for growth, -1 for decay, 0 for no change
        event = 0.0
        if avg_kl > high_threshold:
            self._lr_multiplier = max(0.1, self._lr_multiplier * decay)
            self._kl_low_streak = 0
            event = -1.0
        elif avg_kl < low_threshold:
            self._kl_low_streak += 1
            if self._kl_low_streak >= 10:
                self._lr_multiplier = min(3.0, self._lr_multiplier * growth)
                self._kl_low_streak = 0
                event = 1.0
        else:
            self._kl_low_streak = 0

        # adjust lr based on over or under shooting kl targets
        if event != 0.0:
            self._apply_lr_multiplier()

        metrics["train/lr_multiplier"] = float(self._lr_multiplier)
        metrics["train/adaptive_lr_event"] = event
        metrics["train/kl_low_streak"] = float(self._kl_low_streak)
        return metrics

    def _clip_advantages(self, advantages: torch.Tensor):
        adv_clip = float(getattr(self.config, "PPO_ADV_CLIP", 0.0))
        if adv_clip <= 0:
            return advantages
        return torch.clamp(advantages, min=-adv_clip, max=adv_clip)

    def _compute_style_regularizer(
        self,
        probs: torch.Tensor,
        legal_mask: torch.Tensor,
        scalars: torch.Tensor,
    ):
        if not bool(getattr(self.config, "STYLE_REG_ENABLE", False)):
            zero = probs.new_zeros(())
            return zero, zero, zero
        if self._style_target is None:
            zero = probs.new_zeros(())
            return zero, zero, zero

        abstraction = str(getattr(self.config, "BETTING_ABSTRACTION", "fcpa")).lower()
        if abstraction not in {"fchpa", "fcpha"}:
            zero = probs.new_zeros(())
            return zero, zero, zero

        if scalars.dim() != 2 or scalars.shape[1] < 9:
            zero = probs.new_zeros(())
            return zero, zero, zero

        preflop_mask = scalars[:, 5] > 0.5
        if not bool(preflop_mask.any().item()):
            zero = probs.new_zeros(())
            return zero, zero, zero

        preflop_probs = probs[preflop_mask]
        preflop_legal = legal_mask[preflop_mask]
        preflop_fraction = preflop_probs.shape[0] / max(1, probs.shape[0])

        bucket_idx = self._preflop_bucket_indices()
        ordered = [
            bucket_idx["fold"],
            bucket_idx["call_check"],
            bucket_idx["half_pot"],
            bucket_idx["pot_raise"],
            bucket_idx["allin"],
        ]
        bucket_probs = []
        for idx in ordered:
            if idx is None or idx >= preflop_probs.shape[1]:
                bucket_probs.append(preflop_probs.new_zeros((preflop_probs.shape[0],)))
            else:
                bucket_probs.append(preflop_probs[:, idx])
        pred_dist = torch.stack(bucket_probs, dim=1).mean(dim=0)
        pred_dist = pred_dist / torch.clamp(pred_dist.sum(), min=1e-8)
        pred_dist = torch.clamp(pred_dist, min=1e-8, max=1.0)

        target_dist = torch.as_tensor(
            self._style_target["target_dist"],
            device=pred_dist.device,
            dtype=pred_dist.dtype,
        )
        target_dist = target_dist / torch.clamp(target_dist.sum(), min=1e-8)
        target_dist = torch.clamp(target_dist, min=1e-8, max=1.0)

        style_kl = torch.sum(target_dist * (torch.log(target_dist) - torch.log(pred_dist)))

        pot_idx = bucket_idx["pot_raise"]
        half_idx = bucket_idx["half_pot"]
        pot_prob_given_legal = pred_dist.new_zeros(())
        half_prob_given_legal = pred_dist.new_zeros(())

        if pot_idx is not None and pot_idx < preflop_probs.shape[1]:
            pot_legal = preflop_legal[:, pot_idx] > 0.5
            if bool(pot_legal.any().item()):
                pot_prob_given_legal = preflop_probs[pot_legal, pot_idx].mean()
        if half_idx is not None and half_idx < preflop_probs.shape[1]:
            half_legal = preflop_legal[:, half_idx] > 0.5
            if bool(half_legal.any().item()):
                half_prob_given_legal = preflop_probs[half_legal, half_idx].mean()

        target_pot = float(self._style_target.get("target_pot_given_legal", 0.0))
        target_half = float(self._style_target.get("target_half_given_legal", 0.0))
        hinge_coef = max(float(getattr(self.config, "STYLE_POT_HINGE_COEF", 0.5)), 0.0)

        pot_hinge = torch.relu(probs.new_tensor(target_pot) - pot_prob_given_legal)
        half_hinge = torch.relu(probs.new_tensor(target_half) - half_prob_given_legal)
        style_hinge = pot_hinge + (0.5 * half_hinge)

        return style_kl, (hinge_coef * style_hinge), probs.new_tensor(float(preflop_fraction))

    def _compute_explained_var(self, value_pred: torch.Tensor, scaled_returns: torch.Tensor):
        var_floor = max(float(getattr(self.config, "EXPLAINED_VAR_VAR_FLOOR", 1e-4)), 0.0)
        var_y = torch.var(scaled_returns, unbiased=False)
        residual_var = torch.var(scaled_returns - value_pred.detach(), unbiased=False)
        raw = float((1.0 - (residual_var / (var_y + 1e-8))).item())
        valid = bool(var_y.item() >= var_floor)
        return raw, valid, float(var_y.item())

    def _build_model_clone(self):
        model = AlphaHoldemNetwork(self.num_actions, self.config).to(self.device)
        #never train opponent models
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

    def _collect_rollouts(self, progress: float = 1.0, policy_temperature: float = 1.0):
        self.rollout_buffer.clear()
        returns = []

        # set minimum and target rollouts based on config, with defensive defaults to ensure training progresses and OOM protection
        min_transitions = max(1, int(getattr(self.config, "MIN_ROLLOUT_TRANSITIONS", 1)))
        target_episodes = max(1, int(self.config.ROLLOUT_EPISODES))
        max_episodes = max(target_episodes, min_transitions * 4)
        episodes_collected = 0
        mix = self._resolve_opponent_mix()
        random_opponent_episodes = 0
        exploit_opponent_episodes = 0
        league_opponent_episodes = 0

        #for metrics
        preflop_total = 0
        preflop_fold = 0
        preflop_call_check = 0
        preflop_half_pot = 0
        preflop_pot_raise = 0
        preflop_allin = 0
        preflop_legal_pot_raise = 0
        preflop_legal_half_pot = 0
        preflop_choose_pot_raise_when_legal = 0
        preflop_choose_half_pot_when_legal = 0
        preflop_action_counts = {}
        bucket_idx = self._preflop_bucket_indices()
        fold_idx = bucket_idx["fold"]
        call_check_idx = bucket_idx["call_check"]
        half_pot_idx = bucket_idx["half_pot"]
        pot_raise_idx = bucket_idx["pot_raise"]
        allin_idx = bucket_idx["allin"]

        self.model.eval()
        while episodes_collected < target_episodes or (
            len(self.rollout_buffer) < min_transitions and episodes_collected < max_episodes
        ):
            sampled = random.random()
            #distribtion is a true prob and agent probabs are mutally exculsive
            use_random_opponent = sampled < mix["random"]
            use_exploit_opponent = (sampled >= mix["random"]) and (
                sampled < (mix["random"] + mix["exploit"])
            )
            opponent_entry = None
            opponent_model = None
            opponent_agent = None
            if use_random_opponent:
                opponent_agent = RandomAgent()
                random_opponent_episodes += 1
            elif use_exploit_opponent:
                opponent_agent = random.choice(self._exploit_agents)
                exploit_opponent_episodes += 1
            else:
                opponent_entry = self.league.sample_opponent()
                #defensive check to make sure we always have an opponent model, should never be None due to initial snapshot but just in case
                opponent_model = self.model if opponent_entry is None else self._get_opponent_model(opponent_entry)
                league_opponent_episodes += 1

            episode_transitions, final_return, _ = self.worker.generate_episode(
                policy_model=self.model,
                opponent_model=opponent_model,
                opponent_agent=opponent_agent,
                train_player=None,
                bb_size=self.config.BB_SIZE,
                curriculum=self.curriculum,
                progress=progress,
                policy_temperature=policy_temperature,
            )

            #record rollout transitions in dict
            #for metrics
            for transition in episode_transitions:
                self.rollout_buffer.add(
                    state=transition["state"],
                    action=transition["action"],
                    reward=transition["reward"],
                    done=bool(transition["done"]),
                    log_prob=transition["log_prob"],
                    value=transition["value"],
                )
                scalars = transition["state"].get("scalars")
                is_preflop = False
                if torch.is_tensor(scalars) and scalars.numel() >= 9:
                    is_preflop = bool(float(scalars[5].item()) > 0.5)
                if is_preflop:
                    action = int(transition["action"])
                    legal_mask = transition["state"].get("legal_action_mask")
                    preflop_total += 1
                    preflop_action_counts[action] = int(preflop_action_counts.get(action, 0)) + 1
                    if fold_idx is not None and action == fold_idx:
                        preflop_fold += 1
                    if call_check_idx is not None and action == call_check_idx:
                        preflop_call_check += 1
                    if half_pot_idx is not None and action == half_pot_idx:
                        preflop_half_pot += 1
                    if pot_raise_idx is not None and action == pot_raise_idx:
                        preflop_pot_raise += 1
                    if allin_idx is not None and action == allin_idx:
                        preflop_allin += 1
                    if (
                        torch.is_tensor(legal_mask)
                        and pot_raise_idx is not None
                        and 0 <= int(pot_raise_idx) < int(legal_mask.shape[0])
                        and float(legal_mask[int(pot_raise_idx)].item()) > 0.5
                    ):
                        preflop_legal_pot_raise += 1
                        if action == pot_raise_idx:
                            preflop_choose_pot_raise_when_legal += 1
                    if (
                        torch.is_tensor(legal_mask)
                        and half_pot_idx is not None
                        and 0 <= int(half_pot_idx) < int(legal_mask.shape[0])
                        and float(legal_mask[int(half_pot_idx)].item()) > 0.5
                    ):
                        preflop_legal_half_pot += 1
                        if action == half_pot_idx:
                            preflop_choose_half_pot_when_legal += 1

            returns.append(final_return)

            # elo updates for league opponents based on eps result
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
        if preflop_total > 0:
            probs = [count / float(preflop_total) for count in preflop_action_counts.values()]
            preflop_entropy_bits = float(-sum(p * np.log2(max(p, 1e-12)) for p in probs if p > 0.0))
            preflop_fold_freq = float(preflop_fold) / float(preflop_total)
            preflop_call_check_freq = float(preflop_call_check) / float(preflop_total)
            preflop_half_pot_freq = float(preflop_half_pot) / float(preflop_total)
            preflop_pot_raise_freq = float(preflop_pot_raise) / float(preflop_total)
            preflop_allin_freq = float(preflop_allin) / float(preflop_total)
        else:
            preflop_entropy_bits = 0.0
            preflop_fold_freq = 0.0
            preflop_call_check_freq = 0.0
            preflop_half_pot_freq = 0.0
            preflop_pot_raise_freq = 0.0
            preflop_allin_freq = 0.0
        random_ratio = float(random_opponent_episodes) / float(max(1, episodes_collected))
        exploit_ratio = float(exploit_opponent_episodes) / float(max(1, episodes_collected))
        league_ratio = float(league_opponent_episodes) / float(max(1, episodes_collected))
        preflop_pot_legal_rate = float(preflop_legal_pot_raise) / float(max(1, preflop_total))
        preflop_half_legal_rate = float(preflop_legal_half_pot) / float(max(1, preflop_total))
        preflop_choose_pot_given_legal = float(preflop_choose_pot_raise_when_legal) / float(
            max(1, preflop_legal_pot_raise)
        )
        preflop_choose_half_given_legal = float(preflop_choose_half_pot_when_legal) / float(
            max(1, preflop_legal_half_pot)
        )

        if self.curriculum is not None:
            curriculum_meta = self.curriculum.phase_metadata(progress)
        else:
            curriculum_meta = {
                "curriculum/enabled": 0.0,
                "curriculum/phase": 3.0,
                "curriculum/max_raise_pot_mult": -1.0,
                "curriculum/progress": float(min(max(progress, 0.0), 1.0)),
            }
        return {
            "rollout/episodes": int(episodes_collected),
            "rollout/mean_final_return_bb": mean_return,
            "rollout/transitions": len(self.rollout_buffer),
            "rollout/min_target_transitions": min_transitions,
            "league/current_elo": self.league.current_rating,
            "league/target_opponent_ratio_league": float(mix["league"]),
            "league/target_opponent_ratio_exploit": float(mix["exploit"]),
            "league/target_opponent_ratio_random": float(mix["random"]),
            "league/random_opponent_ratio_observed": random_ratio,
            "league/exploit_opponent_ratio_observed": exploit_ratio,
            "league/league_opponent_ratio_observed": league_ratio,
            "train/policy_temperature": float(policy_temperature),
            "train/preflop_action_entropy_bits": preflop_entropy_bits,
            "train/preflop_fold_freq": preflop_fold_freq,
            "train/preflop_call_check_freq": preflop_call_check_freq,
            "train/preflop_half_pot_freq": preflop_half_pot_freq,
            "train/preflop_pot_raise_freq": preflop_pot_raise_freq,
            "train/preflop_legal_rate/pot_raise": preflop_pot_legal_rate,
            "train/preflop_legal_rate/half_pot": preflop_half_legal_rate,
            "train/preflop_choose_given_legal/pot_raise": preflop_choose_pot_given_legal,
            "train/preflop_choose_given_legal/half_pot": preflop_choose_half_given_legal,
            "train/preflop_allin_freq": preflop_allin_freq,
            **curriculum_meta,
        }

    def _to_device(self, batch_states: Dict[str, torch.Tensor]):
        return {k: v.to(self.device) for k, v in batch_states.items()}

    def _ppo_update(self, entropy_coef: float = None, style_coef: float = None):
        # no eps collected
        if len(self.rollout_buffer) == 0:
            return {
                "train/updates": 0,
                "train/nonfinite_grad_skips": 0,
                "train/nonfinite_loss_batches": 0,
                "train/value_target_var": 0.0,
                "train/explained_var": 0.0,
                "train/explained_var_raw": 0.0,
                "train/explained_var_valid_fraction": 0.0,
                "train/kl_spike_events": 0.0,
                "train/lr_multiplier": float(self._lr_multiplier),
                "train/adaptive_lr_event": 0.0,
                "train/kl_low_streak": float(self._kl_low_streak),
                "train/entropy_coef_effective": float(
                    self.config.PPO_ENTROPY_COEF if entropy_coef is None else entropy_coef
                ),
                "train/style_coef_effective": float(
                    0.0 if style_coef is None else style_coef
                ),
                "train/style_kl": 0.0,
                "train/style_hinge": 0.0,
                "train/style_loss": 0.0,
                "train/style_preflop_fraction": 0.0,
            }

        self.model.train()
        #get advs and returns from buffer
        advantages, returns = self.rollout_buffer.compute_advantages(
            gamma=self.config.PPO_GAMMA,
            gae_lambda=self.config.PPO_GAE_LAMBDA,
        )

        #normalize and clip advs to stabalize updates
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = self._clip_advantages(advantages)
        value_scale = max(float(getattr(self.config, "VALUE_TARGET_SCALE", 1.0)), 1e-6)
        value_loss_type = str(getattr(self.config, "VALUE_LOSS_TYPE", "huber")).lower()
        huber_delta = max(float(getattr(self.config, "VALUE_HUBER_DELTA", 1.0)) / value_scale, 1e-6)
        entropy_coef_effective = float(
            self.config.PPO_ENTROPY_COEF if entropy_coef is None else entropy_coef
        )
        # style_coef_effective = float(
        #     self._style_reg_coef(1.0) if style_coef is None else style_coef
        # )
        skip_nonfinite_grad = bool(getattr(self.config, "SKIP_NONFINITE_GRAD", True))
        kl_spike_threshold = max(float(self.config.PPO_TARGET_KL), 1e-8) * max(
            float(getattr(self.config, "PPO_ADAPTIVE_KL_HIGH", 2.0)),
            1.0,
        )

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_clipfrac = 0.0
        total_explained_var = 0.0
        total_explained_var_raw = 0.0
        total_value_target_var = 0.0
        total_grad_norm = 0.0
        total_style_kl = 0.0
        total_style_hinge = 0.0
        total_style_loss = 0.0
        total_style_preflop_fraction = 0.0
        minibatch_count = 0
        explained_var_valid_count = 0
        kl_spike_events = 0
        optimizer_updates = 0
        nonfinite_grad_skips = 0
        nonfinite_loss_batches = 0

        # dealloc to save memory, will be reallocated in loop as needed
        self.optimizer.zero_grad(set_to_none=True)
        grad_accum = max(1, int(self.config.GRAD_ACCUM_STEPS))
        accum = 0
        early_stop = False

        #ppo update with multiple epochs and minibatches, with early stopping based on kl spikes
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

                #amp to autocast to fp16 for faster trianign and lower memory usage, 
                with torch.amp.autocast(device_type="cuda", enabled=self.amp_enabled):
                    # fw pass through model to get new log probs, entropy, and value preds
                    outputs = self.model(states)
                    logits = masked_logits(outputs["policy_logits"], states["legal_action_mask"])
                    dist = Categorical(logits=logits)

                    new_log_probs = dist.log_prob(actions)
                    entropy = dist.entropy().mean()

                    #PPO loss calc w/ clipping and huber
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(
                        ratio,
                        1.0 - self.config.PPO_CLIP_EPS,
                        1.0 + self.config.PPO_CLIP_EPS,
                    ) * adv
                    #batch to scalar with mean and then clip(based on percentage of batch)
                    policy_loss = -torch.min(surr1, surr2).mean()
                    clipfrac = ((ratio - 1.0).abs() > self.config.PPO_CLIP_EPS).float().mean()

                    # normalize value preds and targets to stabalize grads bc we scale reutrns
                    value_pred = outputs["state_value"] / value_scale
                    #dont use mse 
                    if value_loss_type == "mse":
                        value_loss = F.mse_loss(value_pred, scaled_returns)
                    else:
                        value_loss = F.huber_loss(value_pred, scaled_returns, delta=huber_delta)

                    style_kl, style_hinge, style_preflop_fraction = self._compute_style_regularizer(
                        probs=dist.probs,
                        legal_mask=states["legal_action_mask"],
                        scalars=states["scalars"],
                    )
                    style_loss = style_kl + style_hinge

                    
                    loss = (
                        policy_loss
                        + self.config.PPO_VALUE_COEF * value_loss
                        - entropy_coef_effective * entropy
                    #grad accumed should be 1
                    ) / grad_accum

                #defensive inf loss protection
                if not torch.isfinite(loss.detach()).item():
                    nonfinite_loss_batches += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    accum = 0
                    continue

                #for early stopping, first order taylor approx,
                approx_kl = (old_log_probs - new_log_probs).mean().item()
                with torch.no_grad():
                    explained_var_raw, explained_var_valid, value_target_var = self._compute_explained_var(
                        value_pred,
                        scaled_returns,
                    )
                    explained_var = explained_var_raw if explained_var_valid else 0.0
                if approx_kl > kl_spike_threshold:
                    kl_spike_events += 1

                #multiply loss by scale factor to prevend underflow wiht fp16
                if self.amp_enabled:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum += 1
                minibatch_count += 1
                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                total_style_kl += float(style_kl.item())
                total_style_hinge += float(style_hinge.item())
                total_style_loss += float(style_loss.item())
                total_style_preflop_fraction += float(style_preflop_fraction.item())
                total_kl += float(approx_kl)
                total_clipfrac += float(clipfrac.item())
                total_explained_var += explained_var
                total_explained_var_raw += explained_var_raw
                total_value_target_var += value_target_var
                if explained_var_valid:
                    explained_var_valid_count += 1

                if accum % grad_accum == 0:
                    if self.amp_enabled:
                        self.scaler.unscale_(self.optimizer)
                    #clip grads to prevent explosion
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.GRAD_CLIP,
                    )
                    grad_norm_value = float(grad_norm.item())
                    grad_is_nonfinite = not np.isfinite(grad_norm_value)

                    #throw away NaNs to protect training, log for debugging
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

                #stop if divergence too high to prevent overfitting
                if approx_kl > self.config.PPO_TARGET_KL:
                    early_stop = True
                    break

            if early_stop:
                break

        # update unstepped accumlated gradents if needed
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
            # scheduler tracks performance (higher is better), use negative value loss proxy.
            self.scheduler.step(-(total_value_loss / minibatch_count))

        denom = max(1, minibatch_count)
        avg_kl = total_kl / denom
        adaptive_metrics = self._maybe_adjust_adaptive_lr(avg_kl)
        return {
            "train/updates": optimizer_updates,
            "train/policy_loss": total_policy_loss / denom,
            "train/value_loss": total_value_loss / denom,
            "train/entropy": total_entropy / denom,
            "train/approx_kl": avg_kl,
            "train/clipfrac": total_clipfrac / denom,
            "train/explained_var": total_explained_var / max(1, explained_var_valid_count),
            "train/explained_var_raw": total_explained_var_raw / denom,
            "train/value_target_var": total_value_target_var / denom,
            "train/explained_var_valid_fraction": float(explained_var_valid_count) / float(denom),
            "train/kl_spike_events": float(kl_spike_events),
            "train/grad_norm": total_grad_norm / max(1, optimizer_updates),
            "train/nonfinite_grad_skips": int(nonfinite_grad_skips),
            "train/nonfinite_loss_batches": int(nonfinite_loss_batches),
            "train/entropy_coef_effective": entropy_coef_effective,
            "train/style_kl": total_style_kl / denom,
            "train/style_hinge": total_style_hinge / denom,
            "train/style_loss": total_style_loss / denom,
            "train/style_preflop_fraction": total_style_preflop_fraction / denom,
            **adaptive_metrics,
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
                    # backwards-compatible aliases.
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

    @staticmethod
    def _extract_primary_eval_score(metrics: Dict[str, float]):
        if "eval/mccfr_es_bb100" in metrics:
            return float(metrics["eval/mccfr_es_bb100"])
        if "eval/cfr_bb100" in metrics:
            return float(metrics["eval/cfr_bb100"])
        return None

    def _best_eval_path(self) -> str:
        return str(getattr(self.config, "CHECKPOINT_BEST_EVAL_PATH", "checkpoints/best_eval.pt"))

    def _maybe_save_best_eval(self, step: int, metrics: Dict[str, float]):
        if not bool(getattr(self.config, "CHECKPOINT_SAVE_BEST_EVAL", True)):
            return None

        score = self._extract_primary_eval_score(metrics)
        if score is None or not np.isfinite(score):
            return None

        if score <= self.best_eval_score:
            return None

        self.best_eval_score = float(score)
        self.best_eval_step = int(step)
        best_path = self._best_eval_path()
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=step,
            path=best_path,
            max_keep=1,
            single_file=False,
            save_optimizer=False,
            save_scheduler=False,
            save_league=False,
            save_extra=True,
            league_state={},
            config=self.config,
            extra={
                "best_eval_score": float(score),
                "best_eval_step": int(step),
                "best_eval_metric": "eval/mccfr_es_bb100",
            },
        )
        return best_path

    def _save_checkpoint(self, step: int, extra_metrics: Dict[str, float]):
        if self.config.CHECKPOINT_SINGLE_FILE:
            path = os.path.join("checkpoints", "latest.pt")
        else:
            path = os.path.join("checkpoints", f"point_{step}.pt")
        keep_files = []
        if self.config.CHECKPOINT_SINGLE_FILE and bool(getattr(self.config, "CHECKPOINT_SAVE_BEST_EVAL", True)):
            keep_files.append(self._best_eval_path())
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
            keep_files=keep_files,
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

        best_path = self._best_eval_path()
        if os.path.exists(best_path):
            try:
                best_payload = torch.load(best_path, map_location=self.device)
                best_extra = best_payload.get("extra", {}) if isinstance(best_payload, dict) else {}
                best_score = best_extra.get("best_eval_score")
                if best_score is not None and np.isfinite(float(best_score)):
                    self.best_eval_score = float(best_score)
                    self.best_eval_step = int(best_extra.get("best_eval_step", 0))
            except Exception:
                pass
        return int(payload.get("step", 0))

    def train(self):
        init_wandb(config=self.config)
        start_step = self._maybe_resume()
        static_metadata = {
            "meta/config_name": str(getattr(self.config, "CONFIG_NAME", "default")),
            "meta/betting_abstraction": str(getattr(self.config, "BETTING_ABSTRACTION", "fcpa")),
            "meta/requested_betting_abstraction": str(self.requested_betting_abstraction),
            "meta/effective_betting_abstraction": str(self.effective_betting_abstraction),
            "meta/strict_abstraction": float(bool(getattr(self.config, "STRICT_ABSTRACTION", True))),
            "meta/seed": int(getattr(self.config, "SEED", 0)),
            "meta/resume_step": int(start_step),
            "meta/style_target_loaded": float(self._style_target is not None),
        }
        print(
            f"Starting training: start_step={start_step}, "
            f"target_iterations={self.config.CFR_ITERATIONS}, "
            f"device={self.device}, "
            f"run_id={self.run_id}",
            flush=True,
        )

        for iteration in range(start_step, self.config.CFR_ITERATIONS):
            #frac used for curriculum
            progress = float(iteration + 1) / float(max(1, int(self.config.CFR_ITERATIONS)))
            schedule_progress = self._effective_schedule_progress(iteration, start_step)
            policy_temperature = self._policy_temperature(schedule_progress)
            entropy_coef = self._entropy_coef(schedule_progress)
            metrics = {
                "iteration": iteration + 1,
                "lr": self.optimizer.param_groups[0]["lr"],
                "run_id": self.run_id,
                "train/schedule_progress": float(min(max(schedule_progress, 0.0), 1.0)),
                **static_metadata,
            }

            rollout_metrics = self._collect_rollouts(
                progress=progress,
                policy_temperature=policy_temperature,
            )
            metrics.update(rollout_metrics)

            update_metrics = self._ppo_update(entropy_coef=entropy_coef)
            metrics.update(update_metrics)

            #snapshot models for league with and refesh cache
            if (iteration + 1) % self.config.SNAPSHOT_INTERVAL == 0:
                snapshot_score = rollout_metrics.get("rollout/mean_final_return_bb", 0.0)
                self.league.add_snapshot(self.model, step=iteration + 1, score=snapshot_score)
                self._opponent_cache.clear()
                metrics["league/size"] = len(self.league.entries)

            if self.config.EVAL_FREQ > 0 and (iteration + 1) % self.config.EVAL_FREQ == 0:
                metrics.update(self._evaluate())
                best_path = self._maybe_save_best_eval(iteration + 1, metrics)
                if best_path is not None:
                    metrics["checkpoint/best_eval_path"] = str(best_path)
                    metrics["checkpoint/best_eval_score"] = float(self.best_eval_score)
                    metrics["checkpoint/best_eval_step"] = int(self.best_eval_step)

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

        # gugrantee a final checkpoint exists even if the last step does not hit CHECKPOINT_FREQ.
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
