import random
import time
from typing import Dict, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from open_spiel.python.algorithms.external_sampling_mccfr import ExternalSamplingSolver
from open_spiel.python.algorithms import exploitability
from open_spiel.python import policy as policy_lib

from ..environment.openspiel_wrapper import PokerEnv
from ..environment.state_representation import StateEncoder
from ..models.model_utils import masked_logits
from .baseline_agents import PolicyAgent


class Evaluator:
    def __init__(self, agent_model, config, device="cpu"):
        if hasattr(config, "sync_legacy_fields"):
            config.sync_legacy_fields()
        self.agent_model = agent_model
        self.config = config
        self.device = device
        try:
            self.model_device = next(self.agent_model.parameters()).device
        except StopIteration:
            self.model_device = torch.device("cpu")
        self.env = PokerEnv(
            game_name=config.GAME_NAME,
            env_preset=config.ENV_PRESET,
            betting_abstraction=config.BETTING_ABSTRACTION,
        )
        self.encoder = StateEncoder(device="cpu", max_action_history=config.MAX_ACTION_HISTORY)
        self._baseline_cache: Dict[Tuple[str, int, int, Tuple[Tuple[str, str], ...]], Dict[str, object]] = {}

    class _ModelPolicyAdapter(policy_lib.Policy):
        def __init__(self, evaluator):
            super().__init__(evaluator.env.game, [0, 1])
            self.evaluator = evaluator

        def action_probabilities(self, state, player_id=None):
            if state.is_terminal() or state.is_chance_node():
                return {}
            if player_id is None:
                player_id = state.current_player()

            encoded = self.evaluator.encoder.encode_state(
                state,
                player_id,
                self.evaluator.env.num_actions(),
            )
            with torch.no_grad():
                batch = {k: v.unsqueeze(0).to(self.evaluator.model_device) for k, v in encoded.items()}
                outputs = self.evaluator.agent_model(batch)
                logits = masked_logits(outputs["policy_logits"], batch["legal_action_mask"])
                probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

            legal = state.legal_actions()
            probs_legal = np.array([probs[a] for a in legal], dtype=np.float64)
            if probs_legal.sum() <= 0:
                probs_legal = np.ones_like(probs_legal) / len(probs_legal)
            else:
                probs_legal /= probs_legal.sum()
            return {a: float(p) for a, p in zip(legal, probs_legal)}

    @torch.no_grad()
    def _sample_model_action(self, state, player_id):
        encoded = self.encoder.encode_state(state, player_id, self.env.num_actions())
        batch = {k: v.unsqueeze(0).to(self.model_device) for k, v in encoded.items()}

        outputs = self.agent_model(batch)
        logits = masked_logits(outputs["policy_logits"], batch["legal_action_mask"])
        dist = Categorical(logits=logits)
        return int(dist.sample().item())

    @staticmethod
    def _seed_rngs(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def evaluate(self, opponent, num_episodes=100, seed=None):
        if seed is not None:
            self._seed_rngs(int(seed))

        returns = []

        for episode_idx in range(num_episodes):
            state = self.env.reset()
            agent_player_id = 0 if episode_idx < (num_episodes / 2) else 1

            while not state.is_terminal():
                if state.is_chance_node():
                    outcomes = state.chance_outcomes()
                    action_list, probs = zip(*outcomes)
                    action = np.random.choice(action_list, p=probs)
                    state.apply_action(int(action))
                    continue

                current_player = state.current_player()
                if current_player == agent_player_id:
                    action = self._sample_model_action(state, current_player)
                else:
                    action = int(opponent.step(state))
                state.apply_action(action)

            returns.append(float(state.returns()[agent_player_id]))

        returns = np.array(returns, dtype=np.float64)
        avg_return = float(np.mean(returns))
        bb_per_100 = float((avg_return / max(self.config.BB_SIZE, 1.0)) * 100.0)
        std_err = float(np.std(returns) / np.sqrt(len(returns))) if len(returns) > 1 else 0.0
        return {
            "avg_return": avg_return,
            "bb_per_100": bb_per_100,
            "std_err": std_err,
            "episodes": int(num_episodes),
        }

    def _game_signature(self) -> Tuple[Tuple[str, str], ...]:
        params = self.env.game_parameters()
        return tuple(sorted((str(k), str(v)) for k, v in params.items()))

    def _build_solver_baseline(self, baseline_algo: str, iterations: int, seed: int):
        algo = str(baseline_algo).lower()
        if algo != "mccfr_external_sampling":
            raise ValueError(f"Unsupported solver baseline algorithm: {baseline_algo}")

        self._seed_rngs(seed)

        build_start = time.perf_counter()
        solver = ExternalSamplingSolver(self.env.game)
        for _ in range(iterations):
            solver.iteration()
        avg_policy = solver.average_policy()
        build_seconds = float(time.perf_counter() - build_start)

        label = getattr(self.config, "EVAL_BASELINE_LABEL", "MCCFR-ES")
        return {
            "agent": PolicyAgent(avg_policy),
            "metadata": {
                "baseline/algo": algo,
                "baseline/label": label,
                "baseline/iters": int(iterations),
                "baseline/seed": int(seed),
                "baseline/build_seconds": build_seconds,
            },
        }

    def evaluate_vs_solver_baseline(
        self,
        baseline_algo=None,
        iterations=None,
        num_episodes=100,
        seed=None,
    ):
        if hasattr(self.config, "sync_legacy_fields"):
            self.config.sync_legacy_fields()

        algo = baseline_algo or getattr(self.config, "EVAL_BASELINE_ALGO", "mccfr_external_sampling")
        iters = int(
            iterations
            if iterations is not None
            else getattr(self.config, "EVAL_BASELINE_ITERS", getattr(self.config, "EVAL_CFR_ITERATIONS", 5000))
        )
        eval_seed = int(seed if seed is not None else getattr(self.config, "SEED", 42))
        cache_mode = str(getattr(self.config, "EVAL_BASELINE_CACHE_MODE", "process")).lower()
        if cache_mode not in {"process", "none"}:
            cache_mode = "process"

        cache_key = (str(algo).lower(), iters, eval_seed, self._game_signature())
        cache_hit = False

        if cache_mode == "process" and cache_key in self._baseline_cache:
            payload = self._baseline_cache[cache_key]
            cache_hit = True
        else:
            payload = self._build_solver_baseline(algo, iters, eval_seed)
            if cache_mode == "process":
                self._baseline_cache[cache_key] = payload

        metrics = self.evaluate(payload["agent"], num_episodes=num_episodes, seed=eval_seed)
        metrics.update(payload["metadata"])
        metrics["baseline/cache_hit"] = bool(cache_hit)
        metrics["baseline/cache_mode"] = cache_mode
        return metrics

    def evaluate_vs_cfr(self, iterations=None, num_episodes=100, seed=None):
        # Backwards-compatible wrapper: the baseline is now explicitly a solver-generated MCCFR policy.
        return self.evaluate_vs_solver_baseline(
            baseline_algo=getattr(self.config, "EVAL_BASELINE_ALGO", "mccfr_external_sampling"),
            iterations=iterations,
            num_episodes=num_episodes,
            seed=seed,
        )

    def evaluate_nash_conv(self):
        model_policy = self._ModelPolicyAdapter(self)
        nash_conv_value = exploitability.nash_conv(
            self.env.game,
            model_policy,
            return_only_nash_conv=True,
            use_cpp_br=False,
        )
        return float(nash_conv_value)
