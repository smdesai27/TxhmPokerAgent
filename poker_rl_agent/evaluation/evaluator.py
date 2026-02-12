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
        self._cached_cfr_agent = None

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

    def evaluate(self, opponent, num_episodes=100):
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

    def _build_cfr_agent(self, iterations: int):
        solver = ExternalSamplingSolver(self.env.game)
        for _ in range(iterations):
            solver.iteration()
        avg_policy = solver.average_policy()
        return PolicyAgent(avg_policy)

    def evaluate_vs_cfr(self, iterations=300, num_episodes=100):
        if self._cached_cfr_agent is None:
            self._cached_cfr_agent = self._build_cfr_agent(iterations)
        return self.evaluate(self._cached_cfr_agent, num_episodes=num_episodes)

    def evaluate_nash_conv(self):
        model_policy = self._ModelPolicyAdapter(self)
        nash_conv_value = exploitability.nash_conv(
            self.env.game,
            model_policy,
            return_only_nash_conv=True,
            use_cpp_br=False,
        )
        return float(nash_conv_value)
