import numpy as np
import torch
from torch.distributions import Categorical

from ..environment.openspiel_wrapper import PokerEnv
from ..environment.state_representation import StateEncoder
from ..models.model_utils import masked_logits


class SelfPlayWorker:
    """Collects on-policy trajectories for PPO self-play."""

    def __init__(self, env: PokerEnv, device="cpu", max_action_history: int = 64):
        self.env = env
        self.device = device
        self.encoder = StateEncoder(device="cpu", max_action_history=max_action_history)

    def _to_model_device(self, state_dict):
        return {
            k: (v.unsqueeze(0).to(self.device) if v.dim() > 0 else v.to(self.device))
            for k, v in state_dict.items()
        }

    @torch.no_grad()
    def _sample_action(self, model, state_dict_cpu, policy_temperature: float = 1.0):
        batch = self._to_model_device(state_dict_cpu)
        outputs = model(batch)
        logits = masked_logits(outputs["policy_logits"], batch["legal_action_mask"])
        temp = max(float(policy_temperature), 1e-3)
        logits = logits / temp
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = outputs["state_value"]

        return int(action.item()), float(log_prob.item()), float(value.item())

    @staticmethod
    def _apply_curriculum_mask(
        curriculum,
        state,
        player_id: int,
        num_actions: int,
        encoded_state,
        progress: float,
    ):
        if curriculum is None:
            return encoded_state
        legal_mask = encoded_state["legal_action_mask"]
        masked = curriculum.mask_for_state(
            state=state,
            player_id=player_id,
            num_actions=num_actions,
            legal_action_mask=legal_mask,
            progress=progress,
        )
        out = dict(encoded_state)
        out["legal_action_mask"] = masked
        return out

    def generate_episode(
        self,
        policy_model,
        opponent_model=None,
        opponent_agent=None,
        train_player=None,
        bb_size: float = 100.0,
        curriculum=None,
        progress: float = 1.0,
        policy_temperature: float = 1.0,
    ):
        if opponent_model is None and opponent_agent is None:
            opponent_model = policy_model
        if train_player is None:
            train_player = np.random.choice([0, 1])

        state = self.env.reset()
        transitions = []

        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                action_list, probs = zip(*outcomes)
                action = np.random.choice(action_list, p=probs)
                state.apply_action(int(action))
                continue

            current_player = state.current_player()
            if current_player != train_player and opponent_agent is not None:
                action = int(opponent_agent.step(state))
                state.apply_action(action)
                continue

            model = policy_model if current_player == train_player else opponent_model

            encoded = self.encoder.encode_state(
                state,
                current_player,
                num_actions=self.env.num_actions(),
            )
            encoded = self._apply_curriculum_mask(
                curriculum=curriculum,
                state=state,
                player_id=current_player,
                num_actions=self.env.num_actions(),
                encoded_state=encoded,
                progress=progress,
            )
            sampled_temp = policy_temperature if current_player == train_player else 1.0
            action, log_prob, value = self._sample_action(
                model,
                encoded,
                policy_temperature=sampled_temp,
            )
            state.apply_action(action)

            if current_player == train_player:
                transitions.append(
                    {
                        "state": encoded,
                        "action": action,
                        "log_prob": log_prob,
                        "value": value,
                        "reward": 0.0,
                        "done": 0.0,
                    }
                )

        final_return = float(state.returns()[train_player]) / max(bb_size, 1.0)
        if transitions:
            transitions[-1]["reward"] = final_return
            transitions[-1]["done"] = 1.0

        return transitions, final_return, int(train_player)
