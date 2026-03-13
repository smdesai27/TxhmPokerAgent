import random
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass
class Transition:
    state: Dict[str, torch.Tensor]
    action: int
    reward: float
    done: float
    log_prob: float
    value: float


class RolloutBuffer:
    def __init__(self):
        self.transitions: List[Transition] = []

    def __len__(self):
        return len(self.transitions)

    def clear(self):
        self.transitions.clear()

    def add(
        self,
        state: Dict[str, torch.Tensor],
        action: int,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ):
        # store on CPU to keep rollout memory compact.
        state_cpu = {
            k: (v.detach().cpu() if torch.is_tensor(v) else v)
            for k, v in state.items()
        }
        # ensure action is int and done is float for consistency
        self.transitions.append(
            Transition(
                state=state_cpu,
                action=int(action),
                reward=float(reward),
                done=float(done),
                log_prob=float(log_prob),
                value=float(value),
            )
        )

    def compute_advantages(self, gamma: float, gae_lambda: float):
        n = len(self.transitions)
        rewards = [t.reward for t in self.transitions]
        dones = [t.done for t in self.transitions]
        values = [t.value for t in self.transitions]

        advantages = [0.0] * n
        gae = 0.0
        next_value = 0.0

        for t in reversed(range(n)):
            #make sure zero out the next value if done
            if t < n - 1:
                next_value = values[t + 1] * (1.0 - dones[t])
            else:
                next_value = 0.0

            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = [advantages[t] + values[t] for t in range(n)]
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

    def _collate_states(self, states: List[Dict[str, torch.Tensor]]):
        collated = {}

        collated["hole_cards"] = torch.stack([s["hole_cards"] for s in states], dim=0)
        collated["community_cards"] = torch.stack([s["community_cards"] for s in states], dim=0)
        collated["scalars"] = torch.stack([s["scalars"] for s in states], dim=0)
        collated["legal_action_mask"] = torch.stack([s["legal_action_mask"] for s in states], dim=0)


        # since action history since its variable lengtg, we need to pad it. 
        # we can use num_actions since we init the LSTM with it as the pad_token
        action_histories = [s["action_history"] for s in states]
        num_actions = int(collated["legal_action_mask"].shape[1])
        collated["action_history"] = pad_sequence(
            action_histories,
            batch_first=True,
            padding_value=num_actions,
        )
        return collated

    def iterate_minibatches(
        self,
        minibatch_size: int,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        shuffle: bool = True,
    ):
        indices = list(range(len(self.transitions)))
        if shuffle:
            random.shuffle(indices)

        actions = torch.tensor([t.action for t in self.transitions], dtype=torch.long)
        old_log_probs = torch.tensor([t.log_prob for t in self.transitions], dtype=torch.float32)
        old_values = torch.tensor([t.value for t in self.transitions], dtype=torch.float32)

        for start in range(0, len(indices), minibatch_size):
            batch_idx = indices[start : start + minibatch_size]
            states = [self.transitions[i].state for i in batch_idx]
            state_batch = self._collate_states(states)

            yield {
                "states": state_batch,
                "actions": actions[batch_idx],
                "old_log_probs": old_log_probs[batch_idx],
                "old_values": old_values[batch_idx],
                "advantages": advantages[batch_idx],
                "returns": returns[batch_idx],
            }
