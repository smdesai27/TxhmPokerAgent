import numpy as np

class Agent:
    def step(self, state):
        raise NotImplementedError

class RandomAgent(Agent):
    def step(self, state):
        legal_actions = state.legal_actions()
        return np.random.choice(legal_actions)

class AlwaysCallAgent(Agent):
    def step(self, state):
        legal_actions = state.legal_actions()
        player = state.current_player()
        preferred = []
        fallback = []
        for action in legal_actions:
            try:
                action_name = state.action_to_string(player, action).lower()
            except Exception:
                action_name = ""
            if "call" in action_name or "check" in action_name:
                preferred.append(action)
            elif "fold" in action_name:
                continue
            else:
                fallback.append(action)

        if preferred:
            return int(preferred[0])
        if fallback:
            return int(fallback[0])
        return int(legal_actions[0])

class RuleBasedAgent(Agent):
    """Simple Tight-Aggressive Agent."""
    def step(self, state):
        # Placeholder for complex rule logic
        # Just plays randomly but prefers raising if holding pairs (if we could parse state)
        legal_actions = state.legal_actions()
        return np.random.choice(legal_actions)


class PolicyAgent(Agent):
    """Adapter for OpenSpiel policies exposing action_probabilities()."""

    def __init__(self, policy):
        self.policy = policy

    def step(self, state):
        probs = self.policy.action_probabilities(state, state.current_player())
        actions = list(probs.keys())
        p = np.array([probs[a] for a in actions], dtype=np.float64)
        p = p / p.sum()
        return int(np.random.choice(actions, p=p))
