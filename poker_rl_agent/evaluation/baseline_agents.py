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
        # Assuming Call is usually index 1 or we check action strings
        # In OpenSpiel, 'Call' index depends on the game structure.
        # For simplicity, we try to find 'Call' or 'Check' in string representation
        # Or just pick the middle action (often Call).
        # Better: Pyspiel constant if available.
        # Here we just pick index 1 if available, else random.
        if 1 in legal_actions:
            return 1
        return legal_actions[0]

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
