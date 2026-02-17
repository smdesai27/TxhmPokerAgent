import numpy as np


class Agent:
    def step(self, state):
        raise NotImplementedError


def _action_name(state, action: int) -> str:
    try:
        return str(state.action_to_string(state.current_player(), int(action))).lower()
    except Exception:
        return ""


class RandomAgent(Agent):
    def step(self, state):
        legal_actions = state.legal_actions()
        return np.random.choice(legal_actions)


class AlwaysCallAgent(Agent):
    def step(self, state):
        legal_actions = state.legal_actions()
        preferred = []
        fallback = []
        for action in legal_actions:
            action_name = _action_name(state, action)
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


class PotPressureAgent(Agent):
    """Prefers bigger raises/check-raises to expose value-leaking passivity."""

    def step(self, state):
        legal_actions = state.legal_actions()
        by_name = {int(a): _action_name(state, int(a)) for a in legal_actions}

        pot = [a for a, n in by_name.items() if "pot" in n or "raise" in n or "bet" in n]
        half = [a for a, n in by_name.items() if "half" in n]
        call = [a for a, n in by_name.items() if "call" in n or "check" in n]
        fold = [a for a, n in by_name.items() if "fold" in n]

        if pot:
            return int(np.random.choice(pot))
        if half:
            return int(np.random.choice(half))
        if call:
            return int(np.random.choice(call))
        non_fold = [a for a in legal_actions if int(a) not in fold]
        if non_fold:
            return int(np.random.choice(non_fold))
        return int(np.random.choice(legal_actions))


class StickyCallAgent(Agent):
    """Over-calls/checks and avoids folding to create an exploitable station baseline."""

    def step(self, state):
        legal_actions = state.legal_actions()
        by_name = {int(a): _action_name(state, int(a)) for a in legal_actions}

        call = [a for a, n in by_name.items() if "call" in n or "check" in n]
        half = [a for a, n in by_name.items() if "half" in n]
        pot = [a for a, n in by_name.items() if "pot" in n or "raise" in n or "bet" in n]
        fold = [a for a, n in by_name.items() if "fold" in n]

        if call:
            return int(np.random.choice(call))
        if half:
            return int(np.random.choice(half))
        if pot:
            return int(np.random.choice(pot))
        non_fold = [a for a in legal_actions if int(a) not in fold]
        if non_fold:
            return int(np.random.choice(non_fold))
        return int(np.random.choice(legal_actions))


class PassiveCallerAgent(Agent):
    """Calls/checks when possible and avoids raises to expose value-betting opportunities."""

    def step(self, state):
        legal_actions = state.legal_actions()
        by_name = {int(a): _action_name(state, int(a)) for a in legal_actions}

        call = [a for a, n in by_name.items() if "call" in n or "check" in n]
        fold = [a for a, n in by_name.items() if "fold" in n]
        non_raise_non_fold = [
            a
            for a, n in by_name.items()
            if ("raise" not in n and "bet" not in n and "pot" not in n and "half" not in n and int(a) not in fold)
        ]

        if call:
            return int(np.random.choice(call))
        if non_raise_non_fold:
            return int(np.random.choice(non_raise_non_fold))
        if fold:
            return int(np.random.choice(fold))
        return int(np.random.choice(legal_actions))


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
