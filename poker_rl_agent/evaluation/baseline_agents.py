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


_RANK_ORDER = {r: i for i, r in enumerate("23456789TJQKA")}  # 2->0 .. A->12


def _parse_hole_cards(state):
    """Best-effort parse of the current player's two hole cards as (rank,suit) chars.

    Returns a list of (rank_char, suit_char) tuples, or [] if the state does not
    expose parseable hole cards (e.g. struct shape differs). Mirrors the compact
    2-char card encoding used by StateEncoder ("As", "Td", ...).
    """
    try:
        struct = state.to_struct()
        hands = getattr(struct, "player_hands", [])
        pid = state.current_player()
        if pid is None or pid < 0 or pid >= len(hands):
            return []
        blob = hands[pid] or ""
        cards = [blob[i : i + 2] for i in range(0, len(blob), 2)]
        out = []
        for c in cards[:2]:
            if len(c) >= 2 and c[0].upper() in _RANK_ORDER:
                out.append((c[0].upper(), c[1].lower()))
        return out
    except Exception:
        return []


def _is_top_opening_range(hole) -> bool:
    """Deterministic top-~20% HUNL opening range from two hole cards.

    Conservative tight-aggressive open: any pair, both cards >= ten (broadway),
    any ace, suited connectors with both cards >= eight, or king + (>= ten).
    Falls through to False (fold/limp) for trashy holdings. This is a heuristic
    anchor, not GTO -- the point is a *deterministic, non-random* opponent.
    """
    if len(hole) < 2:
        return False
    r0 = _RANK_ORDER[hole[0][0]]
    r1 = _RANK_ORDER[hole[1][0]]
    suited = hole[0][1] == hole[1][1]
    hi, lo = max(r0, r1), min(r0, r1)
    ace = _RANK_ORDER["A"]
    king = _RANK_ORDER["K"]
    ten = _RANK_ORDER["T"]
    eight = _RANK_ORDER["8"]

    if r0 == r1:  # any pocket pair
        return True
    if lo >= ten:  # both broadway (KT+, QJ, etc.)
        return True
    if hi == ace:  # any ace
        return True
    if hi == king and lo >= ten:  # KT+
        return True
    if suited and lo >= eight and (hi - lo) <= 2:  # suited connectors 8x+
        return True
    return False


def _made_hand_postflop(hole, state) -> bool:
    """Deterministic 'do I have a piece of this board?' check for postflop play.

    True if either hole card pairs the board, or we hold a pocket pair, or we
    hold an ace (top-pair/overpair potential). Coarse on purpose -- a tight
    deterministic value/aggression trigger, not a hand evaluator.
    """
    if len(hole) < 2:
        return False
    if hole[0][0] == hole[1][0]:  # pocket pair
        return True
    if hole[0][0] == "A" or hole[1][0] == "A":
        return True
    try:
        board_blob = getattr(state.to_struct(), "board_cards", "") or ""
        board_ranks = {board_blob[i].upper() for i in range(0, len(board_blob), 2)}
    except Exception:
        board_ranks = set()
    return hole[0][0] in board_ranks or hole[1][0] in board_ranks


class RuleBasedAgent(Agent):
    """Deterministic tight-aggressive (TAG) heuristic anchor.

    DETERMINISTIC by construction: given a state it always returns the same
    action (no np.random anywhere). This matters because it is used as a
    variance-reduction anchor inside duplicate-hand evaluation -- a random
    opponent would defeat the mirrored-seat card-luck cancellation. Same
    ``step(state) -> action`` interface and zero-arg constructor as before, so
    existing call sites are unaffected.

    Strategy (coarse, intentionally simple):
      * Preflop: raise (largest available raise) with a top-~20% opening range,
        otherwise check/call cheaply, folding only when facing pressure off-range.
      * Postflop: bet/raise when we have a made-hand piece (pair/ace/board pair),
        otherwise check/call, folding off-range when there is no cheap continue.
    Action *selection within a tier* is deterministic: we sort legal actions by
    index and take the first matching one, never a random draw.
    """

    def step(self, state):
        legal_actions = sorted(int(a) for a in state.legal_actions())
        by_name = {a: _action_name(state, a) for a in legal_actions}

        # Deterministic action tiers, sorted by action index for stability.
        fold = [a for a in legal_actions if "fold" in by_name[a]]
        check_call = [a for a in legal_actions if ("call" in by_name[a] or "check" in by_name[a])]
        # Smallest-to-largest raise/bet, so [-1] is the biggest pressure action.
        raises = [a for a in legal_actions if any(k in by_name[a] for k in ("raise", "bet", "pot", "half", "allin"))]
        can_check_free = any("check" in by_name[a] for a in check_call)

        hole = _parse_hole_cards(state)
        preflop = self._is_preflop(state)

        if preflop:
            in_range = _is_top_opening_range(hole)
            if in_range and raises:
                return int(raises[-1])  # value/aggression: largest raise
            if check_call:
                # In-range or able to continue cheaply -> check/call.
                if in_range or can_check_free:
                    return int(check_call[0])
                # Off-range and facing a bet: fold if allowed, else cheapest continue.
                if fold:
                    return int(fold[0])
                return int(check_call[0])
            if fold:
                return int(fold[0])
            return int(legal_actions[0])

        # Postflop.
        made = _made_hand_postflop(hole, state)
        if made and raises:
            return int(raises[-1])  # bet/raise for value
        if check_call:
            if made or can_check_free:
                return int(check_call[0])
            if fold:
                return int(fold[0])
            return int(check_call[0])
        if fold:
            return int(fold[0])
        return int(legal_actions[0])

    @staticmethod
    def _is_preflop(state) -> bool:
        try:
            board_blob = getattr(state.to_struct(), "board_cards", "") or ""
            return len(board_blob) == 0
        except Exception:
            return True


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
