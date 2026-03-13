import math
import re
from typing import Dict, Tuple

import torch


_FOLD_RE = re.compile(r"\bmove=Fold\b", re.IGNORECASE)
_CALL_CHECK_RE = re.compile(r"\bmove=(Call|Check)\b", re.IGNORECASE)
_ALLIN_RE = re.compile(r"\bmove=AllIn\b", re.IGNORECASE)
_RAISE_BET_RE = re.compile(r"\bmove=(RaiseTo|Raise|Bet)\b", re.IGNORECASE)
_AMOUNT_RE = re.compile(r"(?:RaiseTo|Raise|Bet)\s+(-?\d+(?:\.\d+)?)", re.IGNORECASE)


class FullgameActionCurriculum:
    """Progressively expands allowed fullgame raise sizes during training."""

    def __init__(self, config):
        self.enabled = bool(getattr(config, "FULLGAME_CURRICULUM_ENABLE", True))
        self.phase1_end = float(getattr(config, "FULLGAME_CURRICULUM_PHASE1_END", 0.30))
        self.phase2_end = float(getattr(config, "FULLGAME_CURRICULUM_PHASE2_END", 0.70))
        self.max_raise_mult_p1 = float(getattr(config, "FULLGAME_CURRICULUM_MAX_RAISE_POT_MULT_P1", 1.0))
        self.max_raise_mult_p2 = float(getattr(config, "FULLGAME_CURRICULUM_MAX_RAISE_POT_MULT_P2", 2.5))

        self.phase1_end = min(max(self.phase1_end, 0.0), 1.0)
        self.phase2_end = min(max(self.phase2_end, self.phase1_end), 1.0)
        self.max_raise_mult_p1 = max(self.max_raise_mult_p1, 0.0)
        self.max_raise_mult_p2 = max(self.max_raise_mult_p2, self.max_raise_mult_p1)

    def phase_metadata(self, progress: float) -> Dict[str, float]:
        phase, max_raise_mult = self._phase_and_max_raise(progress)
        return {
            "curriculum/enabled": 1.0 if self.enabled else 0.0,
            "curriculum/phase": float(phase),
            "curriculum/max_raise_pot_mult": float(max_raise_mult),
            "curriculum/progress": float(min(max(progress, 0.0), 1.0)),
        }

    def _phase_and_max_raise(self, progress: float) -> Tuple[int, float]:
        progress = float(min(max(progress, 0.0), 1.0))
        if not self.enabled:
            return 3, math.inf
        if progress <= self.phase1_end:
            return 1, self.max_raise_mult_p1
        if progress <= self.phase2_end:
            return 2, self.max_raise_mult_p2
        return 3, math.inf

    @staticmethod
    def _extract_raise_amount(action_text: str):
        match = _AMOUNT_RE.search(action_text)
        if not match:
            return None
        try:
            return float(match.group(1))
        except Exception:
            return None

    @staticmethod
    def _action_name(state, player_id: int, action: int) -> str:
        try:
            return str(state.action_to_string(player_id, action))
        except Exception:
            return ""

    def _is_action_allowed(
        self,
        action_text: str,
        raise_cap: float,
    ) -> bool:
        if _FOLD_RE.search(action_text):
            return True
        if _CALL_CHECK_RE.search(action_text):
            return True
        if _ALLIN_RE.search(action_text):
            return True
        if not _RAISE_BET_RE.search(action_text):
            return math.isinf(raise_cap)

        if math.isinf(raise_cap):
            return True

        amount = self._extract_raise_amount(action_text)
        if amount is None:
            return False
        return bool(amount <= raise_cap + 1e-6)

    def mask_for_state(
        self,
        state,
        player_id: int,
        num_actions: int,
        legal_action_mask: torch.Tensor,
        progress: float,
    ) -> torch.Tensor:
        if not self.enabled:
            return legal_action_mask

        if state.is_terminal() or state.is_chance_node():
            return legal_action_mask

        phase, max_raise_mult = self._phase_and_max_raise(progress)
        if phase >= 3:
            return legal_action_mask

        legal_actions = [int(a) for a in state.legal_actions() if 0 <= int(a) < num_actions]
        if not legal_actions:
            return legal_action_mask

        try:
            pot_size = float(getattr(state.to_struct(), "pot_size", 0.0))
        except Exception:
            pot_size = 0.0
        raise_cap = max(1.0, pot_size) * max_raise_mult

        allowed = []
        fallback_call_check = []
        fallback_fold = []
        for action in legal_actions:
            action_text = self._action_name(state, player_id, action)
            if _CALL_CHECK_RE.search(action_text):
                fallback_call_check.append(action)
            if _FOLD_RE.search(action_text):
                fallback_fold.append(action)
            if self._is_action_allowed(action_text, raise_cap=raise_cap):
                allowed.append(action)

        if not allowed:
            if fallback_call_check:
                allowed = [fallback_call_check[0]]
            elif fallback_fold:
                allowed = [fallback_fold[0]]
            else:
                allowed = [legal_actions[0]]

        mask = torch.zeros_like(legal_action_mask)
        mask[allowed] = 1.0
        return legal_action_mask * mask
