
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class LeagueEntry:
    entry_id: int
    step: int
    rating: float
    score: float
    state_dict: Dict[str, torch.Tensor]


class LeagueManager:
    """Maintains K-best policy snapshots with Elo + PFSP sampling."""

    def __init__(
        self,
        k_best: int,
        init_rating: float = 1200.0,
        k_factor: float = 24.0,
        pfsp_beta: float = 2.0,
        pfsp_mode: str = "loss",
    ):
        self.k_best = k_best
        self.init_rating = init_rating
        self.k_factor = k_factor
        self.pfsp_beta = pfsp_beta
        # PFSP opponent-sampling mode. "loss" (default, correct): weight = (1 - P[current beats opp])
        # **beta, the win-probability GAP from the Elo logistic — adaptive to the learner's current
        # rating and diversity-preserving (matches the validated EloPool in
        # validate_leduc_exploitability.py). "legacy": the old weight ∝ exp(rating/beta), which keys
        # off ABSOLUTE rating, ignores current_rating, and with beta=2 at Elo-scale gaps saturates to
        # ~always picking the single highest-rated snapshot (no diversity, non-adaptive). Kept for A/B.
        self.pfsp_mode = str(pfsp_mode)

        self.entries: List[LeagueEntry] = []
        self.current_rating = init_rating
        self._next_id = 0

    @staticmethod
    def _cpu_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        return {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

    def add_snapshot(self, model: torch.nn.Module, step: int, score: float) -> int:
        entry = LeagueEntry(
            entry_id=self._next_id,
            step=int(step),
            rating=self.init_rating,
            score=float(score),
            state_dict=self._cpu_state_dict(model),
        )
        self._next_id += 1
        self.entries.append(entry)

        # keep top-K by score first, then rating.
        self.entries.sort(key=lambda e: (e.score, e.rating), reverse=True)
        if len(self.entries) > self.k_best:
            self.entries = self.entries[: self.k_best]

        return entry.entry_id

    def has_entries(self) -> bool:
        return len(self.entries) > 0

    def get_entry(self, entry_id: int) -> Optional[LeagueEntry]:
        for entry in self.entries:
            if entry.entry_id == entry_id:
                return entry
        return None

    def sample_opponent(self) -> Optional[LeagueEntry]:
        """Prioritized Fictitious Self-Play opponent sampling.

        "loss" (default, correct): weight = (1 - P[current beats opp])**beta, where P comes from the
        Elo logistic on the GAP (opp.rating - current_rating). This concentrates on the learner's
        hardest current matchups, is ADAPTIVE (re-weights as the learner improves), and preserves a
        smooth distribution over the pool. Matches the validated EloPool in
        validate_leduc_exploitability.py.

        "legacy": the original weight ∝ exp(rating/beta) on ABSOLUTE rating. It ignores current_rating
        and, with beta=2 at Elo-scale rating gaps, saturates to ~always selecting the single
        highest-rated snapshot — degenerate (no opponent diversity, non-adaptive). Kept for an A/B.
        """
        if not self.entries:
            return None

        ratings = np.array([entry.rating for entry in self.entries], dtype=np.float64)
        if self.pfsp_mode == "legacy":  # old: absolute-rating softmax (near-argmax, non-adaptive)
            scaled = ratings / max(self.pfsp_beta, 1e-6)
            scaled = scaled - np.max(scaled)
            probs = np.exp(scaled)
        else:  # "loss" (default, correct): win-probability-gap weighting, adaptive + diverse
            p_win = 1.0 / (1.0 + 10.0 ** ((ratings - self.current_rating) / 400.0))
            probs = np.power(1.0 - p_win, self.pfsp_beta) + 1e-6
        probs = probs / probs.sum()
        idx = np.random.choice(len(self.entries), p=probs)
        return self.entries[int(idx)]

    def update_elo(self, opponent_id: int, result: float):
        """Updates current policy and opponent ratings.

        result: 1.0 win, 0.5 tie, 0.0 loss from current policy perspective.
        """
        opponent = self.get_entry(opponent_id)
        if opponent is None:
            return

        expected_current = 1.0 / (1.0 + 10 ** ((opponent.rating - self.current_rating) / 400.0))
        expected_opp = 1.0 - expected_current

        self.current_rating += self.k_factor * (result - expected_current)
        opponent.rating += self.k_factor * ((1.0 - result) - expected_opp)

    def summary(self):
        return [
            {
                "entry_id": e.entry_id,
                "step": e.step,
                "rating": e.rating,
                "score": e.score,
            }
            for e in self.entries
        ]

    def state_dict(self):
        return {
            "k_best": self.k_best,
            "init_rating": self.init_rating,
            "k_factor": self.k_factor,
            "pfsp_beta": self.pfsp_beta,
            "current_rating": self.current_rating,
            "next_id": self._next_id,
            "entries": [
                {
                    "entry_id": e.entry_id,
                    "step": e.step,
                    "rating": e.rating,
                    "score": e.score,
                    "state_dict": e.state_dict,
                }
                for e in self.entries
            ],
        }

    def load_state_dict(self, payload):
        self.current_rating = float(payload.get("current_rating", self.init_rating))
        self._next_id = int(payload.get("next_id", 0))
        self.entries = []
        for raw in payload.get("entries", []):
            self.entries.append(
                LeagueEntry(
                    entry_id=int(raw["entry_id"]),
                    step=int(raw["step"]),
                    rating=float(raw["rating"]),
                    score=float(raw["score"]),
                    state_dict=raw["state_dict"],
                )
            )
