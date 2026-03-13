
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
    ):
        self.k_best = k_best
        self.init_rating = init_rating
        self.k_factor = k_factor
        self.pfsp_beta = pfsp_beta

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
        #pfsp sampling based on ratings (form alphastar), stronger = more likely to sample
        if not self.entries:
            return None

        ratings = np.array([entry.rating for entry in self.entries], dtype=np.float64)
        scaled = ratings / max(self.pfsp_beta, 1e-6)
        scaled = scaled - np.max(scaled)
        probs = np.exp(scaled)
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
