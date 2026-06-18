"""Unit tests for LeagueManager PFSP opponent sampling (the S3 fix).

Verifies the corrected "loss" mode is adaptive + diversity-preserving, and that "legacy" reproduces
the old non-adaptive near-argmax behavior. (league_manager imports torch, so run where torch is
importable, e.g. on the OSCAR venv: `pytest tests/test_league_pfsp.py -xq`.)
"""
import numpy as np

from poker_rl_agent.training.league_manager import LeagueManager, LeagueEntry


def _set_entries(lm, ratings):
    # bypass add_snapshot (which needs a torch model) — inject entries directly for sampling tests
    lm.entries = [
        LeagueEntry(entry_id=i, step=i, rating=float(r), score=0.0, state_dict={})
        for i, r in enumerate(ratings)
    ]


def _empirical_probs(lm, n=40000, seed=0):
    np.random.seed(seed)
    counts = np.zeros(len(lm.entries))
    for _ in range(n):
        counts[lm.sample_opponent().entry_id] += 1
    return counts / counts.sum()


def test_loss_mode_favors_harder_but_keeps_diversity():
    lm = LeagueManager(k_best=8, pfsp_beta=2.0, pfsp_mode="loss")
    lm.current_rating = 1200.0
    _set_entries(lm, [1000.0, 1200.0, 1400.0])  # 0=weak, 1=even, 2=hard
    p = _empirical_probs(lm)
    assert p[2] > p[1] > p[0]   # harder (higher-rated) opponents weighted more
    assert p[0] > 0.02          # but the weak one still appears -> diversity preserved (not argmax)


def test_legacy_mode_collapses_to_top_rated():
    lm = LeagueManager(k_best=8, pfsp_beta=2.0, pfsp_mode="legacy")
    lm.current_rating = 1200.0
    _set_entries(lm, [1000.0, 1200.0, 1400.0])
    p = _empirical_probs(lm)
    assert p[2] > 0.95          # near-argmax on the single highest-rated snapshot (degenerate)


def test_loss_mode_is_adaptive_legacy_is_not():
    ratings = [1000.0, 1200.0, 1400.0]

    def probs_at(mode, rating, seed=1):
        lm = LeagueManager(k_best=8, pfsp_beta=2.0, pfsp_mode=mode)
        lm.current_rating = float(rating)
        _set_entries(lm, ratings)
        return _empirical_probs(lm, seed=seed)

    # "loss" adapts to the learner's own rating: as it strengthens (1200 -> 1600), mass concentrates
    # further on the hardest remaining opponent.
    assert probs_at("loss", 1600)[2] > probs_at("loss", 1200)[2]
    # "legacy" ignores current_rating entirely -> distribution is invariant to it.
    assert np.allclose(probs_at("legacy", 1200), probs_at("legacy", 1600), atol=0.02)
