"""Local Best Response (LBR) opponent -- a tractable LOWER BOUND on exploitability.

WHAT THIS IS (and is NOT)
-------------------------
LBR (Lisy & Bowling 2017, "Equilibrium Approximation Quality of Current No-Limit
Poker Bots", arXiv:1612.07547) is a *cheap, greedy* best-response opponent: at each of
its own decision points it looks one (or a few) steps ahead, rolls the OPPONENT'S OWN
policy forward to estimate the value of each candidate action, and plays the argmax.
Its average winnings against an agent are a guaranteed LOWER BOUND on that agent's
true exploitability -- because LBR is one specific (suboptimal, depth-limited) strategy
in the agent's opponent's strategy space, the *best* response can only do at least as
well. Formally:  LBR_value <= exploitability  (in the same per-player units).

HONESTY CONSTRAINTS -- read before quoting any number:
  * LBR is a LOWER BOUND on exploitability, NEVER "the exploitability". Report it as
    "LBR win-rate >= X", never "the agent is exploitable for exactly X".
  * A LOW LBR value does NOT prove the agent is unexploitable. Lisy & Bowling showed
    bots that were ~tied head-to-head differed by ~1300 mbb/g in true exploitability;
    LBR with a weak/short lookahead can return ~0 against a genuinely exploitable agent
    (it simply failed to find the leak). Low LBR == "this particular probe found no
    leak", not "no leak exists".
  * A HIGH LBR value is a DEFINITIVE DISQUALIFIER: if a one-ply greedy responder already
    wins by a lot, the agent is provably far from equilibrium. This is the value of LBR
    as a screen -- cheap to disprove a bad agent, never sufficient to certify a good one.
  * On full HUNL, LBR is ONLY EVER a lower bound (exact exploitability is intractable).
    This implementation is VALIDATED ON LEDUC FIRST against exact NashConv (see
    poker_rl_agent/scripts/validate_lbr.py): the gate is LBR_value <= NashConv.

HOW THE SEARCH WORKS
--------------------
At an LBR decision node, for each legal action ``a`` we:
  1. CLONE the state (``state.clone()`` if pyspiel exposes it, else replay the action
     history from ``game.new_initial_state()`` -- see ``_clone_state`` below),
  2. apply ``a`` on the clone,
  3. estimate the value of the resulting node to the LBR player by rolling the AGENT'S
     policy (and LBR's own greedy policy at deeper LBR nodes, up to ``max_depth``)
     forward to a terminal or the depth limit, enumerating chance nodes when cheap and
     Monte-Carlo sampling them otherwise (``rollout_samples`` rollouts averaged),
  4. take ``argmax_a`` of the estimated value (LBR plays deterministically).

``max_depth`` counts LBR's OWN remaining lookahead plies. ``max_depth=1`` (the default)
means: try each action once, then let the agent's policy play out the rest -- the
classic cheap LBR. The FCHPA tree branches 5-ways per betting decision, so the cost of
exact lookahead is ~5^depth; keep ``max_depth`` small (0/1) and lean on the MC rollout
plus a node budget to stay tractable.

NODE / TIME BUDGET
------------------
``max_nodes`` bounds the number of state expansions inside a single ``step`` call
(across all candidate actions and rollouts). If exhausted, the search stops expanding
deeper and falls back to the value estimates gathered so far. This guarantees ``step``
returns in bounded time even on the HUNL tree; on Leduc the tree is tiny so the budget
is effectively never binding (and the bound holds exactly).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from .baseline_agents import Agent


def state_clone_available(game) -> bool:
    """Probe whether pyspiel states for ``game`` support ``.clone()`` without mutation.

    Verified at runtime (the repo does not use ``state.clone()`` anywhere, so we cannot
    assume it exists). Returns True iff a fresh initial state exposes a callable
    ``clone`` AND cloning + the clone existing does not raise. Callers use this to pick
    the fast clone path vs. the history-replay fallback.
    """
    try:
        state = game.new_initial_state()
        clone = getattr(state, "clone", None)
        if not callable(clone):
            return False
        cloned = clone()
        # Touch the clone to confirm it is a usable, independent State.
        _ = cloned.history()
        return True
    except Exception:
        return False


class LocalBestResponseAgent(Agent):
    """LBR opponent implementing the project ``Agent.step(state) -> action`` interface.

    Parameters
    ----------
    policy_adapter:
        An object exposing ``action_probabilities(state) -> {action: prob}`` for the
        AGENT being probed (e.g. ``Evaluator._ModelPolicyAdapter`` on HUNL, or a
        ``NetPolicy``-style adapter on Leduc). LBR rolls THIS policy forward to value
        candidate actions. We only call ``action_probabilities``; we do not assume any
        particular encoder, so the same LBR works on universal_poker and leduc_poker.
    lbr_player:
        Seat index (0 or 1) that LBR controls. Returns are read off
        ``state.returns()[lbr_player]``.
    max_depth:
        Number of LBR's OWN lookahead plies. 0 -> pure rollout from the current node
        (no action search; degenerate, used only for the depth=0 self-play oracle).
        1 -> classic one-ply LBR (try each action, then roll the agent out). Cost grows
        ~5^max_depth on FCHPA; keep small.
    rollout_samples:
        Monte-Carlo rollouts averaged per evaluated node when chance is sampled rather
        than enumerated. More samples -> tighter value estimate, more nodes spent.
    rng:
        ``numpy.random.RandomState`` (or compatible) for chance / policy sampling.
        Passed in so the validation harness controls the seed.
    enumerate_chance_threshold:
        If a chance node has <= this many outcomes, enumerate them exactly (weighting by
        their probabilities) instead of sampling. Cheap exact chance handling on small
        games (Leduc's deck) tightens the bound; large chance nodes (HUNL deals) are
        sampled.
    max_nodes:
        Per-``step`` node-expansion budget (see module docstring). ``None`` -> unbounded
        (safe on Leduc; do NOT use unbounded on HUNL).
    """

    def __init__(
        self,
        policy_adapter,
        lbr_player: int,
        max_depth: int = 1,
        rollout_samples: int = 1,
        rng: Optional[np.random.RandomState] = None,
        enumerate_chance_threshold: int = 8,
        max_nodes: Optional[int] = 200_000,
    ):
        if max_depth < 0:
            raise ValueError(f"max_depth must be >= 0, got {max_depth}")
        if rollout_samples < 1:
            raise ValueError(f"rollout_samples must be >= 1, got {rollout_samples}")
        self.policy_adapter = policy_adapter
        self.lbr_player = int(lbr_player)
        self.max_depth = int(max_depth)
        self.rollout_samples = int(rollout_samples)
        self.rng = rng if rng is not None else np.random.RandomState()
        self.enumerate_chance_threshold = int(enumerate_chance_threshold)
        self.max_nodes = max_nodes
        self._nodes_expanded = 0
        # Resolved lazily on the first step (we need a state to find the game handle).
        self._clone_supported: Optional[bool] = None

    # ------------------------------------------------------------------ cloning

    def _ensure_clone_probe(self, state) -> None:
        if self._clone_supported is None:
            try:
                game = state.get_game()
                self._clone_supported = state_clone_available(game)
            except Exception:
                self._clone_supported = False

    def _clone_state(self, state):
        """Return an independent copy of ``state`` that is safe to mutate.

        Fast path: ``state.clone()`` (pyspiel States expose this in builds where it is
        compiled in). Fallback: replay the full action ``history()`` from
        ``game.new_initial_state()`` -- correct for both betting and chance actions
        because ``history()`` records every applied action index in order. The fallback
        is the contract the repo's own ``StateEncoder._extract_decision_history`` relies
        on, so it is known to work on these games.
        """
        self._ensure_clone_probe(state)
        if self._clone_supported:
            try:
                return state.clone()
            except Exception:
                self._clone_supported = False  # demote permanently and fall through
        game = state.get_game()
        replay = game.new_initial_state()
        for action in state.history():
            replay.apply_action(int(action))
        return replay

    def _budget_left(self) -> bool:
        if self.max_nodes is None:
            return True
        return self._nodes_expanded < self.max_nodes

    # ------------------------------------------------------------------ chance

    def _advance_chance(self, state) -> None:
        """Resolve a chance node in place: enumerate-weighted-sample if small, else MC."""
        outcomes = state.chance_outcomes()
        actions = [int(a) for a, _ in outcomes]
        probs = np.array([float(p) for _, p in outcomes], dtype=np.float64)
        probs = probs / probs.sum()
        idx = self.rng.choice(len(actions), p=probs)
        state.apply_action(actions[int(idx)])

    # ------------------------------------------------------------------ rollout

    def _agent_action(self, state) -> int:
        """Sample the AGENT'S action at ``state`` from its policy adapter."""
        probs = self.policy_adapter.action_probabilities(state)
        if not probs:
            legal = state.legal_actions()
            return int(self.rng.choice(legal))
        actions = list(probs.keys())
        p = np.array([probs[a] for a in actions], dtype=np.float64)
        total = p.sum()
        if total <= 0:
            p = np.ones_like(p) / len(p)
        else:
            p = p / total
        return int(self.rng.choice(actions, p=p))

    def _rollout_value(self, state, lbr_depth: int) -> float:
        """Estimate LBR's expected return from ``state`` by playing it out.

        At LBR's own nodes within the remaining ``lbr_depth`` budget, recurse with the
        one-ply best-response search (``_best_action_value``). Once the LBR budget is
        exhausted, LBR also just follows the agent rollout convention -- it samples the
        agent's policy for BOTH players past the lookahead horizon, which is the standard
        LBR approximation (the responder assumes the opponent's average policy fills in
        the tail). Chance is enumerated when small, sampled otherwise.
        """
        node = state
        while True:
            if node.is_terminal():
                return float(node.returns()[self.lbr_player])
            if node.is_chance_node():
                self._nodes_expanded += 1
                self._advance_chance(node)
                continue
            current = node.current_player()
            if current == self.lbr_player and lbr_depth > 0 and self._budget_left():
                # Recurse into LBR's own greedy lookahead one more ply.
                _, value = self._best_action_value(node, lbr_depth)
                return value
            # Past the LBR horizon (or out of budget): everyone plays the agent policy.
            self._nodes_expanded += 1
            action = self._agent_action(node)
            node.apply_action(action)

    def _value_of_node(self, state, lbr_depth: int) -> float:
        """Average value of ``state`` over chance handling and ``rollout_samples`` MC rollouts.

        If the node is a small chance node, enumerate its outcomes exactly and recurse
        (probability-weighted) -- this makes the estimate exact on Leduc-sized chance.
        Otherwise run ``rollout_samples`` Monte-Carlo rollouts and average.
        """
        if state.is_terminal():
            return float(state.returns()[self.lbr_player])

        if state.is_chance_node():
            outcomes = state.chance_outcomes()
            if len(outcomes) <= self.enumerate_chance_threshold and self._budget_left():
                value = 0.0
                for action, prob in outcomes:
                    self._nodes_expanded += 1
                    child = self._clone_state(state)
                    child.apply_action(int(action))
                    value += float(prob) * self._value_of_node(child, lbr_depth)
                return value
            # Large chance node: fall through to sampled rollouts.

        samples = max(1, self.rollout_samples)
        total = 0.0
        for _ in range(samples):
            rollout_state = self._clone_state(state)
            total += self._rollout_value(rollout_state, lbr_depth)
        return total / float(samples)

    # ------------------------------------------------------------------ search

    def _best_action_value(self, state, lbr_depth: int):
        """Return ``(best_action, best_value)`` for LBR at ``state`` with ``lbr_depth`` plies.

        For each legal action: clone, apply, then value the resulting node with one fewer
        LBR ply. argmax over LBR's own return. Deterministic tie-break on the first
        argmax (np.argmax) for reproducibility.
        """
        legal = state.legal_actions()
        if not legal:
            return None, float(state.returns()[self.lbr_player])

        values: List[float] = []
        for action in legal:
            if not self._budget_left():
                # Out of budget: assign the rollout value of the current node so the
                # remaining actions are scored consistently (no deeper expansion).
                values.append(self._rollout_value(self._clone_state(state), 0))
                continue
            self._nodes_expanded += 1
            child = self._clone_state(state)
            child.apply_action(int(action))
            values.append(self._value_of_node(child, lbr_depth - 1))

        best_idx = int(np.argmax(np.array(values, dtype=np.float64)))
        return int(legal[best_idx]), float(values[best_idx])

    # ------------------------------------------------------------------ Agent API

    def step(self, state) -> int:
        """Pick LBR's action at ``state`` (the project ``Agent`` interface)."""
        self._nodes_expanded = 0
        legal = state.legal_actions()
        if not legal:
            raise ValueError("LocalBestResponseAgent.step called at a node with no legal actions")
        if len(legal) == 1:
            return int(legal[0])

        if self.max_depth <= 0:
            # depth=0 oracle: no action search -- play the agent's OWN policy at this
            # node. Against a self-play opponent this reproduces the agent's self-play
            # value (used by the monotonicity oracle: depth=1 >= depth=0).
            return self._agent_action(state)

        best_action, _ = self._best_action_value(state, self.max_depth)
        if best_action is None:
            return int(legal[0])
        return int(best_action)
