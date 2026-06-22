import random
import re
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from open_spiel.python.algorithms.external_sampling_mccfr import ExternalSamplingSolver
from open_spiel.python.algorithms import exploitability
from open_spiel.python import policy as policy_lib

from ..environment.openspiel_wrapper import PokerEnv
from ..environment.state_representation import StateEncoder
from ..models.model_utils import masked_logits
from .baseline_agents import PolicyAgent
from .stats import t95_multiplier


class Evaluator:
    def __init__(self, agent_model, config, device="cpu"):
        if hasattr(config, "sync_legacy_fields"):
            config.sync_legacy_fields()
        self.agent_model = agent_model
        self.config = config
        self.device = device
        try:
            #make sure on the right device 
            self.model_device = next(self.agent_model.parameters()).device
        except StopIteration:
            self.model_device = torch.device("cpu")
        self.env = PokerEnv(
            game_name=config.GAME_NAME,
            env_preset=config.ENV_PRESET,
            betting_abstraction=config.BETTING_ABSTRACTION,
            strict_abstraction=bool(getattr(config, "STRICT_ABSTRACTION", True)),
        )
        self.requested_betting_abstraction = self.env.get_requested_betting_abstraction()
        self.effective_betting_abstraction = self.env.get_effective_betting_abstraction()
        self.config.BETTING_ABSTRACTION = self.effective_betting_abstraction
        self.encoder = StateEncoder(device="cpu", max_action_history=config.MAX_ACTION_HISTORY)
        # for mccfr baselines caching to avoid having to rebuild
        self._baseline_cache: Dict[Tuple[str, int, int, Tuple[Tuple[str, str], ...]], Dict[str, object]] = {}

    class _ModelPolicyAdapter(policy_lib.Policy):
        # adapts to the open spiel policy interface by outputing action probs from the ops state
        # for nash evaluation
        def __init__(self, evaluator):
            super().__init__(evaluator.env.game, [0, 1])
            self.evaluator = evaluator

        def action_probabilities(self, state, player_id=None):
            if state.is_terminal() or state.is_chance_node():
                return {}
            if player_id is None:
                player_id = state.current_player()

            encoded = self.evaluator.encoder.encode_state(
                state,
                player_id,
                self.evaluator.env.num_actions(),
            )
            with torch.no_grad():
                batch = {k: v.unsqueeze(0).to(self.evaluator.model_device) for k, v in encoded.items()}
                outputs = self.evaluator.agent_model(batch)
                logits = masked_logits(outputs["policy_logits"], batch["legal_action_mask"])
                probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

            legal = state.legal_actions()
            probs_legal = np.array([probs[a] for a in legal], dtype=np.float64)
            if probs_legal.sum() <= 0:
                probs_legal = np.ones_like(probs_legal) / len(probs_legal)
            else:
                probs_legal /= probs_legal.sum()
            return {a: float(p) for a, p in zip(legal, probs_legal)}

    @torch.no_grad()
    def _sample_model_action(self, state, player_id):
        encoded = self.encoder.encode_state(state, player_id, self.env.num_actions())
        batch = {k: v.unsqueeze(0).to(self.model_device) for k, v in encoded.items()}

        outputs = self.agent_model(batch)
        logits = masked_logits(outputs["policy_logits"], batch["legal_action_mask"])
        dist = Categorical(logits=logits)
        return int(dist.sample().item())

    @staticmethod
    # seeed them tgther for consistancy and ease of use in slurm scripts
    def _seed_rngs(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _is_preflop(state) -> bool:
        try:
            board_blob = getattr(state.to_struct(), "board_cards", "")
            return len(board_blob) == 0
        except Exception:
            return True

    @staticmethod
    def _extract_action_amount(action_text: str):
        match = re.search(r"(?:RaiseTo|Raise|Bet)\s+(-?\d+(?:\.\d+)?)", str(action_text), re.IGNORECASE)
        if not match:
            return None
        try:
            return float(match.group(1))
        except Exception:
            return None

    @staticmethod
    def _classify_action_bucket(action: int, abstraction: str, action_text: str = "", pot_size: float = 0.0):
        abstraction = str(abstraction).lower()
        if abstraction == "fcpa":
            if action == 0:
                return "fold"
            if action == 1:
                return "call_check"
            if action == 2:
                return "pot_raise"
            if action == 3:
                return "allin"
            return "other"

        if abstraction in {"fcpha", "fchpa"}:
            if action == 0:
                return "fold"
            if action == 1:
                return "call_check"
            if action == 2:
                return "half_pot"
            if action == 3:
                return "pot_raise"
            if action == 4:
                return "allin"
            return "other"

        if abstraction == "fullgame":
            text = str(action_text)
            if re.search(r"\bmove=Fold\b", text, re.IGNORECASE):
                return "fold"
            if re.search(r"\bmove=(Call|Check)\b", text, re.IGNORECASE):
                return "call_check"
            if re.search(r"\bmove=AllIn\b", text, re.IGNORECASE):
                return "allin"
            if re.search(r"\bmove=(RaiseTo|Raise|Bet)\b", text, re.IGNORECASE):
                amount = Evaluator._extract_action_amount(text)
                if amount is not None and float(pot_size) > 0.0 and amount <= 0.75 * float(pot_size):
                    return "half_pot"
                return "pot_raise"
            return "other"

        return "other"

    @staticmethod
    def _preflop_entropy_bits_from_diag(diag: Dict[str, float]) -> float:
        total = float(max(1, int(diag.get("preflop_total", 0))))
        counts = [
            float(diag.get("fold", 0)),
            float(diag.get("call_check", 0)),
            float(diag.get("half_pot", 0)),
            float(diag.get("pot_raise", 0)),
            float(diag.get("allin", 0)),
        ]
        probs = [c / total for c in counts if c > 0.0]
        if not probs:
            return 0.0
        return float(-sum(p * np.log2(max(p, 1e-12)) for p in probs))

    @staticmethod
    def _preflop_dominant_action_freq_from_diag(diag: Dict[str, float]) -> float:
        total = float(max(1, int(diag.get("preflop_total", 0))))
        counts = [
            float(diag.get("fold", 0)),
            float(diag.get("call_check", 0)),
            float(diag.get("half_pot", 0)),
            float(diag.get("pot_raise", 0)),
            float(diag.get("allin", 0)),
        ]
        return float(max(counts) / total) if counts else 0.0

    @staticmethod
    def _preflop_legal_conditioned_metrics(diag: Dict[str, float]) -> Dict[str, float]:
        preflop_total = float(max(1, int(diag.get("preflop_total", 0))))
        legal_pot = float(max(0, int(diag.get("preflop_legal_pot_raise", 0))))
        legal_half = float(max(0, int(diag.get("preflop_legal_half_pot", 0))))
        chosen_pot_when_legal = float(max(0, int(diag.get("preflop_choose_pot_raise_when_legal", 0))))
        chosen_half_when_legal = float(max(0, int(diag.get("preflop_choose_half_pot_when_legal", 0))))

        pot_legal_rate = legal_pot / preflop_total
        half_legal_rate = legal_half / preflop_total
        choose_pot_given_legal = chosen_pot_when_legal / legal_pot if legal_pot > 0.0 else 0.0
        choose_half_given_legal = chosen_half_when_legal / legal_half if legal_half > 0.0 else 0.0

        return {
            "diagnostics/preflop_legal_rate/pot_raise": float(pot_legal_rate),
            "diagnostics/preflop_legal_rate/half_pot": float(half_legal_rate),
            "diagnostics/preflop_choose_given_legal/pot_raise": float(choose_pot_given_legal),
            "diagnostics/preflop_choose_given_legal/half_pot": float(choose_half_given_legal),
            "diagnostics/_preflop_legal_count_pot_raise": int(legal_pot),
            "diagnostics/_preflop_legal_count_half_pot": int(legal_half),
            "diagnostics/_preflop_choose_legal_count_pot_raise": int(chosen_pot_when_legal),
            "diagnostics/_preflop_choose_legal_count_half_pot": int(chosen_half_when_legal),
        }

    def evaluate(self, opponent, num_episodes=100, seed=None, collect_diagnostics: bool = False):
        if seed is not None:
            self._seed_rngs(int(seed))

        abstraction = str(getattr(self.config, "BETTING_ABSTRACTION", "fcpa")).lower()
        returns = []
        diag = {
            "fold": 0,
            "call_check": 0,
            "half_pot": 0,
            "pot_raise": 0,
            "allin": 0,
            "other": 0,
            "preflop_total": 0,
            "preflop_legal_pot_raise": 0,
            "preflop_legal_half_pot": 0,
            "preflop_choose_pot_raise_when_legal": 0,
            "preflop_choose_half_pot_when_legal": 0,
            "showdown_hands": 0,
            "total_hands": 0,
            "total_terminal_pot_bb": 0.0,
        }

        for episode_idx in range(num_episodes):
            state = self.env.reset()
            # alternate seats for fairness
            agent_player_id = 0 if episode_idx < (num_episodes / 2) else 1

            #chance nodes
            while not state.is_terminal():
                if state.is_chance_node():
                    outcomes = state.chance_outcomes()
                    action_list, probs = zip(*outcomes)
                    action = np.random.choice(action_list, p=probs)
                    state.apply_action(int(action))
                    continue

                current_player = state.current_player()
                if current_player == agent_player_id:
                    action = self._sample_model_action(state, current_player)
                    if collect_diagnostics and self._is_preflop(state):
                        try:
                            pot_size = float(getattr(state.to_struct(), "pot_size", 0.0))
                        except Exception:
                            pot_size = 0.0
                        try:
                            action_text = str(state.action_to_string(current_player, int(action)))
                        except Exception:
                            action_text = ""
                        bucket = self._classify_action_bucket(
                            int(action),
                            abstraction=abstraction,
                            action_text=action_text,
                            pot_size=pot_size,
                        )
                        legal_buckets: List[str] = []
                        for legal_action in state.legal_actions():
                            try:
                                legal_action_text = str(state.action_to_string(current_player, int(legal_action)))
                            except Exception:
                                legal_action_text = ""
                            legal_buckets.append(
                                self._classify_action_bucket(
                                    int(legal_action),
                                    abstraction=abstraction,
                                    action_text=legal_action_text,
                                    pot_size=pot_size,
                                )
                            )
                        #metrics
                        pot_legal = any(b == "pot_raise" for b in legal_buckets)
                        half_legal = any(b == "half_pot" for b in legal_buckets)
                        if pot_legal:
                            diag["preflop_legal_pot_raise"] += 1
                        if half_legal:
                            diag["preflop_legal_half_pot"] += 1
                        if pot_legal and bucket == "pot_raise":
                            diag["preflop_choose_pot_raise_when_legal"] += 1
                        if half_legal and bucket == "half_pot":
                            diag["preflop_choose_half_pot_when_legal"] += 1
                        diag[bucket] += 1
                        diag["preflop_total"] += 1
                else:
                    action = int(opponent.step(state))
                state.apply_action(action)

            returns.append(float(state.returns()[agent_player_id]))
            if collect_diagnostics:
                diag["total_hands"] += 1
                try:
                    struct = state.to_struct()
                    board_blob = getattr(struct, "board_cards", "")
                    if len(board_blob) >= 10:
                        diag["showdown_hands"] += 1
                    pot = float(getattr(struct, "pot_size", 0.0))
                except Exception:
                    pot = 0.0
                bb = max(float(getattr(self.config, "BB_SIZE", 100.0)), 1e-6)
                diag["total_terminal_pot_bb"] += pot / bb

        returns = np.array(returns, dtype=np.float64)
        avg_return = float(np.mean(returns))
        bb_per_100 = float((avg_return / max(self.config.BB_SIZE, 1.0)) * 100.0)
        std_err = float(np.std(returns) / np.sqrt(len(returns))) if len(returns) > 1 else 0.0
        metrics = {
            "avg_return": avg_return,
            "bb_per_100": bb_per_100,
            "std_err": std_err,
            "episodes": int(num_episodes),
            "meta/requested_betting_abstraction": str(self.requested_betting_abstraction),
            "meta/effective_betting_abstraction": str(self.effective_betting_abstraction),
            "meta/strict_abstraction": bool(getattr(self.config, "STRICT_ABSTRACTION", True)),
        }
        if collect_diagnostics:
            denom = max(1, int(diag["preflop_total"]))
            hand_denom = max(1, int(diag["total_hands"]))
            preflop_entropy_bits = self._preflop_entropy_bits_from_diag(diag)
            preflop_dominant_action_freq = self._preflop_dominant_action_freq_from_diag(diag)
            legal_metrics = self._preflop_legal_conditioned_metrics(diag)
            metrics.update(
                {
                    "diagnostics/action_freq_preflop/fold": float(diag["fold"]) / float(denom),
                    "diagnostics/action_freq_preflop/call_check": float(diag["call_check"]) / float(denom),
                    "diagnostics/action_freq_preflop/half_pot": float(diag["half_pot"]) / float(denom),
                    "diagnostics/action_freq_preflop/pot_raise": float(diag["pot_raise"]) / float(denom),
                    "diagnostics/action_freq_preflop/allin": float(diag["allin"]) / float(denom),
                    "diagnostics/preflop_action_entropy_bits": preflop_entropy_bits,
                    "diagnostics/preflop_dominant_action_freq": preflop_dominant_action_freq,
                    "diagnostics/showdown_rate": float(diag["showdown_hands"]) / float(hand_denom),
                    "diagnostics/avg_pot_size_bb": float(diag["total_terminal_pot_bb"]) / float(hand_denom),
                    "diagnostics/_preflop_count_fold": int(diag["fold"]),
                    "diagnostics/_preflop_count_call_check": int(diag["call_check"]),
                    "diagnostics/_preflop_count_half_pot": int(diag["half_pot"]),
                    "diagnostics/_preflop_count_pot_raise": int(diag["pot_raise"]),
                    "diagnostics/_preflop_count_allin": int(diag["allin"]),
                    "diagnostics/_preflop_total_count": int(diag["preflop_total"]),
                    "diagnostics/_showdown_hands": int(diag["showdown_hands"]),
                    "diagnostics/_total_hands": int(diag["total_hands"]),
                    "diagnostics/_total_terminal_pot_bb": float(diag["total_terminal_pot_bb"]),
                    **legal_metrics,
                }
            )
        return metrics

    def _game_signature(self) -> Tuple[Tuple[str, str], ...]:
        params = self.env.game_parameters()
        return tuple(sorted((str(k), str(v)) for k, v in params.items()))

    def _build_solver_baseline(self, baseline_algo: str, iterations: int, seed: int):
        algo = str(baseline_algo).lower()
        if algo != "mccfr_external_sampling":
            raise ValueError(f"Unsupported solver baseline algorithm: {baseline_algo}")
        if int(iterations) < 0:
            raise ValueError(f"Iterations must be >= 0, got {iterations}")

        self._seed_rngs(seed)

        build_start = time.perf_counter()
        solver = ExternalSamplingSolver(self.env.game)
        for _ in range(int(iterations)):
            solver.iteration()
        avg_policy = solver.average_policy()
        build_seconds = float(time.perf_counter() - build_start)

        label = getattr(self.config, "EVAL_BASELINE_LABEL", "MCCFR-ES")
        return {
            "agent": PolicyAgent(avg_policy),
            "metadata": {
                "baseline/algo": algo,
                "baseline/label": label,
                "baseline/iters": int(iterations),
                "baseline/seed": int(seed),
                "baseline/build_seconds": build_seconds,
            },
        }

    def evaluate_vs_solver_baseline(
        self,
        baseline_algo=None,
        iterations=None,
        num_episodes=100,
        seed=None,
        collect_diagnostics: bool = False,
    ):
        if hasattr(self.config, "sync_legacy_fields"):
            self.config.sync_legacy_fields()

        algo = baseline_algo or getattr(self.config, "EVAL_BASELINE_ALGO", "mccfr_external_sampling")
        iters = int(
            iterations
            if iterations is not None
            else getattr(self.config, "EVAL_BASELINE_ITERS", getattr(self.config, "EVAL_CFR_ITERATIONS", 5000))
        )
        eval_seed = int(seed if seed is not None else getattr(self.config, "SEED", 42))
        cache_mode = str(getattr(self.config, "EVAL_BASELINE_CACHE_MODE", "process")).lower()
        if cache_mode not in {"process", "none"}:
            cache_mode = "process"

        cache_key = (str(algo).lower(), iters, eval_seed, self._game_signature())
        cache_hit = False

        if cache_mode == "process" and cache_key in self._baseline_cache:
            payload = self._baseline_cache[cache_key]
            cache_hit = True
        else:
            payload = self._build_solver_baseline(algo, iters, eval_seed)
            if cache_mode == "process":
                self._baseline_cache[cache_key] = payload

        metrics = self.evaluate(
            payload["agent"],
            num_episodes=num_episodes,
            seed=eval_seed,
            collect_diagnostics=collect_diagnostics,
        )
        metrics.update(payload["metadata"])
        metrics["baseline/cache_hit"] = bool(cache_hit)
        metrics["baseline/cache_mode"] = cache_mode
        return metrics

    def evaluate_vs_cfr(self, iterations=None, num_episodes=100, seed=None, collect_diagnostics: bool = False):
        # Backwards-compatible wrapper: the baseline is now explicitly a solver-generated MCCFR policy.
        return self.evaluate_vs_solver_baseline(
            baseline_algo=getattr(self.config, "EVAL_BASELINE_ALGO", "mccfr_external_sampling"),
            iterations=iterations,
            num_episodes=num_episodes,
            seed=seed,
            collect_diagnostics=collect_diagnostics,
        )

    def build_solver_style_profile(
        self,
        iterations: int,
        seeds: int,
        episodes_per_seed: int,
        baseline_algo: str | None = None,
        seed_base: int | None = None,
    ) -> Dict[str, object]:
        """Build a solver-derived preflop style target from MCCFR policy self-play. Legacy, for curiousity"""

        algo = baseline_algo or getattr(self.config, "EVAL_BASELINE_ALGO", "mccfr_external_sampling")
        base_seed = int(seed_base if seed_base is not None else getattr(self.config, "SEED", 42))
        abstractions = str(getattr(self.config, "BETTING_ABSTRACTION", "fcpa")).lower()

        diag_total = {
            "fold": 0,
            "call_check": 0,
            "half_pot": 0,
            "pot_raise": 0,
            "allin": 0,
            "preflop_total": 0,
            "preflop_legal_pot_raise": 0,
            "preflop_legal_half_pot": 0,
            "preflop_choose_pot_raise_when_legal": 0,
            "preflop_choose_half_pot_when_legal": 0,
        }
        build_seconds: List[float] = []

        for offset in range(max(1, int(seeds))):
            eval_seed = base_seed + offset
            payload = self._build_solver_baseline(str(algo), int(iterations), int(eval_seed))
            build_seconds.append(float(payload["metadata"]["baseline/build_seconds"]))
            solver_agent = payload["agent"]
            self._seed_rngs(eval_seed)
            for _ in range(int(episodes_per_seed)):
                state = self.env.reset()
                while not state.is_terminal():
                    if state.is_chance_node():
                        outcomes = state.chance_outcomes()
                        action_list, probs = zip(*outcomes)
                        action = np.random.choice(action_list, p=probs)
                        state.apply_action(int(action))
                        continue

                    player = state.current_player()
                    pot_size = 0.0
                    if self._is_preflop(state):
                        try:
                            pot_size = float(getattr(state.to_struct(), "pot_size", 0.0))
                        except Exception:
                            pot_size = 0.0
                        legal_buckets = []
                        for legal_action in state.legal_actions():
                            try:
                                legal_text = str(state.action_to_string(player, int(legal_action)))
                            except Exception:
                                legal_text = ""
                            legal_buckets.append(
                                self._classify_action_bucket(
                                    int(legal_action),
                                    abstraction=abstractions,
                                    action_text=legal_text,
                                    pot_size=pot_size,
                                )
                            )
                        if any(b == "pot_raise" for b in legal_buckets):
                            diag_total["preflop_legal_pot_raise"] += 1
                        if any(b == "half_pot" for b in legal_buckets):
                            diag_total["preflop_legal_half_pot"] += 1

                    action = int(solver_agent.step(state))
                    if self._is_preflop(state):
                        try:
                            action_text = str(state.action_to_string(player, int(action)))
                        except Exception:
                            action_text = ""
                        bucket = self._classify_action_bucket(
                            int(action),
                            abstraction=abstractions,
                            action_text=action_text,
                            pot_size=pot_size,
                        )
                        if bucket in diag_total:
                            diag_total[bucket] += 1
                        diag_total["preflop_total"] += 1

                        if bucket == "pot_raise":
                            diag_total["preflop_choose_pot_raise_when_legal"] += 1
                        if bucket == "half_pot":
                            diag_total["preflop_choose_half_pot_when_legal"] += 1

                    state.apply_action(action)

        preflop_total = float(max(1, int(diag_total["preflop_total"])))
        target = {
            "fold": float(diag_total["fold"]) / preflop_total,
            "call_check": float(diag_total["call_check"]) / preflop_total,
            "half_pot": float(diag_total["half_pot"]) / preflop_total,
            "pot_raise": float(diag_total["pot_raise"]) / preflop_total,
            "allin": float(diag_total["allin"]) / preflop_total,
        }
        legal_metrics = self._preflop_legal_conditioned_metrics(diag_total)

        return {
            "baseline/algo": str(algo),
            "baseline/iters": int(iterations),
            "baseline/seeds": int(seeds),
            "baseline/episodes_per_seed": int(episodes_per_seed),
            "baseline/seed_base": int(base_seed),
            "baseline/build_seconds_mean": float(np.mean(build_seconds)) if build_seconds else 0.0,
            "target": target,
            "target/preflop_choose_given_legal/pot_raise": float(
                legal_metrics["diagnostics/preflop_choose_given_legal/pot_raise"]
            ),
            "target/preflop_choose_given_legal/half_pot": float(
                legal_metrics["diagnostics/preflop_choose_given_legal/half_pot"]
            ),
            "diagnostics/_preflop_total_count": int(diag_total["preflop_total"]),
            **legal_metrics,
        }

    @staticmethod
    def _estimate_preflop_hand_strength(player_hands) -> float:
        """Coarse preflop equity proxy from hole cards (no external deps).

        Pairs:     score = 0.6 + 0.4 * (rank / 12)  → range [0.60, 1.00]
        Non-pairs: score = (rank0 + rank1) / 24 + 0.05 if suited → range [0.0, ~1.0]

        Card indices follow the OpenSpiel convention: index = rank * 4 + suit
        where rank ∈ [0..12] (2=0 … A=12) and suit ∈ [0..3].
        """
        try:
            cards = list(player_hands)
            if len(cards) < 2:
                return 0.5
            c0, c1 = int(cards[0]), int(cards[1])
            if c0 < 0 or c0 > 51 or c1 < 0 or c1 > 51:
                return 0.5
            rank0, suit0 = divmod(c0, 4)
            rank1, suit1 = divmod(c1, 4)
            if rank0 == rank1:
                return 0.6 + 0.4 * (rank0 / 12.0)
            high_rank = max(rank0, rank1)
            low_rank = min(rank0, rank1)
            score = (high_rank + low_rank) / 24.0
            if suit0 == suit1:
                score += 0.05
            return min(1.0, max(0.0, score))
        except Exception:
            return 0.5

    @torch.no_grad()
    def evaluate_hand_strength_correlation(
        self,
        opponent,
        num_episodes: int = 2000,
        seed: int = 42,
        num_bins: int = 5,
    ) -> dict:
        """Bin preflop decisions by coarse hand strength and record action
        frequency distributions per bin. probe for some strategy."""
        self._seed_rngs(seed)
        abstraction = str(getattr(self.config, "BETTING_ABSTRACTION", "fcpa")).lower()

        bin_edges = [i / num_bins for i in range(num_bins + 1)]
        action_labels = ["fold", "call_check", "half_pot", "pot_raise", "allin"]
        bins = [{label: 0 for label in action_labels} for _ in range(num_bins)]
        bin_counts = [0] * num_bins

        for episode_idx in range(num_episodes):
            state = self.env.reset()
            agent_player_id = 0 if episode_idx < (num_episodes / 2) else 1

            while not state.is_terminal():
                if state.is_chance_node():
                    outcomes = state.chance_outcomes()
                    action_list, probs = zip(*outcomes)
                    action = np.random.choice(action_list, p=probs)
                    state.apply_action(int(action))
                    continue

                current_player = state.current_player()
                if current_player == agent_player_id:
                    if self._is_preflop(state):
                        try:
                            player_hands = state.to_struct().player_hands[agent_player_id]
                            strength = self._estimate_preflop_hand_strength(player_hands)
                        except Exception:
                            strength = 0.5

                        bin_idx = min(int(strength * num_bins), num_bins - 1)

                        action = self._sample_model_action(state, current_player)
                        try:
                            action_text = str(state.action_to_string(current_player, int(action)))
                        except Exception:
                            action_text = ""
                        try:
                            pot_size = float(getattr(state.to_struct(), "pot_size", 0.0))
                        except Exception:
                            pot_size = 0.0
                        bucket = self._classify_action_bucket(
                            int(action),
                            abstraction=abstraction,
                            action_text=action_text,
                            pot_size=pot_size,
                        )
                        if bucket in bins[bin_idx]:
                            bins[bin_idx][bucket] += 1
                        bin_counts[bin_idx] += 1
                    else:
                        action = self._sample_model_action(state, current_player)
                else:
                    action = int(opponent.step(state))
                state.apply_action(action)

        table = []
        for i in range(num_bins):
            total = max(1, bin_counts[i])
            row = {
                "equity_bin": f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}",
                "count": bin_counts[i],
            }
            for label in action_labels:
                row[f"freq_{label}"] = round(bins[i][label] / total, 4)
            table.append(row)

        return {
            "hand_strength_table": table,
            "num_episodes": num_episodes,
            "num_bins": num_bins,
            "seed": seed,
        }

    def evaluate_nash_conv(self):
        # NOTE: exploitability.nash_conv does an EXACT full game-tree traversal. It is
        # only tractable on small games (e.g. leduc_poker), NOT full-deck HUNL even
        # under the FCPA/FCHPA betting abstraction (the card chance nodes are
        # astronomically large). Use this for method validation on a solvable game;
        # for HUNL rely on duplicate-hand win-rate (evaluate_duplicate) instead.
        model_policy = self._ModelPolicyAdapter(self)
        nash_conv_value = exploitability.nash_conv(
            self.env.game,
            model_policy,
            return_only_nash_conv=True,
            use_cpp_br=False,
        )
        return float(nash_conv_value)

    def _play_one_hand(self, opponent, agent_player_id, seed):
        """Play one hand with the agent seated at ``agent_player_id`` under a fixed
        RNG seed; return the agent's terminal chip return."""
        self._seed_rngs(int(seed))
        state = self.env.reset()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                action_list, probs = zip(*outcomes)
                action = np.random.choice(action_list, p=probs)
                state.apply_action(int(action))
                continue
            current_player = state.current_player()
            if current_player == agent_player_id:
                action = self._sample_model_action(state, current_player)
            else:
                action = int(opponent.step(state))
            state.apply_action(int(action))
        return float(state.returns()[agent_player_id])

    def evaluate_duplicate(self, opponent, num_pairs=2500, seed=None):
        """Variance-reduced evaluation via duplicate (mirrored-seat) hands.

        Each "pair" plays the SAME deal twice under a common RNG seed with the
        agent's seat swapped: once in seat 0, once in seat 1. The hole-card chance
        nodes are drawn before any betting, so both plays deal identical hole cards
        to identical slots -- swapping the agent's seat makes it play BOTH holdings
        of the same deal against the same opponent. Averaging the agent's two
        returns cancels most of the card-luck variance (antithetic / duplicate
        poker), shrinking the bb/100 confidence interval by roughly an order of
        magnitude vs i.i.d. sampling. The per-pair averages are the i.i.d. sampling
        unit, so the CI is a Student-t interval over ``num_pairs`` values.

        Hole-card pairing is exact; board cards drawn after betting can differ once
        the two plays' action sequences diverge, so this removes most (not all)
        variance. Returns the same metric shape as ``evaluate`` plus paired-CI fields.
        """
        base_seed = int(seed) if seed is not None else int(getattr(self.config, "SEED", 42))
        bb = max(float(getattr(self.config, "BB_SIZE", 100.0)), 1.0)
        pair_returns = []  # agent chip return averaged over the two mirrored seats
        for pair_idx in range(int(num_pairs)):
            deal_seed = base_seed + pair_idx
            r0 = self._play_one_hand(opponent, agent_player_id=0, seed=deal_seed)
            r1 = self._play_one_hand(opponent, agent_player_id=1, seed=deal_seed)
            pair_returns.append(0.5 * (r0 + r1))

        arr = np.array(pair_returns, dtype=np.float64)
        n = int(arr.size)
        avg_return = float(arr.mean()) if n else 0.0
        bb_per_100 = float((avg_return / bb) * 100.0)
        pair_std = float(arr.std(ddof=1)) if n > 1 else 0.0
        pair_stderr = float(pair_std / np.sqrt(n)) if n > 1 else 0.0
        bb100_stderr = float((pair_stderr / bb) * 100.0)
        bb100_ci95 = float(t95_multiplier(n) * bb100_stderr)
        return {
            "avg_return": avg_return,
            "bb_per_100": bb_per_100,
            "std_err": pair_stderr,
            "bb_per_100_stderr": bb100_stderr,
            "bb_per_100_ci95": bb100_ci95,
            "bb_per_100_ci95_lower": bb_per_100 - bb100_ci95,
            "bb_per_100_ci95_upper": bb_per_100 + bb100_ci95,
            "pairs": n,
            "episodes": 2 * n,
            "method": "duplicate_mirrored_seat",
            "ci_method": "normal" if (n - 1) > 30 else "student_t",
            "meta/requested_betting_abstraction": str(self.requested_betting_abstraction),
            "meta/effective_betting_abstraction": str(self.effective_betting_abstraction),
            "meta/strict_abstraction": bool(getattr(self.config, "STRICT_ABSTRACTION", True)),
        }
