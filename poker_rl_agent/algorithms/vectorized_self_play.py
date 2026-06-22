"""Vectorized batched self-play collection (the S1 throughput fix).

The serial SelfPlayWorker (self_play.py) generates one episode at a time and runs ONE batch=1
neural inference per decision, so during collection the GPU sits at ~0% util (collection dominates
wall-clock). This collector runs B independent OpenSpiel games in LOCKSTEP: at each "tick" it gathers
every game currently awaiting a NEURAL decision, batches them into a single forward pass (grouped by
(model, seat) since opponents may be different league snapshots and use a different temperature),
scatters the sampled actions back, and steps the games. Chance nodes and opponent-AGENT decisions are
handled per-game on CPU. This turns batch=1 into batch≈B (30-100x collection throughput) with no new
dependencies and WITHOUT changing the on-policy trajectory distribution — each slot is still a real
self-play hand.

CONTRACT (semantic parity with self_play.generate_episode): collect() returns a list of
(transitions, final_return, train_player, opponent_entry) tuples, one per episode, IDENTICAL in
structure to calling generate_episode() target_episodes times. The trainer's downstream loop (buffer
add, preflop diagnostics, Elo update) therefore stays unchanged — only HOW transitions are produced
differs. A `sample_opponent_spec` closure (supplied by the trainer) keeps opponent resolution as the
single source of truth (same random/exploit/league mix as the serial path).
"""
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence

from ..environment.state_representation import StateEncoder
from ..models.model_utils import masked_logits


class _Slot:
    """One in-flight game: its state, seat assignment, opponent spec, and recorded transitions."""

    __slots__ = ("state", "train_player", "opp_entry", "opp_model", "opp_agent", "transitions", "active")

    def __init__(self):
        self.active = False
        self.state = None
        self.train_player = 0
        self.opp_entry = None
        self.opp_model = None
        self.opp_agent = None
        self.transitions: List[Dict] = []


class VectorizedSelfPlayCollector:
    def __init__(self, env, device="cpu", num_games: int = 256, max_action_history: int = 64,
                 curriculum=None):
        self.env = env
        self.device = device
        self.num_games = int(num_games)
        self.curriculum = curriculum
        # encoding stays on CPU (matches SelfPlayWorker); only the batched tensors hit the GPU.
        self.encoder = StateEncoder(device="cpu", max_action_history=max_action_history)
        self.num_actions = int(env.num_actions())

    # --- helpers -----------------------------------------------------------------
    def _encode(self, state, player, progress):
        encoded = self.encoder.encode_state(state, player, num_actions=self.num_actions)
        if self.curriculum is not None:
            masked = self.curriculum.mask_for_state(
                state=state, player_id=player, num_actions=self.num_actions,
                legal_action_mask=encoded["legal_action_mask"], progress=progress,
            )
            encoded = dict(encoded)
            encoded["legal_action_mask"] = masked
        return encoded

    def _collate(self, states: List[Dict[str, torch.Tensor]]):
        # mirror RolloutBuffer._collate_states so batched inference sees identical inputs.
        collated = {
            "hole_cards": torch.stack([s["hole_cards"] for s in states], dim=0),
            "community_cards": torch.stack([s["community_cards"] for s in states], dim=0),
            "scalars": torch.stack([s["scalars"] for s in states], dim=0),
            "legal_action_mask": torch.stack([s["legal_action_mask"] for s in states], dim=0),
        }
        collated["action_history"] = pad_sequence(
            [s["action_history"] for s in states], batch_first=True, padding_value=self.num_actions,
        )
        return {k: v.to(self.device) for k, v in collated.items()}

    def _advance(self, slot: _Slot) -> Tuple[str, Optional[int], bool]:
        """Step a slot through chance nodes + opponent-AGENT decisions until it is terminal or at a
        NEURAL decision. Returns ("terminal", None, False) or ("neural", player, is_train_player)."""
        state = slot.state
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                action_list, probs = zip(*outcomes)
                state.apply_action(int(np.random.choice(action_list, p=np.asarray(probs, dtype=np.float64))))
                continue
            cur = state.current_player()
            if cur == slot.train_player:
                return ("neural", cur, True)          # learner -> policy_model
            if slot.opp_agent is not None:
                state.apply_action(int(slot.opp_agent.step(state)))  # scripted opponent, CPU
                continue
            return ("neural", cur, False)             # model opponent (league snapshot or self)
        return ("terminal", None, False)

    # --- main --------------------------------------------------------------------
    @torch.no_grad()
    def collect(
        self,
        policy_model,
        sample_opponent_spec: Callable[[], Tuple[object, object, object]],
        target_episodes: int,
        bb_size: float = 100.0,
        progress: float = 1.0,
        policy_temperature: float = 1.0,
    ) -> List[Tuple[List[Dict], float, int]]:
        """sample_opponent_spec() -> (opponent_entry, opponent_model, opponent_agent); both model and
        agent None means self-play (opponent = policy_model). Returns list of (transitions,
        final_return, train_player, opponent_entry)."""
        target = max(1, int(target_episodes))
        B = max(1, min(self.num_games, target))
        results: List[Tuple[List[Dict], float, object]] = []

        slots = [_Slot() for _ in range(B)]
        started = 0

        def start_slot(slot: _Slot):
            nonlocal started
            entry, opp_model, opp_agent = sample_opponent_spec()
            slot.state = self.env.reset()
            slot.train_player = int(np.random.choice([0, 1]))
            slot.opp_entry = entry
            # self-play: opponent uses the live policy_model (matches generate_episode default)
            slot.opp_model = policy_model if (opp_model is None and opp_agent is None) else opp_model
            slot.opp_agent = opp_agent
            slot.transitions = []
            slot.active = True
            started += 1

        for slot in slots:
            if started < target:
                start_slot(slot)

        def finalize(slot: _Slot):
            final_return = float(slot.state.returns()[slot.train_player]) / max(bb_size, 1.0)
            if slot.transitions:
                slot.transitions[-1]["reward"] = final_return
                slot.transitions[-1]["done"] = 1.0
            results.append((slot.transitions, final_return, slot.train_player, slot.opp_entry))

        while len(results) < target:
            # 1) advance every active slot to a neural decision or terminal; harvest terminals.
            pending = []  # (slot, player, is_train, encoded_state)
            for slot in slots:
                if not slot.active:
                    continue
                status, player, is_train = self._advance(slot)
                if status == "terminal":
                    finalize(slot)
                    slot.active = False
                    if started < target:
                        start_slot(slot)
                        # the freshly started slot may itself immediately need a decision; re-advance
                        status2, player2, is_train2 = self._advance(slot)
                        if status2 == "terminal":           # degenerate instant-terminal hand
                            finalize(slot)
                            slot.active = False
                        else:
                            pending.append((slot, player2, is_train2,
                                            self._encode(slot.state, player2, progress)))
                    continue
                pending.append((slot, player, is_train, self._encode(slot.state, player, progress)))

            if not pending:
                break  # nothing left to start or decide

            # 2) group pending decisions by (model, seat) and run one batched forward per group.
            groups: Dict[Tuple[int, bool], List[int]] = {}
            for i, (slot, player, is_train, _enc) in enumerate(pending):
                model = policy_model if is_train else slot.opp_model
                groups.setdefault((id(model), is_train), []).append(i)

            for (_mid, is_train), idxs in groups.items():
                model = policy_model if is_train else pending[idxs[0]][0].opp_model
                temp = max(float(policy_temperature if is_train else 1.0), 1e-3)
                batch = self._collate([pending[i][3] for i in idxs])
                outputs = model(batch)
                logits = masked_logits(outputs["policy_logits"], batch["legal_action_mask"]) / temp
                dist = Categorical(logits=logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                values = outputs["state_value"].reshape(-1)
                for j, i in enumerate(idxs):
                    slot, player, _is_train, encoded = pending[i]
                    a = int(actions[j].item())
                    if is_train:
                        slot.transitions.append({
                            "state": encoded, "action": a,
                            "log_prob": float(log_probs[j].item()),
                            "value": float(values[j].item()),
                            "reward": 0.0, "done": 0.0,
                        })
                    slot.state.apply_action(a)

        return results
