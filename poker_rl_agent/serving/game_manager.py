import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from poker_rl_agent.environment.openspiel_wrapper import PokerEnv
from poker_rl_agent.environment.state_representation import StateEncoder
from poker_rl_agent.models.alpha_holdem_net import AlphaHoldemNetwork
from poker_rl_agent.models.model_utils import masked_logits

from .schemas import (
    ActionEvent,
    CardInfo,
    GameState,
    HandResult,
    LegalAction,
    SessionStats,
    cards_from_compact,
)

STREET_NAMES = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}


def _safe_action_name(state, player: int, action: int) -> str:
    try:
        return state.action_to_string(player, action)
    except Exception:
        return str(action)


def _temperature_scale_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    temp = max(float(temperature), 1e-3)
    return logits / temp


@dataclass
class SessionData:
    session_id: str
    env: PokerEnv
    state: object  # pyspiel.State — not serializable
    human_seat: int
    hand_number: int = 1
    hands_played: int = 0
    total_human_chips: float = 0.0
    total_bot_chips: float = 0.0
    last_accessed: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)
    events: List[ActionEvent] = field(default_factory=list)


class GameManager:
    def __init__(
        self,
        model: AlphaHoldemNetwork,
        encoder: StateEncoder,
        config,
        device: torch.device,
        num_actions: int,
        game_name: str,
        env_preset: str,
        betting_abstraction: str,
        strict_abstraction: bool,
        bot_policy: str = "sample",
        policy_temperature: float = 1.0,
        session_ttl: float = 1200.0,
        bb_size: float = 100.0,
        max_sessions: int = 64,
    ):
        self._model = model
        self._encoder = encoder
        self._config = config
        self._device = device
        self._num_actions = num_actions
        self._game_name = game_name
        self._env_preset = env_preset
        self._betting_abstraction = betting_abstraction
        self._strict_abstraction = strict_abstraction
        self._bot_policy = bot_policy
        self._policy_temperature = policy_temperature
        self._session_ttl = session_ttl
        self._bb_size = bb_size
        self._max_sessions = max_sessions

        self._sessions: Dict[str, SessionData] = {}
        self._sessions_lock = threading.Lock()

    @property
    def active_session_count(self) -> int:
        with self._sessions_lock:
            return len(self._sessions)

    @property
    def num_actions(self) -> int:
        return self._num_actions

    def _get_session(self, session_id: str) -> SessionData:
        with self._sessions_lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError("Session not found")
            session.last_accessed = time.monotonic()
        return session

    def create_session(self, human_seat: int = 0) -> Tuple[GameState, SessionStats]:
        if human_seat not in (0, 1):
            raise ValueError(f"human_seat must be 0 or 1, got {human_seat}")

        session_id = str(uuid.uuid4())
        env = PokerEnv(
            game_name=self._game_name,
            env_preset=self._env_preset,
            betting_abstraction=self._betting_abstraction,
            strict_abstraction=self._strict_abstraction,
        )
        state = env.reset()
        session = SessionData(
            session_id=session_id,
            env=env,
            state=state,
            human_seat=human_seat,
        )
        self._advance_to_human_or_terminal(session)

        with self._sessions_lock:
            if len(self._sessions) >= self._max_sessions:
                raise ValueError("Too many active sessions")
            self._sessions[session_id] = session

        return self._build_game_state(session), self._build_stats(session)

    def get_state(self, session_id: str) -> Tuple[GameState, SessionStats]:
        session = self._get_session(session_id)
        return self._build_game_state(session), self._build_stats(session)

    def apply_human_action(self, session_id: str, action_id: int) -> Tuple[GameState, SessionStats]:
        session = self._get_session(session_id)
        with session.lock:
            state = session.state
            if state.is_terminal():
                raise ValueError("Hand is already terminal")
            if state.is_chance_node():
                raise ValueError("Current node is a chance node, not human's turn")
            if state.current_player() != session.human_seat:
                raise ValueError("It is not the human's turn")
            legal = state.legal_actions()
            if action_id not in legal:
                raise ValueError(
                    f"Action {action_id} is not legal. Legal actions: {legal}"
                )

            action_name = _safe_action_name(state, session.human_seat, action_id)
            session.events.append(
                ActionEvent(actor="human", action_id=action_id, action_name=action_name)
            )
            state.apply_action(action_id)

            self._advance_to_human_or_terminal(session)

            if state.is_terminal():
                self._update_cumulative_stats(session)

        return self._build_game_state(session), self._build_stats(session)

    def new_hand(self, session_id: str) -> Tuple[GameState, SessionStats]:
        session = self._get_session(session_id)
        with session.lock:
            if not session.state.is_terminal():
                raise ValueError("Cannot start new hand: current hand is not terminal")
            session.state = session.env.reset()
            session.hand_number += 1
            session.events = []
            self._advance_to_human_or_terminal(session)
        return self._build_game_state(session), self._build_stats(session)

    def _advance_to_human_or_terminal(self, session: SessionData) -> None:
        state = session.state
        bot_seat = 1 - session.human_seat

        while not state.is_terminal():
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                action_list, probs = zip(*outcomes)
                sampled = int(np.random.choice(action_list, p=probs))
                state.apply_action(sampled)
                continue

            if state.current_player() == bot_seat:
                action = self._sample_bot_action(state, bot_seat)
                action_name = _safe_action_name(state, bot_seat, action)
                session.events.append(
                    ActionEvent(actor="bot", action_id=action, action_name=action_name)
                )
                state.apply_action(action)
                continue

            break  # human's turn

        if state.is_terminal():
            self._update_cumulative_stats(session)

    def _update_cumulative_stats(self, session: SessionData) -> None:
        if session.hands_played >= session.hand_number:
            return  # already counted
        returns = session.state.returns()
        human_return = float(returns[session.human_seat])
        bot_return = float(returns[1 - session.human_seat])
        session.total_human_chips += human_return
        session.total_bot_chips += bot_return
        session.hands_played = session.hand_number

    @torch.no_grad()
    def _sample_bot_action(self, state, bot_seat: int) -> int:
        encoded = self._encoder.encode_state(state, bot_seat, self._num_actions)
        batch = {k: v.unsqueeze(0) for k, v in encoded.items()}
        outputs = self._model(batch)
        logits = masked_logits(outputs["policy_logits"], batch["legal_action_mask"])
        logits = _temperature_scale_logits(logits, self._policy_temperature)
        if self._bot_policy == "argmax":
            return int(torch.argmax(logits, dim=-1).item())
        return int(Categorical(logits=logits).sample().item())

    def _build_game_state(self, session: SessionData) -> GameState:
        state = session.state
        is_terminal = state.is_terminal()
        struct = state.to_struct()

        # Cards
        hands = list(getattr(struct, "player_hands", []))
        hero_hand = hands[session.human_seat] if session.human_seat < len(hands) else ""
        hero_cards = cards_from_compact(hero_hand)

        bot_seat = 1 - session.human_seat
        if is_terminal:
            bot_hand = hands[bot_seat] if bot_seat < len(hands) else ""
            bot_cards = cards_from_compact(bot_hand)
        else:
            bot_cards = []

        board_blob = getattr(struct, "board_cards", "")
        board_cards = cards_from_compact(board_blob)

        # Pot / stacks / contributions
        contrib = list(getattr(struct, "player_contributions", [0.0, 0.0]))
        starts = list(getattr(struct, "starting_stacks", [20000.0, 20000.0]))
        pot_size = float(getattr(struct, "pot_size", 0.0))

        human_contrib = float(contrib[session.human_seat]) if session.human_seat < len(contrib) else 0.0
        bot_contrib = float(contrib[bot_seat]) if bot_seat < len(contrib) else 0.0
        human_start = float(starts[session.human_seat]) if session.human_seat < len(starts) else 20000.0
        bot_start = float(starts[bot_seat]) if bot_seat < len(starts) else 20000.0
        human_stack = max(0.0, human_start - human_contrib)
        bot_stack = max(0.0, bot_start - bot_contrib)

        # Street
        board_count = len(board_cards)
        street = STREET_NAMES.get(board_count, f"unknown({board_count})")

        # Current player label
        current_player: Optional[str] = None
        if not is_terminal and not state.is_chance_node():
            cp = state.current_player()
            if cp == session.human_seat:
                current_player = "human"
            else:
                current_player = "bot"

        # Legal actions
        legal_actions: List[LegalAction] = []
        if not is_terminal and not state.is_chance_node():
            for a in state.legal_actions():
                name = _safe_action_name(state, state.current_player(), a)
                legal_actions.append(LegalAction(action_id=a, name=name))

        # Hand result
        hand_result: Optional[HandResult] = None
        if is_terminal:
            returns = state.returns()
            human_chips = float(returns[session.human_seat])
            bot_chips = float(returns[bot_seat])
            hand_result = HandResult(
                human_chips=human_chips,
                bot_chips=bot_chips,
                human_bb=human_chips / self._bb_size if self._bb_size > 0 else 0.0,
                bot_bb=bot_chips / self._bb_size if self._bb_size > 0 else 0.0,
            )

        return GameState(
            session_id=session.session_id,
            hand_number=session.hand_number,
            is_terminal=is_terminal,
            street=street,
            current_player=current_player,
            human_seat=session.human_seat,
            hero_cards=hero_cards,
            bot_cards=bot_cards,
            board_cards=board_cards,
            pot_size=pot_size,
            human_stack=human_stack,
            bot_stack=bot_stack,
            human_contribution=human_contrib,
            bot_contribution=bot_contrib,
            legal_actions=legal_actions,
            events_since_last_action=list(session.events),
            hand_result=hand_result,
        )

    def _build_stats(self, session: SessionData) -> SessionStats:
        return SessionStats(
            hands_played=session.hands_played,
            total_human_chips=session.total_human_chips,
            total_bot_chips=session.total_bot_chips,
            total_human_bb=session.total_human_chips / self._bb_size if self._bb_size > 0 else 0.0,
            total_bot_bb=session.total_bot_chips / self._bb_size if self._bb_size > 0 else 0.0,
        )

    def cleanup_expired(self) -> int:
        now = time.monotonic()
        expired = []
        with self._sessions_lock:
            for sid, session in self._sessions.items():
                if now - session.last_accessed > self._session_ttl:
                    expired.append(sid)
            for sid in expired:
                del self._sessions[sid]
        return len(expired)
