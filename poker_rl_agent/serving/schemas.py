from typing import List, Optional

from pydantic import BaseModel, Field

SUIT_UNICODE = {"h": "\u2665", "d": "\u2666", "c": "\u2663", "s": "\u2660"}


def card_to_display(raw: str) -> str:
    if len(raw) != 2:
        return raw
    rank = raw[0].upper()
    suit_char = raw[1].lower()
    symbol = SUIT_UNICODE.get(suit_char, suit_char)
    return f"{rank}{symbol}"


def cards_from_compact(blob: str) -> List["CardInfo"]:
    if not blob:
        return []
    raw_cards = [blob[i : i + 2] for i in range(0, len(blob), 2)]
    return [CardInfo(raw=c, display=card_to_display(c)) for c in raw_cards]


class CardInfo(BaseModel):
    raw: str
    display: str


class LegalAction(BaseModel):
    action_id: int
    name: str


class ActionEvent(BaseModel):
    actor: str
    action_id: int
    action_name: str


class HandResult(BaseModel):
    human_chips: float
    bot_chips: float
    human_bb: float
    bot_bb: float


class GameState(BaseModel):
    session_id: str
    hand_number: int
    is_terminal: bool
    street: str
    current_player: Optional[str] = None
    human_seat: int
    hero_cards: List[CardInfo] = Field(default_factory=list)
    bot_cards: List[CardInfo] = Field(default_factory=list)
    board_cards: List[CardInfo] = Field(default_factory=list)
    pot_size: float = 0.0
    human_stack: float = 0.0
    bot_stack: float = 0.0
    human_contribution: float = 0.0
    bot_contribution: float = 0.0
    legal_actions: List[LegalAction] = Field(default_factory=list)
    events_since_last_action: List[ActionEvent] = Field(default_factory=list)
    hand_result: Optional[HandResult] = None


class SessionStats(BaseModel):
    hands_played: int = 0
    total_human_chips: float = 0.0
    total_bot_chips: float = 0.0
    total_human_bb: float = 0.0
    total_bot_bb: float = 0.0


class CreateSessionRequest(BaseModel):
    human_seat: int = 0


class CreateSessionResponse(BaseModel):
    game_state: GameState
    stats: SessionStats


class ActionRequest(BaseModel):
    action_id: int


class ActionResponse(BaseModel):
    game_state: GameState
    stats: SessionStats


class NewHandResponse(BaseModel):
    game_state: GameState
    stats: SessionStats


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    active_sessions: int
