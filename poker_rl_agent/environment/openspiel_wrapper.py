import logging
from typing import Any, Dict, Optional, Tuple

import pyspiel


LOGGER = logging.getLogger(__name__)


class PokerEnv:
    """Thin OpenSpiel wrapper with strict HUNL validation."""

    def __init__(
        self,
        game_name: str = "universal_poker",
        config: Optional[Dict[str, Any]] = None,
        env_preset: str = "hunl_fcpa",
        betting_abstraction: str = "fcpa",
        strict_abstraction: bool = True,
    ):
        self.game_name = game_name
        self.env_preset = env_preset
        self.betting_abstraction = betting_abstraction
        self.strict_abstraction = bool(strict_abstraction)
        self.requested_betting_abstraction = self._resolve_betting_abstraction()
        self.effective_betting_abstraction: Optional[str] = None
        self.game = self._load_game(config)
        self._validate_game_configuration()
        self.state = None

    @staticmethod
    def _normalize_betting_abstraction(raw: str) -> str:
        abstraction = str(raw).lower().strip()
        if abstraction == "fcpha":
            return "fchpa"
        return abstraction

    def _resolve_betting_abstraction(self) -> str:
        abstraction = self._normalize_betting_abstraction(self.betting_abstraction)
        preset = str(self.env_preset).lower().strip()

        if abstraction == "fullgame" or preset == "hunl_fullgame":
            return "fullgame"
        if abstraction == "fchpa" or preset in {"hunl_fchpa", "hunl_fcpha"}:
            return "fchpa"
        if abstraction == "fcpa" or preset == "hunl_fcpa":
            return "fcpa"

        raise ValueError(
            f"Unsupported betting abstraction '{self.betting_abstraction}'. "
            "Supported values: fcpa, fchpa, fullgame (fcpha accepted as alias)."
        )

    def _load_game(self, config: Optional[Dict[str, Any]]):
        if self.game_name != "universal_poker":
            if config is None:
                return pyspiel.load_game(self.game_name)
            return pyspiel.load_game(self.game_name, config)

        if config is None:
            abstraction = self.requested_betting_abstraction
            game, effective = self._load_hunl_game(abstraction)
            self.effective_betting_abstraction = effective
            return game

        game = pyspiel.load_game("universal_poker", config)
        params = game.get_parameters()
        self.effective_betting_abstraction = self._normalize_betting_abstraction(
            str(params.get("bettingAbstraction", ""))
        )
        return game

    @staticmethod
    def _try_load_hunl_from_string(abstraction: str) -> Tuple[Optional[Any], Optional[str]]:
        try:
            game_string = pyspiel.hunl_game_string(abstraction)
            game = pyspiel.load_game(game_string)
            return game, None
        except Exception as exc:
            return None, str(exc)

    @staticmethod
    def _try_load_fchpa_from_params() -> Tuple[Optional[Any], Optional[str]]:
        try:
            base_game = pyspiel.load_game(pyspiel.hunl_game_string("fcpa"))
            params = dict(base_game.get_parameters())
            params["bettingAbstraction"] = "fchpa"
            game = pyspiel.load_game("universal_poker", params)
            return game, None
        except Exception as exc:
            return None, str(exc)

    def _load_hunl_game(self, abstraction: str) -> Tuple[Any, str]:
        errors = []

        game, error = self._try_load_hunl_from_string(abstraction)
        if game is not None:
            return game, abstraction
        errors.append(f"{abstraction}: {error}")

        # Some OpenSpiel builds support fchpa through universal_poker parameters
        # even when hunl_game_string helper support is inconsistent.
        if abstraction == "fchpa":
            game, error = self._try_load_fchpa_from_params()
            if game is not None:
                return game, "fchpa"
            errors.append(f"fchpa(param): {error}")

        if self.strict_abstraction:
            raise ValueError(
                "Requested betting abstraction is unsupported in this OpenSpiel build and "
                f"STRICT_ABSTRACTION=true. requested={abstraction}, errors={' | '.join(errors)}"
            )

        fallback = "fcpa"
        if abstraction != fallback:
            game, error = self._try_load_hunl_from_string(fallback)
            if game is not None:
                msg = (
                    f"Requested abstraction '{abstraction}' unavailable; "
                    f"falling back to '{fallback}' because STRICT_ABSTRACTION=false."
                )
                LOGGER.warning(msg)
                print(f"[PokerEnv] WARNING: {msg}", flush=True)
                return game, fallback
            errors.append(f"{fallback}: {error}")

        raise ValueError(
            f"Failed to initialize HUNL game for abstraction '{abstraction}'. "
            f"Details: {' | '.join(errors)}"
        )

    def _validate_game_configuration(self):
        if self.game_name != "universal_poker":
            return

        params = self.game.get_parameters()
        failures = []

        if int(params.get("numHoleCards", 0)) != 2:
            failures.append(f"numHoleCards={params.get('numHoleCards')}")
        if int(params.get("numRanks", 0)) != 13:
            failures.append(f"numRanks={params.get('numRanks')}")
        if int(params.get("numSuits", 0)) != 4:
            failures.append(f"numSuits={params.get('numSuits')}")

        expected_abstraction = self.get_effective_betting_abstraction()
        abstraction = self._normalize_betting_abstraction(str(params.get("bettingAbstraction", "")))
        if abstraction != expected_abstraction:
            failures.append(f"bettingAbstraction={abstraction} (expected {expected_abstraction})")

        if self.strict_abstraction and abstraction != self.requested_betting_abstraction:
            failures.append(
                f"strict_abstraction_mismatch={abstraction} "
                f"(requested {self.requested_betting_abstraction})"
            )

        if failures:
            details = ", ".join(failures)
            raise ValueError(
                "Invalid universal_poker configuration for HUNL training: " + details
            )

    def reset(self):
        self.state = self.game.new_initial_state()
        return self.state

    def step(self, action: int):
        self.state.apply_action(action)
        return self.state

    def get_legal_actions(self):
        return self.state.legal_actions()

    def is_terminal(self):
        return self.state.is_terminal()

    def get_rewards(self):
        return self.state.returns()

    def get_observation_tensor(self):
        current_player = self.state.current_player()
        if current_player < 0:
            return []
        return self.state.information_state_tensor(current_player)

    def get_state(self):
        return self.state

    def num_actions(self):
        return self.game.num_distinct_actions()

    def observation_tensor_shape(self):
        return self.game.information_state_tensor_shape()

    def game_parameters(self):
        return self.game.get_parameters()

    def get_effective_betting_abstraction(self) -> str:
        if self.effective_betting_abstraction:
            return self.effective_betting_abstraction
        return self.requested_betting_abstraction

    def get_requested_betting_abstraction(self) -> str:
        return self.requested_betting_abstraction
