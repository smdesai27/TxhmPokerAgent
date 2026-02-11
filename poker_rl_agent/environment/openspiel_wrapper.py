import pyspiel
from typing import Dict, Any, Optional


class PokerEnv:
    """Thin OpenSpiel wrapper with strict HUNL validation."""

    def __init__(
        self,
        game_name: str = "universal_poker",
        config: Optional[Dict[str, Any]] = None,
        env_preset: str = "hunl_fcpa",
        betting_abstraction: str = "fcpa",
    ):
        self.game_name = game_name
        self.env_preset = env_preset
        self.betting_abstraction = betting_abstraction
        self.game = self._load_game(config)
        self._validate_game_configuration()
        self.state = None

    def _load_game(self, config: Optional[Dict[str, Any]]):
        if self.game_name != "universal_poker":
            if config is None:
                return pyspiel.load_game(self.game_name)
            return pyspiel.load_game(self.game_name, config)

        if config is None:
            if self.env_preset == "hunl_fullgame" or self.betting_abstraction == "fullgame":
                game_string = pyspiel.hunl_game_string("fullgame")
            else:
                game_string = pyspiel.hunl_game_string("fcpa")
            return pyspiel.load_game(game_string)

        # Explicit parameterized construction when caller passes overrides.
        return pyspiel.load_game("universal_poker", config)

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

        expected_abstraction = "fullgame" if self.betting_abstraction == "fullgame" else "fcpa"
        abstraction = params.get("bettingAbstraction", "")
        if abstraction != expected_abstraction:
            failures.append(f"bettingAbstraction={abstraction} (expected {expected_abstraction})")

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
