import pyspiel
import numpy as np
from typing import List, Dict, Any, Tuple

class PokerEnv:
    def __init__(self, game_name="universal_poker", config: Dict[str, Any] = None):
        """
        Wraps OpenSpiel's Universal Poker for RL training.
        
        Args:
            game_name (str): Name of the game (default 'universal_poker').
            config (dict): Configuration dictionary for the game.
                           Defaults to standard HUNL-like settings if None.
        """
        if config is None:
            # Default to a simplified HUNL or Leduc if full HUNL is too heavy directly
            # For this task, we aim for HUNL. 
            # Note: Full HUNL string definition is complex in universal_poker.
            # We will use a standard approximation or rely on pyspiel's parameterized loading.
            # This config attempts to mimic standard HUNL:
            # 2 players, No Limit, 100/200 blinds (example), stack 20000.
            self.game_config = {
                "betting": "nolimit",
                "numPlayers": 2,
                "stack": "20000 20000",
                "blind": "50 100",
                "firstPlayer": "1",
                "numRounds": 4,  # Preflop, Flop, Turn, River
                "numBoardCards": "0 3 1 1",
                "maxRaises": "0",   # 0 usually means unlimited in some definitions, or we set high
            }
        else:
            self.game_config = config
            
        # Pyspiel `universal_poker` loading can be tricky without an ACPC file.
        # Often `load_game("universal_poker", ...)` works if arguments are supported.
        # Fallback to 'leduc_poker' if debugging, but main goal is HUNL.
        try:
            self.game = pyspiel.load_game(game_name, self.game_config)
        except Exception as e:
            print(f"Error loading {game_name} with config {self.game_config}: {e}")
            print("Falling back to 'leduc_poker' for stability if HUNL fails (Update this in production).")
            self.game = pyspiel.load_game("leduc_poker")

        self.state = None

    def reset(self):
        """Resets the game to a new state."""
        self.state = self.game.new_initial_state()
        return self.state

    def step(self, action: int):
        """Applies an action to the current state."""
        if self.state.is_chance_node():
            # Chance nodes should be handled by the environment/external loop usually,
            # but here we might just want to apply the action provided (if we control chance for seeding)
            # or usually in RL loop we expect the environment to handle chance.
            # OpenSpiel states handle chance sampling via apply_action usually if we know the outcome.
            # But normally we just call state.apply_action(action).
            pass
        
        self.state.apply_action(action)
        return self.state

    def get_legal_actions(self):
        """Returns list of legal action indices."""
        return self.state.legal_actions()

    def is_terminal(self):
        return self.state.is_terminal()
    
    def get_rewards(self):
        return self.state.returns()

    def get_observation_tensor(self):
        """Returns the standard OpenSpiel observation tensor (flat)."""
        return self.state.information_state_tensor()
    
    def get_state(self):
        return self.state

    def num_actions(self):
        return self.game.num_distinct_actions()

    def observation_tensor_shape(self):
        return self.game.information_state_tensor_shape()
