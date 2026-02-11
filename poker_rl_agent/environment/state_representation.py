import torch
import pyspiel
import numpy as np

class StateEncoder:
    def __init__(self, device='cpu'):
        self.device = device
        
        # Mapping for cards to indices 0-51 (plus special tokens if needed)
        # Ranks: 2=0, ..., A=12
        # Suits: C=0, D=1, H=2, S=3
        self.rank_map = {r: i for i, r in enumerate("23456789TJQKA")}
        self.suit_map = {s: i for i, s in enumerate("cdhs")}

    def parse_card_string(self, card_str: str) -> int:
        """Converts 'As', 'Th' etc. to index 0-51."""
        if not card_str or card_str == "?": return 52 # Unknown/None
        rank = self.rank_map[card_str[0]]
        suit = self.suit_map[card_str[1]]
        return rank * 4 + suit

    def encode_state(self, state: pyspiel.State, player_id: int, num_actions: int = None):
        """
        Encodes the OpenSpiel state into tensors for the Pseudo-Siamese Network.
        
        Args:
            num_actions: Max valid action index for clamping/masking.
        """
        # 1. Parse Cards (Placeholder for robust parsing)
        # ...
        hole_cards = [52, 52] 
        community_cards = [52] * 5
        try:
             # Very basic string parsing for Leduc (Private: X, Public: Y)
             # Leduc string: "Round: 1 ... P0: J" or similar.
             # This is just a placeholder to prevent crash, actual card logic needed for perf.
             pass
        except:
            pass
            
        hole_cards_tensor = torch.tensor(hole_cards, dtype=torch.long, device=self.device)
        community_cards_tensor = torch.tensor(community_cards, dtype=torch.long, device=self.device)

        # 2. Action History (Action Tower)
        history = state.history()
        
        # Mask actions that are out of bounds (e.g. chance actions > num_actions)
        # If num_actions is provided, we use it to clip/mask.
        # ActionTower expects indices < num_actions.
        # We map everything else to num_actions (which is the padding/unknown idx).
        if num_actions is not None:
            safe_history = [a if a < num_actions else num_actions for a in history]
        else:
            safe_history = history
            
        action_seq = torch.tensor(safe_history, dtype=torch.long, device=self.device)


        # 3. Context/Scalars (Pot, Stack)
        # Extracted from state or observation tensor
        # Placeholder:
        pot = 0.0 # Extract
        stack = 0.0 # Extract
        
        # Normalization (Phase 1 Fixes)
        pot_norm = pot / 10000.0
        stack_norm = stack / 10000.0
        # If bet amount exists, normalize it relative to pot
        # bet_norm = bet / max(pot, 1.0) 
        
        scalars = torch.tensor([pot_norm, stack_norm], dtype=torch.float32, device=self.device)

        return {
            "hole_cards": hole_cards_tensor,
            "community_cards": community_cards_tensor,
            "action_history": action_seq,
            "scalars": scalars
        }
