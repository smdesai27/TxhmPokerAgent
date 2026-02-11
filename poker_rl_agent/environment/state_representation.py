import torch
import pyspiel
from typing import List


class StateEncoder:
    """Encodes OpenSpiel universal poker states for the pseudo-siamese model."""

    UNKNOWN_CARD = 52

    def __init__(self, device: str = "cpu", max_action_history: int = 64):
        self.device = device
        self.max_action_history = max_action_history
        self.rank_map = {r: i for i, r in enumerate("23456789TJQKA")}
        self.suit_map = {s: i for i, s in enumerate("cdhs")}

    def parse_card_string(self, card_str: str) -> int:
        if not card_str or card_str == "?":
            return self.UNKNOWN_CARD

        rank = card_str[0].upper()
        suit = card_str[1].lower()
        if rank not in self.rank_map or suit not in self.suit_map:
            return self.UNKNOWN_CARD

        return self.rank_map[rank] * 4 + self.suit_map[suit]

    @staticmethod
    def _cards_from_compact(card_blob: str) -> List[str]:
        if not card_blob:
            return []
        return [card_blob[i : i + 2] for i in range(0, len(card_blob), 2)]

    def _extract_decision_history(self, state: pyspiel.State, num_actions: int) -> List[int]:
        history = state.history()
        if not history:
            return [num_actions]

        try:
            replay_state = state.get_game().new_initial_state()
        except Exception:
            # Conservative fallback if game handle is unavailable.
            filtered = [a for a in history if 0 <= a < num_actions]
            return filtered[-self.max_action_history :] if filtered else [num_actions]

        decisions = []
        for action in history:
            if not replay_state.is_chance_node() and replay_state.current_player() >= 0:
                decisions.append(action if 0 <= action < num_actions else num_actions)
            replay_state.apply_action(action)

        if not decisions:
            return [num_actions]

        return decisions[-self.max_action_history :]

    def encode_state(self, state: pyspiel.State, player_id: int, num_actions: int):
        # OpenSpiel universal_poker state struct carries stable parseable fields.
        struct = state.to_struct()

        hands = getattr(struct, "player_hands", [])
        hero_hand = hands[player_id] if player_id < len(hands) else ""
        hero_cards = self._cards_from_compact(hero_hand)

        hole_cards = [self.UNKNOWN_CARD, self.UNKNOWN_CARD]
        for i, card in enumerate(hero_cards[:2]):
            hole_cards[i] = self.parse_card_string(card)

        board_blob = getattr(struct, "board_cards", "")
        board_cards = self._cards_from_compact(board_blob)
        community = [self.UNKNOWN_CARD] * 5
        for i, card in enumerate(board_cards[:5]):
            community[i] = self.parse_card_string(card)

        hole_cards_tensor = torch.tensor(hole_cards, dtype=torch.long, device=self.device)
        community_cards_tensor = torch.tensor(community, dtype=torch.long, device=self.device)

        action_seq = self._extract_decision_history(state, num_actions)
        action_seq_tensor = torch.tensor(action_seq, dtype=torch.long, device=self.device)

        legal_mask = torch.zeros(num_actions, dtype=torch.float32, device=self.device)
        if not state.is_terminal() and not state.is_chance_node():
            for action in state.legal_actions():
                if 0 <= action < num_actions:
                    legal_mask[action] = 1.0

        starting_stacks = getattr(struct, "starting_stacks", [20000, 20000])
        contributions = getattr(struct, "player_contributions", [0, 0])
        pot_size = float(getattr(struct, "pot_size", sum(contributions)))

        hero_contrib = float(contributions[player_id]) if player_id < len(contributions) else 0.0
        opp_id = 1 - player_id
        opp_contrib = float(contributions[opp_id]) if opp_id < len(contributions) else 0.0

        hero_start = float(starting_stacks[player_id]) if player_id < len(starting_stacks) else 20000.0
        opp_start = float(starting_stacks[opp_id]) if opp_id < len(starting_stacks) else 20000.0

        hero_stack = max(0.0, hero_start - hero_contrib)
        opp_stack = max(0.0, opp_start - opp_contrib)
        stack_norm = max(hero_start, opp_start, 1.0)

        board_count = len(board_cards)
        # 0=preflop, 3=flop, 4=turn, 5=river
        street_idx = 0
        if board_count >= 5:
            street_idx = 3
        elif board_count == 4:
            street_idx = 2
        elif board_count >= 3:
            street_idx = 1

        street_one_hot = [0.0, 0.0, 0.0, 0.0]
        street_one_hot[street_idx] = 1.0

        current_player = state.current_player()
        acting_flag = 1.0 if current_player == player_id else 0.0
        seat_flag = float(player_id)

        scalars = [
            pot_size / stack_norm,
            hero_stack / stack_norm,
            opp_stack / stack_norm,
            hero_contrib / stack_norm,
            opp_contrib / stack_norm,
            *street_one_hot,
            seat_flag,
            acting_flag,
        ]

        scalars_tensor = torch.tensor(scalars, dtype=torch.float32, device=self.device)

        return {
            "hole_cards": hole_cards_tensor,
            "community_cards": community_cards_tensor,
            "action_history": action_seq_tensor,
            "scalars": scalars_tensor,
            "legal_action_mask": legal_mask,
        }
