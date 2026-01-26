import torch
import torch.nn as nn
from .card_tower import CardTower
from .action_tower import ActionTower
from .model_utils import init_weights

class AlphaHoldemNetwork(nn.Module):
    def __init__(self, num_actions, config):
        """
        Pseudo-Siamese Network Architecture.
        """
        super(AlphaHoldemNetwork, self).__init__()
        
        self.card_tower = CardTower(
            card_embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS_CARD
        )
        
        self.action_tower = ActionTower(
            num_actions=num_actions,
            action_embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS_ACTION
        )
        
        # Combined Representation
        combined_dim = self.card_tower.output_dim + self.action_tower.output_dim + 2 # +2 for Pot, Stack
        self.fc_layers = nn.Sequential(
            nn.Linear(combined_dim, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.ReLU()
        )
        
        # Output Head: Predicts Regret Values for each action
        # In CFR+, we often predict cumulative regrets or immediate counterfactual values.
        # Output dim = num_actions (one value per action)
        self.regret_head = nn.Linear(config.HIDDEN_DIM, num_actions)
        
        # Optional: Policy Head if we want to predict strategy directly
        # self.policy_head = nn.Linear(config.HIDDEN_DIM, num_actions)

        self.apply(init_weights)

    def forward(self, state_dict):
        """
        Args:
           state_dict: Dict from StateEncoder
             - hole_cards
             - community_cards
             - action_history
             - scalars
        """
        hole_cards = state_dict['hole_cards']
        community_cards = state_dict['community_cards']
        action_history = state_dict['action_history']
        scalars = state_dict['scalars']
        
        card_features = self.card_tower(hole_cards, community_cards)
        action_features = self.action_tower(action_history)
        
        combined = torch.cat([card_features, action_features, scalars], dim=1)
        x = self.fc_layers(combined)
        
        regrets = self.regret_head(x)
        return regrets
