import torch
import torch.nn as nn
from .card_tower import CardTower
from .action_tower import ActionTower
from .model_utils import init_weights

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.fc(x))

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
        
        # Main MLP
        layers = []
        # Projection to hidden dim first
        layers.append(nn.Linear(combined_dim, config.HIDDEN_DIM))
        if config.USE_LAYERNORM:
            layers.append(nn.LayerNorm(config.HIDDEN_DIM))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(config.DROPOUT))
        
        # Residual Blocks or Standard Layers
        if config.USE_RESIDUAL:
            # Force max 2 residual blocks for stability
            layers.append(ResidualBlock(config.HIDDEN_DIM, config.DROPOUT))
            layers.append(ResidualBlock(config.HIDDEN_DIM, config.DROPOUT))
        else:
            # Standard MLP - reduce depth if needed
            layers.append(nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM))
            layers.append(nn.LayerNorm(config.HIDDEN_DIM))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.DROPOUT))
            
            # Explicit second layer for depth
            layers.append(nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM))
            layers.append(nn.LayerNorm(config.HIDDEN_DIM))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.DROPOUT))
            
        self.fc_layers = nn.Sequential(*layers)
        
        # Output Head: Predicts Regret Values for each action
        self.regret_head = nn.Linear(config.HIDDEN_DIM, num_actions)
        
        # Initialization
        # Initialization
        self.apply(self._init_weights)

        # Custom init for regret head to start near 0
        nn.init.uniform_(self.regret_head.weight, -0.01, 0.01)
        nn.init.constant_(self.regret_head.bias, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Xavier/He initialization with small gain for stability
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, state_dict):
        """
        Args:
           state_dict: Dict from StateEncoder
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
