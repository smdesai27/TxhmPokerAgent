import torch
import torch.nn as nn
from .card_tower import CardTower
from .action_tower import ActionTower
from .model_utils import masked_logits


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
    """Pseudo-siamese trunk with policy and value heads."""

    def __init__(self, num_actions, config):
        super().__init__()
        self.num_actions = num_actions

        self.card_tower = CardTower(
            card_embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS_CARD,
        )

        self.action_tower = ActionTower(
            num_actions=num_actions,
            action_embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_LAYERS_ACTION,
        )

        scalar_dim = getattr(config, "SCALAR_DIM", 11)
        combined_dim = self.card_tower.output_dim + self.action_tower.output_dim + scalar_dim

        layers = [
            nn.Linear(combined_dim, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM) if config.USE_LAYERNORM else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
        ]

        if config.USE_RESIDUAL:
            layers.extend([
                ResidualBlock(config.HIDDEN_DIM, config.DROPOUT),
                ResidualBlock(config.HIDDEN_DIM, config.DROPOUT),
            ])
        else:
            layers.extend([
                nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
                nn.LayerNorm(config.HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT),
                nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
                nn.LayerNorm(config.HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT),
            ])

        self.fc_layers = nn.Sequential(*layers)

        self.policy_head = nn.Linear(config.HIDDEN_DIM, num_actions)
        self.value_head = nn.Linear(config.HIDDEN_DIM, 1)

        self.apply(self._init_weights)

        nn.init.uniform_(self.policy_head.weight, -0.01, 0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.uniform_(self.value_head.weight, -0.01, 0.01)
        nn.init.zeros_(self.value_head.bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, state_dict):
        hole_cards = state_dict["hole_cards"]
        community_cards = state_dict["community_cards"]
        action_history = state_dict["action_history"]
        scalars = state_dict["scalars"]

        if scalars.dim() == 1:
            scalars = scalars.unsqueeze(0)

        card_features = self.card_tower(hole_cards, community_cards)

        if action_history.dim() == 1:
            action_history = action_history.unsqueeze(0)
        seq_lengths = (action_history != self.num_actions).sum(dim=1)
        seq_lengths = torch.clamp(seq_lengths, min=1)
        action_features = self.action_tower(action_history, sequence_lengths=seq_lengths)

        x = torch.cat([card_features, action_features, scalars], dim=1)
        x = self.fc_layers(x)

        policy_logits = self.policy_head(x)
        state_value = self.value_head(x).squeeze(-1)

        return {
            "policy_logits": policy_logits,
            "state_value": state_value,
        }

    def masked_policy_logits(self, state_dict):
        outputs = self.forward(state_dict)
        legal_mask = state_dict["legal_action_mask"]
        outputs["policy_logits"] = masked_logits(outputs["policy_logits"], legal_mask)
        return outputs
