import torch
import torch.nn as nn
import torch.nn.functional as F

class CardTower(nn.Module):
    def __init__(self, card_embedding_dim=64, hidden_dim=256, num_layers=3):
        super(CardTower, self).__init__()
        # 52 cards + 1 unknown/none token = 53 embeddings
        self.card_embedding = nn.Embedding(53, card_embedding_dim)
        
        # We process 2 hole cards and 5 community cards.
        # Simple architecture: Concatenate embeddings and pass through MLP.
        # Advanced: Transformer or Set-invariant architecture (DeepSets).
        # For simplicity and effectiveness in Poker: Flattened embeddings -> MLP is common.
        
        self.input_dim = (2 + 5) * card_embedding_dim
        
        layers = []
        input_d = self.input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(input_d, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            input_d = hidden_dim
            
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dim

    def forward(self, hole_cards, community_cards):
        """
        Args:
            hole_cards: [Batch, 2] indices
            community_cards: [Batch, 5] indices
        """
        # Embed
        hc_emb = self.card_embedding(hole_cards) # [B, 2, Emb]
        cc_emb = self.card_embedding(community_cards) # [B, 5, Emb]
        
        # Flatten
        hc_flat = hc_emb.view(hc_emb.size(0), -1)
        cc_flat = cc_emb.view(cc_emb.size(0), -1)
        
        x = torch.cat([hc_flat, cc_flat], dim=1)
        
        return self.mlp(x)
