import torch
import torch.nn as nn


class CardTower(nn.Module):
    def __init__(self, card_embedding_dim=64, hidden_dim=256, num_layers=3):
        super().__init__()
        # 52 standard cards + unknown token.
        self.card_embedding = nn.Embedding(53, card_embedding_dim)
        self.input_dim = 7 * card_embedding_dim  # 2 hole + 5 community

        layers = []
        in_dim = self.input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dim

    def forward(self, hole_cards, community_cards):
        if hole_cards.dim() == 1:
            hole_cards = hole_cards.unsqueeze(0)
        if community_cards.dim() == 1:
            community_cards = community_cards.unsqueeze(0)

        hc_emb = self.card_embedding(hole_cards)
        cc_emb = self.card_embedding(community_cards)

        x = torch.cat([
            hc_emb.reshape(hc_emb.size(0), -1),
            cc_emb.reshape(cc_emb.size(0), -1),
        ], dim=1)
        return self.mlp(x)
