import torch
import torch.nn as nn


class ActionTower(nn.Module):
    def __init__(self, num_actions, action_embedding_dim=64, hidden_dim=256, num_layers=2):
        super().__init__()
        self.num_actions = num_actions
        self.padding_idx = num_actions
        self.action_embedding = nn.Embedding(
            num_actions + 1,
            action_embedding_dim,
            padding_idx=self.padding_idx,
        )
        self.lstm = nn.LSTM(
            input_size=action_embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_dim = hidden_dim

    def forward(self, action_history, sequence_lengths=None):
        if action_history.dim() == 1:
            action_history = action_history.unsqueeze(0)

        if sequence_lengths is None:
            # Non-padding token count per sequence; clamp so packed sequence is valid.
            sequence_lengths = (action_history != self.padding_idx).sum(dim=1)
            sequence_lengths = torch.clamp(sequence_lengths, min=1)

        embedded = self.action_embedding(action_history)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            sequence_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hx, _) = self.lstm(packed)
        return hx[-1]
