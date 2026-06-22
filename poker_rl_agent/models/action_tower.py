import torch
import torch.nn as nn


class ActionTower(nn.Module):
    def __init__(self, num_actions, action_embedding_dim=64, hidden_dim=256, num_layers=2):
        super().__init__()
        self.num_actions = num_actions
        self.padding_idx = num_actions
        self.action_embedding = nn.Embedding(
            #for padding, we use num_actions index since legal actions are 0 to num_actions-1.
            num_actions + 1,
            action_embedding_dim,
            #grads don't flow through padding tkn
            padding_idx=self.padding_idx,
        )
        #lstm since action history is variable length, and we want to capture temporal relations. 
        # could use transformer but could be overkill and unstable.
        self.lstm = nn.LSTM(
            input_size=action_embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_dim = hidden_dim

    def forward(self, action_history, sequence_lengths=None):
        # fail safes if lenghts are not provided
        if action_history.dim() == 1:
            action_history = action_history.unsqueeze(0)

        if sequence_lengths is None:
            # non-padding token count per sequence; clamp so packed sequence is valid.
            sequence_lengths = (action_history != self.padding_idx).sum(dim=1)
            sequence_lengths = torch.clamp(sequence_lengths, min=1)

        embedded = self.action_embedding(action_history)
        #rm padding tokens for packing
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            sequence_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # (output, (hx, cx)), we take the last layer's hidden state as the output. (num_layers ,B, 256) -> (B, 256)
        _, (hx, _) = self.lstm(packed)
        return hx[-1]
