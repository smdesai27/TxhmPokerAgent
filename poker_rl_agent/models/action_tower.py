import torch
import torch.nn as nn

class ActionTower(nn.Module):
    def __init__(self, num_actions, action_embedding_dim=64, hidden_dim=256, num_layers=2):
        super(ActionTower, self).__init__()
        # Embedding for action IDs (Fold, Call, Raise, etc.)
        # If we just have index history.
        # Assuming a reasonable max number of distinct action IDs handled by OpenSpiel.
        # Or we map (Type, Amount) -> Dense Vector.
        # For now, we assume discrete action indices.
        
        self.action_embedding = nn.Embedding(num_actions + 1, action_embedding_dim, padding_idx=num_actions)
        
        self.lstm = nn.LSTM(
            input_size=action_embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.output_dim = hidden_dim

    def forward(self, action_history, sequence_lengths=None):
        """
        Args:
            action_history: [Batch, SeqLen] indices
            sequence_lengths: [Batch] lengths of valid actions
        """
        x = self.action_embedding(action_history) # [B, S, Emb]
        
        # If sequences are variable length, packing is ideal.
        # For simplicity here, we might just take last output or max pool.
        # Using LSTM output.
        
        output, (hx, cx) = self.lstm(x)
        
        # Return the final hidden state (or max over time)
        # Standard: Last Hidden State
        return hx[-1] # [B, Hidden]
