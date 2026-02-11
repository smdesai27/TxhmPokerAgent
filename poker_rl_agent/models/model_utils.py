import torch
import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)


def masked_logits(policy_logits: torch.Tensor, legal_action_mask: torch.Tensor) -> torch.Tensor:
    """Applies a legal-action mask to logits by forcing illegal actions to -inf."""
    if legal_action_mask.dtype != torch.bool:
        mask = legal_action_mask > 0.5
    else:
        mask = legal_action_mask
    invalid = ~mask
    return policy_logits.masked_fill(invalid, -1e9)
