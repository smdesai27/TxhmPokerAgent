import torch
import torch.nn.functional as F

def regret_matching_plus(regrets: torch.Tensor) -> torch.Tensor:
    """
    Computes current strategy from regrets using Regret Matching+.
    Strategy is proportional to max(0, R).
    
    Args:
        regrets: [Batch, NumActions]
        
    Returns:
        strategy: [Batch, NumActions] (Probability distribution)
    """
    # RM+: Strategy is proportional to positive regrets
    positive_regrets = torch.relu(regrets)
    sum_pos_regrets = torch.sum(positive_regrets, dim=1, keepdim=True)
    
    # Check for zero sums (exploration)
    # If all regrets <= 0, use uniform random strategy
    uniform = torch.ones_like(regrets) / regrets.shape[1]
    
    strategy = torch.where(
        sum_pos_regrets > 1e-8,
        positive_regrets / sum_pos_regrets,
        uniform
    )
    
    return strategy

def get_action_from_strategy(strategy: torch.Tensor) -> torch.Tensor:
    """Samples action from strategy distribution."""
    # strategy: [Batch, NumActions] or [NumActions]
    if strategy.dim() == 1:
        return torch.multinomial(strategy, 1).item()
    else:
        return torch.multinomial(strategy, 1).squeeze()
