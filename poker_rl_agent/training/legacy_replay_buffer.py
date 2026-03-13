import collections
import random
import torch

class ReplayBuffer:
    "unused : legacy replay buffer for DeepCFR. Kept for reference but not used in current training loop."
    def __init__(self, capacity: int = 100000):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state_dict, action, regret, reward):
        """
        Stores an experience tuple.
        Arg format depends on what we want to train.
        For DeepCFR, we store (State, RegretVector) mostly.
        """
        self.buffer.append((state_dict, action, regret, reward))

    def sample(self, batch_size: int, pad_config: dict = None):
        batch = random.sample(self.buffer, batch_size)
        # Assuming batch is list of tuples
        # Need to collate state_dicts
        state_dicts, actions, regrets, rewards = zip(*batch)
        
        # Collate state_dicts
        collated_states = {}
        keys = state_dicts[0].keys()
        
        from torch.nn.utils.rnn import pad_sequence
        
        for k in keys:
            # Stack tensors
            tensors = [s[k] for s in state_dicts]
            
            # Check for padding based on dimensionality or explicit config
            # 'action_history' is variable length [1, SeqLen]
            if pad_config and k in pad_config:
                # Squeeze the [1, L] to [L] for padding
                tensors = [t.squeeze(0) if t.dim() == 2 else t for t in tensors]
                pad_val = pad_config[k]
                collated_states[k] = pad_sequence(tensors, batch_first=True, padding_value=pad_val)
            else:
                collated_states[k] = torch.stack(tensors) 
            
        if actions[0] is not None:
            actions = torch.tensor(actions, dtype=torch.long)
        else:
            actions = None
            
        regrets = torch.stack(regrets) if torch.is_tensor(regrets[0]) else torch.tensor(regrets, dtype=torch.float32)
        
        if rewards[0] is not None:
            rewards = torch.tensor(rewards, dtype=torch.float32)
        else:
            rewards = None
        
        return collated_states, actions, regrets, rewards

    def __len__(self):
        return len(self.buffer)
