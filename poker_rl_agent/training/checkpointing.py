import torch
import os

def save_checkpoint(model, optimizer, step, buffer, path):
    """Saves training state."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        # Saving buffer might be too large, usually we don't or save simplified
    }, path)

def load_checkpoint(path, model, optimizer=None):
    """Loads training state."""
    if not os.path.exists(path):
        return 0
        
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint.get('step', 0)
