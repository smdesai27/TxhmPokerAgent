import torch
import os

def save_checkpoint(model, optimizer, step, buffer, path, max_keep=5):
    """Saves training state and cleans up old ones."""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
        # buffer omitted for disk space
    }, path)
    
    # Cleanup
    try:
        dirname = os.path.dirname(path)
        checkpoints = []
        for f in os.listdir(dirname):
            if f.startswith("point_") and f.endswith(".pt"):
                full_path = os.path.join(dirname, f)
                try:
                    # Extract step
                    s = int(f.split("_")[1].split(".")[0])
                    checkpoints.append((s, full_path))
                except:
                    pass
        
        checkpoints.sort(key=lambda x: x[0])
        
        if len(checkpoints) > max_keep:
            to_remove = checkpoints[:-max_keep]
            for _, p in to_remove:
                try:
                    os.remove(p)
                    print(f"Removed old checkpoint: {p}")
                except Exception as e:
                    print(f"Failed to remove {p}: {e}")
    except Exception as e:
        print(f"Error during checkpoint cleanup: {e}")


def load_checkpoint(path, model, optimizer=None):
    """Loads training state."""
    if not os.path.exists(path):
        return 0
        
    checkpoint = torch.load(path, map_location='cpu') # Safer to load to cpu first
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint.get('step', 0)
