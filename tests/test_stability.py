import torch
import torch.nn as nn
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from poker_rl_agent.training.trainer import Trainer
from poker_rl_agent.utils.config import Config

def test_stability():
    print("Initializing Trainer for Stability Test...")
    config = Config()
    
    # Assert V2 Configs
    assert config.LR == 5e-5, f"LR should be 5e-5, got {config.LR}"
    assert config.HIDDEN_DIM == 128, f"HIDDEN should be 128, got {config.HIDDEN_DIM}"
    assert config.GRAD_CLIP == 0.5, f"CLIP should be 0.5, got {config.GRAD_CLIP}"
    
    trainer = Trainer(config)
    
    print("Running 100 training steps...")
    
    # Mock training loop logic to access internals or just run a short train
    # Since train() loops forever/large iterations, we'll manually step if possible
    # Or just run trainer.train() but we'd need to mock the loop or interrupt it.
    # Actually trainer.train() runs CFR_ITERATIONS.
    # Let's override CFR_ITERATIONS to 2 (since each iter has multiple steps)
    
    trainer.config.CFR_ITERATIONS = 2
    trainer.config.SELF_PLAY_GAMES = 2 # Minimal data
    trainer.config.BATCH_SIZE = 16 # Small batch for quick test
    trainer.config.LOG_INTERVAL = 1
    
    try:
        trainer.train()
        print("Training finished without crash.")
    except Exception as e:
        print(f"FAILED: Training crashed with {e}")
        raise e
        
    print("Checking Model Weights for NaNs...")
    for name, param in trainer.model.named_parameters():
        if torch.isnan(param).any():
            print(f"FAILED: NaN found in {name}")
            sys.exit(1)
        if param.grad is not None and torch.isnan(param.grad).any():
             print(f"FAILED: NaN gradient in {name}")
             sys.exit(1)
             
    print("SUCCESS: Stability Test Passed")

if __name__ == "__main__":
    test_stability()
