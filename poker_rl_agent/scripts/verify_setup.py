import sys
import os
import torch
import shutil
import time

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poker_rl_agent.training.trainer import Trainer
from poker_rl_agent.utils.config import Config
from poker_rl_agent.evaluation.evaluator import Evaluator
from poker_rl_agent.evaluation.baseline_agents import RandomAgent
from poker_rl_agent.models.alpha_holdem_net import AlphaHoldemNetwork
from poker_rl_agent.training.checkpointing import load_checkpoint

class TestConfig(Config):
    # Override for quick CPU test
    GAME_NAME = "leduc_poker"
    PLAYERS = 2
    
    # Small Network
    EMBEDDING_DIM = 16
    HIDDEN_DIM = 32
    NUM_LAYERS_CARD = 1
    NUM_LAYERS_ACTION = 1
    
    # Very short training
    LR = 1e-3
    BATCH_SIZE = 16
    CFR_ITERATIONS = 20 # Enough to run a few loops
    SELF_PLAY_GAMES = 2
    BUFFER_SIZE = 1000
    
    CHECKPOINT_FREQ = 10
    LOG_INTERVAL = 5
    DEVICE = "cpu"

def main():
    print(">>> Starting Test Run (Leduc Poker, CPU, Fast)...")
    start_time = time.time()
    
    # 1. Setup Trainer with Test Config
    print(">>> Initializing Trainer...")
    trainer = Trainer(TestConfig)
    
    # 2. Train
    print(f">>> Training for {TestConfig.CFR_ITERATIONS} iterations...")
    try:
        trainer.train()
        print(">>> Training loop finished successfully.")
    except Exception as e:
        print(f"!!! Training FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # 3. Check Checkpoint
    checkpoint_path = f"checkpoints/point_{TestConfig.CFR_ITERATIONS}.pt"
    if os.path.exists(checkpoint_path):
        print(f">>> Checkpoint found at {checkpoint_path}")
    else:
        print(f"!!! Checkpoint NOT found at {checkpoint_path}")
        # Look for any checkpoint
        if os.path.exists("checkpoints"):
             print(f"Checkpoints dir content: {os.listdir('checkpoints')}")
    
    # 4. Evaluate
    print(">>> Running Evaluation...")
    try:
        # Re-load model from checkpoint to verify loading
        # Hack: need simple way to get env num actions without opening env again
        # but Evaluator opens env.
        from poker_rl_agent.environment.openspiel_wrapper import PokerEnv
        env = PokerEnv(TestConfig.GAME_NAME)
        num_actions = env.num_actions()
        
        model = AlphaHoldemNetwork(num_actions, TestConfig)
        step = load_checkpoint(checkpoint_path, model)
        print(f">>> Loaded checkpoint from step {step}")
        
        evaluator = Evaluator(model, device="cpu")
        # Point Evaluator to Leduc (Evaluator inside uses PokerEnv() which defaults to Config or "universal_poker")
        # We need to ensure Evaluator uses "leduc_poker" too.
        # Currently Evaluator uses `PokerEnv()` which uses default config logic.
        # We should modify Evaluator or just manually patch it.
        # Patching env in evaluator:
        evaluator.env = PokerEnv("leduc_poker") 
        
        avg_ret = evaluator.evaluate(RandomAgent(), num_episodes=50)
        print(f">>> Evaluation vs Random (50 games): Avg Reward = {avg_ret:.3f}")
        
        print(">>> Test Run COMPLETED SUCCESSFULLY.")
        print(f"Total Time: {time.time() - start_time:.2f}s")
        
    except Exception as e:
        print(f"!!! Evaluation FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
