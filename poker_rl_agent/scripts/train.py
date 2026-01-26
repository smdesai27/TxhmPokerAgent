import sys
import os

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poker_rl_agent.training.trainer import Trainer

def main():
    trainer = Trainer()
    print("Starting Training...")
    trainer.train()

if __name__ == "__main__":
    main()
