import sys
import os
import torch
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poker_rl_agent.models.alpha_holdem_net import AlphaHoldemNetwork
from poker_rl_agent.evaluation.evaluator import Evaluator
from poker_rl_agent.evaluation.baseline_agents import RandomAgent, AlwaysCallAgent
from poker_rl_agent.utils.config import Config
from poker_rl_agent.training.checkpointing import load_checkpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    config = Config
    # Hack: Need num_actions without instantiating full environment in main if possible,
    # but easier to just use Env wrapper
    from poker_rl_agent.environment.openspiel_wrapper import PokerEnv
    env = PokerEnv(config.GAME_NAME)
    num_actions = env.num_actions()
    
    model = AlphaHoldemNetwork(num_actions, config)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    
    evaluator = Evaluator(model)
    
    print(f"Evaluating against Random Agent for {args.episodes} episodes...")
    avg_return = evaluator.evaluate(RandomAgent(), args.episodes)
    print(f"Average Return vs Random: {avg_return}")

    print(f"Evaluating against Always Call Agent for {args.episodes} episodes...")
    avg_return = evaluator.evaluate(AlwaysCallAgent(), args.episodes)
    print(f"Average Return vs AlwaysCall: {avg_return}")

if __name__ == "__main__":
    main()
