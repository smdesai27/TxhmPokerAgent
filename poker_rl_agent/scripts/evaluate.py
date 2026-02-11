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

    config = Config()
    # Hack: Need num_actions without instantiating full environment in main if possible,
    # but easier to just use Env wrapper
    from poker_rl_agent.environment.openspiel_wrapper import PokerEnv
    env = PokerEnv(
        game_name=config.GAME_NAME,
        env_preset=config.ENV_PRESET,
        betting_abstraction=config.BETTING_ABSTRACTION,
    )
    num_actions = env.num_actions()
    
    model = AlphaHoldemNetwork(num_actions, config)
    load_checkpoint(args.checkpoint, model)
    model.eval()
    
    evaluator = Evaluator(model, config=config, device=config.DEVICE)
    
    print(f"Evaluating against Random Agent for {args.episodes} episodes...")
    random_stats = evaluator.evaluate(RandomAgent(), args.episodes)
    print(f"Average Return vs Random: {random_stats['avg_return']:.3f}")
    print(f"BB/100 vs Random: {random_stats['bb_per_100']:.3f}")

    print(f"Evaluating against Always Call Agent for {args.episodes} episodes...")
    call_stats = evaluator.evaluate(AlwaysCallAgent(), args.episodes)
    print(f"Average Return vs AlwaysCall: {call_stats['avg_return']:.3f}")
    print(f"BB/100 vs AlwaysCall: {call_stats['bb_per_100']:.3f}")

    print(f"Evaluating against CFR baseline for {args.episodes} episodes...")
    cfr_stats = evaluator.evaluate_vs_cfr(
        iterations=config.EVAL_CFR_ITERATIONS,
        num_episodes=args.episodes,
    )
    print(f"Average Return vs CFR: {cfr_stats['avg_return']:.3f}")
    print(f"BB/100 vs CFR: {cfr_stats['bb_per_100']:.3f} (stderr {cfr_stats['std_err']:.3f})")

if __name__ == "__main__":
    main()
