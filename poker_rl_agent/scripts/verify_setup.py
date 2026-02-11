import os
import sys
import time

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poker_rl_agent.training.trainer import Trainer
from poker_rl_agent.utils.config import Config
from poker_rl_agent.evaluation.evaluator import Evaluator
from poker_rl_agent.evaluation.baseline_agents import RandomAgent
from poker_rl_agent.models.alpha_holdem_net import AlphaHoldemNetwork
from poker_rl_agent.training.checkpointing import load_checkpoint
from poker_rl_agent.environment.openspiel_wrapper import PokerEnv


def build_test_config():
    config = Config()
    config.ENV_PRESET = "hunl_fcpa"
    config.BETTING_ABSTRACTION = "fcpa"
    config.CFR_ITERATIONS = 8
    config.ROLLOUT_EPISODES = 16
    config.PPO_EPOCHS = 2
    config.PPO_MINIBATCH_SIZE = 64
    config.HIDDEN_DIM = 128
    config.CHECKPOINT_FREQ = 4
    config.LOG_INTERVAL = 1
    config.EVAL_FREQ = 4
    config.EVAL_EPISODES = 20
    config.EVAL_CFR_ITERATIONS = 25
    config.WANDB_MODE = "disabled"
    config.OFFLINE_LOGGING = True
    config.DEVICE = "cpu"
    return config


def main():
    print(">>> Starting verify_setup smoke run...")
    start_time = time.time()

    config = build_test_config()
    trainer = Trainer(config)

    try:
        trainer.train()
        print(">>> Training loop completed.")
    except Exception as exc:
        print(f"!!! Training FAILED: {exc}")
        raise

    checkpoint_path = "checkpoints/point_8.pt"
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"Expected checkpoint not found: {checkpoint_path}")

    env = PokerEnv(
        game_name=config.GAME_NAME,
        env_preset=config.ENV_PRESET,
        betting_abstraction=config.BETTING_ABSTRACTION,
    )
    model = AlphaHoldemNetwork(env.num_actions(), config)
    payload = load_checkpoint(checkpoint_path, model)
    print(f">>> Loaded checkpoint step={payload['step']} format={payload['format_version']}")

    evaluator = Evaluator(model, config=config, device="cpu")
    random_stats = evaluator.evaluate(RandomAgent(), num_episodes=20)
    print(f">>> Random baseline bb/100: {random_stats['bb_per_100']:.3f}")

    elapsed = time.time() - start_time
    print(f">>> verify_setup completed in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
