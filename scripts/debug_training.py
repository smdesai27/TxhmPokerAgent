import yaml
import argparse
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poker_rl_agent.training.trainer import Trainer
from poker_rl_agent.utils.config import Config

def load_config(config_path, config_name="default"):
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Start with default config params
    base_config = Config()
    base_dict = base_config.to_dict_instance()
    
    # Update with 'default' section from yaml
    if 'default' in data:
        base_dict.update(data['default'])
        
    # Update with specific config
    if config_name != 'default' and config_name in data:
         base_dict.update(data[config_name])
    
    # Re-create Config object
    # We need to filter out keys that don't belong if strictly enforcing, 
    # but Dataclasses just accept what we pass if we use **kwargs logic or setattr
    
    for k, v in base_dict.items():
        if hasattr(base_config, k):
            setattr(base_config, k, v)
        else:
            print(f"Warning: Config key {k} not found in Config class")
            
    return base_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/training_configs.yaml")
    parser.add_argument("--config_name", type=str, default="debug")
    args = parser.parse_args()
    
    config = load_config(args.config_file, args.config_name)
    print(f"Loaded Config: {args.config_name}")
    print(config)
    
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
