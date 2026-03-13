import yaml
import argparse
import sys
import os

# make sure project root is in path to prevent import errors 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poker_rl_agent.training.trainer import Trainer
from poker_rl_agent.utils.config import Config

def load_config(config_path, config_name="default"):
    with open(config_path, 'r') as f:
        # safe load yaml
        data = yaml.safe_load(f)
    
    # load defulats
    base_config = Config()
    base_dict = base_config.to_dict_instance()
    
    # update with defaults section from yaml
    if 'default' in data:
        base_dict.update(data['default'])
        
    # update with the given spec config
    if config_name != 'default' and config_name in data:
         base_dict.update(data[config_name])
    
    # re-create Config object
    # we need to filter out keys that don't belong if strictly enforcing, 
    # but Dataclasses just accept what we pass if we use **kwargs logic or setattr
    
    for k, v in base_dict.items():
        if hasattr(base_config, k):
            setattr(base_config, k, v)
        else:
            print(f"Warning: Config key {k} not found in Config class")
    # for wandb naming and logging
    base_config.CONFIG_NAME = str(config_name)
    if hasattr(base_config, "sync_legacy_fields"):
        base_config.sync_legacy_fields()
            
    return base_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/training_configs.yaml")
    parser.add_argument("--config_name", type=str, default="debug")
    args = parser.parse_args()
    
    config = load_config(args.config_file, args.config_name)
    #allow for slurm env vars
    strict_abstraction_env = os.environ.get("STRICT_ABSTRACTION")
    if strict_abstraction_env is not None:
        # allow any tuthy representatino so slurm scirpts can set this easily 
        config.STRICT_ABSTRACTION = str(strict_abstraction_env).strip().lower() in {"1", "true", "yes", "on"}

    # special case fail safe bc openspiel names "fchpa"
    if args.config_name == "quadro_stage_d_fcpha":
        print(
            "using canonical FCHPA abstraction behavior."
        )
    print(f"Loaded Config: {args.config_name}")
    print(config)
    
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
