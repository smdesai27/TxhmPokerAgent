import wandb
import logging
from .config import Config

def setup_logger(name="PokerRL"):
    """Sets up a standard python logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

def init_wandb(project_name="alpha-holdem-poker", run_name=None, config=None):
    """Initializes WandB logging."""
    if config:
        # Convert dataclass instance to dict if possible
        if hasattr(config, 'to_dict_instance'):
            config_dict = config.to_dict_instance()
        elif hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
        else:
            try:
                config_dict = config.__dict__
            except:
                config_dict = {}
    else:
        config_dict = Config.to_dict()

    wandb.init(
        project=project_name,
        name=run_name,
        config=config_dict,
        monitor_gym=False
    )

def log_metrics(metrics: dict, step: int):
    """Logs metrics to WandB."""
    if wandb.run is not None:
        wandb.log(metrics, step=step)
