import json
import logging
import os
from datetime import datetime

import wandb

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
    """Initializes W&B, but never hard-fails training when unavailable."""
    if config is None:
        config = Config()
    elif isinstance(config, type):
        config = config()

    if hasattr(config, "to_dict_instance"):
        config_dict = config.to_dict_instance()
    elif hasattr(config, "__dict__"):
        config_dict = dict(config.__dict__)
    else:
        config_dict = Config.to_dict()

    mode = str(config_dict.get("WANDB_MODE", "offline")).lower()
    if mode == "disabled":
        return None

    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("WANDB_MODE", mode)

    try:
        return wandb.init(
            project=getattr(config, "WANDB_PROJECT", project_name),
            name=getattr(config, "WANDB_RUN_NAME", run_name),
            config=config_dict,
            mode=mode,
            monitor_gym=False,
            reinit=True,
        )
    except Exception:
        # Fallback to no-op logging in network/sandbox restricted environments.
        return None

def log_metrics(metrics: dict, step: int):
    """Logs metrics to WandB."""
    if wandb.run is not None:
        wandb.log(metrics, step=step)


def log_metrics_local(metrics: dict, step: int, out_dir: str = "logs"):
    """Append JSONL metrics for offline-safe experiment tracking."""
    os.makedirs(out_dir, exist_ok=True)
    record = dict(metrics)
    record["_step"] = int(step)
    record["_timestamp"] = datetime.utcnow().isoformat() + "Z"
    run_id = str(record.get("run_id", "")).strip()
    if run_id:
        safe_run_id = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in run_id)
        filename = f"metrics_{safe_run_id}.jsonl"
        record["_run_id"] = safe_run_id
    else:
        filename = "metrics.jsonl"
    path = os.path.join(out_dir, filename)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
