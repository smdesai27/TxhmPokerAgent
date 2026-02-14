import json
import logging
import os
from datetime import datetime
from typing import List

try:
    import wandb as _wandb
    _WANDB_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    _wandb = None
    _WANDB_IMPORT_ERROR = exc


class _MissingWandb:
    """Minimal stand-in so tests can monkeypatch attributes safely."""

    run = None

    def init(self, *args, **kwargs):  # pragma: no cover - defensive fallback
        raise ModuleNotFoundError(
            "The 'wandb' package is not installed. Install it with "
            "'pip install wandb' or 'pip install -r requirements.txt'."
        ) from _WANDB_IMPORT_ERROR

    def log(self, *args, **kwargs):  # pragma: no cover - no-op
        return None


wandb = _wandb if _wandb is not None else _MissingWandb()

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


def _parse_tags(raw_tags) -> List[str]:
    if raw_tags is None:
        return []
    if isinstance(raw_tags, (list, tuple)):
        return [str(t).strip() for t in raw_tags if str(t).strip()]
    return [part.strip() for part in str(raw_tags).split(",") if part.strip()]


def init_wandb(project_name="alpha-holdem-poker", run_name=None, config=None):
    """Initializes W&B with optional strict online preflight."""
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

    mode = str(os.environ.get("WANDB_MODE", config_dict.get("WANDB_MODE", "offline"))).lower()
    if mode not in {"online", "offline", "disabled"}:
        mode = "offline"
    if mode == "disabled":
        return None

    if _WANDB_IMPORT_ERROR is not None:
        require_online = bool(config_dict.get("WANDB_REQUIRE_ONLINE", True))
        if mode == "online" and require_online:
            raise RuntimeError(
                "W&B logging requested in online mode, but the 'wandb' package is not installed. "
                "Install it with 'pip install wandb' (or 'pip install -r requirements.txt')."
            ) from _WANDB_IMPORT_ERROR
        # Keep offline-safe behavior if wandb is unavailable and strict online mode is not required.
        return None

    require_online = bool(config_dict.get("WANDB_REQUIRE_ONLINE", True))
    if mode == "online" and require_online and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "WANDB_MODE=online requires WANDB_API_KEY when WANDB_REQUIRE_ONLINE=true. "
            "Set WANDB_API_KEY in your shell or SLURM environment."
        )

    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ["WANDB_MODE"] = mode

    project = str(os.environ.get("WANDB_PROJECT", getattr(config, "WANDB_PROJECT", project_name))).strip()
    run_title = str(os.environ.get("WANDB_RUN_NAME", getattr(config, "WANDB_RUN_NAME", run_name) or "")).strip() or None
    entity = str(os.environ.get("WANDB_ENTITY", getattr(config, "WANDB_ENTITY", ""))).strip() or None
    tags = _parse_tags(os.environ.get("WANDB_TAGS", getattr(config, "WANDB_TAGS", "")))
    run_group = os.environ.get("WANDB_RUN_GROUP", "").strip() or None
    try:
        return wandb.init(
            project=project,
            name=run_title,
            entity=entity,
            tags=tags or None,
            group=run_group,
            config=config_dict,
            mode=mode,
            monitor_gym=False,
            reinit=True,
        )
    except Exception:
        if mode == "online" and require_online:
            raise
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
