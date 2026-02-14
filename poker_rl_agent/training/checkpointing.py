import hashlib
import json
import os
from typing import Any, Dict, Optional

import torch


CHECKPOINT_FORMAT_VERSION = 2


def _config_hash(config: Optional[Any]) -> str:
    if config is None:
        return ""
    if hasattr(config, "to_dict_instance"):
        payload = config.to_dict_instance()
    elif hasattr(config, "__dict__"):
        payload = dict(config.__dict__)
    else:
        payload = {}
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def save_checkpoint(
    model,
    optimizer,
    step,
    scheduler,
    path,
    max_keep=5,
    single_file=False,
    save_optimizer=True,
    save_scheduler=True,
    save_league=True,
    save_extra=True,
    league_state: Optional[Dict[str, Any]] = None,
    config: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
    keep_files: Optional[list] = None,
):
    """Saves training state and prunes old checkpoints."""
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if (save_optimizer and optimizer is not None) else None
        ),
        "scheduler_state_dict": (
            scheduler.state_dict() if (save_scheduler and scheduler is not None) else None
        ),
        "step": int(step),
        "league_state": league_state if save_league else {},
        "config_hash": _config_hash(config),
        "extra": extra if save_extra else {},
    }
    torch.save(payload, path)

    if single_file:
        # Minimal-storage mode: keep only the rolling file.
        try:
            keep_name = os.path.basename(path)
            keep_set = {keep_name}
            if keep_files:
                keep_set.update(os.path.basename(str(name)) for name in keep_files)
            for file_name in os.listdir(dirname or "."):
                if file_name.endswith(".pt") and file_name not in keep_set:
                    full_path = os.path.join(dirname, file_name) if dirname else file_name
                    try:
                        os.remove(full_path)
                    except OSError:
                        pass
        except Exception:
            pass
        return

    # Cleanup old point_<step>.pt style checkpoints.
    try:
        checkpoints = []
        for file_name in os.listdir(dirname or "."):
            if file_name.startswith("point_") and file_name.endswith(".pt"):
                try:
                    s = int(file_name.split("_")[1].split(".")[0])
                except Exception:
                    continue
                full_path = os.path.join(dirname, file_name) if dirname else file_name
                checkpoints.append((s, full_path))

        checkpoints.sort(key=lambda x: x[0])
        if len(checkpoints) > max_keep:
            for _, old_path in checkpoints[:-max_keep]:
                try:
                    os.remove(old_path)
                except OSError:
                    pass
    except Exception:
        pass


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    """Loads training state.

    Returns a dict with fields: step, league_state, format_version, extra, config_hash.
    """
    if not os.path.exists(path):
        return {
            "step": 0,
            "league_state": {},
            "format_version": 0,
            "extra": {},
            "config_hash": "",
        }

    checkpoint = torch.load(path, map_location=map_location)

    # v1 compatibility.
    if "format_version" not in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return {
            "step": int(checkpoint.get("step", 0)),
            "league_state": {},
            "format_version": 1,
            "extra": {},
            "config_hash": "",
        }

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "step": int(checkpoint.get("step", 0)),
        "league_state": checkpoint.get("league_state", {}),
        "format_version": int(checkpoint.get("format_version", 0)),
        "extra": checkpoint.get("extra", {}),
        "config_hash": checkpoint.get("config_hash", ""),
    }
