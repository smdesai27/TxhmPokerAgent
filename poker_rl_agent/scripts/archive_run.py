import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from poker_rl_agent.utils.config import Config


def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_") or "run"


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_config(config_file: str, config_name: str) -> Config:
    config = Config()
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        merged = {}
        if "default" in data and isinstance(data["default"], dict):
            merged.update(data["default"])
        if config_name != "default" and config_name in data and isinstance(data[config_name], dict):
            merged.update(data[config_name])

        for key, value in merged.items():
            if hasattr(config, key):
                setattr(config, key, value)

    config.CONFIG_NAME = str(config_name)
    if hasattr(config, "sync_legacy_fields"):
        config.sync_legacy_fields()
    return config


def _copy_required(src_path: str, dst_path: Path):
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"Required path does not exist: {src}")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_path)


def _copy_optional(src_path: str, dst_path: Path):
    src = Path(src_path)
    if not src.exists():
        return False
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_path)
    return True


def _read_metrics_summary(metrics_path: Path) -> Dict[str, float]:
    rows = []
    with metrics_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    if not rows:
        return {
            "records": 0,
            "iteration_min": None,
            "iteration_max": None,
            "nonfinite_grad_skips_final": None,
            "nonfinite_loss_batches_final": None,
            "rollout_mean_final_return_bb_last": None,
        }

    iterations = [int(r["iteration"]) for r in rows if isinstance(r.get("iteration"), int)]
    nonfinite_grad = [float(r["train/nonfinite_grad_skips"]) for r in rows if isinstance(r.get("train/nonfinite_grad_skips"), (int, float))]
    nonfinite_loss = [float(r["train/nonfinite_loss_batches"]) for r in rows if isinstance(r.get("train/nonfinite_loss_batches"), (int, float))]
    rollout = [float(r["rollout/mean_final_return_bb"]) for r in rows if isinstance(r.get("rollout/mean_final_return_bb"), (int, float))]

    return {
        "records": len(rows),
        "iteration_min": min(iterations) if iterations else None,
        "iteration_max": max(iterations) if iterations else None,
        "nonfinite_grad_skips_final": nonfinite_grad[-1] if nonfinite_grad else None,
        "nonfinite_loss_batches_final": nonfinite_loss[-1] if nonfinite_loss else None,
        "rollout_mean_final_return_bb_last": rollout[-1] if rollout else None,
    }


def _read_eval_summary(eval_path: Path) -> Dict[str, object]:
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    return {
        "pass_all": payload.get("overall/pass_all"),
        "pass_default_cfr_gate": payload.get("overall/pass_default_cfr_gate"),
        "pass_robustness_gate": payload.get("overall/pass_robustness_gate"),
        "pass_holdout_gate": payload.get("overall/pass_holdout_gate"),
        "solver_training_verified": payload.get("verification/solver_training_verified"),
    }


def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        return out
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--stage", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest.pt")
    parser.add_argument("--metrics", type=str, nargs="+", required=True)
    parser.add_argument("--eval_json", type=str, nargs="+", required=True)
    parser.add_argument("--slurm_out", type=str, nargs="*", default=[])
    parser.add_argument("--slurm_err", type=str, nargs="*", default=[])
    parser.add_argument("--config_file", type=str, default="configs/training_configs.yaml")
    parser.add_argument("--config_name", type=str, default="default")
    parser.add_argument("--archive_root", type=str, default="artifacts/runs")
    args = parser.parse_args()

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_slug = _slugify(args.run_name)
    archive_root = Path(args.archive_root)
    archive_dir = archive_root / f"{timestamp}_{run_slug}"
    (archive_dir / "model").mkdir(parents=True, exist_ok=True)
    (archive_dir / "eval").mkdir(parents=True, exist_ok=True)
    (archive_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (archive_dir / "slurm").mkdir(parents=True, exist_ok=True)
    (archive_dir / "config").mkdir(parents=True, exist_ok=True)
    (archive_dir / "report").mkdir(parents=True, exist_ok=True)

    # Copy core artifacts.
    _copy_required(args.checkpoint, archive_dir / "model" / "latest.pt")
    for path in args.eval_json:
        _copy_required(path, archive_dir / "eval" / Path(path).name)
    for path in args.metrics:
        _copy_required(path, archive_dir / "metrics" / Path(path).name)
    for path in args.slurm_out:
        _copy_optional(path, archive_dir / "slurm" / Path(path).name)
    for path in args.slurm_err:
        _copy_optional(path, archive_dir / "slurm" / Path(path).name)

    config = _load_config(args.config_file, args.config_name)
    with (archive_dir / "config" / "config_snapshot.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config.to_dict_instance(), fh, sort_keys=True)

    checkpoint_payload = torch.load(archive_dir / "model" / "latest.pt", map_location="cpu")
    checkpoint_step = int(checkpoint_payload.get("step", 0))

    metrics_summary = _read_metrics_summary(archive_dir / "metrics" / Path(args.metrics[0]).name)
    eval_summary = _read_eval_summary(archive_dir / "eval" / Path(args.eval_json[0]).name)

    report = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "run_name": args.run_name,
        "stage": args.stage,
        "checkpoint_step": checkpoint_step,
        "training_stability": metrics_summary,
        "benchmark_gates": eval_summary,
        "key_plots": [
            "rollout/mean_final_return_bb",
            "train/policy_loss",
            "train/value_loss",
            "train/approx_kl",
            "train/clipfrac",
            "train/grad_norm",
        ],
    }
    (archive_dir / "report" / "champion_summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    label = "stage_c_fcpa_pass_all_snapshot"
    if not (str(args.stage).lower().startswith("stage_c") and bool(eval_summary.get("pass_all"))):
        label = f"{_slugify(args.stage)}_snapshot"
    (archive_dir / "report" / "snapshot_label.txt").write_text(label + "\n", encoding="utf-8")

    md = (
        f"# Snapshot Report\n\n"
        f"- Run name: `{args.run_name}`\n"
        f"- Stage: `{args.stage}`\n"
        f"- Checkpoint step: `{checkpoint_step}`\n"
        f"- Snapshot label: `{label}`\n\n"
        f"## What Was Trained\n"
        f"- Config preset: `{args.config_name}`\n"
        f"- Abstraction: `{getattr(config, 'BETTING_ABSTRACTION', 'unknown')}`\n\n"
        f"## What Passed\n"
        f"- pass_all: `{eval_summary.get('pass_all')}`\n"
        f"- pass_default_cfr_gate: `{eval_summary.get('pass_default_cfr_gate')}`\n"
        f"- pass_robustness_gate: `{eval_summary.get('pass_robustness_gate')}`\n"
        f"- pass_holdout_gate: `{eval_summary.get('pass_holdout_gate')}`\n"
        f"- solver_training_verified: `{eval_summary.get('solver_training_verified')}`\n\n"
        f"## Known Limitations\n"
        f"- Results are scoped to the configured abstraction.\n"
        f"- Fullgame free-bet-size performance is not implied by this snapshot.\n"
    )
    (archive_dir / "report" / "champion_summary.md").write_text(md, encoding="utf-8")

    files = []
    for path in sorted(archive_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(archive_dir))
        if rel == "manifest.json":
            continue
        files.append(
            {
                "path": rel,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    manifest = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "run_name": args.run_name,
        "stage": args.stage,
        "archive_dir": str(archive_dir),
        "checkpoint_step": checkpoint_step,
        "config_name": args.config_name,
        "git_commit": _git_commit(),
        "source_paths": {
            "checkpoint": args.checkpoint,
            "metrics": args.metrics,
            "eval_json": args.eval_json,
            "slurm_out": args.slurm_out,
            "slurm_err": args.slurm_err,
            "config_file": args.config_file,
        },
        "files": files,
    }
    (archive_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Archived snapshot to: {archive_dir}")


if __name__ == "__main__":
    main()
