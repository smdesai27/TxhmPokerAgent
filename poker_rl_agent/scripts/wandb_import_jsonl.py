import argparse
import json
import math
import os
import re
from pathlib import Path

try:
    import wandb
    _WANDB_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    wandb = None
    _WANDB_IMPORT_ERROR = exc


def _should_keep_key(key: str, allow_re):
    if key == "":
        return False
    if key.startswith("_"):
        return False
    if allow_re is None:
        return True
    return bool(allow_re.search(key))


def _to_wandb_payload(record: dict, step_key: str, allow_re):
    payload = {}
    step = record.get(step_key, None)
    if isinstance(step, bool):
        step = int(step)
    if not isinstance(step, (int, float)) or not math.isfinite(float(step)):
        return None, None
    step = int(step)

    for key, value in record.items():
        if key == step_key:
            continue
        if not _should_keep_key(str(key), allow_re):
            continue
        if isinstance(value, bool):
            payload[str(key)] = bool(value)
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            payload[str(key)] = float(value)
    return step, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_jsonl", type=str, required=True)
    parser.add_argument("--project", type=str, required=True)
    parser.add_argument("--entity", type=str, default="")
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--step_key", type=str, default="_step")
    parser.add_argument("--allow_keys_regex", type=str, default="")
    args = parser.parse_args()

    if wandb is None:
        raise RuntimeError(
            "Missing dependency: 'wandb' is not installed in this environment. "
            "Run 'pip install wandb' or 'pip install -r requirements.txt', then retry."
        ) from _WANDB_IMPORT_ERROR

    metrics_path = Path(args.metrics_jsonl)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    mode = os.environ.get("WANDB_MODE", "online").lower()
    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "WANDB_MODE=online requires WANDB_API_KEY in the environment. "
            "Set WANDB_API_KEY or set WANDB_MODE=offline for local import."
        )

    allow_re = re.compile(args.allow_keys_regex) if args.allow_keys_regex else None
    run = wandb.init(
        project=args.project,
        entity=(args.entity.strip() or None),
        name=args.run_name,
        mode=mode,
        config={
            "source_metrics_jsonl": str(metrics_path),
            "step_key": args.step_key,
            "allow_keys_regex": args.allow_keys_regex,
        },
        reinit=True,
    )

    total_rows = 0
    logged_rows = 0
    with metrics_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total_rows += 1
            try:
                record = json.loads(line)
            except Exception:
                continue
            step, payload = _to_wandb_payload(record, args.step_key, allow_re)
            if step is None or not payload:
                continue
            wandb.log(payload, step=step)
            logged_rows += 1

    run.summary["import/total_rows"] = total_rows
    run.summary["import/logged_rows"] = logged_rows
    run.finish()
    print(
        f"Imported metrics into W&B run '{args.run_name}': "
        f"logged_rows={logged_rows}, total_rows={total_rows}"
    )


if __name__ == "__main__":
    main()
