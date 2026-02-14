#!/usr/bin/env python3
"""Checkpoint organization and indexing utilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


ACTIVE_NAMES = {"latest.pt", "best_eval.pt"}
INDEX_FILE = "checkpoint_index.json"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _classify(root: Path, path: Path) -> str:
    rel = path.relative_to(root)
    if rel.parent == Path(".") and (path.name in ACTIVE_NAMES or path.name.startswith("best_eval")):
        return "active"
    if "snapshots" in rel.parts:
        return "snapshot"
    return "loose"


@dataclass
class OpResult:
    moved: int = 0
    copied: int = 0
    removed_dups: int = 0


def _resolve_conflict(dst: Path, src: Path) -> Path:
    if not dst.exists():
        return dst
    if _sha256(dst) == _sha256(src):
        return dst
    stem = dst.stem
    suffix = dst.suffix
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return dst.with_name(f"{stem}_dup_{ts}{suffix}")


def build_index(root: Path) -> Dict[str, object]:
    records: List[Dict[str, object]] = []
    for path in sorted(root.rglob("*.pt")):
        rel = path.relative_to(root).as_posix()
        stat = path.stat()
        records.append(
            {
                "path": rel,
                "role": _classify(root, path),
                "size_bytes": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "sha256": _sha256(path),
            }
        )
    return {
        "generated_utc": _now_utc(),
        "root": str(root),
        "count": len(records),
        "records": records,
    }


def write_index(root: Path) -> Path:
    idx = build_index(root)
    output = root / INDEX_FILE
    output.write_text(json.dumps(idx, indent=2), encoding="utf-8")
    return output


def organize(root: Path, dry_run: bool = True) -> OpResult:
    result = OpResult()
    manual_dir = root / "snapshots" / "manual"
    manual_dir.mkdir(parents=True, exist_ok=True)

    for path in sorted(root.glob("*.pt")):
        if path.name in ACTIVE_NAMES or path.name.startswith("best_eval"):
            continue
        target = _resolve_conflict(manual_dir / path.name, path)
        if target == manual_dir / path.name and target.exists() and _sha256(target) == _sha256(path):
            if dry_run:
                print(f"[dry-run] remove duplicate {path}")
            else:
                path.unlink()
                result.removed_dups += 1
            continue
        if dry_run:
            print(f"[dry-run] move {path} -> {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))
            result.moved += 1
    return result


def snapshot(root: Path, source: Path, stage: str, label: str, active_alias: str | None, dry_run: bool = True) -> OpResult:
    result = OpResult()
    if not source.exists():
        raise FileNotFoundError(f"Source checkpoint does not exist: {source}")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage_dir = root / "snapshots" / stage
    dst = stage_dir / f"{ts}_{label}.pt"
    stage_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"[dry-run] copy {source} -> {dst}")
    else:
        shutil.copy2(source, dst)
        result.copied += 1

    if active_alias:
        alias_path = root / active_alias
        if dry_run:
            print(f"[dry-run] copy {source} -> {alias_path}")
        else:
            shutil.copy2(source, alias_path)
            result.copied += 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage checkpoint organization and indexing.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default="checkpoints", help="Checkpoint root directory.")
    common.add_argument("--apply", action="store_true", help="Apply changes. Default is dry-run.")

    p_org = sub.add_parser("organize", parents=[common], help="Move loose checkpoints into snapshots/manual.")
    p_idx = sub.add_parser("index", parents=[common], help="Generate checkpoint index JSON.")

    p_snap = sub.add_parser("snapshot", parents=[common], help="Snapshot a checkpoint into a stage folder.")
    p_snap.add_argument("--source", required=True, help="Source checkpoint path.")
    p_snap.add_argument("--stage", default="stage_d", help="Snapshot stage folder.")
    p_snap.add_argument("--label", required=True, help="Snapshot label.")
    p_snap.add_argument("--active-alias", default="", help="Optional alias path under checkpoint root, e.g. ablate_a_latest.pt")

    args = parser.parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    dry_run = not args.apply

    if args.cmd == "organize":
        result = organize(root, dry_run=dry_run)
        if not dry_run:
            idx_path = write_index(root)
            print(f"Moved={result.moved}, RemovedDuplicates={result.removed_dups}")
            print(f"Wrote index: {idx_path}")
        return

    if args.cmd == "snapshot":
        source = Path(args.source).resolve()
        active_alias = args.active_alias.strip() or None
        result = snapshot(root, source, args.stage, args.label, active_alias, dry_run=dry_run)
        if not dry_run:
            idx_path = write_index(root)
            print(f"Copied={result.copied}")
            print(f"Wrote index: {idx_path}")
        return

    if args.cmd == "index":
        idx_path = write_index(root)
        print(f"Wrote index: {idx_path}")
        return

    raise RuntimeError(f"Unhandled command: {args.cmd}")


if __name__ == "__main__":
    main()
