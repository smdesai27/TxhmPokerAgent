#!/usr/bin/env python3
"""Resolve Stage D selected 20k config from winner metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


DEFAULT_SELECTED_CONFIG_CONSERVATIVE = "quadro_stage_d_fchpa_evfirst_selected_21k_h2"
DEFAULT_SELECTED_CONFIG_BALANCED = "quadro_stage_d_fchpa_evfirst_selected_21k_h1"


def _parse_winner_fields_from_env(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    if not path.exists():
        return fields
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in {"WINNER_NAME", "WINNER_CONFIG_NAME"}:
            continue
        value = value.strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        if value:
            fields[key] = value
    return fields


def _parse_winner_fields_from_json(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    if not path.exists():
        return fields
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected")
    if not isinstance(selected, dict):
        return fields
    name = selected.get("name")
    if isinstance(name, str) and name:
        fields["WINNER_NAME"] = name
    config_name = selected.get("config_name")
    if isinstance(config_name, str) and config_name:
        fields["WINNER_CONFIG_NAME"] = config_name
    return fields


def resolve_selected_config(
    winner_name: str | None,
    conservative_config: str = DEFAULT_SELECTED_CONFIG_CONSERVATIVE,
    balanced_config: str = DEFAULT_SELECTED_CONFIG_BALANCED,
    fallback_config: str | None = None,
    winner_config_name: str | None = None,
) -> str:
    fallback = fallback_config or conservative_config
    config_hint = (winner_config_name or "").strip().lower()
    if config_hint:
        if "balanced" in config_hint:
            return balanced_config
        if (
            "conservative" in config_hint
            or "_h2_" in config_hint
            or "_c1_" in config_hint
            or "_c2_" in config_hint
            or "halfpot_nudge" in config_hint
        ):
            return conservative_config
    normalized = (winner_name or "").strip().lower()
    if normalized == "probe_b":
        return conservative_config
    if normalized == "probe_a":
        return balanced_config
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve Stage D selected config based on winner name.")
    parser.add_argument("--winner_name", default="", help="Explicit winner name (probe_a/probe_b).")
    parser.add_argument("--winner_config_name", default="", help="Explicit winner config name.")
    parser.add_argument("--winner_env_path", default="", help="Path to selection .env file.")
    parser.add_argument("--winner_json_path", default="", help="Path to selection .json file.")
    parser.add_argument(
        "--selected_config_conservative",
        default=DEFAULT_SELECTED_CONFIG_CONSERVATIVE,
    )
    parser.add_argument(
        "--selected_config_balanced",
        default=DEFAULT_SELECTED_CONFIG_BALANCED,
    )
    parser.add_argument(
        "--fallback_config",
        default="",
        help="Optional fallback config when winner name is unknown.",
    )
    parser.add_argument(
        "--print_winner_name",
        action="store_true",
        help="Print winner name on stderr for debugging.",
    )
    args = parser.parse_args()

    winner_name = args.winner_name.strip() or None
    winner_config_name = args.winner_config_name.strip() or None

    if args.winner_env_path:
        env_fields = _parse_winner_fields_from_env(Path(args.winner_env_path))
        if winner_name is None:
            winner_name = env_fields.get("WINNER_NAME")
        if winner_config_name is None:
            winner_config_name = env_fields.get("WINNER_CONFIG_NAME")
    if args.winner_json_path:
        json_fields = _parse_winner_fields_from_json(Path(args.winner_json_path))
        if winner_name is None:
            winner_name = json_fields.get("WINNER_NAME")
        if winner_config_name is None:
            winner_config_name = json_fields.get("WINNER_CONFIG_NAME")

    resolved = resolve_selected_config(
        winner_name=winner_name,
        winner_config_name=winner_config_name,
        conservative_config=args.selected_config_conservative,
        balanced_config=args.selected_config_balanced,
        fallback_config=args.fallback_config or None,
    )
    if args.print_winner_name:
        print(winner_name or "", file=sys.stderr)
    print(resolved)


if __name__ == "__main__":
    main()
