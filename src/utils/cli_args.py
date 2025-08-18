"""
CLI Arguments Utility
=====================

Provides a universal CLI flag to choose between mock or live data modes
without editing `config/master_config.yaml`.

Usage:
    from src.utils.cli_args import parse_mode, print_mode_banner

    mode = parse_mode()
    print_mode_banner(mode)

Behavior:
- If `--mode {mock,live}` is provided, it takes precedence.
- If omitted, falls back to `config/master_config.yaml` → `data.mode`.
- Defaults to "mock" if the config is missing or unreadable.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import yaml


ModeLiteral = Literal["mock", "live"]


def _read_mode_from_config() -> ModeLiteral:
    """Read default mode from master_config.yaml (data.mode)."""
    try:
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / "config" / "master_config.yaml"
        if not config_path.exists():
            return "mock"
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return (cfg.get("data", {}) or {}).get("mode", "mock") in ("live",) and "live" or "mock"
    except Exception:
        return "mock"


def parse_mode(argv: list[str] | None = None) -> ModeLiteral:
    """Parse the run mode from CLI args, with YAML fallback.

    Args:
        argv: Optional list of arguments (for testing). If None, uses sys.argv.

    Returns:
        "mock" or "live"
    """
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--mode",
        choices=["mock", "live"],
        help="Run mode: 'mock' uses simulated data; 'live' connects to MT5",
    )

    # Parse known args so this can coexist with other scripts' args
    args, _ = parser.parse_known_args(argv)

    if getattr(args, "mode", None) in ("mock", "live"):
        return args.mode  # type: ignore[return-value]

    # Fallback to config
    return _read_mode_from_config()


def print_mode_banner(mode: ModeLiteral) -> None:
    """Print a clear banner indicating the selected mode."""
    line = "=" * 60
    if mode == "live":
        print(line)
        print("RUN MODE: LIVE - connecting to MT5 (production)")
        print(line)
    else:
        print(line)
        print("RUN MODE: MOCK - using simulated OHLCV data")
        print(line)


