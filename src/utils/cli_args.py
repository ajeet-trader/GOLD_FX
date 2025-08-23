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

- If omitted, returns None (so config can be used).

- Defaults to "mock" only if config is missing or unreadable in _read_mode_from_config.

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from typing import Literal, Optional

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

def parse_mode(argv: list[str] | None = None) -> Optional[ModeLiteral]:
    """Parse the run mode from CLI args, returns None if not specified.

    Args:
        argv: Optional list of arguments (for testing). If None, uses sys.argv.

    Returns:
        "mock", "live", or None if --mode not provided
    """
    
    # Use provided argv or sys.argv
    args_to_parse = argv if argv is not None else sys.argv[1:]
    
    # Quick check: if no --mode argument, return None immediately
    if not args_to_parse or '--mode' not in args_to_parse:
        return None
    
    parser = argparse.ArgumentParser(add_help=False)  # Don't interfere with other parsers
    parser.add_argument(
        "--mode",
        choices=["mock", "live", "test"],
        help="Run mode: 'mock' uses simulated data; 'live' connects to MT5; 'test' runs quick tests",
    )

    try:
        # Parse known args so this can coexist with other scripts' args
        args, _ = parser.parse_known_args(args_to_parse)
        
        # Return the mode if explicitly provided, otherwise None
        return getattr(args, "mode", None)
        
    except (SystemExit, argparse.ArgumentError):
        # If parsing fails, don't override config
        return None

# Global banner tracking to prevent duplicates
_BANNER_DISPLAYED = False

def print_mode_banner(mode: ModeLiteral) -> None:
    """Print a clear banner indicating the selected mode (only once per session)."""
    global _BANNER_DISPLAYED
    
    # Only print banner if it hasn't been displayed yet
    if not _BANNER_DISPLAYED:
        line = "=" * 60
        
        if mode == "live":
            print(line)
            print("RUN MODE: LIVE - connecting to MT5 (production)")
            print(line)
        else:
            print(line)
            print("RUN MODE: MOCK - using simulated OHLCV data")
            print(line)
        
        _BANNER_DISPLAYED = True

def reset_banner_flag():
    """Reset the banner flag for testing purposes."""
    global _BANNER_DISPLAYED
    _BANNER_DISPLAYED = False
