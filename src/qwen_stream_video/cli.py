"""Command-line interface for qwen-stream-video."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .config import load_config, summarize_config
from .exceptions import ConfigurationError


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="qwen-stream-video",
        description="Analyze local video in streaming-style windows.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to YAML configuration file (optional).",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Resolve configuration, print a summary, and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return its exit status."""
    args = build_parser().parse_args(argv)

    try:
        config = load_config(config_path=args.config)
    except ConfigurationError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 1

    if args.print_config:
        source = f"YAML: {args.config}" if args.config else "代码默认值"
        print(summarize_config(config, source=source))
        return 0

    return 0
