"""Command-line interface for qwen-stream-video."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .config import load_config, summarize_config
from .exceptions import ConfigurationError, VideoOpenError
from .inference import FakeQwenClient, PromptBuilder, QwenClient, ResponseParser
from .pipeline import StreamingVideoPipeline

logger = logging.getLogger(__name__)


DEFAULT_FAKE_RESPONSE = """{
  "schema_version": "1.0",
  "window": {
    "global_index": 0,
    "start_seconds": 0.0,
    "end_seconds": 1.0
  },
  "summary": "No observations in this window.",
  "scene": {
    "camera_change": false,
    "view_type": "unknown",
    "visibility": "unknown",
    "description": "A test scene."
  },
  "entities": [],
  "actions": [],
  "attribute_observations": [],
  "uncertainties": []
}"""


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
        "--video",
        type=Path,
        default=None,
        help="Path to the local MP4 video to analyse.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Root directory for run outputs (overrides config).",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Ignore windows ending before this time (seconds).",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="Ignore windows starting at or after this time (seconds).",
    )
    parser.add_argument(
        "--start-window",
        type=int,
        default=None,
        help="Ignore windows with a global index below this.",
    )
    parser.add_argument(
        "--end-window",
        type=int,
        default=None,
        help="Ignore windows with a global index above this.",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Process at most this many windows.",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Wait until the video's logical time reaches each window end.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Sample frames and build prompts without calling the model.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check the video and report windows without calling the model.",
    )
    parser.add_argument(
        "--no-state",
        action="store_true",
        help="Do not carry the previous window summary into the next prompt.",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Persist sampled frames for this run.",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Resolve configuration, print a summary, and exit.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def _resolve_config_path(args: argparse.Namespace) -> Path | None:
    """Return the effective config path, defaulting to config.yaml if present."""
    if args.config is not None:
        return args.config
    default = Path("config.yaml")
    if default.is_file():
        return default
    return None


def _build_config(args: argparse.Namespace) -> Any:
    """Load the resolved configuration from CLI arguments."""
    cli_overrides: dict[str, Any] = {}
    if args.realtime:
        cli_overrides["runtime.realtime"] = True
    if args.no_state:
        cli_overrides["runtime.carry_previous_state"] = False
    if args.save_frames:
        cli_overrides["runtime.save_sampled_frames"] = True
        cli_overrides["storage.save_sampled_frames"] = True
    if args.output_dir:
        cli_overrides["storage.output_root"] = args.output_dir
    if args.max_windows:
        cli_overrides["runtime.max_windows"] = args.max_windows

    return load_config(
        config_path=_resolve_config_path(args),
        cli_overrides=cli_overrides or None,
    )


def _build_client(args: argparse.Namespace, config: Any) -> Any:
    """Create a real or fake inference client depending on the run mode."""
    if args.dry_run or args.validate_only:
        return FakeQwenClient(response_text=DEFAULT_FAKE_RESPONSE)
    if not config.model.api_key:
        print(
            "错误: 未配置 API Key。请设置 DASHSCOPE_API_KEY 环境变量或在配置中提供 model.api_key。",
            file=sys.stderr,
        )
        return None
    return QwenClient(config.model)


def _build_video_context(config: Any) -> dict[str, Any]:
    """Build the video context dict from the resolved configuration."""
    return {
        "video_name": config.video_context.video_name,
        "video_category": config.video_context.video_category,
        "task_background": config.video_context.task_background,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return its exit status."""
    args = build_parser().parse_args(argv)

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    try:
        config = _build_config(args)
    except ConfigurationError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 1

    if args.print_config:
        resolved_config_path = _resolve_config_path(args)
        source = f"YAML: {resolved_config_path}" if resolved_config_path else "代码默认值"
        print(summarize_config(config, video_path=args.video, source=source))
        return 0

    if not args.video:
        print("错误: 必须提供 --video 参数。", file=sys.stderr)
        return 1

    if not Path(args.video).is_file():
        print(f"错误: 视频文件不存在: {args.video}", file=sys.stderr)
        return 1

    client = _build_client(args, config)
    if client is None:
        return 1

    pipeline = StreamingVideoPipeline(
        config,
        args.video,
        client=client,
        prompt_builder=PromptBuilder(),
        parser=ResponseParser(),
        video_context=_build_video_context(config),
    )

    try:
        storage = pipeline.run(
            output_dir=args.output_dir,
            start_time=args.start_time,
            end_time=args.end_time,
            start_window=args.start_window,
            end_window=args.end_window,
            max_windows=args.max_windows,
            dry_run=args.dry_run,
            validate_only=args.validate_only,
        )
    except VideoOpenError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1

    if storage is not None:
        print(f"完成。输出目录: {storage.run_dir}")
    return 0


def entrypoint() -> None:
    """Console-script entry point that loads environment variables from .env."""
    load_dotenv()
    raise SystemExit(main())
