"""Backward-compatible entry point for qwen-stream-video."""

from qwen_stream_video.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
