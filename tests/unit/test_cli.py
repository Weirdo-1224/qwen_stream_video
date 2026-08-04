"""Unit tests for the command-line interface."""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from qwen_stream_video.cli import main


def _make_test_video(path: Path, duration: float = 2.0, fps: float = 5.0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 64, 64
    total_frames = int(duration * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(total_frames):
        frame = np.full((height, width, 3), (i % 256, 128, 128), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _write_config(path: Path) -> Path:
    config_path = path / "config.yaml"
    config_path.write_text(
        """
experiment:
  name: cli_test
video:
  window_seconds: 2.0
  stride_seconds: 1.0
sampling:
  sample_fps: 1.0
  min_frames: 2
  max_frames: 4
  max_image_side: 64
storage:
  output_root: outputs
""",
        encoding="utf-8",
    )
    return config_path


def _extract_output_dir(stdout: str) -> Path:
    match = re.search(r"outputs[/\\][^\s]+", stdout)
    if match is None:
        msg = f"Could not find output directory in stdout: {stdout!r}"
        raise AssertionError(msg)
    return Path(match.group(0))


def test_print_config_returns_zero_and_hides_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "super-secret-key")
    video_path = _make_test_video(tmp_path / "video.mp4")
    exit_code = main(["--print-config", "--video", str(video_path)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "super-secret-key" not in captured.out
    assert "API Key: 已配置" in captured.out
    assert str(video_path) in captured.out


def test_missing_video_returns_error(capsys: pytest.CaptureFixture) -> None:
    exit_code = main(["--video", "nonexistent_video.mp4"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "nonexistent_video.mp4" in captured.err


def test_validate_only_reports_windows(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    config_path = _write_config(tmp_path)
    video_path = _make_test_video(tmp_path / "video.mp4", duration=5.0)
    exit_code = main(
        [
            "--config",
            str(config_path),
            "--video",
            str(video_path),
            "--validate-only",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "视频:" in captured.out
    assert "窗口数:" in captured.out


def test_dry_run_creates_output_without_api_key(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    config_path = _write_config(tmp_path)
    video_path = _make_test_video(tmp_path / "video.mp4", duration=5.0)
    exit_code = main(
        [
            "--config",
            str(config_path),
            "--video",
            str(video_path),
            "--dry-run",
            "--max-windows",
            "2",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    output_dir = _extract_output_dir(captured.out)
    windows_path = output_dir / "windows.jsonl"
    assert windows_path.exists()
    windows = [json.loads(line) for line in windows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(windows) == 2


def test_dry_run_with_save_frames_persists_sampled_frames(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    config_path = _write_config(tmp_path)
    video_path = _make_test_video(tmp_path / "video.mp4", duration=5.0)
    exit_code = main(
        [
            "--config",
            str(config_path),
            "--video",
            str(video_path),
            "--dry-run",
            "--save-frames",
            "--max-windows",
            "1",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    output_dir = _extract_output_dir(captured.out)
    frame_dirs = [p for p in (output_dir / "sampled_frames").iterdir() if p.is_dir()]
    assert len(frame_dirs) == 1
    assert len(list(frame_dirs[0].glob("*.jpg"))) > 0


def test_normal_run_requires_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    config_path = _write_config(tmp_path)
    video_path = _make_test_video(tmp_path / "video.mp4")
    exit_code = main(["--config", str(config_path), "--video", str(video_path)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "API Key" in captured.err
