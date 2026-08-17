"""Unit tests for video metadata and window generation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from qwen_stream_video.exceptions import VideoOpenError
from qwen_stream_video.video import (
    VideoMetadata,
    VideoWindow,
    build_video_windows,
    calculate_realtime_target,
    read_video_metadata,
    select_windows,
)


def _make_video(
    tmp_path: Path,
    duration: float,
    fps: int = 10,
    width: int = 160,
    height: int = 120,
) -> Path:
    """Create a tiny synthetic MP4 for testing."""
    path = tmp_path / f"test_{duration}s.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    total_frames = int(duration * fps)
    for i in range(total_frames):
        frame = np.full((height, width, 3), (i % 256, 0, 0), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _make_metadata(tmp_path: Path, duration: float, fps: int = 10) -> VideoMetadata:
    path = _make_video(tmp_path, duration, fps)
    return read_video_metadata(path)


def test_read_video_metadata(tmp_path: Path) -> None:
    metadata = _make_metadata(tmp_path, 15.0, fps=24)
    assert metadata.duration_seconds == pytest.approx(15.0, abs=0.05)
    assert metadata.fps == 24.0
    assert metadata.frame_count == 360
    assert metadata.width == 160
    assert metadata.height == 120


def test_missing_video_raises(tmp_path: Path) -> None:
    with pytest.raises(VideoOpenError):
        read_video_metadata(tmp_path / "does-not-exist.mp4")


def test_regular_windows(tmp_path: Path) -> None:
    metadata = _make_metadata(tmp_path, 15.0)
    windows = build_video_windows(metadata, window_seconds=6.0, stride_seconds=3.0)

    assert len(windows) == 4
    for idx, window in enumerate(windows):
        assert window.global_index == idx
        assert window.run_index == idx
        assert window.window_type == "regular"

    assert windows[0] == VideoWindow(
        global_index=0, run_index=0, start_seconds=0.0, end_seconds=6.0
    )
    assert windows[-1] == VideoWindow(
        global_index=3,
        run_index=3,
        start_seconds=9.0,
        commit_start_seconds=12.0,
        end_seconds=15.0,
    )


def test_short_video(tmp_path: Path) -> None:
    metadata = _make_metadata(tmp_path, 2.0)
    windows = build_video_windows(metadata, window_seconds=6.0, stride_seconds=3.0)

    assert len(windows) == 1
    assert windows[0].window_type == "tail_completion"
    assert windows[0].start_seconds == 0.0
    assert windows[0].end_seconds == pytest.approx(2.0, abs=0.05)


def test_tail_completion_window(tmp_path: Path) -> None:
    metadata = _make_metadata(tmp_path, 16.0)
    windows = build_video_windows(metadata, window_seconds=6.0, stride_seconds=6.0)

    assert len(windows) == 3
    assert windows[0].window_type == "regular"
    assert windows[1].window_type == "regular"
    assert windows[2].window_type == "tail_completion"
    assert windows[2].start_seconds == pytest.approx(10.0, abs=1e-9)
    assert windows[2].end_seconds == pytest.approx(16.0, abs=0.05)


def test_windows_do_not_exceed_duration(tmp_path: Path) -> None:
    metadata = _make_metadata(tmp_path, 15.0)
    windows = build_video_windows(metadata, window_seconds=6.0, stride_seconds=3.0)

    for window in windows:
        assert window.end_seconds <= metadata.duration_seconds + 1e-9
        assert window.start_seconds < window.end_seconds

    starts = [w.start_seconds for w in windows]
    assert starts == sorted(starts)


def test_global_index_preserved_after_selection(tmp_path: Path) -> None:
    metadata = _make_metadata(tmp_path, 30.0)
    windows = build_video_windows(metadata, window_seconds=6.0, stride_seconds=3.0)
    selected = select_windows(windows, start_time=12.0, end_time=18.0)

    global_indices = [w.global_index for w in selected]
    run_indices = [w.run_index for w in selected]

    assert selected
    assert global_indices[0] > 0
    assert run_indices == list(range(len(selected)))
    assert global_indices == sorted(global_indices)


def test_realtime_target_from_nonzero_start() -> None:
    assert calculate_realtime_target(1000.0, 480.0, 486.0) == pytest.approx(1006.0)


def test_select_by_window_index(tmp_path: Path) -> None:
    metadata = _make_metadata(tmp_path, 30.0)
    windows = build_video_windows(metadata, window_seconds=6.0, stride_seconds=6.0)
    selected = select_windows(windows, start_window=1, end_window=2)

    assert [w.global_index for w in selected] == [1, 2]
    assert [w.run_index for w in selected] == [0, 1]


def test_select_max_windows(tmp_path: Path) -> None:
    metadata = _make_metadata(tmp_path, 30.0)
    windows = build_video_windows(metadata, window_seconds=6.0, stride_seconds=6.0)
    selected = select_windows(windows, max_windows=2)

    assert len(selected) == 2
    assert [w.run_index for w in selected] == [0, 1]


def test_window_end_must_be_after_start() -> None:
    with pytest.raises(ValueError):
        VideoWindow(global_index=0, run_index=0, start_seconds=5.0, end_seconds=5.0)
