"""Unit tests for frame sampling and image encoding."""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
import pytest

from qwen_stream_video.exceptions import FrameReadError
from qwen_stream_video.video import (
    SampledFrame,
    VideoMetadata,
    VideoWindow,
    build_video_windows,
    encode_frame_to_data_url,
    read_video_metadata,
    sample_window_frames,
)


def _make_video(
    tmp_path: Path,
    duration: float,
    fps: int = 10,
    width: int = 160,
    height: int = 120,
) -> Path:
    """Create a tiny synthetic MP4 for testing."""
    path = tmp_path / f"sample_{duration}s_{fps}fps.mp4"
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


def _first_window(metadata: VideoMetadata) -> VideoWindow:
    windows = build_video_windows(metadata, window_seconds=6.0, stride_seconds=3.0)
    return windows[0]


def test_sample_window_frame_count(tmp_path: Path) -> None:
    metadata = _make_metadata(tmp_path, 15.0)
    window = _first_window(metadata)

    frames = sample_window_frames(
        metadata,
        window,
        sample_fps=1.0,
        min_frames=4,
        max_frames=12,
    )

    # duration 6s * 1 fps = 6, clamped to [4, 12] -> 6
    assert len(frames) == 6
    for i, frame in enumerate(frames):
        assert isinstance(frame, SampledFrame)
        assert frame.global_index == window.global_index
        assert frame.run_index == window.run_index
        assert frame.frame_index == i


def test_sample_window_respects_boundaries(tmp_path: Path) -> None:
    metadata = _make_metadata(tmp_path, 15.0)
    window = _first_window(metadata)

    frames = sample_window_frames(
        metadata,
        window,
        sample_fps=2.0,
        min_frames=4,
        max_frames=12,
    )

    # 6s * 2 fps = 12, clamped to 12
    assert len(frames) == 12
    for frame in frames:
        assert window.start_seconds <= frame.timestamp < window.end_seconds

    # No frame should sit exactly on the right boundary.
    assert all(frame.timestamp < window.end_seconds for frame in frames)


def test_sample_window_max_frames_clamped(tmp_path: Path) -> None:
    metadata = _make_metadata(tmp_path, 15.0)
    window = _first_window(metadata)

    frames = sample_window_frames(
        metadata,
        window,
        sample_fps=10.0,
        min_frames=4,
        max_frames=8,
    )

    # 6s * 10 fps = 60, clamped to 8
    assert len(frames) == 8


def test_sample_window_insufficient_frames(tmp_path: Path) -> None:
    # 1 second video at 10 fps only has 10 distinct real frames.
    metadata = _make_metadata(tmp_path, 1.0, fps=10)
    window = VideoWindow(
        global_index=0,
        run_index=0,
        start_seconds=0.0,
        end_seconds=1.0,
    )

    with pytest.raises(FrameReadError):
        sample_window_frames(
            metadata,
            window,
            sample_fps=1.0,
            min_frames=20,
            max_frames=30,
        )


def test_sample_window_distinct_frames(tmp_path: Path) -> None:
    metadata = _make_metadata(tmp_path, 15.0)
    window = _first_window(metadata)

    frames = sample_window_frames(
        metadata,
        window,
        sample_fps=1.0,
        min_frames=4,
        max_frames=12,
    )

    # Synthetic frames have different blue channel values, so no duplicates.
    for i in range(1, len(frames)):
        assert not np.array_equal(frames[i].image, frames[i - 1].image)


def test_sampled_frame_hides_image_on_serialization() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    frame = SampledFrame(
        run_index=0,
        global_index=0,
        timestamp=0.0,
        frame_index=0,
        image=image,
    )
    data = frame.model_dump()
    assert "image" not in data
    assert data["timestamp"] == 0.0


def test_encode_small_image_not_upscaled() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    data_url = encode_frame_to_data_url(image, max_image_side=768, jpeg_quality=80)

    assert data_url.startswith("data:image/jpeg;base64,")
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(data_url.split(",")[1]), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded.shape[:2] == (100, 100)


def test_encode_large_image_downscaled() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    data_url = encode_frame_to_data_url(image, max_image_side=100, jpeg_quality=80)

    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(data_url.split(",")[1]), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    # Largest side should be 100; the other side scales proportionally.
    height, width = decoded.shape[:2]
    assert max(width, height) == 100
    assert height == 75


def test_encode_from_sampled_frame(tmp_path: Path) -> None:
    metadata = _make_metadata(tmp_path, 15.0)
    window = _first_window(metadata)
    frames = sample_window_frames(
        metadata,
        window,
        sample_fps=1.0,
        min_frames=4,
        max_frames=12,
    )

    data_url = encode_frame_to_data_url(frames[0], max_image_side=768, jpeg_quality=80)
    assert data_url.startswith("data:image/jpeg;base64,")


def test_encode_invalid_arguments() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        encode_frame_to_data_url(image, max_image_side=0, jpeg_quality=80)
    with pytest.raises(ValueError):
        encode_frame_to_data_url(image, max_image_side=100, jpeg_quality=0)
    with pytest.raises(ValueError):
        encode_frame_to_data_url(image, max_image_side=100, jpeg_quality=101)
