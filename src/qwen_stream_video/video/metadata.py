"""Video metadata extraction and model."""

from __future__ import annotations

from pathlib import Path

import cv2
from pydantic import BaseModel, ConfigDict, Field

from ..exceptions import VideoMetadataError, VideoOpenError


class VideoMetadata(BaseModel):
    """Immutable description of a local video file."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    path: str
    fps: float = Field(gt=0)
    frame_count: int = Field(ge=0)
    duration_seconds: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)


def read_video_metadata(path: str | Path) -> VideoMetadata:
    """Open a local video with OpenCV and read its metadata.

    Raises:
        VideoOpenError: if the file is missing or cannot be opened.
        VideoMetadataError: if the reported metadata is invalid.
    """
    path = Path(path)
    if not path.is_file():
        raise VideoOpenError(f"Video file not found: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoOpenError(f"Cannot open video file: {path}")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()

    if not fps or fps <= 0:
        raise VideoMetadataError(f"Invalid FPS for {path}: {fps}")
    if width <= 0 or height <= 0:
        raise VideoMetadataError(f"Invalid video dimensions for {path}: {width}x{height}")

    if frame_count > 0:
        duration_seconds = frame_count / fps
    else:
        # Fallback: count frames by reading the stream.
        duration_seconds = _estimate_duration_by_reading(path, fps)

    if duration_seconds <= 0:
        raise VideoMetadataError(
            f"Unable to determine positive duration for {path}: {duration_seconds}s"
        )

    return VideoMetadata(
        path=str(path),
        fps=fps,
        frame_count=max(0, frame_count),
        duration_seconds=duration_seconds,
        width=width,
        height=height,
    )


def _estimate_duration_by_reading(path: Path, fps: float) -> float:
    """Count frames manually when CAP_PROP_FRAME_COUNT is not available."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoOpenError(f"Cannot reopen video file for duration estimation: {path}")

    try:
        frame_count = 0
        while True:
            success, _ = capture.read()
            if not success:
                break
            frame_count += 1
    finally:
        capture.release()

    if frame_count <= 0 or fps <= 0:
        raise VideoMetadataError(
            f"Unable to estimate duration for {path}: {frame_count} frames, {fps} fps"
        )
    return frame_count / fps
