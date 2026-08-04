"""Frame sampling from video windows and JPEG/Base64 encoding."""

from __future__ import annotations

import base64

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..exceptions import FrameReadError, VideoOpenError
from .metadata import VideoMetadata
from .window import VideoWindow


class SampledFrame(BaseModel):
    """A single real frame sampled from a video window.

    The raw ``image`` is kept as a NumPy array for downstream use and is
    excluded from serialization. The ``sample_index`` is the position within
    the sampled frames for this window; ``frame_index`` is the approximate
    original video frame number.
    """

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    run_index: int = Field(ge=0)
    global_index: int = Field(ge=0)
    sample_index: int = Field(ge=0)
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    encoded_image: str | None = None
    image: np.ndarray = Field(exclude=True)


_FrameInput = SampledFrame | np.ndarray


def sample_window_frames(
    metadata: VideoMetadata,
    window: VideoWindow,
    sample_fps: float,
    min_frames: int,
    max_frames: int,
) -> list[SampledFrame]:
    """Sample real, distinct frames from ``window`` without crossing its end.

    The number of frames is derived from the window duration and
    ``sample_fps``, then clamped to ``[min_frames, max_frames]``. Frames are
    positioned uniformly so that every timestamp satisfies
    ``start <= timestamp < end``. If the video cannot provide enough distinct
    real frames, a :class:`FrameReadError` is raised.

    Args:
        metadata: Video metadata including the file path.
        window: Temporal window to sample.
        sample_fps: Target number of frames per second within the window.
        min_frames: Minimum frames required.
        max_frames: Maximum frames allowed.

    Returns:
        A list of sampled frames ordered by timestamp.

    Raises:
        ValueError: If any argument is invalid.
        VideoOpenError: If the video file cannot be opened.
        FrameReadError: If a frame cannot be read or distinct frames are
            insufficient.
    """
    duration = window.end_seconds - window.start_seconds
    if duration <= 0:
        raise ValueError("window duration must be positive")
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    if min_frames < 1:
        raise ValueError("min_frames must be at least 1")
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")
    if min_frames > max_frames:
        raise ValueError("min_frames must not exceed max_frames")

    raw_count = int(duration * sample_fps)
    count = max(min_frames, min(raw_count, max_frames))

    step = duration / count
    timestamps = [window.start_seconds + i * step for i in range(count)]

    capture = cv2.VideoCapture(metadata.path)
    if not capture.isOpened():
        raise VideoOpenError(f"Cannot open video file: {metadata.path}")

    frames: list[SampledFrame] = []
    try:
        for i, timestamp in enumerate(timestamps):
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            success, image = capture.read()
            if not success or image is None:
                raise FrameReadError(
                    f"Failed to read frame at {timestamp:.3f}s "
                    f"for window {window.global_index}"
                )

            if _is_duplicate(image, frames):
                raise FrameReadError(
                    f"Insufficient distinct frames in window {window.global_index}: "
                    f"needed {count}, got only {len(frames)} unique frames"
                )

            frame_index = round(timestamp * metadata.fps)
            frames.append(
                SampledFrame(
                    run_index=window.run_index,
                    global_index=window.global_index,
                    sample_index=i,
                    frame_index=frame_index,
                    timestamp_seconds=timestamp,
                    image=image,
                )
            )
    finally:
        capture.release()

    return frames


def _is_duplicate(image: np.ndarray, frames: list[SampledFrame]) -> bool:
    """Return True if ``image`` is identical to any already sampled frame."""
    for existing in frames:
        if image.shape == existing.image.shape and np.array_equal(image, existing.image):
            return True
    return False


def encode_frame_to_data_url(
    frame: _FrameInput,
    max_image_side: int,
    jpeg_quality: int,
) -> str:
    """Encode a frame as a JPEG Data URL.

    The image is resized proportionally only when its largest side exceeds
    ``max_image_side``; smaller images are never upscaled.

    Args:
        frame: Either a :class:`SampledFrame` or a raw ``np.ndarray`` image.
        max_image_side: Maximum allowed width or height in pixels.
        jpeg_quality: JPEG quality factor (1-100).

    Returns:
        A ``data:image/jpeg;base64,...`` string.

    Raises:
        ValueError: If encoding parameters are invalid.
        FrameReadError: If JPEG encoding fails.
    """
    if max_image_side <= 0:
        raise ValueError("max_image_side must be positive")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")

    image = frame.image if isinstance(frame, SampledFrame) else frame
    if not isinstance(image, np.ndarray):
        raise TypeError("frame must contain a numpy image array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a BGR image with shape (H, W, 3)")

    height, width = image.shape[:2]
    max_side = max(width, height)
    if max_side > max_image_side:
        scale = max_image_side / max_side
        new_width = round(width * scale)
        new_height = round(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
    )
    if not success:
        raise FrameReadError("Failed to encode frame to JPEG")

    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"
