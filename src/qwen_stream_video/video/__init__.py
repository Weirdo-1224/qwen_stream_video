"""Video processing components for qwen-stream-video."""

from .metadata import VideoMetadata, read_video_metadata
from .sampling import SampledFrame, encode_frame_to_data_url, sample_window_frames
from .window import (
    VideoWindow,
    WindowType,
    build_video_windows,
    calculate_realtime_target,
    select_windows,
)

__all__ = [
    "SampledFrame",
    "VideoMetadata",
    "VideoWindow",
    "WindowType",
    "build_video_windows",
    "calculate_realtime_target",
    "encode_frame_to_data_url",
    "read_video_metadata",
    "sample_window_frames",
    "select_windows",
]
