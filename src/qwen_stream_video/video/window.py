"""Causal sliding window generation and selection for local videos."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from .metadata import VideoMetadata

WindowType = Literal["regular", "tail_completion"]


class VideoWindow(BaseModel):
    """A single temporal window into a video.

    ``global_index`` is the window's position in the full video; ``run_index`` is
    its position after user-supplied time/window selection and is recalculated
    each run.
    """

    model_config = ConfigDict(extra="ignore")

    global_index: int = Field(ge=0)
    run_index: int = Field(ge=0)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    window_type: WindowType = "regular"

    @field_validator("end_seconds", mode="after")
    @classmethod
    def _end_after_start(cls, value: float, info: ValidationInfo) -> float:
        if value <= info.data.get("start_seconds", 0.0):
            raise ValueError("end_seconds must be greater than start_seconds")
        return value


def build_video_windows(
    metadata: VideoMetadata,
    window_seconds: float,
    stride_seconds: float,
) -> list[VideoWindow]:
    """Generate causal, ordered, non-empty video windows.

    The generated windows never exceed the video duration. When the video is
    shorter than ``window_seconds`` a single ``tail_completion`` window covers
    the whole video. When the final regular window leaves a gap at the end, a
    full-width ``tail_completion`` window ending at the video duration is added.
    """
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive")

    duration = metadata.duration_seconds
    if duration <= window_seconds:
        return [
            VideoWindow(
                global_index=0,
                run_index=0,
                start_seconds=0.0,
                end_seconds=duration,
                window_type="tail_completion",
            )
        ]

    windows: list[VideoWindow] = []
    start = 0.0
    index = 0
    while start + window_seconds <= duration + 1e-9:
        windows.append(
            VideoWindow(
                global_index=index,
                run_index=index,
                start_seconds=start,
                end_seconds=start + window_seconds,
                window_type="regular",
            )
        )
        start += stride_seconds
        index += 1

    last_end = windows[-1].end_seconds if windows else 0.0
    if last_end < duration - 1e-9:
        tail_start = max(0.0, duration - window_seconds)
        # Only add a tail window if it is not identical to the last regular one.
        if tail_start < last_end - 1e-9 or abs(tail_start - last_end) > 1e-9:
            windows.append(
                VideoWindow(
                    global_index=index,
                    run_index=index,
                    start_seconds=tail_start,
                    end_seconds=duration,
                    window_type="tail_completion",
                )
            )

    return windows


def select_windows(
    windows: list[VideoWindow],
    *,
    start_time: float | None = None,
    end_time: float | None = None,
    start_window: int | None = None,
    end_window: int | None = None,
    max_windows: int | None = None,
) -> list[VideoWindow]:
    """Select a subset of windows and recompute ``run_index``.

    Selection preserves ``global_index`` while recalculating ``run_index`` from
    zero so that downstream code sees a contiguous run.
    """
    selected = windows

    if start_time is not None:
        selected = [w for w in selected if w.end_seconds > start_time]
    if end_time is not None:
        selected = [w for w in selected if w.start_seconds < end_time]
    if start_window is not None:
        selected = [w for w in selected if w.global_index >= start_window]
    if end_window is not None:
        selected = [w for w in selected if w.global_index <= end_window]
    if max_windows is not None:
        selected = selected[:max_windows]

    return [
        w.model_copy(update={"run_index": run_index})
        for run_index, w in enumerate(selected)
    ]


def calculate_realtime_target(
    wall_start: float,
    video_origin: float,
    window_end: float,
) -> float:
    """Return the wall-clock time at which a window should be processed.

    The real-time origin is the start of the first selected window, not the
    beginning of the video. This avoids unnecessary waiting when the run starts
    from a non-zero ``start_time``.
    """
    return wall_start + window_end - video_origin
