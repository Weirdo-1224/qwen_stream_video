"""Causal sliding window generation and selection for local videos."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from .metadata import VideoMetadata

if TYPE_CHECKING:
    from .sampling import SampledFrame

WindowType = Literal["regular", "tail_completion"]


class VideoWindow(BaseModel):
    """A single temporal window into a video.

    ``global_index`` is the window's position in the full video; ``run_index`` is
    its position after user-supplied time/window selection and is recalculated
    each run.
    """

    model_config = ConfigDict(extra="forbid")

    global_index: int = Field(ge=0)
    run_index: int = Field(ge=0)
    start_seconds: float = Field(ge=0)
    commit_start_seconds: float | None = Field(default=None, ge=0)
    end_seconds: float = Field(gt=0)
    window_type: WindowType = "regular"
    processing_role: Literal["warmup", "commit"] = "commit"

    @field_validator("end_seconds", mode="after")
    @classmethod
    def _end_after_start(cls, value: float, info: ValidationInfo) -> float:
        if value <= info.data.get("start_seconds", 0.0):
            raise ValueError("end_seconds must be greater than start_seconds")
        return value

    @model_validator(mode="after")
    def _commit_interval_is_valid(self) -> VideoWindow:
        if self.commit_start_seconds is None:
            self.commit_start_seconds = self.start_seconds
        if not self.start_seconds <= self.commit_start_seconds < self.end_seconds:
            raise ValueError("start_seconds <= commit_start_seconds < end_seconds is required")
        return self


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
                commit_start_seconds=0.0,
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
                commit_start_seconds=start if not windows else max(start, windows[-1].end_seconds),
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
                    commit_start_seconds=(
                        tail_start if not windows else max(tail_start, windows[-1].end_seconds)
                    ),
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


def select_windows_with_warmup(
    windows: list[VideoWindow],
    *,
    start_window: int | None = None,
    end_window: int | None = None,
    warmup_windows: int = 0,
    max_windows: int | None = None,
) -> tuple[list[VideoWindow], bool, tuple[int, int] | None]:
    """Select a window range and prepend available windows as warmup.

    The return value is ``(windows, cold_start, warmup_range)``.  ``run_index``
    remains contiguous while ``global_index`` remains the source-video index.
    """
    if warmup_windows < 0:
        raise ValueError("warmup_windows must be non-negative")
    eligible = [w for w in windows if (start_window is None or w.global_index >= start_window)]
    eligible = [w for w in eligible if end_window is None or w.global_index <= end_window]
    if max_windows is not None:
        eligible = eligible[:max_windows]
    if not eligible:
        return [], start_window is not None, None
    first_index = eligible[0].global_index
    previous = [w for w in windows if w.global_index < first_index]
    warmup = previous[-warmup_windows:] if warmup_windows else []
    cold_start = bool(warmup_windows and len(warmup) < warmup_windows)
    selected = warmup + eligible
    warmup_ids = {w.global_index for w in warmup}
    result = [
        w.model_copy(
            update={
                "run_index": i,
                "processing_role": "warmup" if w.global_index in warmup_ids else "commit",
            }
        )
        for i, w in enumerate(selected)
    ]
    warmup_range = (warmup[0].global_index, warmup[-1].global_index) if warmup else None
    return result, cold_start, warmup_range


def evidence_timestamps(
    evidence_frames: list[int], sampled_frames: list[SampledFrame]
) -> list[float]:
    """Map sample indices to timestamps, rejecting invalid evidence."""
    by_index = {frame.sample_index: frame.timestamp_seconds for frame in sampled_frames}
    missing = sorted(set(evidence_frames) - set(by_index))
    if missing:
        raise ValueError(f"Evidence sample indices not found: {missing}")
    return [by_index[index] for index in sorted(set(evidence_frames))]


def evidence_intersects_commit_interval(
    evidence_frames: list[int], sampled_frames: list[SampledFrame], window: VideoWindow
) -> bool:
    """Return whether any evidence sample lies in the half-open commit interval."""
    return any(
        window.commit_start_seconds <= timestamp < window.end_seconds
        for timestamp in evidence_timestamps(evidence_frames, sampled_frames)
    )


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
