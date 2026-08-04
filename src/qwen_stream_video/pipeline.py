"""Streaming video pipeline for incremental observations."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .config import AppConfig
from .inference import PromptBuilder, ResponseParser
from .storage import RunStorage
from .video import (
    VideoMetadata,
    build_video_windows,
    calculate_realtime_target,
    encode_frame_to_data_url,
    read_video_metadata,
    sample_window_frames,
    select_windows,
)

logger = logging.getLogger(__name__)


class StreamingVideoPipeline:
    """Run the incremental observation pipeline over a local video."""

    def __init__(
        self,
        config: AppConfig,
        video_path: str | Path,
        client: Any | None = None,
        prompt_builder: PromptBuilder | None = None,
        parser: ResponseParser | None = None,
        video_context: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.video_path = Path(video_path)
        self.client = client
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.parser = parser or ResponseParser()
        self.video_context = video_context or {}
        self._previous_summary: str | None = None
        self._previous_entities: list[dict[str, Any]] = []

    def run(
        self,
        *,
        output_dir: str | Path | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        start_window: int | None = None,
        end_window: int | None = None,
        max_windows: int | None = None,
        dry_run: bool = False,
        validate_only: bool = False,
        realtime: bool | None = None,
        carry_previous_state: bool | None = None,
        progress_reporter: Any | None = None,
    ) -> RunStorage | None:
        """Run the pipeline.

        Args:
            output_dir: Override the configured output root.
            start_time: Skip windows ending before this time (seconds).
            end_time: Skip windows starting at or after this time (seconds).
            start_window: Skip windows with global_index below this.
            end_window: Skip windows with global_index above this.
            max_windows: Process at most this many selected windows.
            dry_run: Sample frames and build prompts without calling the model.
            validate_only: Check the video and report windows without sampling.
            realtime: Override the configured realtime flag.
            carry_previous_state: Override the configured state-carry flag.
            progress_reporter: Optional callable(window, total, message) for progress.

        Returns:
            The :class:`RunStorage` for the run, or ``None`` for validate-only runs.
        """
        realtime = self.config.runtime.realtime if realtime is None else realtime
        carry_previous_state = (
            self.config.runtime.carry_previous_state
            if carry_previous_state is None
            else carry_previous_state
        )
        max_windows = max_windows or self.config.runtime.max_windows

        metadata = read_video_metadata(self.video_path)
        windows = build_video_windows(
            metadata,
            self.config.video.window_seconds,
            self.config.video.stride_seconds,
        )
        selected = select_windows(
            windows,
            start_time=start_time,
            end_time=end_time,
            start_window=start_window,
            end_window=end_window,
            max_windows=max_windows,
        )

        if validate_only:
            self._report_validation(metadata, selected)
            return None

        output_root = output_dir if output_dir is not None else self.config.storage.output_root
        storage_config = self._config_with_output_root(output_root)
        storage = RunStorage(storage_config, metadata, prompt_builder=self.prompt_builder)
        storage.initialize()
        storage.write_windows(selected)

        wall_start = time.perf_counter()
        video_origin = selected[0].start_seconds if selected else 0.0

        for run_index, window in enumerate(selected):
            if progress_reporter is not None:
                progress_reporter(
                    window,
                    len(selected),
                    f"processing window {run_index}/{len(selected)}",
                )

            if realtime:
                target = calculate_realtime_target(
                    wall_start, video_origin, window.end_seconds
                )
                now = time.perf_counter()
                wait = target - now
                if wait > 0:
                    time.sleep(wait)

            self._process_window(
                storage,
                window,
                metadata,
                dry_run=dry_run,
                carry_previous_state=carry_previous_state,
            )

        storage.finalize()
        return storage

    def _config_with_output_root(self, output_root: str | Path) -> AppConfig:
        """Return a copy of the config with the resolved output root."""
        updated_storage = self.config.storage.model_copy(
            update={"output_root": str(output_root)}
        )
        return self.config.model_copy(update={"storage": updated_storage})

    def _process_window(
        self,
        storage: RunStorage,
        window: Any,
        metadata: VideoMetadata,
        *,
        dry_run: bool,
        carry_previous_state: bool,
    ) -> None:
        sampled_frames: list[Any] = []
        raw_result: Any | None = None
        observation: Any | None = None
        error: Exception | None = None

        try:
            sampled_frames = sample_window_frames(
                metadata,
                window,
                self.config.sampling.sample_fps,
                self.config.sampling.min_frames,
                self.config.sampling.max_frames,
            )

            if dry_run:
                # Exercise the encoding path without sending anything.
                for frame in sampled_frames:
                    encode_frame_to_data_url(
                        frame,
                        self.config.sampling.max_image_side,
                        self.config.sampling.jpeg_quality,
                    )
                storage.write_window_result(
                    window,
                    sampled_frames,
                    raw_result=None,
                    observation=None,
                    error=None,
                )
                return

            images = [
                encode_frame_to_data_url(
                    frame,
                    self.config.sampling.max_image_side,
                    self.config.sampling.jpeg_quality,
                )
                for frame in sampled_frames
            ]

            user_prompt = self.prompt_builder.build_user_prompt(
                window,
                sampled_frames,
                video_context=self.video_context,
                previous_summary=self._previous_summary if carry_previous_state else None,
                previous_entities=self._previous_entities if carry_previous_state else None,
            )

            if self.client is None:
                raise RuntimeError("No inference client provided for the non-dry run")

            raw_result = self.client.infer(
                self.prompt_builder.system_prompt,
                user_prompt,
                images,
            )

            batch, warnings = self.parser.parse(
                raw_result.raw_text,
                sampled_frames,
                window=window,
            )
            for warning in warnings:
                logger.warning("Window %d: %s", window.global_index, warning)

            observation = batch
            storage.write_window_result(
                window,
                sampled_frames,
                raw_result=raw_result,
                observation=observation,
                error=None,
            )

            if carry_previous_state and observation is not None:
                if observation.summary:
                    self._previous_summary = observation.summary
                self._previous_entities = [
                    {
                        "candidate_global_id": entity.candidate_global_id,
                        "entity_type": entity.entity_type.value,
                        "description": entity.description or entity.name,
                    }
                    for entity in observation.entities
                    if entity.candidate_global_id is not None
                ]

        except KeyboardInterrupt:
            raise
        except Exception as exc:
            error = exc
            logger.exception("Window %d failed", window.global_index)
            storage.write_window_result(
                window,
                sampled_frames,
                raw_result=raw_result,
                observation=None,
                error=error,
            )

    def _report_validation(
        self,
        metadata: VideoMetadata,
        windows: list[Any],
    ) -> None:
        """Print a human-readable validation report for the selected windows."""
        print(f"视频: {metadata.path}")
        print(f"  时长: {metadata.duration_seconds:.3f}s, FPS: {metadata.fps:.2f}")
        print(f"  窗口数: {len(windows)}")
        for window in windows:
            duration = window.end_seconds - window.start_seconds
            estimated = max(
                self.config.sampling.min_frames,
                min(
                    int(duration * self.config.sampling.sample_fps),
                    self.config.sampling.max_frames,
                ),
            )
            print(
                f"  window {window.global_index} "
                f"({window.start_seconds:.3f}s - {window.end_seconds:.3f}s): "
                f"预估 {estimated} 帧"
            )
