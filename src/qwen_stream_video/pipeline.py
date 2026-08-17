"""Ordered Observation Generator -> deterministic State Reducer pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .config import AppConfig
from .domain import GlobalState
from .inference import ObservationNormalizer, PromptBuilder, ResponseParser
from .state import ContextBuilder, StateReducer
from .storage import RunStorage, StateStorage
from .video import (
    VideoMetadata,
    VideoWindow,
    build_video_windows,
    calculate_realtime_target,
    encode_frame_to_data_url,
    read_video_metadata,
    sample_window_frames,
    select_windows,
)

logger = logging.getLogger(__name__)


class StreamingVideoPipeline:
    """Process windows sequentially; state algorithms stay outside this class."""

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
        self.prompt_builder = prompt_builder or PromptBuilder(
            context_policy=config.observation.context_policy
        )
        self.parser = parser or ResponseParser()
        self.normalizer = ObservationNormalizer()
        self.context_builder = ContextBuilder(config.context)
        self.state_reducer = StateReducer(config)
        self.video_context = video_context or {}
        self._previous_summary: str | None = None
        self._previous_entities: list[dict[str, Any]] = []
        self._state: GlobalState | None = None
        self._state_storage: StateStorage | None = None
        self._state_enabled = config.state.enabled

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
        state_enabled: bool | None = None,
        warmup_windows: int | None = None,
        progress_reporter: Any | None = None,
    ) -> RunStorage | None:
        realtime = self.config.runtime.realtime if realtime is None else realtime
        carry_previous_state = (
            self.config.runtime.carry_previous_state
            if carry_previous_state is None
            else carry_previous_state
        )
        self._state_enabled = self.config.state.enabled if state_enabled is None else state_enabled
        max_windows = max_windows or self.config.runtime.max_windows
        metadata = read_video_metadata(self.video_path)
        all_windows = build_video_windows(
            metadata, self.config.video.window_seconds, self.config.video.stride_seconds
        )
        requested = select_windows(
            all_windows,
            start_time=start_time,
            end_time=end_time,
            start_window=start_window,
            end_window=end_window,
            max_windows=max_windows,
        )
        selected = self._add_warmup(all_windows, requested, warmup_windows)
        if validate_only:
            self._report_validation(metadata, selected)
            return None
        output_root = output_dir if output_dir is not None else self.config.storage.output_root
        storage = RunStorage(self._config_with_output_root(output_root), metadata, prompt_builder=self.prompt_builder)
        storage.initialize()
        storage.write_windows(selected)
        state_storage = StateStorage(storage.run_dir, self.config) if self._state_enabled else None
        if state_storage is not None:
            state_storage.initialize(prompt_builder=self.prompt_builder)
        self._state_storage = state_storage
        self._state = GlobalState(run_id=storage.run_id) if self._state_enabled else None
        storage.update_run_meta(
            {
                "requested_start_window": start_window,
                "requested_end_window": end_window,
                "warmup_windows": warmup_windows if warmup_windows is not None else self.config.video.warmup_windows,
                "warmup_start_window": selected[0].global_index if selected and selected[0].processing_role == "warmup" else None,
                "warmup_end_window": next((window.global_index - 1 for window in selected if window.processing_role == "commit"), None),
                "first_committed_window": next((window.global_index for window in selected if window.processing_role == "commit"), None),
                "last_committed_window": next((window.global_index for window in reversed(selected) if window.processing_role == "commit"), None),
                "cold_start": bool(
                    selected
                    and selected[0].processing_role == "commit"
                    and selected[0].global_index > 0
                ),
                "observation_schema_version": self.config.observation.schema_version,
                "state_schema_version": "2.0" if self._state_enabled else None,
                "state_enabled": self._state_enabled,
            }
        )
        wall_start = time.perf_counter()
        video_origin = selected[0].start_seconds if selected else 0.0
        try:
            for run_index, window in enumerate(selected):
                if progress_reporter is not None:
                    progress_reporter(window, len(selected), f"processing window {run_index}/{len(selected)}")
                if realtime:
                    target = calculate_realtime_target(wall_start, video_origin, window.end_seconds)
                    wait = target - time.perf_counter()
                    if wait > 0:
                        time.sleep(wait)
                self._process_window(
                    storage,
                    window,
                    metadata,
                    dry_run=dry_run,
                    carry_previous_state=carry_previous_state,
                )
        finally:
            if state_storage is not None:
                state_storage.finalize(self._state)
            storage.finalize(
                stats={
                    "state_enabled": self._state_enabled,
                    "state_last_committed_window": self._state.last_committed_window if self._state else None,
                }
            )
        return storage

    def _add_warmup(
        self,
        all_windows: list[VideoWindow],
        requested: list[VideoWindow],
        warmup_windows: int | None,
    ) -> list[VideoWindow]:
        count = self.config.video.warmup_windows if warmup_windows is None else warmup_windows
        if count <= 0 or not requested:
            return [
                window.model_copy(
                    update={
                        "run_index": i,
                        # A filtered run has no usable predecessor unless it
                        # explicitly prepended warmup windows.
                        "commit_start_seconds": window.start_seconds if i == 0 else window.commit_start_seconds,
                    }
                )
                for i, window in enumerate(requested)
            ]
        first = requested[0].global_index
        prior = [window for window in all_windows if window.global_index < first]
        warmup = prior[-count:]
        warmup_ids = {window.global_index for window in warmup}
        selected = warmup + requested
        result = [
            window.model_copy(
                update={
                    "run_index": index,
                    "processing_role": "warmup" if window.global_index in warmup_ids else "commit",
                }
            )
            for index, window in enumerate(selected)
        ]
        if not warmup and result:
            result[0] = result[0].model_copy(update={"commit_start_seconds": result[0].start_seconds})
        return result

    def _config_with_output_root(self, output_root: str | Path) -> AppConfig:
        updated_storage = self.config.storage.model_copy(update={"output_root": str(output_root)})
        return self.config.model_copy(update={"storage": updated_storage})

    def _process_window(
        self,
        storage: RunStorage,
        window: VideoWindow,
        metadata: VideoMetadata,
        *,
        dry_run: bool,
        carry_previous_state: bool,
    ) -> None:
        sampled_frames: list[Any] = []
        raw_result: Any | None = None
        try:
            sampled_frames = sample_window_frames(
                metadata,
                window,
                self.config.sampling.sample_fps,
                self.config.sampling.min_frames,
                self.config.sampling.max_frames,
            )
            context = (
                self.context_builder.build(self._state, window)
                if self._state_enabled and self._state is not None
                else None
            )
            if dry_run:
                for frame in sampled_frames:
                    encode_frame_to_data_url(
                        frame,
                        self.config.sampling.max_image_side,
                        self.config.sampling.jpeg_quality,
                    )
                storage.write_window_result(window, sampled_frames)
                return
            images = [
                encode_frame_to_data_url(
                    frame, self.config.sampling.max_image_side, self.config.sampling.jpeg_quality
                )
                for frame in sampled_frames
            ]
            user_prompt = self.prompt_builder.build_user_prompt(
                window,
                sampled_frames,
                video_context=self.video_context,
                previous_summary=self._previous_summary if carry_previous_state else None,
                previous_entities=self._previous_entities if carry_previous_state else None,
                context=context,
            )
            if self.client is None:
                raise RuntimeError("No inference client provided for the non-dry run")
            raw_result = self.client.infer(self.prompt_builder.system_prompt, user_prompt, images)
            batch, parser_warnings = self.parser.parse(raw_result.raw_text, sampled_frames, window=window)
            normalized = self.normalizer.normalize(batch)
            for warning in parser_warnings + [warning.message for warning in normalized.warnings]:
                logger.warning("Window %d: %s", window.global_index, warning)
            storage.write_window_result(window, sampled_frames, raw_result=raw_result, observation=normalized.batch)
            if self._state_enabled and self._state is not None and self._state_storage is not None:
                self._state_storage.write_normalization_warnings(window.global_index, normalized.warnings)
                reduction = self.state_reducer.apply_observation(
                    self._state, normalized.batch, sampled_frames, window
                )
                self._state = reduction.state
                self._state_storage.write_reduction(
                    reduction,
                    window_global_index=window.global_index,
                    warmup=window.processing_role == "warmup",
                )
            if carry_previous_state:
                self._previous_summary = normalized.batch.summary
                self._previous_entities = [
                    {
                        "candidate_global_id": entity.candidate_global_id,
                        "entity_type": entity.entity_type.value,
                        "description": entity.description or entity.name,
                    }
                    for entity in normalized.batch.entities
                    if entity.candidate_global_id is not None
                ]
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            logger.exception("Window %d failed", window.global_index)
            storage.write_window_result(window, sampled_frames, raw_result=raw_result, error=exc)
            if self._state_enabled and self._state is not None and self._state_storage is not None:
                gap = self.state_reducer.apply_observation_gap(self._state, window, reason=str(exc))
                self._state = gap.state
                self._state_storage.write_reduction(
                    gap, window_global_index=window.global_index, warmup=window.processing_role == "warmup"
                )
                self._state_storage.write_error(
                    window.global_index, stage="observation", error=exc, observation_succeeded=False
                )

    def _report_validation(self, metadata: VideoMetadata, windows: list[VideoWindow]) -> None:
        print(f"视频: {metadata.path}")
        print(f"  时长: {metadata.duration_seconds:.3f}s, FPS: {metadata.fps:.2f}")
        print(f"  窗口数: {len(windows)}")
        for window in windows:
            duration = window.end_seconds - window.start_seconds
            estimated = max(
                self.config.sampling.min_frames,
                min(int(duration * self.config.sampling.sample_fps), self.config.sampling.max_frames),
            )
            print(
                f"  window {window.global_index} ({window.start_seconds:.3f}s - {window.end_seconds:.3f}s): "
                f"commit from {window.commit_start_seconds:.3f}s, estimated {estimated} frames, {window.processing_role}"
            )
