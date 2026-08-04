"""Unit tests for the run result storage layer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from qwen_stream_video.config import AppConfig
from qwen_stream_video.domain import (
    Action,
    Entity,
    EntityType,
    SceneObservation,
    WindowObservation,
)
from qwen_stream_video.inference import RawInferenceResult
from qwen_stream_video.storage import RunStorage
from qwen_stream_video.video import SampledFrame, VideoMetadata, VideoWindow


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "experiment": {"name": "test_storage", "seed": 7},
            "storage": {"output_root": str(tmp_path), "save_raw_responses": True},
            "model": {"api_key": "secret-key", "name": "test-model"},
        }
    )


@pytest.fixture
def video_metadata() -> VideoMetadata:
    return VideoMetadata(
        path="/fake/video.mp4",
        fps=30.0,
        frame_count=300,
        duration_seconds=10.0,
        width=1920,
        height=1080,
    )


@pytest.fixture
def video_window() -> VideoWindow:
    return VideoWindow(
        global_index=3,
        run_index=1,
        start_seconds=6.0,
        end_seconds=12.0,
    )


@pytest.fixture
def sampled_frames(video_window: VideoWindow) -> list[SampledFrame]:
    return [
        SampledFrame(
            run_index=video_window.run_index,
            global_index=video_window.global_index,
            timestamp=video_window.start_seconds + i,
            frame_index=i,
            image=np.zeros((8, 8, 3), dtype=np.uint8),
        )
        for i in range(4)
    ]


@pytest.fixture
def raw_result() -> RawInferenceResult:
    return RawInferenceResult(
        raw_text='{"observations": []}',
        model="test-model",
        latency_seconds=1.23,
        request_id="req-123",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        attempts=1,
    )


@pytest.fixture
def window_observation(video_window: VideoWindow) -> WindowObservation:
    entity = Entity(
        local_id="e1",
        entity_type=EntityType.OBJECT,
        label="tool",
        confidence=0.9,
    )
    action = Action(
        local_id="a1",
        actor_id="e1",
        action_type="unknown",
        confidence=0.8,
        evidence_frame_sample_indices=[0, 1],
    )
    return WindowObservation(
        schema_version="1.0",
        window_run_index=video_window.run_index,
        window_global_index=video_window.global_index,
        window_start_seconds=video_window.start_seconds,
        window_end_seconds=video_window.end_seconds,
        scene=SceneObservation(description="test scene"),
        entities=[entity],
        actions=[action],
    )


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_unique_run_directories(app_config: AppConfig, video_metadata: VideoMetadata) -> None:
    """Two runs with the same experiment must not share a directory."""
    storage1 = RunStorage(app_config, video_metadata)
    storage2 = RunStorage(app_config, video_metadata)
    assert storage1.run_id != storage2.run_id
    assert storage1.run_dir != storage2.run_dir
    assert storage1.run_dir.exists()
    assert storage2.run_dir.exists()


def test_run_directory_collision(app_config: AppConfig, video_metadata: VideoMetadata) -> None:
    """Reusing an explicit run id must raise before any files are overwritten."""
    storage = RunStorage(app_config, video_metadata)
    storage.initialize()
    with pytest.raises(FileExistsError):
        RunStorage(app_config, video_metadata, run_id=storage.run_id)


def test_jsonl_outputs(
    app_config: AppConfig,
    video_metadata: VideoMetadata,
    video_window: VideoWindow,
    sampled_frames: list[SampledFrame],
    raw_result: RawInferenceResult,
    window_observation: WindowObservation,
) -> None:
    """Windows, observations, metrics and errors are written as JSONL."""
    storage = RunStorage(app_config, video_metadata)
    storage.initialize()
    storage.write_windows([video_window])
    storage.write_window_result(
        window=video_window,
        sampled_frames=sampled_frames,
        raw_result=raw_result,
        observation=window_observation,
    )
    error_window = VideoWindow(
        global_index=4, run_index=2, start_seconds=12.0, end_seconds=18.0
    )
    storage.write_window_result(
        window=error_window,
        sampled_frames=[],
        error=ValueError("sample failure"),
    )
    storage.finalize()

    windows = _read_jsonl(storage.run_dir / "windows.jsonl")
    assert len(windows) == 1
    assert windows[0]["global_index"] == video_window.global_index

    observations = _read_jsonl(storage.run_dir / "observations.jsonl")
    assert len(observations) == 1
    assert observations[0]["window_global_index"] == video_window.global_index

    metrics = _read_jsonl(storage.run_dir / "metrics.jsonl")
    assert len(metrics) == 1
    assert metrics[0]["window_run_index"] == video_window.run_index
    assert metrics[0]["model"] == raw_result.model
    assert metrics[0]["latency_seconds"] == raw_result.latency_seconds

    errors = _read_jsonl(storage.run_dir / "errors.jsonl")
    assert len(errors) == 1
    assert errors[0]["window_run_index"] == error_window.run_index
    assert errors[0]["error_type"] == "ValueError"
    assert "sample failure" in errors[0]["error_message"]

    metadata = json.loads((storage.run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["run_id"] == storage.run_id
    assert metadata["final_stats"]["window_count"] == 2
    assert metadata["final_stats"]["observation_count"] == 1
    assert metadata["final_stats"]["error_count"] == 1


def test_raw_response_saved(
    app_config: AppConfig,
    video_metadata: VideoMetadata,
    video_window: VideoWindow,
    sampled_frames: list[SampledFrame],
    raw_result: RawInferenceResult,
) -> None:
    """Raw responses are saved to disk and referenced in metrics and errors."""
    storage = RunStorage(app_config, video_metadata)
    storage.initialize()
    storage.write_window_result(
        window=video_window,
        sampled_frames=sampled_frames,
        raw_result=raw_result,
        error=RuntimeError("parse failed"),
    )
    storage.finalize()

    raw_files = list(storage._raw_dir.iterdir())
    assert len(raw_files) == 1
    raw_file = raw_files[0]
    assert raw_file.read_text(encoding="utf-8") == raw_result.raw_text

    metrics = _read_jsonl(storage.run_dir / "metrics.jsonl")
    assert metrics[0]["raw_response_path"] == f"raw_responses/{raw_file.name}"

    errors = _read_jsonl(storage.run_dir / "errors.jsonl")
    assert errors[0]["raw_response_path"] == f"raw_responses/{raw_file.name}"


def test_config_secrets_redacted(
    app_config: AppConfig, video_metadata: VideoMetadata
) -> None:
    """The written configuration must not contain the API key."""
    storage = RunStorage(app_config, video_metadata)
    storage.initialize()
    config_path = storage.run_dir / "config.json"
    written = json.loads(config_path.read_text(encoding="utf-8"))
    assert written["model"]["api_key"] == "***REDACTED***"


def test_save_sampled_frames(
    app_config: AppConfig,
    video_metadata: VideoMetadata,
    video_window: VideoWindow,
    sampled_frames: list[SampledFrame],
) -> None:
    """Sampled frames are persisted when explicitly requested."""
    storage = RunStorage(app_config, video_metadata)
    storage.initialize()
    storage.write_window_result(
        window=video_window,
        sampled_frames=sampled_frames,
        save_frames=True,
    )
    storage.finalize()

    frame_dirs = [p for p in storage._frame_dir.iterdir() if p.is_dir()]
    assert len(frame_dirs) == 1
    saved_frames = list(frame_dirs[0].glob("*.jpg"))
    assert len(saved_frames) == len(sampled_frames)


def test_observation_without_error_only_writes_observation(
    app_config: AppConfig,
    video_metadata: VideoMetadata,
    video_window: VideoWindow,
    sampled_frames: list[SampledFrame],
    window_observation: WindowObservation,
) -> None:
    """A successful window writes an observation but no error record."""
    storage = RunStorage(app_config, video_metadata)
    storage.initialize()
    storage.write_window_result(
        window=video_window,
        sampled_frames=sampled_frames,
        observation=window_observation,
    )
    storage.finalize()

    assert len(_read_jsonl(storage.run_dir / "observations.jsonl")) == 1
    assert len(_read_jsonl(storage.run_dir / "errors.jsonl")) == 0
