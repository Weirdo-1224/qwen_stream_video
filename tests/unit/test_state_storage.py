"""Unit tests for deterministic state persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from qwen_stream_video.config import AppConfig
from qwen_stream_video.domain import (
    EntityObservation,
    EntityResolutionBatch,
    EntityType,
    GlobalState,
    ObservationBatch,
    SceneObservation,
    StateDelta,
    StateEvent,
    WindowObservation,
)
from qwen_stream_video.inference import PromptBuilder
from qwen_stream_video.state import SceneTracker, StateReducer, StateReductionResult
from qwen_stream_video.storage import StateStorage
from qwen_stream_video.video import SampledFrame, VideoMetadata, VideoWindow


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@pytest.fixture
def base_config(tmp_path: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "storage": {
                "output_root": str(tmp_path),
                "save_entity_resolutions": True,
                "save_state_events": True,
                "save_state_deltas": True,
                "save_state_snapshots": True,
            },
            "state": {"enabled": True, "snapshot_interval_windows": 1},
        }
    )


@pytest.fixture
def state_storage(tmp_path: Path, base_config: AppConfig) -> StateStorage:
    return StateStorage(tmp_path / "run", base_config)


@pytest.fixture
def sample_reduction() -> StateReductionResult:
    state = GlobalState(run_id="sample")
    return StateReductionResult(
        state=state,
        resolution=EntityResolutionBatch(window_global_index=0),
        events=[
            StateEvent(
                event_id="event_000001",
                event_type="test_event",
                window_global_index=0,
                scene_id="scene_0001",
            )
        ],
        delta=StateDelta(
            window_global_index=0,
            scene_id="scene_0001",
            entity_updates=["entity_0001"],
        ),
    )



def test_save_switches_control_file_generation(
    tmp_path: Path, base_config: AppConfig, sample_reduction: StateReductionResult
) -> None:
    """When a save_* flag is false, the corresponding JSONL must not be created."""
    storage = StateStorage(tmp_path / "run", base_config)
    storage.initialize(prompt_builder=PromptBuilder())
    storage.write_reduction(sample_reduction, window_global_index=0, warmup=False)
    storage.finalize(sample_reduction.state)

    run_dir = storage.run_dir
    assert (run_dir / "entity_resolutions.jsonl").exists()
    assert (run_dir / "state_events.jsonl").exists()
    assert (run_dir / "state_deltas.jsonl").exists()
    assert (run_dir / "state_snapshots.jsonl").exists()
    assert (run_dir / "state_errors.jsonl").exists()
    assert (run_dir / "normalization_warnings.jsonl").exists()


def test_save_entity_resolutions_false_suppresses_file(
    tmp_path: Path, base_config: AppConfig, sample_reduction: StateReductionResult
) -> None:
    base_config.storage.save_entity_resolutions = False
    storage = StateStorage(tmp_path / "run", base_config)
    storage.initialize(prompt_builder=PromptBuilder())
    storage.write_reduction(sample_reduction, window_global_index=0, warmup=False)
    storage.finalize(sample_reduction.state)

    assert not (storage.run_dir / "entity_resolutions.jsonl").exists()
    assert (storage.run_dir / "state_events.jsonl").exists()


def test_save_state_events_false_still_allows_errors(
    tmp_path: Path, base_config: AppConfig, sample_reduction: StateReductionResult
) -> None:
    base_config.storage.save_state_events = False
    storage = StateStorage(tmp_path / "run", base_config)
    storage.initialize(prompt_builder=PromptBuilder())
    storage.write_reduction(sample_reduction, window_global_index=0, warmup=False)
    storage.finalize(sample_reduction.state)

    assert not (storage.run_dir / "state_events.jsonl").exists()
    assert (storage.run_dir / "state_errors.jsonl").exists()


def test_save_state_deltas_false_suppresses_file(
    tmp_path: Path, base_config: AppConfig, sample_reduction: StateReductionResult
) -> None:
    base_config.storage.save_state_deltas = False
    storage = StateStorage(tmp_path / "run", base_config)
    storage.initialize(prompt_builder=PromptBuilder())
    storage.write_reduction(sample_reduction, window_global_index=0, warmup=False)
    storage.finalize(sample_reduction.state)

    assert not (storage.run_dir / "state_deltas.jsonl").exists()
    assert (storage.run_dir / "state_events.jsonl").exists()


def test_save_state_snapshots_false_still_writes_final_snapshot(
    tmp_path: Path, base_config: AppConfig, sample_reduction: StateReductionResult
) -> None:
    """Even with periodic snapshots disabled, the final state must be snapshotted."""
    base_config.storage.save_state_snapshots = False
    storage = StateStorage(tmp_path / "run", base_config)
    storage.initialize(prompt_builder=PromptBuilder())
    storage.write_reduction(sample_reduction, window_global_index=0, warmup=False)
    storage.finalize(sample_reduction.state)

    snapshots = _read_jsonl(storage.run_dir / "state_snapshots.jsonl")
    assert len(snapshots) == 1
    final = json.loads((storage.run_dir / "final_state.json").read_text(encoding="utf-8"))
    assert snapshots[0] == final


def test_warmup_writes_resolutions_not_events_or_deltas(
    tmp_path: Path, base_config: AppConfig
) -> None:
    """Warmup windows persist entity resolutions when enabled but never formal events."""
    storage = StateStorage(tmp_path / "run", base_config)
    storage.initialize(prompt_builder=PromptBuilder())
    state = GlobalState(run_id="warmup")
    result = StateReductionResult(
        state=state,
        resolution=EntityResolutionBatch(window_global_index=0),
        events=[
            StateEvent(
                event_id="event_000001",
                event_type="warmup_event",
                window_global_index=0,
                scene_id="scene_0001",
            )
        ],
        delta=StateDelta(window_global_index=0, scene_id="scene_0001"),
    )
    storage.write_reduction(result, window_global_index=0, warmup=True)
    storage.finalize(result.state)

    assert len(_read_jsonl(storage.run_dir / "entity_resolutions.jsonl")) == 1
    assert len(_read_jsonl(storage.run_dir / "state_events.jsonl")) == 0
    assert len(_read_jsonl(storage.run_dir / "state_deltas.jsonl")) == 0


def test_warmup_suppresses_entity_resolutions_when_disabled(
    tmp_path: Path, base_config: AppConfig
) -> None:
    base_config.storage.save_entity_resolutions = False
    storage = StateStorage(tmp_path / "run", base_config)
    storage.initialize(prompt_builder=PromptBuilder())
    state = GlobalState(run_id="warmup")
    result = StateReductionResult(
        state=state,
        resolution=EntityResolutionBatch(window_global_index=0),
    )
    storage.write_reduction(result, window_global_index=0, warmup=True)
    storage.finalize(result.state)

    assert not (storage.run_dir / "entity_resolutions.jsonl").exists()


class _RaisingSceneTracker(SceneTracker):
    """SceneTracker that always fails, forcing the StateReducer to roll back."""

    def update(self, state: GlobalState, observation: ObservationBatch) -> Any:
        raise RuntimeError("forced scene failure")


def _observation_for_window(index: int) -> ObservationBatch:
    return ObservationBatch(
        window=WindowObservation(
            global_index=index,
            start_seconds=index * 3.0,
            commit_start_seconds=index * 3.0,
            end_seconds=index * 3.0 + 3.0,
        ),
        scene=SceneObservation(),
        entities=[
            EntityObservation(
                local_id="P",
                entity_type=EntityType.PERSON,
                name="worker",
                confidence=0.9,
                evidence_frames=[0],
            )
        ],
    )


def _frames_for_window(index: int) -> list[SampledFrame]:
    return [
        SampledFrame(
            run_index=index,
            global_index=index,
            sample_index=0,
            frame_index=0,
            timestamp_seconds=index * 3.0,
            image=np.zeros((4, 4, 3), dtype="uint8"),
        )
    ]


def test_state_error_state_affected_false_on_reducer_rollback(
    tmp_path: Path, base_config: AppConfig
) -> None:
    """When the StateReducer catches an exception, the original state is preserved."""
    base_config.state.fail_on_state_error = False
    reducer = StateReducer(base_config, scene_tracker=_RaisingSceneTracker())
    state = GlobalState(run_id="rollback")
    window = VideoWindow(global_index=0, run_index=0, start_seconds=0.0, end_seconds=3.0)
    result = reducer.apply_observation(
        state, _observation_for_window(0), _frames_for_window(0), window
    )
    assert result.error is not None
    assert result.state.model_dump(mode="json") == state.model_dump(mode="json")

    storage = StateStorage(tmp_path / "run", base_config)
    storage.initialize(prompt_builder=PromptBuilder())
    storage.write_reduction(result, window_global_index=0, warmup=False)
    storage.finalize(result.state)

    errors = _read_jsonl(storage.run_dir / "state_errors.jsonl")
    assert len(errors) == 1
    assert errors[0]["stage"] == "state_reducer"
    assert errors[0]["state_affected"] is False


def test_successful_reduction_produces_no_state_error(
    tmp_path: Path, base_config: AppConfig
) -> None:
    reducer = StateReducer(base_config)
    state = GlobalState(run_id="success")
    window = VideoWindow(global_index=0, run_index=0, start_seconds=0.0, end_seconds=3.0)
    result = reducer.apply_observation(
        state, _observation_for_window(0), _frames_for_window(0), window
    )
    assert result.error is None

    storage = StateStorage(tmp_path / "run", base_config)
    storage.initialize(prompt_builder=PromptBuilder())
    storage.write_reduction(result, window_global_index=0, warmup=False)
    storage.finalize(result.state)

    assert len(_read_jsonl(storage.run_dir / "state_errors.jsonl")) == 0


def test_final_state_matches_last_snapshot(
    tmp_path: Path, base_config: AppConfig
) -> None:
    storage = StateStorage(tmp_path / "run", base_config)
    storage.initialize(prompt_builder=PromptBuilder())

    state1 = GlobalState(run_id="final")
    storage.write_reduction(
        StateReductionResult(
            state=state1,
            events=[
                StateEvent(
                    event_id="event_000001",
                    event_type="test_event",
                    window_global_index=0,
                    scene_id="scene_0001",
                )
            ],
            delta=StateDelta(window_global_index=0, scene_id="scene_0001"),
        ),
        window_global_index=0,
        warmup=False,
    )

    state2 = state1.model_copy(update={"current_scene_id": "scene_0001"})
    storage.write_reduction(
        StateReductionResult(
            state=state2,
            events=[
                StateEvent(
                    event_id="event_000002",
                    event_type="test_event",
                    window_global_index=1,
                    scene_id="scene_0001",
                )
            ],
            delta=StateDelta(window_global_index=1, scene_id="scene_0001"),
        ),
        window_global_index=1,
        warmup=False,
    )
    storage.finalize(state2)

    final_path = storage.run_dir / "final_state.json"
    assert final_path.exists()
    final_state = json.loads(final_path.read_text(encoding="utf-8"))
    snapshots = _read_jsonl(storage.run_dir / "state_snapshots.jsonl")
    assert snapshots[-1] == final_state
    assert final_state["current_scene_id"] == "scene_0001"


def test_artifacts_contain_prompts_schema_and_vocabularies(
    tmp_path: Path, base_config: AppConfig
) -> None:
    storage = StateStorage(tmp_path / "run", base_config)
    prompt_builder = PromptBuilder(
        system_prompt="system prompt body",
        user_prompt_template="user prompt body",
    )
    storage.initialize(prompt_builder=prompt_builder)

    artifacts = storage.run_dir / "artifacts"
    saved_system_prompt = (artifacts / "prompts" / "observation_system.txt").read_text(
        encoding="utf-8"
    )
    assert saved_system_prompt.startswith("system prompt body")
    assert "Observation Schema 2.0" in saved_system_prompt
    assert (artifacts / "prompts" / "observation_user.txt").read_text(
        encoding="utf-8"
    ) == "user prompt body"

    schema_text = (artifacts / "schemas" / "observation_v2.schema.json").read_text(
        encoding="utf-8"
    )
    schema = json.loads(schema_text)
    assert "$defs" in schema or "title" in schema

    for filename in ("actions.yaml", "attributes.yaml", "entity_types.yaml"):
        vocab_path = artifacts / "vocabularies" / filename
        assert vocab_path.exists()
        assert vocab_path.read_text(encoding="utf-8")


def test_run_meta_records_covered_seconds_and_warmup(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Pipeline writes accurate covered/warmup metadata even with warmup windows."""
    from qwen_stream_video.pipeline import StreamingVideoPipeline

    config = AppConfig.model_validate(
        {
            "storage": {
                "output_root": str(tmp_path),
                "save_raw_responses": False,
                "save_sampled_frames": False,
            },
            "video": {"window_seconds": 6.0, "stride_seconds": 3.0, "warmup_windows": 2},
            "state": {"enabled": False},
            "sampling": {"sample_fps": 1.0, "min_frames": 1, "max_frames": 1},
        }
    )

    video_path = tmp_path / "fake.mp4"
    metadata = VideoMetadata(
        path=str(video_path),
        fps=10.0,
        frame_count=150,
        duration_seconds=15.0,
        width=640,
        height=480,
    )
    monkeypatch.setattr(
        "qwen_stream_video.pipeline.read_video_metadata", lambda _path: metadata
    )

    def fake_sample(_metadata: VideoMetadata, window: VideoWindow, *args: Any, **kwargs: Any) -> list[SampledFrame]:
        return [
            SampledFrame(
                run_index=window.run_index,
                global_index=window.global_index,
                sample_index=0,
                frame_index=0,
                timestamp_seconds=window.start_seconds,
                image=np.zeros((8, 8, 3), dtype="uint8") + window.global_index,
            )
        ]

    monkeypatch.setattr("qwen_stream_video.pipeline.sample_window_frames", fake_sample)

    pipeline = StreamingVideoPipeline(config, video_path=video_path)
    storage = pipeline.run(dry_run=True, start_window=2, max_windows=3)
    assert storage is not None

    meta = json.loads((storage.run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["requested_start_window"] == 2
    assert meta["requested_end_window"] is None
    assert meta["warmup_start_window"] == 0
    assert meta["warmup_end_window"] == 1
    assert meta["first_committed_window"] == 2
    assert meta["last_committed_window"] == 3
    # covered_* must describe the formal commit range, not the warmup prefix.
    assert meta["covered_start_seconds"] == 6.0
    assert meta["covered_end_seconds"] == 15.0
    assert meta["cold_start"] is False
    assert meta["observation_schema_version"] == config.observation.schema_version
    assert meta["state_enabled"] is False
    assert meta["state_schema_version"] is None
    assert "api_key" not in meta


def test_run_meta_cold_start_when_warmup_prefix_is_incomplete(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from qwen_stream_video.pipeline import StreamingVideoPipeline

    config = AppConfig.model_validate(
        {
            "storage": {
                "output_root": str(tmp_path),
                "save_raw_responses": False,
                "save_sampled_frames": False,
            },
            "video": {"window_seconds": 6.0, "stride_seconds": 3.0, "warmup_windows": 2},
            "state": {"enabled": False},
            "sampling": {"sample_fps": 1.0, "min_frames": 1, "max_frames": 1},
        }
    )

    video_path = tmp_path / "fake.mp4"
    metadata = VideoMetadata(
        path=str(video_path),
        fps=10.0,
        frame_count=150,
        duration_seconds=15.0,
        width=640,
        height=480,
    )
    monkeypatch.setattr(
        "qwen_stream_video.pipeline.read_video_metadata", lambda _path: metadata
    )

    def fake_sample(_metadata: VideoMetadata, window: VideoWindow, *args: Any, **kwargs: Any) -> list[SampledFrame]:
        return [
            SampledFrame(
                run_index=window.run_index,
                global_index=window.global_index,
                sample_index=0,
                frame_index=0,
                timestamp_seconds=window.start_seconds,
                image=np.zeros((8, 8, 3), dtype="uint8") + window.global_index,
            )
        ]

    monkeypatch.setattr("qwen_stream_video.pipeline.sample_window_frames", fake_sample)

    pipeline = StreamingVideoPipeline(config, video_path=video_path)
    # Start at window 1: only one prior window is available, less than warmup_windows=2.
    storage = pipeline.run(dry_run=True, start_window=1, max_windows=3)
    assert storage is not None

    meta = json.loads((storage.run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["warmup_start_window"] == 0
    assert meta["warmup_end_window"] == 0
    assert meta["first_committed_window"] == 1
    # Formal coverage starts at the first committed window, excluding warmup.
    assert meta["covered_start_seconds"] == 3.0
    assert meta["covered_end_seconds"] == 15.0
    assert meta["cold_start"] is True
