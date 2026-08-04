"""Unit tests for the observation semantic validator."""

from __future__ import annotations

import pytest

from qwen_stream_video.domain import (
    Action,
    ActionPhase,
    Entity,
    EntityType,
    ObservationBatch,
    SceneObservation,
    WindowObservation,
)
from qwen_stream_video.exceptions import ModelOutputSemanticError
from qwen_stream_video.inference import ObservationSemanticValidator
from qwen_stream_video.video import SampledFrame, VideoWindow


@pytest.fixture
def validator() -> ObservationSemanticValidator:
    return ObservationSemanticValidator()


@pytest.fixture
def video_window() -> VideoWindow:
    return VideoWindow(
        global_index=5,
        run_index=1,
        start_seconds=10.0,
        end_seconds=16.0,
    )


@pytest.fixture
def sampled_frames() -> list[SampledFrame]:
    return [
        SampledFrame(
            run_index=1,
            global_index=5,
            timestamp=10.0 + i,
            frame_index=i,
            image=__import__("numpy").zeros((10, 10, 3), dtype="uint8"),
        )
        for i in range(4)
    ]


@pytest.fixture
def valid_observation() -> WindowObservation:
    return WindowObservation(
        window_run_index=0,
        window_global_index=0,
        window_start_seconds=0.0,
        window_end_seconds=3.0,
        scene=SceneObservation(description="test"),
        entities=[
            Entity(
                local_id="E1",
                entity_type=EntityType.PERSON,
                label="technician",
                confidence=0.9,
            ),
            Entity(
                local_id="E2",
                entity_type=EntityType.EQUIPMENT,
                label="breaker",
                confidence=0.8,
            ),
        ],
        actions=[
            Action(
                local_id="A1",
                actor_id="E1",
                action_type="touch",
                phase=ActionPhase.CONTINUE,
                target_id="E2",
                evidence_frame_sample_indices=[0, 1],
                confidence=0.85,
            ),
        ],
    )


def test_valid_batch_passes(
    validator: ObservationSemanticValidator,
    valid_observation: WindowObservation,
    sampled_frames: list[SampledFrame],
) -> None:
    batch = ObservationBatch(observations=[valid_observation])
    warnings = validator.validate(batch, sampled_frames)
    assert warnings == []


def test_window_fields_overwritten(
    validator: ObservationSemanticValidator,
    valid_observation: WindowObservation,
    sampled_frames: list[SampledFrame],
    video_window: VideoWindow,
) -> None:
    batch = ObservationBatch(observations=[valid_observation])
    validator.validate(batch, sampled_frames, window=video_window)
    obs = batch.observations[0]
    assert obs.window_global_index == 5
    assert obs.window_run_index == 1
    assert obs.window_start_seconds == 10.0
    assert obs.window_end_seconds == 16.0


def test_duplicate_entity_id_raises(
    validator: ObservationSemanticValidator,
    valid_observation: WindowObservation,
    sampled_frames: list[SampledFrame],
) -> None:
    valid_observation.entities[1].local_id = "E1"
    batch = ObservationBatch(observations=[valid_observation])
    with pytest.raises(ModelOutputSemanticError, match="Duplicate entity local_id"):
        validator.validate(batch, sampled_frames)


def test_duplicate_action_id_raises(
    validator: ObservationSemanticValidator,
    valid_observation: WindowObservation,
    sampled_frames: list[SampledFrame],
) -> None:
    valid_observation.actions.append(
        Action(
            local_id="A1",
            actor_id="E1",
            action_type="hold",
            confidence=0.6,
        )
    )
    batch = ObservationBatch(observations=[valid_observation])
    with pytest.raises(ModelOutputSemanticError, match="Duplicate action local_id"):
        validator.validate(batch, sampled_frames)


def test_missing_actor_reference_raises(
    validator: ObservationSemanticValidator,
    valid_observation: WindowObservation,
    sampled_frames: list[SampledFrame],
) -> None:
    valid_observation.actions[0].actor_id = "E99"
    batch = ObservationBatch(observations=[valid_observation])
    with pytest.raises(ModelOutputSemanticError, match="missing actor"):
        validator.validate(batch, sampled_frames)


def test_missing_target_reference_raises(
    validator: ObservationSemanticValidator,
    valid_observation: WindowObservation,
    sampled_frames: list[SampledFrame],
) -> None:
    valid_observation.actions[0].target_id = "E99"
    batch = ObservationBatch(observations=[valid_observation])
    with pytest.raises(ModelOutputSemanticError, match="missing target"):
        validator.validate(batch, sampled_frames)


def test_out_of_range_evidence_frame_raises(
    validator: ObservationSemanticValidator,
    valid_observation: WindowObservation,
    sampled_frames: list[SampledFrame],
) -> None:
    valid_observation.actions[0].evidence_frame_sample_indices = [0, 10]
    batch = ObservationBatch(observations=[valid_observation])
    with pytest.raises(ModelOutputSemanticError, match="out-of-range evidence frame"):
        validator.validate(batch, sampled_frames)


def test_negative_evidence_frame_raises(
    validator: ObservationSemanticValidator,
    valid_observation: WindowObservation,
    sampled_frames: list[SampledFrame],
) -> None:
    valid_observation.actions[0].evidence_frame_sample_indices = [-1]
    batch = ObservationBatch(observations=[valid_observation])
    with pytest.raises(ModelOutputSemanticError, match="out-of-range evidence frame"):
        validator.validate(batch, sampled_frames)


def test_evidence_frames_deduplicated_and_sorted(
    validator: ObservationSemanticValidator,
    valid_observation: WindowObservation,
    sampled_frames: list[SampledFrame],
) -> None:
    valid_observation.actions[0].evidence_frame_sample_indices = [2, 1, 2, 0]
    batch = ObservationBatch(observations=[valid_observation])
    validator.validate(batch, sampled_frames)
    assert batch.observations[0].actions[0].evidence_frame_sample_indices == [0, 1, 2]


def test_unknown_action_type_mapped_to_unknown_with_warning(
    validator: ObservationSemanticValidator,
    valid_observation: WindowObservation,
    sampled_frames: list[SampledFrame],
) -> None:
    valid_observation.actions[0].action_type = "dance"
    batch = ObservationBatch(observations=[valid_observation])
    warnings = validator.validate(batch, sampled_frames)
    assert len(warnings) == 1
    assert "mapped to 'unknown'" in warnings[0]
    assert batch.observations[0].actions[0].action_type == "unknown"


def test_target_none_is_allowed(
    validator: ObservationSemanticValidator,
    valid_observation: WindowObservation,
    sampled_frames: list[SampledFrame],
) -> None:
    valid_observation.actions[0].target_id = None
    batch = ObservationBatch(observations=[valid_observation])
    warnings = validator.validate(batch, sampled_frames)
    assert warnings == []
