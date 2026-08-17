"""Unit tests for the observation semantic validator."""

from __future__ import annotations

import pytest

from qwen_stream_video.domain import (
    ActionObservation,
    ActionPhaseObservation,
    AttributeObservation,
    EntityObservation,
    EntityType,
    ObservationBatch,
    SceneObservation,
    ViewType,
    VisibilityQuality,
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
            sample_index=i,
            frame_index=i,
            timestamp_seconds=10.0 + i,
            image=__import__("numpy").zeros((10, 10, 3), dtype="uint8"),
        )
        for i in range(4)
    ]


@pytest.fixture
def valid_batch() -> ObservationBatch:
    return ObservationBatch(
        schema_version="1.0",
        window=WindowObservation(
            global_index=0,
            start_seconds=0.0,
            end_seconds=3.0,
        ),
        summary="test",
        scene=SceneObservation(
            camera_change=False,
            view_type=ViewType.UNKNOWN,
            visibility=VisibilityQuality.UNKNOWN,
            description="test scene",
        ),
        entities=[
            EntityObservation(
                local_id="E1",
                entity_type=EntityType.PERSON,
                name="technician",
                confidence=0.9,
                evidence_frames=[0],
            ),
            EntityObservation(
                local_id="E2",
                entity_type=EntityType.DEVICE,
                name="breaker",
                confidence=0.8,
                evidence_frames=[1],
            ),
        ],
        actions=[
            ActionObservation(
                local_id="A1",
                actor_local_id="E1",
                action_type="touch",
                target_local_id="E2",
                phase_observation=ActionPhaseObservation.ONGOING,
                description="test",
                confidence=0.85,
                evidence_frames=[0, 1],
            ),
        ],
    )


def test_valid_batch_passes(
    validator: ObservationSemanticValidator,
    valid_batch: ObservationBatch,
    sampled_frames: list[SampledFrame],
) -> None:
    warnings = validator.validate(valid_batch, sampled_frames)
    assert warnings == []


def test_window_fields_overwritten(
    validator: ObservationSemanticValidator,
    valid_batch: ObservationBatch,
    sampled_frames: list[SampledFrame],
    video_window: VideoWindow,
) -> None:
    validator.validate(valid_batch, sampled_frames, window=video_window)
    assert valid_batch.window.global_index == 5
    assert valid_batch.window.start_seconds == 10.0
    assert valid_batch.window.end_seconds == 16.0


def test_duplicate_entity_local_id(
    validator: ObservationSemanticValidator,
    valid_batch: ObservationBatch,
    sampled_frames: list[SampledFrame],
) -> None:
    valid_batch.entities[1].local_id = "E1"
    with pytest.raises(ModelOutputSemanticError, match="Duplicate entity local_id"):
        validator.validate(valid_batch, sampled_frames)


def test_duplicate_action_local_id(
    validator: ObservationSemanticValidator,
    valid_batch: ObservationBatch,
    sampled_frames: list[SampledFrame],
) -> None:
    valid_batch.actions.append(
        ActionObservation(
            local_id="A1",
            actor_local_id="E1",
            action_type="hold",
            confidence=0.6,
        )
    )
    with pytest.raises(ModelOutputSemanticError, match="Duplicate action local_id"):
        validator.validate(valid_batch, sampled_frames)


def test_missing_actor_reference(
    validator: ObservationSemanticValidator,
    valid_batch: ObservationBatch,
    sampled_frames: list[SampledFrame],
) -> None:
    valid_batch.actions[0].actor_local_id = "E99"
    with pytest.raises(ModelOutputSemanticError, match="missing actor"):
        validator.validate(valid_batch, sampled_frames)


def test_missing_target_reference(
    validator: ObservationSemanticValidator,
    valid_batch: ObservationBatch,
    sampled_frames: list[SampledFrame],
) -> None:
    valid_batch.actions[0].target_local_id = "E99"
    with pytest.raises(ModelOutputSemanticError, match="missing target"):
        validator.validate(valid_batch, sampled_frames)


def test_missing_tool_reference(
    validator: ObservationSemanticValidator,
    valid_batch: ObservationBatch,
    sampled_frames: list[SampledFrame],
) -> None:
    valid_batch.actions[0].tool_local_id = "E99"
    with pytest.raises(ModelOutputSemanticError, match="missing tool"):
        validator.validate(valid_batch, sampled_frames)


def test_missing_attribute_entity_reference(
    validator: ObservationSemanticValidator,
    valid_batch: ObservationBatch,
    sampled_frames: list[SampledFrame],
) -> None:
    valid_batch.attribute_observations.append(
        AttributeObservation(
            entity_local_id="E99",
            attribute="state",
            value="open",
            confidence=0.8,
        )
    )
    with pytest.raises(ModelOutputSemanticError, match="missing entity"):
        validator.validate(valid_batch, sampled_frames)


def test_invalid_evidence_frame(
    validator: ObservationSemanticValidator,
    valid_batch: ObservationBatch,
    sampled_frames: list[SampledFrame],
) -> None:
    valid_batch.actions[0].evidence_frames = [0, 10]
    with pytest.raises(ModelOutputSemanticError, match="out-of-range evidence frame"):
        validator.validate(valid_batch, sampled_frames)


def test_negative_evidence_frame(
    validator: ObservationSemanticValidator,
    valid_batch: ObservationBatch,
    sampled_frames: list[SampledFrame],
) -> None:
    valid_batch.entities[0].evidence_frames = [-1]
    with pytest.raises(ModelOutputSemanticError, match="out-of-range evidence frame"):
        validator.validate(valid_batch, sampled_frames)


def test_evidence_frames_deduplicated_and_sorted(
    validator: ObservationSemanticValidator,
    valid_batch: ObservationBatch,
    sampled_frames: list[SampledFrame],
) -> None:
    valid_batch.actions[0].evidence_frames = [2, 1, 2, 0]
    validator.validate(valid_batch, sampled_frames)
    assert valid_batch.actions[0].evidence_frames == [0, 1, 2]


def test_unknown_action_mapped_to_unknown_with_warning(
    validator: ObservationSemanticValidator,
    valid_batch: ObservationBatch,
    sampled_frames: list[SampledFrame],
) -> None:
    valid_batch.actions[0].action_type = "dance"
    warnings = validator.validate(valid_batch, sampled_frames)
    assert len(warnings) == 1
    assert "mapped to 'unknown'" in warnings[0]
    assert valid_batch.actions[0].action_type == "dance"


def test_target_none_is_allowed(
    validator: ObservationSemanticValidator,
    valid_batch: ObservationBatch,
    sampled_frames: list[SampledFrame],
) -> None:
    valid_batch.actions[0].target_local_id = None
    warnings = validator.validate(valid_batch, sampled_frames)
    assert warnings == []


def test_valid_references_pass(
    validator: ObservationSemanticValidator,
    valid_batch: ObservationBatch,
    sampled_frames: list[SampledFrame],
) -> None:
    warnings = validator.validate(valid_batch, sampled_frames)
    assert warnings == []
    assert valid_batch.entities[0].evidence_frames == [0]
    assert valid_batch.actions[0].evidence_frames == [0, 1]


def test_multiple_attributes_for_same_entity_are_allowed(
    validator: ObservationSemanticValidator,
    valid_batch: ObservationBatch,
    sampled_frames: list[SampledFrame],
) -> None:
    """A single entity may have several different attributes in one window."""
    valid_batch.attribute_observations = [
        AttributeObservation(
            entity_local_id="E2",
            attribute="state",
            value="closed",
            confidence=0.8,
        ),
        AttributeObservation(
            entity_local_id="E2",
            attribute="position",
            value="vertical",
            confidence=0.7,
        ),
    ]
    warnings = validator.validate(valid_batch, sampled_frames)
    assert warnings == []


def test_duplicate_attribute_for_same_entity_emits_warning(
    validator: ObservationSemanticValidator,
    valid_batch: ObservationBatch,
    sampled_frames: list[SampledFrame],
) -> None:
    """Observing the exact same attribute twice on one entity is only a warning."""
    valid_batch.attribute_observations = [
        AttributeObservation(
            entity_local_id="E2",
            attribute="state",
            value="closed",
            confidence=0.8,
        ),
        AttributeObservation(
            entity_local_id="E2",
            attribute="state",
            value="open",
            confidence=0.6,
        ),
    ]
    warnings = validator.validate(valid_batch, sampled_frames)
    assert len(warnings) == 1
    assert "Duplicate attribute observation" in warnings[0]
