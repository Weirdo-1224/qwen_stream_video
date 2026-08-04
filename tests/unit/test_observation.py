"""Unit tests for the incremental observation schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from qwen_stream_video.domain import (
    Action,
    ActionPhase,
    Attribute,
    Entity,
    EntityType,
    ObservationBatch,
    SceneObservation,
    Uncertainty,
    Viewpoint,
    Visibility,
    WindowObservation,
)


def _valid_window_observation() -> WindowObservation:
    return WindowObservation(
        window_run_index=0,
        window_global_index=1,
        window_start_seconds=0.0,
        window_end_seconds=3.0,
        scene=SceneObservation(
            description="A technician operates a circuit breaker.",
            setting="indoor substation",
            lighting="well_lit",
            viewpoint=Viewpoint.CLOSE_UP,
        ),
        entities=[
            Entity(
                local_id="E1",
                entity_type=EntityType.PERSON,
                label="technician",
                candidate_global_id="G1",
                viewpoint=Viewpoint.FRONT,
                visibility=Visibility.FULLY_VISIBLE,
                bounding_box=[100.0, 200.0, 300.0, 400.0],
                attributes=[
                    Attribute(name="role", value="operator", confidence=0.9),
                ],
                confidence=0.95,
            ),
            Entity(
                local_id="E2",
                entity_type=EntityType.EQUIPMENT,
                label="circuit breaker",
                confidence=0.88,
            ),
        ],
        actions=[
            Action(
                local_id="A1",
                actor_id="E1",
                action_type="operate",
                phase=ActionPhase.CONTINUE,
                target_id="E2",
                start_time_seconds=0.5,
                end_time_seconds=2.5,
                evidence_frame_sample_indices=[0, 1, 2],
                attributes=[
                    Attribute(name="tool", value="wrench"),
                ],
                confidence=0.85,
            ),
        ],
        uncertainties=[
            Uncertainty(
                category="occlusion",
                description="The lower half of the breaker is occluded.",
                severity="medium",
                confidence=0.7,
            ),
        ],
        summary="Technician continues to operate the circuit breaker.",
    )


def test_valid_window_observation() -> None:
    obs = _valid_window_observation()
    assert obs.window_run_index == 0
    assert obs.entities[0].local_id == "E1"
    assert obs.actions[0].evidence_frame_sample_indices == [0, 1, 2]
    assert len(obs.uncertainties) == 1


def test_invalid_confidence_rejected() -> None:
    with pytest.raises(ValidationError):
        Entity(
            local_id="E1",
            entity_type=EntityType.PERSON,
            label="technician",
            confidence=1.1,
        )


def test_negative_confidence_rejected() -> None:
    with pytest.raises(ValidationError):
        Action(
            local_id="A1",
            actor_id="E1",
            action_type="operate",
            confidence=-0.01,
        )


def test_window_end_must_be_after_start() -> None:
    with pytest.raises(ValidationError):
        WindowObservation(
            window_run_index=0,
            window_global_index=0,
            window_start_seconds=5.0,
            window_end_seconds=5.0,
            scene=SceneObservation(description="test"),
        )


def test_action_end_must_be_after_start() -> None:
    with pytest.raises(ValidationError):
        Action(
            local_id="A1",
            actor_id="E1",
            action_type="operate",
            start_time_seconds=2.0,
            end_time_seconds=1.0,
            confidence=0.8,
        )


def test_invalid_enum_value_rejected() -> None:
    with pytest.raises(ValidationError):
        Entity(
            local_id="E1",
            entity_type="robot",  # type: ignore[arg-type]
            label="technician",
            confidence=0.9,
        )


def test_bounding_box_must_have_four_values() -> None:
    with pytest.raises(ValidationError):
        Entity(
            local_id="E1",
            entity_type=EntityType.PERSON,
            label="technician",
            bounding_box=[1.0, 2.0],
            confidence=0.9,
        )


def test_default_factories_create_empty_collections() -> None:
    obs = WindowObservation(
        window_run_index=0,
        window_global_index=0,
        window_start_seconds=0.0,
        window_end_seconds=1.0,
        scene=SceneObservation(description="empty window"),
    )
    assert obs.entities == []
    assert obs.actions == []
    assert obs.uncertainties == []

    # Ensure independent defaults across instances.
    obs2 = WindowObservation(
        window_run_index=1,
        window_global_index=1,
        window_start_seconds=1.0,
        window_end_seconds=2.0,
        scene=SceneObservation(description="second window"),
    )
    obs.entities.append(
        Entity(
            local_id="E1",
            entity_type=EntityType.OTHER,
            label="x",
            confidence=0.5,
        )
    )
    assert obs2.entities == []


def test_observation_batch_accepts_multiple_windows() -> None:
    obs1 = _valid_window_observation()
    obs2 = obs1.model_copy(update={"window_run_index": 1, "window_global_index": 2})
    batch = ObservationBatch(observations=[obs1, obs2])
    assert len(batch.observations) == 2
    assert batch.schema_version == "1.0"


def test_missing_required_scene_description_rejected() -> None:
    with pytest.raises(ValidationError):
        WindowObservation(
            window_run_index=0,
            window_global_index=0,
            window_start_seconds=0.0,
            window_end_seconds=1.0,
            scene=SceneObservation(),  # type: ignore[call-arg]
        )
