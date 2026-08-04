"""Unit tests for the incremental observation schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from qwen_stream_video.domain import (
    ActionObservation,
    ActionPhaseObservation,
    AttributeObservation,
    EntityObservation,
    EntityType,
    ObservationBatch,
    SceneObservation,
    UncertaintyObservation,
    ViewType,
    VisibilityQuality,
    WindowObservation,
)


def _valid_window_observation() -> WindowObservation:
    return WindowObservation(
        global_index=1,
        start_seconds=0.0,
        end_seconds=3.0,
    )


def _valid_entity() -> EntityObservation:
    return EntityObservation(
        local_id="E1",
        entity_type=EntityType.PERSON,
        name="operator",
        description="A technician.",
        appearance={"role": "operator"},
        spatial_region="center",
        candidate_global_id="person_1",
        confidence=0.95,
        evidence_frames=[0, 1],
    )


def _valid_action() -> ActionObservation:
    return ActionObservation(
        local_id="A1",
        actor_local_id="E1",
        action_type="touch",
        target_local_id="E2",
        tool_local_id=None,
        phase_observation=ActionPhaseObservation.ONGOING,
        description="Operator touches the breaker.",
        confidence=0.85,
        evidence_frames=[1, 2],
    )


def _valid_batch() -> ObservationBatch:
    return ObservationBatch(
        schema_version="1.0",
        window=_valid_window_observation(),
        summary="Operator touches the breaker.",
        scene=SceneObservation(
            camera_change=False,
            view_type=ViewType.MEDIUM,
            visibility=VisibilityQuality.CLEAR,
            description="Indoor substation scene.",
        ),
        entities=[
            _valid_entity(),
            EntityObservation(
                local_id="E2",
                entity_type=EntityType.DEVICE,
                name="breaker",
                confidence=0.88,
                evidence_frames=[2],
            ),
        ],
        actions=[_valid_action()],
        attribute_observations=[
            AttributeObservation(
                entity_local_id="E2",
                attribute="state",
                value="closed",
                confidence=0.8,
                evidence_frames=[2],
            )
        ],
        uncertainties=[
            UncertaintyObservation(
                description="Unable to confirm the breaker is fully closed.",
                related_local_ids=["E2"],
                evidence_frames=[],
            )
        ],
    )


def test_valid_observation_batch() -> None:
    batch = _valid_batch()
    assert batch.schema_version == "1.0"
    assert batch.window.global_index == 1
    assert batch.entities[0].local_id == "E1"
    assert batch.actions[0].evidence_frames == [1, 2]
    assert len(batch.attribute_observations) == 1
    assert len(batch.uncertainties) == 1


def test_confidence_above_one_fails() -> None:
    with pytest.raises(ValidationError):
        EntityObservation(
            local_id="E1",
            entity_type=EntityType.PERSON,
            name="operator",
            confidence=1.1,
        )


def test_confidence_below_zero_fails() -> None:
    with pytest.raises(ValidationError):
        ActionObservation(
            local_id="A1",
            actor_local_id="E1",
            action_type="touch",
            confidence=-0.01,
        )


def test_invalid_enum_fails() -> None:
    with pytest.raises(ValidationError):
        EntityObservation(
            local_id="E1",
            entity_type="robot",  # type: ignore[arg-type]
            name="operator",
            confidence=0.9,
        )


def test_mutable_defaults_are_isolated() -> None:
    batch1 = ObservationBatch(
        schema_version="1.0",
        window=_valid_window_observation(),
    )
    batch2 = ObservationBatch(
        schema_version="1.0",
        window=_valid_window_observation(),
    )
    batch1.entities.append(_valid_entity())
    assert batch2.entities == []
    assert batch1.scene.description == ""


def test_default_factories_create_empty_collections() -> None:
    batch = ObservationBatch(
        schema_version="1.0",
        window=_valid_window_observation(),
    )
    assert batch.entities == []
    assert batch.actions == []
    assert batch.attribute_observations == []
    assert batch.uncertainties == []
    assert batch.scene.description == ""


def test_batch_does_not_accept_observations_list() -> None:
    with pytest.raises(ValidationError):
        ObservationBatch(
            schema_version="1.0",
            window=_valid_window_observation(),
            observations=[{}],  # type: ignore[call-arg]
        )
