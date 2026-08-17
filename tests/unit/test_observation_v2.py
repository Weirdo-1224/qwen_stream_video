from __future__ import annotations

import pytest
from pydantic import ValidationError

from qwen_stream_video.domain import (
    EntityObservation,
    EntityType,
    ObservationBatch,
    RelationObservation,
    WindowObservation,
)


def test_valid_observation_v2_and_mutable_defaults() -> None:
    first = ObservationBatch(
        window=WindowObservation(global_index=0, start_seconds=0, commit_start_seconds=0, end_seconds=1)
    )
    second = ObservationBatch(
        window=WindowObservation(global_index=0, start_seconds=0, commit_start_seconds=0, end_seconds=1)
    )
    assert first.schema_version == "2.0"
    first.entities.append(EntityObservation(local_id="E1", entity_type=EntityType.PERSON, confidence=0.8))
    assert second.entities == []


def test_unknown_schema_and_bad_commit_fail() -> None:
    with pytest.raises(ValidationError):
        ObservationBatch.model_validate(
            {"schema_version": "9.0", "window": {"global_index": 0, "start_seconds": 0, "end_seconds": 1}}
        )
    with pytest.raises(ValidationError):
        WindowObservation(global_index=0, start_seconds=0, commit_start_seconds=1, end_seconds=1)


def test_relation_is_serializable() -> None:
    batch = ObservationBatch(
        window=WindowObservation(global_index=0, start_seconds=0, commit_start_seconds=0, end_seconds=1),
        entities=[EntityObservation(local_id="A", entity_type=EntityType.PERSON, confidence=0.9)],
        relations=[RelationObservation(subject_local_id="A", object_local_id="A", relation_type="near", confidence=0.5)],
    )
    assert batch.model_validate_json(batch.model_dump_json()).relations[0].relation_type == "near"
