"""Unit tests for EntityRegistry deterministic lifecycle management."""

from __future__ import annotations

import pytest

from qwen_stream_video.config import EntityRegistryConfig
from qwen_stream_video.domain import (
    EntityLifecycleStatus,
    EntityObservation,
    EntityType,
    GlobalState,
)
from qwen_stream_video.state import EntityRegistry


@pytest.fixture
def config() -> EntityRegistryConfig:
    return EntityRegistryConfig(
        confident_match_threshold=0.78,
        ambiguous_match_threshold=0.58,
        ambiguous_margin=0.08,
        max_missing_windows=2,
        temporary_entity_prefix="temp",
        candidate_hint_weight=0.05,
        allow_delayed_merge=True,
        delayed_merge_support_windows=2,
    )


@pytest.fixture
def registry(config: EntityRegistryConfig) -> EntityRegistry:
    return EntityRegistry(config)


@pytest.fixture
def empty_state() -> GlobalState:
    return GlobalState(run_id="test")


@pytest.fixture
def person_obs() -> EntityObservation:
    return EntityObservation(
        local_id="P1",
        entity_type=EntityType.PERSON,
        name="worker",
        confidence=0.9,
        evidence_frames=[],
    )


def test_global_ids_are_monotonic(registry: EntityRegistry, empty_state: GlobalState) -> None:
    e1 = registry.create_entity(empty_state, EntityType.PERSON)
    e2 = registry.create_entity(empty_state, EntityType.PERSON)
    e3 = registry.create_entity(empty_state, EntityType.DEVICE)
    assert e1.entity_id == "person_0001"
    assert e2.entity_id == "person_0002"
    assert e3.entity_id == "device_0001"


def test_ids_are_not_reused(registry: EntityRegistry, empty_state: GlobalState) -> None:
    e1 = registry.create_entity(empty_state, EntityType.PERSON)
    _ = registry.create_entity(empty_state, EntityType.PERSON)
    e3 = registry.create_entity(empty_state, EntityType.PERSON)
    assert e3.entity_id == "person_0003"
    assert e1.entity_id not in ("person_0002", "person_0003")


def test_temporary_ids_use_prefix(registry: EntityRegistry, empty_state: GlobalState) -> None:
    t1 = registry.create_entity(empty_state, EntityType.PERSON, temporary=True)
    t2 = registry.create_entity(empty_state, EntityType.PERSON, temporary=True)
    assert t1.entity_id == "temp_person_0001"
    assert t2.entity_id == "temp_person_0002"


def test_entity_not_seen_is_not_deleted(registry: EntityRegistry, empty_state: GlobalState, person_obs: EntityObservation) -> None:
    entity = registry.create_entity(empty_state, EntityType.PERSON)
    registry.update_from_observation(
        empty_state, entity.entity_id, person_obs, scene_id="scene_0001", run_id="test"
    )
    registry.mark_not_observed(empty_state, set(), current_window=1)
    assert entity.entity_id in empty_state.entities
    assert empty_state.entities[entity.entity_id].lifecycle_status == EntityLifecycleStatus.TEMPORARILY_MISSING


def test_entity_becomes_temporarily_missing(registry: EntityRegistry, empty_state: GlobalState, person_obs: EntityObservation) -> None:
    entity = registry.create_entity(empty_state, EntityType.PERSON)
    registry.update_from_observation(
        empty_state, entity.entity_id, person_obs, scene_id="scene_0001", run_id="test"
    )
    changed, events = registry.mark_not_observed(empty_state, set(), current_window=1)
    assert entity.entity_id in changed
    assert empty_state.entities[entity.entity_id].lifecycle_status == EntityLifecycleStatus.TEMPORARILY_MISSING
    assert any(e.event_type == "entity_temporarily_missing" for e in events)


def test_entity_becomes_inactive_after_threshold(registry: EntityRegistry, empty_state: GlobalState, person_obs: EntityObservation) -> None:
    entity = registry.create_entity(empty_state, EntityType.PERSON)
    registry.update_from_observation(
        empty_state, entity.entity_id, person_obs, scene_id="scene_0001", run_id="test"
    )
    registry.mark_not_observed(empty_state, set(), current_window=1)
    registry.mark_not_observed(empty_state, set(), current_window=2)
    changed, _ = registry.mark_not_observed(empty_state, set(), current_window=3)
    assert entity.entity_id in changed
    assert empty_state.entities[entity.entity_id].lifecycle_status == EntityLifecycleStatus.INACTIVE


def test_temporary_entity_merge_preserves_history(registry: EntityRegistry, empty_state: GlobalState) -> None:
    temp = registry.create_entity(empty_state, EntityType.PERSON, temporary=True, window_index=0, scene_id="scene_0001")
    formal = registry.create_entity(empty_state, EntityType.PERSON, window_index=0, scene_id="scene_0001")
    temp_obs = EntityObservation(
        local_id="P1",
        entity_type=EntityType.PERSON,
        name="worker",
        confidence=0.9,
        evidence_frames=[],
    )
    registry.update_from_observation(empty_state, temp.entity_id, temp_obs, scene_id="scene_0001", run_id="test", window_index=0)
    merge_result, events = registry.merge_temporary_entity(
        empty_state, temp.entity_id, formal.entity_id, window_index=1
    )
    assert merge_result.temporary_entity_id == temp.entity_id
    assert merge_result.merged_into == formal.entity_id
    assert empty_state.entities[temp.entity_id].lifecycle_status == EntityLifecycleStatus.MERGED
    assert empty_state.entities[temp.entity_id].merged_into == formal.entity_id
    assert temp.entity_id in empty_state.entities[formal.entity_id].aliases
    assert len(empty_state.entities[formal.entity_id].evidence) >= len(temp.evidence)
    assert any(e.event_type == "entity_merged" for e in events)


def test_merged_entity_is_excluded_from_candidates(registry: EntityRegistry, empty_state: GlobalState) -> None:
    temp = registry.create_entity(empty_state, EntityType.PERSON, temporary=True, window_index=0)
    formal = registry.create_entity(empty_state, EntityType.PERSON, window_index=0)
    registry.merge_temporary_entity(empty_state, temp.entity_id, formal.entity_id, window_index=1)
    candidates = registry.find_candidates(empty_state, EntityType.PERSON, "scene_0001", 2)
    assert temp.entity_id not in {c.entity_id for c in candidates}


def test_find_candidates_excludes_inactive(registry: EntityRegistry, empty_state: GlobalState, person_obs: EntityObservation) -> None:
    entity = registry.create_entity(empty_state, EntityType.PERSON, window_index=0)
    registry.update_from_observation(empty_state, entity.entity_id, person_obs, scene_id="scene_0001", run_id="test", window_index=0)
    registry.mark_not_observed(empty_state, set(), current_window=1)
    registry.mark_not_observed(empty_state, set(), current_window=2)
    registry.mark_not_observed(empty_state, set(), current_window=3)
    candidates = registry.find_candidates(empty_state, EntityType.PERSON, "scene_0001", 4)
    assert entity.entity_id not in {c.entity_id for c in candidates}


def test_find_candidates_respects_max_missing(registry: EntityRegistry, empty_state: GlobalState, person_obs: EntityObservation) -> None:
    entity = registry.create_entity(empty_state, EntityType.PERSON, window_index=0)
    registry.update_from_observation(empty_state, entity.entity_id, person_obs, scene_id="scene_0001", run_id="test", window_index=0)
    registry.mark_not_observed(empty_state, set(), current_window=1)
    candidates = registry.find_candidates(empty_state, EntityType.PERSON, "scene_0001", 2, max_missing_windows=3)
    assert entity.entity_id in {c.entity_id for c in candidates}
    candidates = registry.find_candidates(empty_state, EntityType.PERSON, "scene_0001", 2, max_missing_windows=0)
    assert entity.entity_id not in {c.entity_id for c in candidates}


def test_find_candidates_across_scenes_when_preserve_enabled(
    registry: EntityRegistry, empty_state: GlobalState, person_obs: EntityObservation
) -> None:
    entity = registry.create_entity(empty_state, EntityType.PERSON, window_index=0)
    registry.update_from_observation(
        empty_state, entity.entity_id, person_obs, scene_id="scene_0001", run_id="test", window_index=0
    )
    registry.mark_not_observed(empty_state, set(), current_window=1)
    # entity is from scene_0001 and currently not_visible; new scene is scene_0002
    candidates = registry.find_candidates(
        empty_state, EntityType.PERSON, "scene_0002", 2, preserve_entities_across_scenes=True
    )
    assert entity.entity_id in {c.entity_id for c in candidates}
    candidates = registry.find_candidates(
        empty_state, EntityType.PERSON, "scene_0002", 2, preserve_entities_across_scenes=False
    )
    assert entity.entity_id not in {c.entity_id for c in candidates}


def test_update_emits_reactivated_event(registry: EntityRegistry, empty_state: GlobalState, person_obs: EntityObservation) -> None:
    entity = registry.create_entity(empty_state, EntityType.PERSON, window_index=0)
    registry.update_from_observation(empty_state, entity.entity_id, person_obs, scene_id="scene_0001", run_id="test", window_index=0)
    registry.mark_not_observed(empty_state, set(), current_window=1)
    _, events = registry.update_from_observation(
        empty_state, entity.entity_id, person_obs, scene_id="scene_0001", run_id="test", window_index=2
    )
    assert any(e.event_type == "entity_reactivated" for e in events)
