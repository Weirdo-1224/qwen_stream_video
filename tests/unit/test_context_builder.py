"""Unit tests for ContextBuilder bounded prompt context."""

from __future__ import annotations

import json

import pytest

from qwen_stream_video.config import ContextConfig
from qwen_stream_video.domain import (
    ActionLifecycle,
    AttributeConfirmationStatus,
    EntityLifecycleStatus,
    EntityType,
    GlobalActionState,
    GlobalEntityState,
    GlobalState,
    SceneState,
    ViewType,
    VisibilityState,
)
from qwen_stream_video.state import ContextBuilder
from qwen_stream_video.video import VideoWindow


def _window(global_index: int = 10) -> VideoWindow:
    return VideoWindow(
        global_index=global_index,
        run_index=global_index,
        start_seconds=float(global_index * 3),
        commit_start_seconds=float(global_index * 3),
        end_seconds=float(global_index * 3 + 6),
    )


def _entity(
    entity_id: str,
    last_seen_window: int,
    lifecycle: EntityLifecycleStatus = EntityLifecycleStatus.ACTIVE,
    visibility: VisibilityState = VisibilityState.VISIBLE,
) -> GlobalEntityState:
    return GlobalEntityState(
        entity_id=entity_id,
        entity_type=EntityType.PERSON,
        canonical_name=f"entity_{entity_id}",
        first_seen_window=last_seen_window,
        last_seen_window=last_seen_window,
        lifecycle_status=lifecycle,
        visibility=visibility,
        appearance_signature={"color": entity_id},
    )


def _action(
    action_id: str,
    last_observed_window: int,
    lifecycle: ActionLifecycle = ActionLifecycle.ONGOING,
    actor_id: str = "person_0001",
) -> GlobalActionState:
    return GlobalActionState(
        action_id=action_id,
        actor_id=actor_id,
        action_type="inspect",
        last_observed_window=last_observed_window,
        lifecycle=lifecycle,
    )


@pytest.fixture
def config() -> ContextConfig:
    return ContextConfig(
        max_entities=10,
        recent_window_count=5,
        max_active_actions=10,
        max_pending_attributes=10,
        max_serialized_characters=6000,
    )


@pytest.fixture
def builder(config: ContextConfig) -> ContextBuilder:
    return ContextBuilder(config)


@pytest.fixture
def empty_state() -> GlobalState:
    return GlobalState(run_id="test")


def test_recent_entities_are_included(
    builder: ContextBuilder, empty_state: GlobalState
) -> None:
    empty_state.entities["person_0001"] = _entity("person_0001", 9)
    empty_state.entities["person_0002"] = _entity("person_0002", 4)

    context = builder.build(empty_state, _window(10))

    ids = {e["entity_id"] for e in context.candidate_entities}
    assert "person_0001" in ids
    # person_0002 is within the default recent_window_count of 5 (10 - 4 = 6 > 5)
    assert "person_0002" not in ids


def test_active_actions_are_included(
    builder: ContextBuilder, empty_state: GlobalState
) -> None:
    empty_state.entities["person_0001"] = _entity("person_0001", 9)
    empty_state.actions["action_0001"] = _action("action_0001", 9, ActionLifecycle.ONGOING)
    empty_state.actions["action_0002"] = _action(
        "action_0002", 9, ActionLifecycle.ENDED
    )

    context = builder.build(empty_state, _window(10))

    action_ids = {a["action_id"] for a in context.active_actions}
    assert "action_0001" in action_ids
    assert "action_0002" not in action_ids


def test_excludes_old_inactive(
    builder: ContextBuilder, empty_state: GlobalState
) -> None:
    empty_state.entities["person_0001"] = _entity("person_0001", 10)
    empty_state.entities["person_0002"] = _entity(
        "person_0002", 2, EntityLifecycleStatus.INACTIVE
    )

    context = builder.build(empty_state, _window(10))

    ids = {e["entity_id"] for e in context.candidate_entities}
    assert "person_0001" in ids
    assert "person_0002" not in ids


def test_candidate_ids_valid(
    builder: ContextBuilder, empty_state: GlobalState
) -> None:
    empty_state.entities["person_0001"] = _entity("person_0001", 10)
    empty_state.entities["person_0002"] = _entity("person_0002", 10)
    empty_state.entities["person_0003"] = _entity(
        "person_0003", 10, EntityLifecycleStatus.MERGED
    )

    context = builder.build(empty_state, _window(10))
    candidate_ids = builder.candidate_entity_ids(empty_state, _window(10))

    entity_ids = {e["entity_id"] for e in context.candidate_entities}
    assert entity_ids == candidate_ids
    assert "person_0001" in candidate_ids
    assert "person_0002" in candidate_ids
    assert "person_0003" not in candidate_ids
    assert all(isinstance(eid, str) for eid in candidate_ids)


def test_respects_character_limit(
    builder: ContextBuilder, empty_state: GlobalState
) -> None:
    # Fill state with many entities so the context is certainly larger than the limit.
    for i in range(20):
        entity_id = f"person_{i:04d}"
        empty_state.entities[entity_id] = _entity(entity_id, 10)
    empty_state.current_scene_id = "scene_0001"
    empty_state.scenes["scene_0001"] = SceneState(
        scene_id="scene_0001",
        view_type=ViewType.WIDE,
        continuity="continuous",
        start_window=0,
        last_active_window=10,
    )

    limited_config = ContextConfig(
        max_entities=20,
        recent_window_count=5,
        max_active_actions=1,
        max_pending_attributes=1,
        max_serialized_characters=400,
    )
    limited_builder = ContextBuilder(limited_config)

    context = limited_builder.build(empty_state, _window(10))

    serialized = context.to_json()
    assert len(serialized) <= limited_config.max_serialized_characters
    assert context.truncated is True
    assert len(context.candidate_entities) < 20


def test_truncation_keeps_valid_json(
    builder: ContextBuilder, empty_state: GlobalState
) -> None:
    for i in range(20):
        entity_id = f"person_{i:04d}"
        empty_state.entities[entity_id] = _entity(entity_id, 10)
    empty_state.current_scene_id = "scene_0001"
    empty_state.scenes["scene_0001"] = SceneState(
        scene_id="scene_0001",
        view_type=ViewType.WIDE,
        continuity="continuous",
        start_window=0,
        last_active_window=10,
    )

    limited_config = ContextConfig(
        max_entities=20,
        recent_window_count=5,
        max_active_actions=1,
        max_pending_attributes=1,
        max_serialized_characters=400,
    )
    limited_builder = ContextBuilder(limited_config)

    context = limited_builder.build(empty_state, _window(10))
    serialized = context.to_json()

    parsed = json.loads(serialized)
    assert isinstance(parsed, dict)
    assert "candidate_entities" in parsed
    assert "scene" in parsed
    assert isinstance(parsed["candidate_entities"], list)


def test_pending_attributes_are_included(
    builder: ContextBuilder, empty_state: GlobalState
) -> None:
    entity = _entity("person_0001", 10)
    entity.attributes["status"] = entity.attributes.get(
        "status",
        None,
    )
    from qwen_stream_video.domain import AttributeState

    entity.attributes["status"] = AttributeState(
        attribute_key="status",
        value="unknown",
        confidence=0.8,
        first_observed_window=10,
        last_observed_window=10,
        status=AttributeConfirmationStatus.PENDING,
        pending_value="running",
        pending_confidence=0.7,
    )
    empty_state.entities["person_0001"] = entity

    context = builder.build(empty_state, _window(10))

    pending_keys = {(p["entity_id"], p["attribute_key"]) for p in context.pending_attributes}
    assert ("person_0001", "status") in pending_keys


def test_recent_scene_change_is_included(
    builder: ContextBuilder, empty_state: GlobalState
) -> None:
    empty_state.current_scene_id = "scene_0002"
    empty_state.scenes["scene_0001"] = SceneState(
        scene_id="scene_0001",
        view_type=ViewType.WIDE,
        continuity="continuous",
        start_window=0,
        last_active_window=4,
    )
    empty_state.scenes["scene_0002"] = SceneState(
        scene_id="scene_0002",
        view_type=ViewType.CLOSEUP,
        continuity="camera_change",
        start_window=5,
        last_active_window=10,
    )

    context = builder.build(empty_state, _window(10))

    assert len(context.recent_scene_changes) == 1
    change = context.recent_scene_changes[0]
    assert change["scene_id"] == "scene_0002"
    assert change["continuity"] == "camera_change"
    assert change["view_type"] == "closeup"
    assert change["start_window"] == 5


def test_truncation_prunes_oldest_entities_first(
    builder: ContextBuilder, empty_state: GlobalState
) -> None:
    # Create one recent active entity and many old active entities.
    empty_state.entities["person_new"] = _entity("person_new", 10)
    for i in range(15):
        empty_state.entities[f"person_old_{i:02d}"] = _entity(f"person_old_{i:02d}", 1)

    limited_config = ContextConfig(
        max_entities=5,
        recent_window_count=10,
        max_active_actions=1,
        max_pending_attributes=1,
        max_serialized_characters=2000,
    )
    limited_builder = ContextBuilder(limited_config)
    context = limited_builder.build(empty_state, _window(10))

    assert "person_new" in {e["entity_id"] for e in context.candidate_entities}
    assert all("person_old" in e["entity_id"] for e in context.candidate_entities[1:])


def test_truncation_prunes_possible_ended_actions_before_active(
    builder: ContextBuilder, empty_state: GlobalState
) -> None:
    empty_state.entities["person_0001"] = _entity("person_0001", 10)
    empty_state.actions["action_active"] = _action("action_active", 10, ActionLifecycle.ONGOING)
    empty_state.actions["action_ended"] = _action("action_ended", 10, ActionLifecycle.POSSIBLE_ENDED)

    limited_config = ContextConfig(
        max_entities=10,
        recent_window_count=5,
        max_active_actions=1,
        max_pending_attributes=1,
        max_serialized_characters=600,
    )
    limited_builder = ContextBuilder(limited_config)
    context = limited_builder.build(empty_state, _window(10))

    assert len(context.active_actions) == 1
    assert context.active_actions[0]["action_id"] == "action_active"


def test_truncation_prunes_low_confidence_pending_first(
    builder: ContextBuilder, empty_state: GlobalState
) -> None:
    from qwen_stream_video.domain import AttributeState

    entity = _entity("person_0001", 10)
    entity.attributes["low"] = AttributeState(
        attribute_key="low",
        value="unknown",
        confidence=0.8,
        first_observed_window=10,
        last_observed_window=10,
        status=AttributeConfirmationStatus.PENDING,
        pending_value="a",
        pending_confidence=0.4,
    )
    entity.attributes["high"] = AttributeState(
        attribute_key="high",
        value="unknown",
        confidence=0.8,
        first_observed_window=10,
        last_observed_window=10,
        status=AttributeConfirmationStatus.PENDING,
        pending_value="b",
        pending_confidence=0.9,
    )
    empty_state.entities["person_0001"] = entity

    limited_config = ContextConfig(
        max_entities=10,
        recent_window_count=5,
        max_active_actions=1,
        max_pending_attributes=1,
        max_serialized_characters=600,
    )
    limited_builder = ContextBuilder(limited_config)
    context = limited_builder.build(empty_state, _window(10))

    assert len(context.pending_attributes) == 1
    assert context.pending_attributes[0]["attribute_key"] == "high"
