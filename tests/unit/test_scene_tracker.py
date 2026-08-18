"""Unit tests for SceneTracker scene continuity and visibility events."""

from __future__ import annotations

import pytest

from qwen_stream_video.config import SceneTrackerConfig
from qwen_stream_video.domain import (
    EntityType,
    GlobalState,
    ObservationBatch,
    SceneObservation,
    ViewType,
    VisibilityState,
    WindowObservation,
)
from qwen_stream_video.state import EntityRegistry, SceneTracker


def _observation(
    window_index: int,
    view_type: ViewType = ViewType.WIDE,
    camera_change: bool = False,
    continuity_hint: str = "continuous",
    target_visibility: str = "unknown",
) -> ObservationBatch:
    return ObservationBatch(
        window=WindowObservation(
            global_index=window_index,
            start_seconds=float(window_index * 3),
            commit_start_seconds=float(window_index * 3),
            end_seconds=float(window_index * 3 + 6),
        ),
        scene=SceneObservation(
            camera_change=camera_change,
            view_type=view_type,
            continuity_hint=continuity_hint,  # type: ignore[arg-type]
            target_visibility=target_visibility,  # type: ignore[arg-type]
        ),
        entities=[],
    )


@pytest.fixture
def config() -> SceneTrackerConfig:
    return SceneTrackerConfig(
        enabled=True,
        camera_change_starts_new_scene=True,
        preserve_entities_across_scenes=True,
    )


@pytest.fixture
def tracker(config: SceneTrackerConfig) -> SceneTracker:
    return SceneTracker(config)


@pytest.fixture
def empty_state() -> GlobalState:
    return GlobalState(run_id="test")


@pytest.fixture
def registry() -> EntityRegistry:
    return EntityRegistry()


def test_first_window_creates_scene(
    tracker: SceneTracker, empty_state: GlobalState
) -> None:
    obs = _observation(0, ViewType.WIDE)
    result = tracker.update(empty_state, obs)

    assert result.scene_id == "scene_0001"
    assert empty_state.current_scene_id == "scene_0001"
    assert result.scene_changed is True
    assert result.previous_scene_id is None
    assert result.continuity == "continuous"
    assert any(
        e.event_type == "scene_started" and e.scene_id == "scene_0001"
        for e in result.events
    )
    assert result.preserve_entities_across_scenes is True


def test_camera_change_creates_new_scene(
    tracker: SceneTracker, empty_state: GlobalState
) -> None:
    first = _observation(0, ViewType.WIDE)
    tracker.update(empty_state, first)
    first_scene_id = empty_state.current_scene_id

    second = _observation(1, ViewType.WIDE, camera_change=True, continuity_hint="camera_change")
    result = tracker.update(empty_state, second)

    assert result.scene_id == "scene_0002"
    assert result.scene_id != first_scene_id
    assert empty_state.current_scene_id == "scene_0002"
    assert result.scene_changed is True
    assert result.previous_scene_id == first_scene_id
    assert result.continuity == "camera_change"
    assert any(
        e.event_type == "scene_changed" and e.scene_id == "scene_0002"
        for e in result.events
    )


def test_camera_change_respects_enabled_flag(empty_state: GlobalState) -> None:
    disabled_config = SceneTrackerConfig(
        enabled=False,
        camera_change_starts_new_scene=True,
        preserve_entities_across_scenes=True,
    )
    tracker = SceneTracker(disabled_config)

    first = _observation(0, ViewType.WIDE)
    tracker.update(empty_state, first)
    first_scene_id = empty_state.current_scene_id

    second = _observation(1, ViewType.WIDE, camera_change=True, continuity_hint="camera_change")
    result = tracker.update(empty_state, second)

    assert result.scene_id == first_scene_id
    assert result.scene_changed is False


def test_reframe_keeps_scene_id(
    tracker: SceneTracker, empty_state: GlobalState
) -> None:
    first = _observation(0, ViewType.WIDE)
    tracker.update(empty_state, first)
    first_scene_id = empty_state.current_scene_id

    second = _observation(1, ViewType.WIDE, continuity_hint="reframed")
    result = tracker.update(empty_state, second)

    assert result.scene_id == first_scene_id
    assert result.scene_changed is False
    assert result.continuity == "reframed"
    assert any(
        e.event_type == "scene_reframed" and e.scene_id == first_scene_id
        for e in result.events
    )


def test_camera_change_does_not_delete_entities(
    tracker: SceneTracker,
    registry: EntityRegistry,
    empty_state: GlobalState,
) -> None:
    first = _observation(0, ViewType.WIDE)
    scene_result = tracker.update(empty_state, first)
    entity = registry.create_entity(
        empty_state,
        EntityType.PERSON,
        name="worker",
        window_index=0,
        scene_id=scene_result.scene_id,
    )
    empty_state.scenes[scene_result.scene_id].visible_entity_ids.append(entity.entity_id)
    empty_state.scenes[scene_result.scene_id].visible_entity_ids.sort()

    second = _observation(1, ViewType.WIDE, camera_change=True, continuity_hint="camera_change")
    tracker.update(empty_state, second)

    assert entity.entity_id in empty_state.entities
    assert empty_state.entities[entity.entity_id].lifecycle_status.value == "active"
    assert empty_state.entities[entity.entity_id].merged_into is None


def test_closeup_marks_previous_entities_not_visible(
    tracker: SceneTracker,
    registry: EntityRegistry,
    empty_state: GlobalState,
) -> None:
    first = _observation(0, ViewType.WIDE)
    scene_result = tracker.update(empty_state, first)
    entity = registry.create_entity(
        empty_state,
        EntityType.PERSON,
        name="worker",
        window_index=0,
        scene_id=scene_result.scene_id,
    )
    empty_state.scenes[scene_result.scene_id].visible_entity_ids.append(entity.entity_id)
    empty_state.scenes[scene_result.scene_id].visible_entity_ids.sort()
    assert entity.visibility == VisibilityState.VISIBLE

    second = _observation(1, ViewType.CLOSEUP)
    result = tracker.update(empty_state, second)

    assert result.scene_id == scene_result.scene_id
    assert empty_state.entities[entity.entity_id].visibility in {
        VisibilityState.NOT_VISIBLE,
        VisibilityState.PARTIAL,
    }
    assert empty_state.entities[entity.entity_id].lifecycle_status.value == "active"
    assert any(
        e.event_type == "scene_visibility_changed"
        and e.metadata.get("entity_id") == entity.entity_id
        for e in result.events
    )


def test_return_to_wide_allows_reactivation(
    tracker: SceneTracker,
    registry: EntityRegistry,
    empty_state: GlobalState,
) -> None:
    wide_obs = _observation(0, ViewType.WIDE)
    wide_scene = tracker.update(empty_state, wide_obs)
    entity = registry.create_entity(
        empty_state,
        EntityType.PERSON,
        name="worker",
        window_index=0,
        scene_id=wide_scene.scene_id,
    )
    wide_scene_state = empty_state.scenes[wide_scene.scene_id]
    wide_scene_state.visible_entity_ids.append(entity.entity_id)
    wide_scene_state.visible_entity_ids.sort()

    closeup_obs = _observation(1, ViewType.CLOSEUP)
    tracker.update(empty_state, closeup_obs)
    assert empty_state.entities[entity.entity_id].visibility in {
        VisibilityState.NOT_VISIBLE,
        VisibilityState.PARTIAL,
    }

    return_obs = _observation(2, ViewType.WIDE, camera_change=True, continuity_hint="camera_change")
    result = tracker.update(empty_state, return_obs)
    assert result.scene_id != wide_scene.scene_id

    candidates = registry.find_candidates(
        empty_state,
        EntityType.PERSON,
        result.scene_id,
        current_window=2,
        preserve_entities_across_scenes=True,
    )
    assert entity.entity_id in {c.entity_id for c in candidates}

    # Without preservation the cross-scene not_visible entity is excluded.
    candidates_no_preserve = registry.find_candidates(
        empty_state,
        EntityType.PERSON,
        result.scene_id,
        current_window=2,
        preserve_entities_across_scenes=False,
    )
    assert entity.entity_id not in {c.entity_id for c in candidates_no_preserve}


def test_find_candidates_respects_preserve_flag_via_scene_result(
    tracker: SceneTracker,
    registry: EntityRegistry,
    empty_state: GlobalState,
) -> None:
    config_no_preserve = SceneTrackerConfig(
        enabled=True,
        camera_change_starts_new_scene=True,
        preserve_entities_across_scenes=False,
    )
    tracker_no_preserve = SceneTracker(config_no_preserve)

    wide_obs = _observation(0, ViewType.WIDE)
    scene_result = tracker_no_preserve.update(empty_state, wide_obs)
    entity = registry.create_entity(
        empty_state,
        EntityType.PERSON,
        name="worker",
        window_index=0,
        scene_id=scene_result.scene_id,
    )
    empty_state.scenes[scene_result.scene_id].visible_entity_ids.append(entity.entity_id)
    empty_state.scenes[scene_result.scene_id].visible_entity_ids.sort()

    closeup_obs = _observation(1, ViewType.CLOSEUP)
    tracker_no_preserve.update(empty_state, closeup_obs)

    change_obs = _observation(2, ViewType.WIDE, camera_change=True, continuity_hint="camera_change")
    result = tracker_no_preserve.update(empty_state, change_obs)

    assert result.preserve_entities_across_scenes is False
