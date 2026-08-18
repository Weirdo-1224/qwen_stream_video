"""Unit tests for ActionTracker lifecycle and identity rules."""

from __future__ import annotations

import pytest

from qwen_stream_video.config import ActionTrackerConfig
from qwen_stream_video.domain import (
    ActionLifecycle,
    ActionObservation,
    EntityResolution,
    EntityResolutionBatch,
    GlobalState,
    ObservationBatch,
    SceneObservation,
    TimeInterval,
    WindowObservation,
)
from qwen_stream_video.state import ActionTracker, SceneUpdateResult
from qwen_stream_video.video import SampledFrame, VideoWindow


def _window(
    global_index: int,
    start_seconds: float,
    commit_start_seconds: float,
    end_seconds: float,
) -> WindowObservation:
    return WindowObservation(
        global_index=global_index,
        start_seconds=start_seconds,
        commit_start_seconds=commit_start_seconds,
        end_seconds=end_seconds,
    )


def _sampled_frames(
    global_index: int,
    start_seconds: float,
    count: int,
    step: float = 1.0,
    run_index: int = 0,
) -> list[SampledFrame]:
    return [
        SampledFrame(
            run_index=run_index,
            global_index=global_index,
            sample_index=i,
            frame_index=int((start_seconds + i * step) * 1),
            timestamp_seconds=start_seconds + i * step,
        )
        for i in range(count)
    ]


def _action(
    local_id: str,
    action_type: str,
    evidence_frames: list[int],
    actor_local_id: str = "A1",
    confidence: float = 0.9,
    target_local_id: str | None = None,
    tool_local_id: str | None = None,
) -> ActionObservation:
    return ActionObservation(
        local_id=local_id,
        actor_local_id=actor_local_id,
        action_type=action_type,
        evidence_frames=evidence_frames,
        confidence=confidence,
        target_local_id=target_local_id,
        tool_local_id=tool_local_id,
    )


def _resolutions(
    window_global_index: int,
    mappings: dict[str, str],
) -> EntityResolutionBatch:
    return EntityResolutionBatch(
        window_global_index=window_global_index,
        mappings=[
            EntityResolution(
                window_global_index=window_global_index,
                local_id=local_id,
                global_entity_id=global_id,
                status="matched",
            )
            for local_id, global_id in sorted(mappings.items())
        ],
    )


def _scene_result(scene_id: str = "scene_0001", camera_change: bool = False) -> SceneUpdateResult:
    return SceneUpdateResult(
        scene_id=scene_id,
        scene_changed=camera_change,
        continuity="camera_change" if camera_change else "continuous",
        camera_change=camera_change,
    )


@pytest.fixture
def empty_state() -> GlobalState:
    return GlobalState(run_id="test")


@pytest.fixture
def tracker() -> ActionTracker:
    return ActionTracker(ActionTrackerConfig())


def test_new_action_started_in_commit_interval(
    empty_state: GlobalState, tracker: ActionTracker
) -> None:
    obs = ObservationBatch(
        window=_window(0, 0.0, 0.0, 6.0),
        scene=SceneObservation(),
        actions=[_action("a1", "use", [1])],
    )
    frames = _sampled_frames(0, 0.0, 3)
    resolution = _resolutions(0, {"A1": "entity_001"})

    result = tracker.update(empty_state, obs, resolution, frames, _scene_result())

    assert result.action_ids == ["action_000001"]
    action = empty_state.actions["action_000001"]
    assert action.lifecycle == ActionLifecycle.STARTED
    assert action.actor_id == "entity_001"
    assert action.start_window == 0
    assert action.start_time_interval == TimeInterval(lower=0.0, upper=1.0)
    assert action.last_observed_time == 1.0
    assert empty_state.active_action_ids == ["action_000001"]

    started = [e for e in result.events if e.event_type == "action_started"]
    assert len(started) == 1
    assert started[0].evidence
    assert started[0].evidence[0].sample_indices == [1]


def test_overlapping_action_is_continued_not_duplicated(
    empty_state: GlobalState, tracker: ActionTracker
) -> None:
    first = ObservationBatch(
        window=_window(0, 0.0, 0.0, 6.0),
        scene=SceneObservation(),
        actions=[_action("a1", "use", [2])],
    )
    frames0 = _sampled_frames(0, 0.0, 3)
    resolution = _resolutions(0, {"A1": "entity_001"})

    tracker.update(empty_state, first, resolution, frames0, _scene_result())

    second = ObservationBatch(
        window=_window(1, 3.0, 6.0, 9.0),
        scene=SceneObservation(),
        actions=[_action("a1", "use", [2])],
    )
    frames1 = _sampled_frames(1, 3.0, 3, step=1.5)  # 3.0, 4.5, 6.0

    result = tracker.update(empty_state, second, resolution, frames1, _scene_result())

    assert result.action_ids == ["action_000001"]
    assert len(empty_state.actions) == 1
    action = empty_state.actions["action_000001"]
    assert action.observed_windows == [0, 1]
    assert action.lifecycle == ActionLifecycle.ONGOING

    continued = [e for e in result.events if e.event_type == "action_continued"]
    assert len(continued) == 1
    assert continued[0].evidence
    assert continued[0].evidence[0].timestamps_seconds == [6.0]


def test_action_possible_ended_after_one_missing_window(
    empty_state: GlobalState, tracker: ActionTracker
) -> None:
    obs = ObservationBatch(
        window=_window(0, 0.0, 0.0, 6.0),
        scene=SceneObservation(),
        actions=[_action("a1", "use", [1])],
    )
    frames = _sampled_frames(0, 0.0, 3)
    resolution = _resolutions(0, {"A1": "entity_001"})
    tracker.update(empty_state, obs, resolution, frames, _scene_result())

    missing_window = VideoWindow(
        global_index=1,
        run_index=1,
        start_seconds=3.0,
        commit_start_seconds=6.0,
        end_seconds=9.0,
    )
    events = tracker.mark_missing(empty_state, missing_window)

    action = empty_state.actions["action_000001"]
    assert action.lifecycle == ActionLifecycle.POSSIBLE_ENDED
    assert action.missing_window_count == 1
    assert empty_state.active_action_ids == ["action_000001"]

    possible = [e for e in events if e.event_type == "action_possible_ended"]
    assert len(possible) == 1
    assert possible[0].evidence


def test_action_ended_after_configured_missing_windows(
    empty_state: GlobalState, tracker: ActionTracker
) -> None:
    obs = ObservationBatch(
        window=_window(0, 0.0, 0.0, 6.0),
        scene=SceneObservation(),
        actions=[_action("a1", "use", [1])],
    )
    frames = _sampled_frames(0, 0.0, 3)
    resolution = _resolutions(0, {"A1": "entity_001"})
    tracker.update(empty_state, obs, resolution, frames, _scene_result())

    action = empty_state.actions["action_000001"]
    last_observed_time = action.last_observed_time

    missing_window_1 = VideoWindow(
        global_index=1,
        run_index=1,
        start_seconds=3.0,
        commit_start_seconds=6.0,
        end_seconds=9.0,
    )
    tracker.mark_missing(empty_state, missing_window_1)
    assert action.lifecycle == ActionLifecycle.POSSIBLE_ENDED

    missing_window_2 = VideoWindow(
        global_index=2,
        run_index=2,
        start_seconds=6.0,
        commit_start_seconds=9.0,
        end_seconds=12.0,
    )
    events = tracker.mark_missing(empty_state, missing_window_2)

    assert action.lifecycle == ActionLifecycle.ENDED
    assert action.end_window == 2
    assert action.end_time_interval == TimeInterval(
        lower=last_observed_time, upper=12.0
    )
    assert empty_state.active_action_ids == []

    ended = [e for e in events if e.event_type == "action_ended"]
    assert len(ended) == 1
    assert ended[0].evidence


def test_camera_change_makes_action_uncertain(
    empty_state: GlobalState, tracker: ActionTracker
) -> None:
    obs = ObservationBatch(
        window=_window(0, 0.0, 0.0, 6.0),
        scene=SceneObservation(),
        actions=[_action("a1", "use", [1])],
    )
    frames = _sampled_frames(0, 0.0, 3)
    resolution = _resolutions(0, {"A1": "entity_001"})
    tracker.update(empty_state, obs, resolution, frames, _scene_result())

    change_window = VideoWindow(
        global_index=1,
        run_index=1,
        start_seconds=3.0,
        commit_start_seconds=6.0,
        end_seconds=9.0,
    )
    events = tracker.mark_missing(
        empty_state, change_window, camera_change=True, observed_action_ids=set()
    )

    action = empty_state.actions["action_000001"]
    assert action.lifecycle == ActionLifecycle.UNCERTAIN
    assert action.action_id in empty_state.active_action_ids

    uncertain = [e for e in events if e.event_type == "action_uncertain"]
    assert len(uncertain) == 1
    assert uncertain[0].evidence


def test_instant_action_not_repeated_from_context_interval(
    empty_state: GlobalState, tracker: ActionTracker
) -> None:
    first = ObservationBatch(
        window=_window(0, 0.0, 0.0, 6.0),
        scene=SceneObservation(),
        actions=[_action("a1", "press", [1])],
    )
    frames0 = _sampled_frames(0, 0.0, 3)
    resolution = _resolutions(0, {"A1": "entity_001"})

    tracker.update(empty_state, first, resolution, frames0, _scene_result())

    assert len(empty_state.actions) == 1
    assert empty_state.active_action_ids == []
    action = empty_state.actions["action_000001"]
    assert action.lifecycle == ActionLifecycle.INSTANT

    second = ObservationBatch(
        window=_window(1, 3.0, 6.0, 9.0),
        scene=SceneObservation(),
        actions=[_action("a1", "press", [0])],
    )
    frames1 = _sampled_frames(1, 3.0, 3)  # evidence at 3.0 is in context interval only
    result = tracker.update(empty_state, second, resolution, frames1, _scene_result())

    assert len(empty_state.actions) == 1
    assert result.warnings
    assert any("Context-only" in w for w in result.warnings)
    assert empty_state.active_action_ids == []


def test_repeated_same_action_after_gap_gets_new_id(
    empty_state: GlobalState,
) -> None:
    config = ActionTrackerConfig(
        end_missing_windows=1,
        repeat_action_min_gap_seconds=1.0,
    )
    tracker = ActionTracker(config)

    obs = ObservationBatch(
        window=_window(0, 0.0, 0.0, 6.0),
        scene=SceneObservation(),
        actions=[_action("a1", "use", [1])],
    )
    frames = _sampled_frames(0, 0.0, 3)
    resolution = _resolutions(0, {"A1": "entity_001"})
    tracker.update(empty_state, obs, resolution, frames, _scene_result())

    gap_window = VideoWindow(
        global_index=1,
        run_index=1,
        start_seconds=3.0,
        commit_start_seconds=6.0,
        end_seconds=9.0,
    )
    tracker.mark_missing(empty_state, gap_window)
    assert empty_state.actions["action_000001"].lifecycle == ActionLifecycle.ENDED
    assert empty_state.active_action_ids == []

    repeat = ObservationBatch(
        window=_window(2, 6.0, 9.0, 12.0),
        scene=SceneObservation(),
        actions=[_action("a1", "use", [3])],
    )
    frames2 = _sampled_frames(2, 6.0, 4, step=1.0)  # 6.0, 7.0, 8.0, 9.0
    result = tracker.update(empty_state, repeat, resolution, frames2, _scene_result())

    assert len(empty_state.actions) == 2
    assert "action_000001" in empty_state.actions
    assert "action_000002" in empty_state.actions
    assert result.action_ids == ["action_000002"]
    assert empty_state.actions["action_000002"].lifecycle == ActionLifecycle.STARTED
    assert empty_state.active_action_ids == ["action_000002"]


def test_unresolved_actor_does_not_create_action(
    empty_state: GlobalState, tracker: ActionTracker
) -> None:
    obs = ObservationBatch(
        window=_window(0, 0.0, 0.0, 6.0),
        scene=SceneObservation(),
        actions=[_action("a1", "use", [1])],
    )
    frames = _sampled_frames(0, 0.0, 3)
    resolution = _resolutions(0, {"other": "entity_001"})  # actor A1 is not resolved

    result = tracker.update(empty_state, obs, resolution, frames, _scene_result())

    assert not empty_state.actions
    assert not empty_state.active_action_ids
    assert any("unresolved actor" in w for w in result.warnings)
    assert not [e for e in result.events if e.event_type == "action_started"]
