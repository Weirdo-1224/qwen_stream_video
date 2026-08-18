from __future__ import annotations

import numpy as np

from qwen_stream_video.config import TransitionEngineConfig
from qwen_stream_video.domain import (
    AttributeConfirmationStatus,
    AttributeObservation,
    AttributeState,
    EntityObservation,
    EntityResolution,
    EntityResolutionBatch,
    EntityResolutionStatus,
    EntityType,
    GlobalActionState,
    GlobalEntityState,
    GlobalState,
    ObservationBatch,
    SceneObservation,
    VisibilityState,
    WindowObservation,
)
from qwen_stream_video.state import TransitionEngine, TransitionUpdateResult
from qwen_stream_video.state.action_tracker import ActionUpdateResult
from qwen_stream_video.video import SampledFrame, VideoWindow

HIGH = 0.9
MEDIUM = 0.7
LOW = 0.4


def _sampled_frames(window_index: int, count: int = 4) -> list[SampledFrame]:
    base_ts = window_index * 3
    return [
        SampledFrame(
            run_index=window_index,
            global_index=window_index,
            sample_index=i,
            frame_index=base_ts + i,
            timestamp_seconds=base_ts + i,
            image=np.zeros((2, 2, 3), dtype="uint8"),
        )
        for i in range(count)
    ]


def _video_window(
    global_index: int,
    start: float | None = None,
    commit_start: float | None = None,
    end: float | None = None,
) -> VideoWindow:
    start = start if start is not None else global_index * 3.0
    end = end if end is not None else start + 3.0
    commit_start = commit_start if commit_start is not None else start
    return VideoWindow(
        global_index=global_index,
        run_index=global_index,
        start_seconds=start,
        commit_start_seconds=commit_start,
        end_seconds=end,
    )


def _fresh_state() -> GlobalState:
    return GlobalState(run_id="test")


def _device_entity(
    state: GlobalState,
    entity_id: str = "device_0001",
    window: int = 0,
    visibility: VisibilityState = VisibilityState.VISIBLE,
) -> GlobalEntityState:
    entity = GlobalEntityState(
        entity_id=entity_id,
        entity_type=EntityType.DEVICE,
        first_seen_window=window,
        last_seen_window=window,
        visibility=visibility,
    )
    state.entities[entity_id] = entity
    return entity


def _resolution(
    local_id: str,
    global_entity_id: str,
    status: EntityResolutionStatus = EntityResolutionStatus.MATCHED,
) -> EntityResolution:
    return EntityResolution(
        window_global_index=0,
        local_id=local_id,
        global_entity_id=global_entity_id,
        status=status,
        selected_score=0.95,
        second_best_score=None,
    )


def _attribute_obs(
    local_id: str,
    key: str,
    value: str,
    confidence: float,
    frames: list[int] | None = None,
    normalization_status: str = "canonical",
) -> AttributeObservation:
    return AttributeObservation(
        entity_local_id=local_id,
        attribute_key=key,
        value=value,
        confidence=confidence,
        evidence_frames=frames if frames is not None else [0],
        normalization_status=normalization_status,  # type: ignore[arg-type]
    )


def _observation_batch(
    window_index: int,
    attributes: list[AttributeObservation],
    camera_change: bool = False,
    entities: list[EntityObservation] | None = None,
) -> ObservationBatch:
    return ObservationBatch(
        window=WindowObservation(
            global_index=window_index,
            start_seconds=window_index * 3.0,
            commit_start_seconds=window_index * 3.0,
            end_seconds=window_index * 3.0 + 3.0,
        ),
        scene=SceneObservation(camera_change=camera_change),
        entities=entities or [],
        attribute_observations=attributes,
    )


def _run_engine(
    state: GlobalState,
    observation: ObservationBatch,
    window: VideoWindow,
    action_ids: list[str] | None = None,
) -> None:
    engine = TransitionEngine(TransitionEngineConfig())
    resolutions = EntityResolutionBatch(
        window_global_index=window.global_index,
        mappings=[
            _resolution(local_id=attr.entity_local_id, global_entity_id=attr.entity_local_id)
            for attr in observation.attribute_observations
        ],
    )
    action_result = ActionUpdateResult(
        window_global_index=window.global_index,
        action_ids=action_ids or [],
    )
    engine.update(state, observation, resolutions, action_result, _sampled_frames(window.global_index), window)


def test_high_confidence_initial_value_is_initialized() -> None:
    state = _fresh_state()
    _device_entity(state)
    obs = _observation_batch(
        0, [_attribute_obs("device_0001", "door.state", "closed", HIGH)]
    )
    _run_engine(state, obs, _video_window(0))

    attr = state.entities["device_0001"].attributes["door.state"]
    assert attr.status == AttributeConfirmationStatus.CONFIRMED
    assert attr.value == "closed"
    assert attr.confidence == HIGH
    assert attr.supporting_observations == 1


def test_initial_value_is_not_transition() -> None:
    state = _fresh_state()
    _device_entity(state)
    obs = _observation_batch(
        0, [_attribute_obs("device_0001", "door.state", "closed", HIGH)]
    )
    engine = TransitionEngine(TransitionEngineConfig())
    resolutions = EntityResolutionBatch(
        window_global_index=0,
        mappings=[_resolution("device_0001", "device_0001")],
    )
    action_result = ActionUpdateResult(window_global_index=0)
    result = engine.update(state, obs, resolutions, action_result, _sampled_frames(0), _video_window(0))

    assert all(event.event_type != "attribute_transition" for event in result.events)
    assert any(event.event_type == "attribute_initialized" for event in result.events)


def test_same_value_does_not_emit_duplicate_transition() -> None:
    state = _fresh_state()
    _device_entity(state)
    # First observation initializes the attribute.
    _run_engine(
        state,
        _observation_batch(0, [_attribute_obs("device_0001", "door.state", "closed", HIGH)]),
        _video_window(0),
    )
    # Second observation with the same value must not emit a transition.
    result_events: list[str] = []
    for idx in (1, 2):
        engine = TransitionEngine(TransitionEngineConfig())
        obs = _observation_batch(
            idx, [_attribute_obs("device_0001", "door.state", "closed", HIGH)]
        )
        resolutions = EntityResolutionBatch(
            window_global_index=idx,
            mappings=[_resolution("device_0001", "device_0001")],
        )
        action_result = ActionUpdateResult(window_global_index=idx)
        result = engine.update(state, obs, resolutions, action_result, _sampled_frames(idx), _video_window(idx))
        result_events.extend([event.event_type for event in result.events])

    assert "attribute_transition" not in result_events
    assert state.entities["device_0001"].attributes["door.state"].supporting_observations == 3


def test_medium_confidence_requires_two_windows() -> None:
    state = _fresh_state()
    _device_entity(state)
    engine = TransitionEngine(TransitionEngineConfig())
    for idx in (0, 1):
        obs = _observation_batch(
            idx, [_attribute_obs("device_0001", "door.state", "closed", MEDIUM)]
        )
        resolutions = EntityResolutionBatch(
            window_global_index=idx,
            mappings=[_resolution("device_0001", "device_0001")],
        )
        action_result = ActionUpdateResult(window_global_index=idx)
        result = engine.update(state, obs, resolutions, action_result, _sampled_frames(idx), _video_window(idx))
        if idx == 0:
            assert any(event.event_type == "attribute_pending" for event in result.events)
            assert state.entities["device_0001"].attributes["door.state"].status == AttributeConfirmationStatus.PENDING
        else:
            assert any(event.event_type == "attribute_confirmed" for event in result.events)
            attr = state.entities["device_0001"].attributes["door.state"]
            assert attr.status == AttributeConfirmationStatus.CONFIRMED
            assert attr.value == "closed"


def test_high_confidence_transition_requires_action_support() -> None:
    state = _fresh_state()
    _device_entity(state)
    # Initialize as closed.
    _run_engine(
        state,
        _observation_batch(0, [_attribute_obs("device_0001", "door.state", "closed", HIGH)]),
        _video_window(0),
    )

    # High-confidence "open" without a supporting action -> pending, not transition.
    _run_engine(
        state,
        _observation_batch(1, [_attribute_obs("device_0001", "door.state", "open", HIGH)]),
        _video_window(1),
    )
    attr = state.entities["device_0001"].attributes["door.state"]
    assert attr.status == AttributeConfirmationStatus.CONFIRMED
    assert attr.value == "closed"
    assert attr.pending_value == "open"

    # Add a supporting action and observe "open" again.
    state.actions["action_000001"] = GlobalActionState(
        action_id="action_000001",
        actor_id="person_0001",
        action_type="open",
        target_id="device_0001",
    )
    _run_engine(
        state,
        _observation_batch(2, [_attribute_obs("device_0001", "door.state", "open", HIGH)]),
        _video_window(2),
        action_ids=["action_000001"],
    )
    attr = state.entities["device_0001"].attributes["door.state"]
    assert attr.status == AttributeConfirmationStatus.CONFIRMED
    assert attr.value == "open"
    assert attr.pending_value is None


def test_context_only_attribute_does_not_create_transition() -> None:
    state = _fresh_state()
    _device_entity(state)
    engine = TransitionEngine(TransitionEngineConfig())

    # Evidence falls entirely before the commit interval -> cannot create a state.
    obs = _observation_batch(
        0, [_attribute_obs("device_0001", "door.state", "closed", HIGH)]
    )
    window = _video_window(0, commit_start=1.5)
    resolutions = EntityResolutionBatch(
        window_global_index=0,
        mappings=[_resolution("device_0001", "device_0001")],
    )
    result = engine.update(
        state,
        obs,
        resolutions,
        ActionUpdateResult(window_global_index=0),
        _sampled_frames(0),
        window,
    )
    assert "door.state" not in state.entities["device_0001"].attributes
    assert any("Context-only attribute" in warning for warning in result.warnings)

    # An existing pending value can be supported by context evidence, but not confirmed.
    state2 = _fresh_state()
    _device_entity(state2)
    _run_engine(
        state2,
        _observation_batch(0, [_attribute_obs("device_0001", "door.state", "open", MEDIUM)]),
        _video_window(0),
    )
    attr = state2.entities["device_0001"].attributes["door.state"]
    assert attr.status == AttributeConfirmationStatus.PENDING

    _run_engine(
        state2,
        _observation_batch(1, [_attribute_obs("device_0001", "door.state", "open", MEDIUM)]),
        _video_window(1, commit_start=4.5),
    )
    attr = state2.entities["device_0001"].attributes["door.state"]
    assert attr.status == AttributeConfirmationStatus.PENDING


def test_newly_visible_attribute_is_not_false_transition() -> None:
    state = _fresh_state()
    entity = _device_entity(state, window=0)
    # Initialize the attribute in window 0.
    _run_engine(
        state,
        _observation_batch(0, [_attribute_obs("device_0001", "door.state", "closed", HIGH)]),
        _video_window(0),
    )
    attr = entity.attributes["door.state"]
    assert attr.status == AttributeConfirmationStatus.CONFIRMED

    # Simulate the attribute reappearing after a visibility gap with the same value.
    attr.last_observed_window = 0
    result = _run_engine_return(
        state,
        _observation_batch(2, [_attribute_obs("device_0001", "door.state", "closed", HIGH)]),
        _video_window(2),
    )

    assert attr.status == AttributeConfirmationStatus.CONFIRMED
    assert attr.value == "closed"
    assert attr.supporting_observations == 2
    assert "attribute_transition" not in [event.event_type for event in result.events]


def _run_engine_return(
    state: GlobalState,
    observation: ObservationBatch,
    window: VideoWindow,
    action_ids: list[str] | None = None,
) -> TransitionUpdateResult:
    engine = TransitionEngine(TransitionEngineConfig())
    resolutions = EntityResolutionBatch(
        window_global_index=window.global_index,
        mappings=[
            _resolution(local_id=attr.entity_local_id, global_entity_id=attr.entity_local_id)
            for attr in observation.attribute_observations
        ],
    )
    action_result = ActionUpdateResult(
        window_global_index=window.global_index,
        action_ids=action_ids or [],
    )
    return engine.update(state, observation, resolutions, action_result, _sampled_frames(window.global_index), window)


def test_conflicting_pending_value_is_cancelled() -> None:
    state = _fresh_state()
    entity = _device_entity(state)
    # confirmed = closed, pending = open
    entity.attributes["door.state"] = AttributeState(
        attribute_key="door.state",
        value="closed",
        confidence=HIGH,
        status=AttributeConfirmationStatus.CONFIRMED,
        first_observed_window=0,
        last_observed_window=0,
        confirmed_window=0,
        supporting_observations=1,
        pending_value="open",
        pending_confidence=MEDIUM,
        pending_support_windows=[1],
    )

    _run_engine(
        state,
        _observation_batch(2, [_attribute_obs("device_0001", "door.state", "closed", HIGH)]),
        _video_window(2),
    )
    attr = entity.attributes["door.state"]
    assert attr.status == AttributeConfirmationStatus.CONFIRMED
    assert attr.value == "closed"
    assert attr.pending_value is None
    assert attr.pending_support_windows == []
    assert attr.supporting_observations == 2


def test_third_conflicting_value_enters_conflicted() -> None:
    state = _fresh_state()
    entity = _device_entity(state)
    entity.attributes["door.state"] = AttributeState(
        attribute_key="door.state",
        value="closed",
        confidence=HIGH,
        status=AttributeConfirmationStatus.CONFIRMED,
        first_observed_window=0,
        last_observed_window=0,
        confirmed_window=0,
        supporting_observations=1,
        pending_value="open",
        pending_confidence=MEDIUM,
        pending_support_windows=[1],
    )

    result = _run_engine_return(
        state,
        _observation_batch(2, [_attribute_obs("device_0001", "door.state", "ajar", MEDIUM)]),
        _video_window(2),
    )
    attr = entity.attributes["door.state"]
    assert attr.status == AttributeConfirmationStatus.CONFLICTED
    assert attr.value == "closed"
    assert attr.pending_value == "open"
    assert any(event.event_type == "attribute_conflict" for event in result.events)


def test_attribute_invalid_for_entity_type_does_not_update_state() -> None:
    state = _fresh_state()
    _device_entity(state)
    result = _run_engine_return(
        state,
        _observation_batch(
            0,
            [
                _attribute_obs(
                    "device_0001",
                    "door.state",
                    "closed",
                    HIGH,
                    normalization_status="invalid_for_entity_type",
                )
            ],
        ),
        _video_window(0),
    )
    assert "door.state" not in state.entities["device_0001"].attributes
    assert any("invalid_for_entity_type" in warning for warning in result.warnings)


def test_cabinet_label_change_does_not_mutate_same_entity_without_match() -> None:
    state = _fresh_state()
    _device_entity(state, entity_id="device_0001")
    _device_entity(state, entity_id="device_0002")
    # Attribute for local_id that resolves to a temporary/ambiguous entity.
    obs = _observation_batch(
        0,
        [
            _attribute_obs("cabinet_new", "door.state", "open", HIGH),
        ],
    )
    resolutions = EntityResolutionBatch(
        window_global_index=0,
        mappings=[
            _resolution(
                "cabinet_new",
                "temp_device_0001",
                status=EntityResolutionStatus.TEMPORARY,
            )
        ],
    )
    engine = TransitionEngine(TransitionEngineConfig())
    engine.update(
        state,
        obs,
        resolutions,
        ActionUpdateResult(window_global_index=0),
        _sampled_frames(0),
        _video_window(0),
    )
    # The original cabinet must not be mutated by an unmatched observation.
    assert "door.state" not in state.entities["device_0001"].attributes
    assert "door.state" not in state.entities["device_0002"].attributes


def test_low_confidence_creates_observed_event() -> None:
    state = _fresh_state()
    _device_entity(state)
    result = _run_engine_return(
        state,
        _observation_batch(0, [_attribute_obs("device_0001", "door.state", "closed", LOW)]),
        _video_window(0),
    )
    attr = state.entities["device_0001"].attributes["door.state"]
    assert attr.status == AttributeConfirmationStatus.OBSERVED
    assert any(event.event_type == "attribute_observed" for event in result.events)


def test_pending_support_windows_are_deduplicated() -> None:
    state = _fresh_state()
    _device_entity(state)
    engine = TransitionEngine(TransitionEngineConfig())
    # Two observations of the same value in the same window should not inflate the support count.
    obs = _observation_batch(
        0,
        [
            _attribute_obs("device_0001", "door.state", "open", MEDIUM),
            _attribute_obs("device_0001", "door.state", "open", MEDIUM),
        ],
    )
    resolutions = EntityResolutionBatch(
        window_global_index=0,
        mappings=[
            _resolution("device_0001", "device_0001"),
            _resolution("device_0001", "device_0001"),
        ],
    )
    engine.update(
        state,
        obs,
        resolutions,
        ActionUpdateResult(window_global_index=0),
        _sampled_frames(0),
        _video_window(0),
    )
    attr = state.entities["device_0001"].attributes["door.state"]
    assert attr.pending_support_windows == [0]


def test_camera_change_breaks_consecutive_support() -> None:
    state = _fresh_state()
    _device_entity(state)
    engine = TransitionEngine(TransitionEngineConfig())
    # Window 0: pending open.
    obs0 = _observation_batch(
        0, [_attribute_obs("device_0001", "door.state", "open", MEDIUM)]
    )
    engine.update(
        state,
        obs0,
        EntityResolutionBatch(
            window_global_index=0,
            mappings=[_resolution("device_0001", "device_0001")],
        ),
        ActionUpdateResult(window_global_index=0),
        _sampled_frames(0),
        _video_window(0),
    )
    # Window 1 with camera_change: same value should not continue the previous chain.
    obs1 = _observation_batch(
        1,
        [_attribute_obs("device_0001", "door.state", "open", MEDIUM)],
        camera_change=True,
    )
    engine.update(
        state,
        obs1,
        EntityResolutionBatch(
            window_global_index=1,
            mappings=[_resolution("device_0001", "device_0001")],
        ),
        ActionUpdateResult(window_global_index=1),
        _sampled_frames(1),
        _video_window(1),
    )
    attr = state.entities["device_0001"].attributes["door.state"]
    assert attr.pending_support_windows == [1]
    assert attr.status == AttributeConfirmationStatus.PENDING


def test_max_pending_gap_windows_expires_pending() -> None:
    state = _fresh_state()
    _device_entity(state)
    engine = TransitionEngine(TransitionEngineConfig())
    # Window 0: pending open.
    engine.update(
        state,
        _observation_batch(0, [_attribute_obs("device_0001", "door.state", "open", MEDIUM)]),
        EntityResolutionBatch(
            window_global_index=0,
            mappings=[_resolution("device_0001", "device_0001")],
        ),
        ActionUpdateResult(window_global_index=0),
        _sampled_frames(0),
        _video_window(0),
    )
    # Window 2 (> max_pending_gap_windows of 1) without support -> pending expires.
    result = engine.update(
        state,
        _observation_batch(2, [_attribute_obs("device_0001", "door.state", "closed", MEDIUM)]),
        EntityResolutionBatch(
            window_global_index=2,
            mappings=[_resolution("device_0001", "device_0001")],
        ),
        ActionUpdateResult(window_global_index=2),
        _sampled_frames(2),
        _video_window(2),
    )
    attr = state.entities["device_0001"].attributes["door.state"]
    assert attr.status == AttributeConfirmationStatus.PENDING
    assert attr.pending_value == "closed"
    assert any(event.event_type == "attribute_pending_expired" for event in result.events)


def test_pending_expires_when_attribute_not_observed_for_gap() -> None:
    """A pending attribute whose observation is completely absent for enough windows expires."""
    state = _fresh_state()
    _device_entity(state)
    engine = TransitionEngine(TransitionEngineConfig())
    # Window 0: medium confidence observation -> pending.
    engine.update(
        state,
        _observation_batch(0, [_attribute_obs("device_0001", "door.state", "open", MEDIUM)]),
        EntityResolutionBatch(
            window_global_index=0,
            mappings=[_resolution("device_0001", "device_0001")],
        ),
        ActionUpdateResult(window_global_index=0),
        _sampled_frames(0),
        _video_window(0),
    )
    attr = state.entities["device_0001"].attributes["door.state"]
    assert attr.status == AttributeConfirmationStatus.PENDING
    assert "device_0001:door.state" in state.pending_attribute_keys

    # Window 2 has no attribute observation at all. The pending value must expire.
    result = engine.update(
        state,
        _observation_batch(2, []),
        EntityResolutionBatch(window_global_index=2, mappings=[]),
        ActionUpdateResult(window_global_index=2),
        _sampled_frames(2),
        _video_window(2),
    )
    attr = state.entities["device_0001"].attributes["door.state"]
    assert attr.pending_value is None
    assert attr.pending_confidence is None
    assert attr.pending_support_windows == []
    assert attr.status == AttributeConfirmationStatus.OBSERVED
    assert attr.value == "open"
    assert any(
        event.event_type == "attribute_pending_expired"
        and event.reason == "pending_support_gap_exceeded"
        for event in result.events
    )
    assert "device_0001:door.state" not in state.pending_attribute_keys


def test_pending_from_confirmed_value_expires_back_to_confirmed() -> None:
    """A pending transition expires back to its confirmed value when unsupported for a gap."""
    state = _fresh_state()
    entity = _device_entity(state)
    entity.attributes["door.state"] = AttributeState(
        attribute_key="door.state",
        value="closed",
        confidence=HIGH,
        status=AttributeConfirmationStatus.CONFIRMED,
        first_observed_window=0,
        last_observed_window=0,
        confirmed_window=0,
        supporting_observations=1,
        pending_value="open",
        pending_confidence=MEDIUM,
        pending_support_windows=[1],
    )
    engine = TransitionEngine(TransitionEngineConfig())
    # Window 3 has no observation for this attribute.
    result = engine.update(
        state,
        _observation_batch(3, []),
        EntityResolutionBatch(window_global_index=3, mappings=[]),
        ActionUpdateResult(window_global_index=3),
        _sampled_frames(3),
        _video_window(3),
    )
    attr = entity.attributes["door.state"]
    assert attr.pending_value is None
    assert attr.status == AttributeConfirmationStatus.CONFIRMED
    assert attr.value == "closed"
    assert any(event.event_type == "attribute_pending_expired" for event in result.events)


def test_transition_event_includes_support_window_evidence() -> None:
    state = _fresh_state()
    _device_entity(state)
    engine = TransitionEngine(TransitionEngineConfig())
    for idx in (0, 1):
        obs = _observation_batch(
            idx, [_attribute_obs("device_0001", "door.state", "open", MEDIUM)]
        )
        resolutions = EntityResolutionBatch(
            window_global_index=idx,
            mappings=[_resolution("device_0001", "device_0001")],
        )
        result = engine.update(
            state,
            obs,
            resolutions,
            ActionUpdateResult(window_global_index=idx),
            _sampled_frames(idx),
            _video_window(idx),
        )

    transition_events = [event for event in result.events if event.event_type == "attribute_confirmed"]
    assert transition_events
    evidence_windows = [ref.window_global_index for ref in transition_events[0].evidence]
    assert 0 in evidence_windows
    assert 1 in evidence_windows


def test_action_support_considers_tool_id() -> None:
    state = _fresh_state()
    _device_entity(state, entity_id="device_0001")
    # Initialize door.state.
    _run_engine(
        state,
        _observation_batch(0, [_attribute_obs("device_0001", "door.state", "closed", HIGH)]),
        _video_window(0),
    )
    # Action where the entity is the tool_id should support a value change.
    state.actions["action_000001"] = GlobalActionState(
        action_id="action_000001",
        actor_id="person_0001",
        action_type="open",
        tool_id="device_0001",
    )
    result = _run_engine_return(
        state,
        _observation_batch(1, [_attribute_obs("device_0001", "door.state", "open", HIGH)]),
        _video_window(1),
        action_ids=["action_000001"],
    )
    attr = state.entities["device_0001"].attributes["door.state"]
    assert attr.status == AttributeConfirmationStatus.CONFIRMED
    assert attr.value == "open"
    assert any(event.event_type == "attribute_transition" for event in result.events)


def test_require_evidence_frames_config_is_respected() -> None:
    state = _fresh_state()
    _device_entity(state, entity_id="device_0001")
    engine = TransitionEngine(TransitionEngineConfig(), require_evidence_frames=False)
    obs = _observation_batch(
        0,
        [
            AttributeObservation(
                entity_local_id="device_0001",
                attribute_key="door.state",
                value="closed",
                confidence=HIGH,
                evidence_frames=[],
            )
        ],
    )
    resolutions = EntityResolutionBatch(
        window_global_index=0,
        mappings=[_resolution("device_0001", "device_0001")],
    )
    result = engine.update(
        state,
        obs,
        resolutions,
        ActionUpdateResult(window_global_index=0),
        _sampled_frames(0),
        _video_window(0),
    )
    attr = state.entities["device_0001"].attributes["door.state"]
    assert attr.status == AttributeConfirmationStatus.CONFIRMED
    assert attr.value == "closed"
    assert any(event.event_type == "attribute_initialized" for event in result.events)


def test_require_evidence_frames_true_rejects_empty_evidence() -> None:
    state = _fresh_state()
    _device_entity(state, entity_id="device_0001")
    engine = TransitionEngine(TransitionEngineConfig(), require_evidence_frames=True)
    obs = _observation_batch(
        0,
        [
            AttributeObservation(
                entity_local_id="device_0001",
                attribute_key="door.state",
                value="closed",
                confidence=HIGH,
                evidence_frames=[],
            )
        ],
    )
    resolutions = EntityResolutionBatch(
        window_global_index=0,
        mappings=[_resolution("device_0001", "device_0001")],
    )
    result = engine.update(
        state,
        obs,
        resolutions,
        ActionUpdateResult(window_global_index=0),
        _sampled_frames(0),
        _video_window(0),
    )
    assert "door.state" not in state.entities["device_0001"].attributes
    assert any("has no evidence" in warning for warning in result.warnings)
