from __future__ import annotations

import numpy as np

from qwen_stream_video.domain import (
    ActionObservation,
    EntityObservation,
    EntityType,
    GlobalState,
    ObservationBatch,
    WindowObservation,
)
from qwen_stream_video.state import StateReducer
from qwen_stream_video.video import SampledFrame, VideoWindow


def _frames(index: int) -> list[SampledFrame]:
    return [
        SampledFrame(
            run_index=index,
            global_index=index,
            sample_index=i,
            frame_index=i,
            timestamp_seconds=index * 3 + i,
            image=np.zeros((2, 2, 3), dtype="uint8"),
        )
        for i in range(3)
    ]


def test_reducer_keeps_identity_and_action_id_across_windows() -> None:
    reducer = StateReducer()
    state = GlobalState(run_id="integration")
    action_ids: list[str] = []
    for index in range(2):
        observation = ObservationBatch(
            window=WindowObservation(
                global_index=index,
                start_seconds=index * 3,
                commit_start_seconds=index * 3,
                end_seconds=index * 3 + 3,
            ),
            entities=[
                EntityObservation(local_id="P", entity_type=EntityType.PERSON, name="worker", confidence=0.95, evidence_frames=[0]),
                EntityObservation(local_id="D", entity_type=EntityType.DEVICE, name="cabinet 4", confidence=0.95, evidence_frames=[0]),
            ],
            actions=[
                ActionObservation(local_id="A", actor_local_id="P", target_local_id="D", action_type="inspect", confidence=0.9, evidence_frames=[1])
            ],
        )
        window = VideoWindow(
            global_index=index,
            run_index=index,
            start_seconds=index * 3,
            commit_start_seconds=index * 3,
            end_seconds=index * 3 + 3,
        )
        result = reducer.apply_observation(state, observation, _frames(index), window)
        state = result.state
        action_ids.extend(result.action_result.action_ids if result.action_result else [])
    assert sorted(state.entities) == ["device_0001", "person_0001"]
    assert action_ids == ["action_000001", "action_000001"]
