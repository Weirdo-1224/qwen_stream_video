from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from qwen_stream_video.config import AppConfig
from qwen_stream_video.domain import (
    ActionObservation,
    AttributeObservation,
    EntityObservation,
    EntityType,
    GlobalState,
    ObservationBatch,
    WindowObservation,
)
from qwen_stream_video.state import StateReducer
from qwen_stream_video.state.replay import ObservationReplay
from qwen_stream_video.video import SampledFrame, VideoWindow


def _frames(index: int, stride_seconds: float = 3.0, frame_count: int = 6) -> list[SampledFrame]:
    return [
        SampledFrame(
            run_index=index,
            global_index=index,
            sample_index=i,
            frame_index=i,
            timestamp_seconds=index * stride_seconds + i,
            image=np.zeros((2, 2, 3), dtype="uint8"),
        )
        for i in range(frame_count)
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


def _person(local_id: str, name: str, evidence_frames: list[int] | None = None) -> EntityObservation:
    regions = {"A": "left", "B": "center", "C": "right"}
    return EntityObservation(
        local_id=local_id,
        entity_type=EntityType.PERSON,
        name=name,
        confidence=0.95,
        evidence_frames=evidence_frames or [0],
        appearance={"uniform": "work_clothes"},
        spatial_region=regions.get(local_id, "unknown"),
    )


def _device(local_id: str, name: str, evidence_frames: list[int] | None = None) -> EntityObservation:
    regions = {"C4": "left", "C5": "right", "CP": "panel"}
    return EntityObservation(
        local_id=local_id,
        entity_type=EntityType.DEVICE,
        name=name,
        confidence=0.95,
        evidence_frames=evidence_frames or [0],
        appearance={"color": "gray"},
        spatial_region=regions.get(local_id, "unknown"),
    )


def _tool(local_id: str, name: str, evidence_frames: list[int] | None = None) -> EntityObservation:
    return EntityObservation(
        local_id=local_id,
        entity_type=EntityType.TOOL,
        name=name,
        confidence=0.95,
        evidence_frames=evidence_frames or [0],
        appearance={"color": "metal"},
        spatial_region="center",
    )


def _component(local_id: str, name: str, evidence_frames: list[int] | None = None) -> EntityObservation:
    return EntityObservation(
        local_id=local_id,
        entity_type=EntityType.COMPONENT,
        name=name,
        confidence=0.95,
        evidence_frames=evidence_frames or [0],
        appearance={"color": "red"},
        spatial_region="panel",
    )


def _action(
    local_id: str,
    action_type: str,
    actor_local_id: str,
    target_local_id: str | None = None,
    tool_local_id: str | None = None,
    evidence_frames: list[int] | None = None,
) -> ActionObservation:
    return ActionObservation(
        local_id=local_id,
        actor_local_id=actor_local_id,
        action_type=action_type,
        target_local_id=target_local_id,
        tool_local_id=tool_local_id,
        confidence=0.9,
        evidence_frames=evidence_frames or [3],
    )


def _attribute(
    entity_local_id: str,
    attribute_key: str,
    value: str,
    evidence_frames: list[int] | None = None,
) -> AttributeObservation:
    return AttributeObservation(
        entity_local_id=entity_local_id,
        attribute_key=attribute_key,
        value=value,
        confidence=0.9,
        evidence_frames=evidence_frames or [3],
    )


def _observation(
    index: int,
    entities: list[EntityObservation],
    actions: list[ActionObservation] | None = None,
    attributes: list[AttributeObservation] | None = None,
    camera_change: bool = False,
    view_type: str = "wide",
    *,
    window_seconds: float = 6.0,
    stride_seconds: float = 3.0,
) -> ObservationBatch:
    start = index * stride_seconds
    end = start + window_seconds
    commit_start = start if index == 0 else start + stride_seconds
    return ObservationBatch(
        window=WindowObservation(
            global_index=index,
            start_seconds=start,
            commit_start_seconds=commit_start,
            end_seconds=end,
        ),
        scene={"camera_change": camera_change, "view_type": view_type},
        entities=entities,
        actions=actions or [],
        attribute_observations=attributes or [],
    )


def _write_replay_input(
    tmp_path: Path,
    observations: list[ObservationBatch],
    *,
    window_seconds: float = 6.0,
    stride_seconds: float = 3.0,
) -> Path:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    observations_path = input_dir / "observations.jsonl"
    windows_path = input_dir / "windows.jsonl"
    with observations_path.open("w", encoding="utf-8") as handle:
        for observation in observations:
            handle.write(observation.model_dump_json() + "\n")
    with windows_path.open("w", encoding="utf-8") as handle:
        for index, observation in enumerate(observations):
            window_data = {
                "global_index": index,
                "run_index": index,
                "start_seconds": observation.window.start_seconds,
                "commit_start_seconds": observation.window.commit_start_seconds,
                "end_seconds": observation.window.end_seconds,
                "window_type": "regular",
                "processing_role": "commit",
                "sampled_frames": [
                    {"sample_index": i, "frame_index": i, "timestamp_seconds": observation.window.start_seconds + i}
                    for i in range(6)
                ],
            }
            handle.write(json.dumps(window_data, ensure_ascii=False, sort_keys=True) + "\n")
    (input_dir / "run_meta.json").write_text(
        json.dumps({"run_id": "stage2_test", "observation_schema_version": "2.0"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return input_dir


def test_eight_window_maintenance_sequence(tmp_path: Path) -> None:
    """Replay a true 8-window maintenance sequence with 6s windows and 3s stride.

    Window layout (context / commit):
      0: [0,6)  / [0,6)
      1: [3,9)  / [6,9)
      2: [6,12) / [9,12)
      3: [9,15) / [12,15)  camera change to control-panel closeup
      4: [12,18)/ [15,18)  indicator first visible
      5: [15,21)/ [18,21)  return wide
      6: [18,24)/ [21,24)  open cabinet 4
      7: [21,27)/ [24,27)  confirm cabinet 4 open
    """
    window_seconds = 6.0
    stride_seconds = 3.0
    observations = [
        # 1. 三人和两个柜体首次出现
        _observation(
            0,
            [
                _person("A", "person A"),
                _person("B", "person B"),
                _person("C", "person C"),
                _device("C4", "cabinet 4"),
                _device("C5", "cabinet 5"),
            ],
            attributes=[_attribute("C4", "door.state", "closed", [0])],
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
        ),
        # 2. 人员持续检查 5 号柜
        _observation(
            1,
            [
                _person("A", "person A"),
                _person("B", "person B"),
                _person("C", "person C"),
                _device("C4", "cabinet 4"),
                _device("C5", "cabinet 5"),
            ],
            actions=[_action("a1", "inspect", "B", "C5")],
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
        ),
        # 3. 中间人员递交工具
        _observation(
            2,
            [
                _person("A", "person A"),
                _person("B", "person B"),
                _person("C", "person C"),
                _device("C4", "cabinet 4"),
                _device("C5", "cabinet 5"),
                _tool("T", "wrench"),
            ],
            actions=[_action("a1", "hand_over", "B", "A", "T")],
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
        ),
        # 4. 镜头切到控制面板特写
        _observation(
            3,
            [
                _device("CP", "control panel"),
            ],
            camera_change=True,
            view_type="closeup",
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
        ),
        # 5. 指示灯首次可见 (continue closeup with indicator)
        _observation(
            4,
            [
                _device("CP", "control panel"),
                _component("IND", "indicator"),
            ],
            attributes=[_attribute("IND", "indicator.energy.color", "red")],
            view_type="closeup",
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
        ),
        # 6. 返回全景
        _observation(
            5,
            [
                _person("A", "person A"),
                _person("B", "person B"),
                _person("C", "person C"),
                _device("C4", "cabinet 4"),
                _device("C5", "cabinet 5"),
                _device("CP", "control panel"),
                _component("IND", "indicator"),
            ],
            view_type="wide",
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
        ),
        # 7. 人员打开 4 号柜门
        _observation(
            6,
            [
                _person("A", "person A"),
                _person("B", "person B"),
                _person("C", "person C"),
                _device("C4", "cabinet 4"),
                _device("C5", "cabinet 5"),
            ],
            actions=[_action("a1", "open", "B", "C4")],
            attributes=[_attribute("C4", "door.state", "open")],
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
        ),
        # 8. 柜门开启再次确认
        _observation(
            7,
            [
                _person("B", "person B"),
                _device("C4", "cabinet 4"),
            ],
            attributes=[_attribute("C4", "door.state", "open")],
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
        ),
    ]
    input_dir = _write_replay_input(
        tmp_path, observations, window_seconds=window_seconds, stride_seconds=stride_seconds
    )
    output = ObservationReplay(AppConfig()).replay(
        input_dir / "observations.jsonl", output_dir=tmp_path / "run"
    )

    events = [
        json.loads(line)
        for line in (output / "state_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    final = json.loads((output / "final_state.json").read_text(encoding="utf-8"))
    resolutions = [
        json.loads(line)
        for line in (output / "entity_resolutions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Build final identity mapping keyed by canonical name.
    person_ids = {e["canonical_name"]: eid for eid, e in final["entities"].items() if e["entity_type"] == "person"}
    device_ids = {e["canonical_name"]: eid for eid, e in final["entities"].items() if e["entity_type"] == "device"}

    # 1. Exactly three formal person entities, and only three persons overall.
    all_person_ids = {eid for eid, e in final["entities"].items() if e["entity_type"] == "person"}
    assert len(all_person_ids) == 3, f"expected exactly 3 persons, got {person_ids}"
    assert len(person_ids) == 3, f"expected 3 distinct person names, got {person_ids}"

    # 2. A/B/C map to the same IDs across the closeup without global drift.
    def _latest_mapping(local_id: str) -> str | None:
        for row in reversed(resolutions):
            for mapping in row["mappings"]:
                if mapping["local_id"] == local_id:
                    return mapping["global_entity_id"]
        return None

    person_a_id = _latest_mapping("A")
    person_b_id = _latest_mapping("B")
    person_c_id = _latest_mapping("C")
    assert person_a_id is not None
    assert person_b_id is not None
    assert person_c_id is not None
    assert len({person_a_id, person_b_id, person_c_id}) == 3, "A/B/C must map to three distinct IDs"
    # Before and after the closeup the same IDs are reused.
    assert person_ids["person A"] == person_a_id
    assert person_ids["person B"] == person_b_id
    assert person_ids["person C"] == person_c_id

    # No extra temporary/formal person entity with the same canonical name.
    seen_names: set[str] = set()
    for ent in final["entities"].values():
        if ent["entity_type"] == "person":
            assert ent["canonical_name"] not in seen_names, f"duplicate person name {ent['canonical_name']}"
            seen_names.add(ent["canonical_name"])

    # 3. Cabinet 4 and cabinet 5 identities are stable and distinct.
    assert len(device_ids) >= 2, f"expected at least 2 devices, got {device_ids}"
    assert "cabinet 4" in device_ids
    assert "cabinet 5" in device_ids
    assert device_ids["cabinet 4"] != device_ids["cabinet 5"]

    # 4. hand_over is preserved and not normalized to unknown.
    hand_over_action_ids = {
        aid
        for aid, action in final["actions"].items()
        if action["action_type"] == "hand_over"
    }
    assert len(hand_over_action_ids) == 1, f"expected one hand_over action, got {hand_over_action_ids}"
    hand_over_id = next(iter(hand_over_action_ids))
    assert any(
        e["event_type"] == "action_instant" and e.get("action_id") == hand_over_id
        for e in events
    ), "hand_over should emit an action_instant event"

    # 5. Overlapping windows do not create a second action ID for the repeated action.
    open_action_ids = {
        aid
        for aid, action in final["actions"].items()
        if action["action_type"] == "open"
    }
    assert len(open_action_ids) == 1, f"expected one open action, got {open_action_ids}"

    # 6. Indicator first initialized exactly once at window 4 (0-indexed).
    indicator_events = [
        e for e in events
        if e["event_type"] == "attribute_initialized"
        and e.get("attribute_key") == "indicator.energy.color"
    ]
    assert len(indicator_events) == 1, f"expected 1 indicator init, got {indicator_events}"
    assert indicator_events[0]["window_global_index"] == 4
    # No spurious transition for the indicator (it stays red).
    indicator_transitions = [
        e for e in events
        if e["event_type"] == "attribute_transition"
        and e.get("attribute_key") == "indicator.energy.color"
    ]
    assert len(indicator_transitions) == 0, f"expected no indicator transition, got {indicator_transitions}"

    # 7. Only one door.state closed -> open transition for cabinet 4.
    door_transitions = [
        e for e in events
        if e["event_type"] == "attribute_transition"
        and e.get("attribute_key") == "door.state"
        and e.get("before") == "closed"
        and e.get("after") == "open"
    ]
    assert len(door_transitions) == 1, f"expected 1 door transition, got {door_transitions}"

    # 8. Transition evidence includes the confirmation window.
    transition = door_transitions[0]
    evidence_windows = {ref["window_global_index"] for ref in transition["evidence"]}
    assert 6 in evidence_windows, f"door transition missing confirmation window: {evidence_windows}"

    # 9. Final cabinet 4 door is open and confirmed.
    cabinet4_id = device_ids["cabinet 4"]
    assert final["entities"][cabinet4_id]["attributes"]["door.state"]["value"] == "open"
    assert final["entities"][cabinet4_id]["attributes"]["door.state"]["status"] == "confirmed"

    # 10. Run metadata identifies the replay.
    run_meta = json.loads((output / "run_meta.json").read_text(encoding="utf-8"))
    assert run_meta["run_id"] == "replay"
    assert run_meta["source_run_id"] == "stage2_test"
    assert run_meta["state_enabled"] is True
    assert run_meta["state_schema_version"] == "2.0"
