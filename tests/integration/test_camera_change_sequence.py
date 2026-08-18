from __future__ import annotations

import json
from pathlib import Path

from qwen_stream_video.config import AppConfig
from qwen_stream_video.domain import (
    ActionObservation,
    AttributeObservation,
    EntityObservation,
    EntityType,
    ObservationBatch,
    WindowObservation,
)
from qwen_stream_video.state.replay import ObservationReplay


def _entity(local_id: str, name: str, entity_type: EntityType = EntityType.PERSON) -> EntityObservation:
    regions = {"A": "left", "B": "center", "C": "right"}
    return EntityObservation(
        local_id=local_id,
        entity_type=entity_type,
        name=name,
        confidence=0.95,
        evidence_frames=[0],
        appearance={"uniform": "work_clothes"},
        spatial_region=regions.get(local_id, "unknown"),
    )


def _action(local_id: str, action_type: str, actor_local_id: str) -> ActionObservation:
    return ActionObservation(
        local_id=local_id,
        actor_local_id=actor_local_id,
        action_type=action_type,
        confidence=0.9,
        evidence_frames=[1],
    )


def _attribute(entity_local_id: str, key: str, value: str) -> AttributeObservation:
    return AttributeObservation(
        entity_local_id=entity_local_id,
        attribute_key=key,
        value=value,
        confidence=0.9,
        evidence_frames=[1],
    )


def _observation(
    index: int,
    entities: list[EntityObservation],
    actions: list[ActionObservation] | None = None,
    attributes: list[AttributeObservation] | None = None,
    camera_change: bool = False,
    view_type: str = "wide",
) -> ObservationBatch:
    return ObservationBatch(
        window=WindowObservation(
            global_index=index,
            start_seconds=index * 3.0,
            commit_start_seconds=index * 3.0,
            end_seconds=index * 3.0 + 3.0,
        ),
        scene={"camera_change": camera_change, "view_type": view_type},
        entities=entities,
        actions=actions or [],
        attribute_observations=attributes or [],
    )


def _write_replay_input(tmp_path: Path) -> Path:
    observations = [
        _observation(
            0,
            [_entity("A", "person A"), _entity("B", "person B"), _entity("C", "person C")],
        ),
        _observation(
            1,
            [_entity("B", "person B")],
            camera_change=True,
            view_type="closeup",
        ),
        _observation(
            2,
            [_entity("A", "person A"), _entity("B", "person B"), _entity("C", "person C")],
        ),
    ]
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    with (input_dir / "observations.jsonl").open("w", encoding="utf-8") as handle:
        for observation in observations:
            handle.write(observation.model_dump_json() + "\n")
    with (input_dir / "windows.jsonl").open("w", encoding="utf-8") as handle:
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
                    for i in range(3)
                ],
            }
            handle.write(json.dumps(window_data, ensure_ascii=False, sort_keys=True) + "\n")
    (input_dir / "run_meta.json").write_text(
        json.dumps({"run_id": "camera_test", "observation_schema_version": "2.0"}, ensure_ascii=False),
        encoding="utf-8",
    )
    return input_dir


def test_camera_change_preserves_person_identity(tmp_path: Path) -> None:
    """A close-up of person B must not cause A->B, B->C, or C->new drift."""
    input_dir = _write_replay_input(tmp_path)
    output = ObservationReplay(AppConfig()).replay(
        input_dir / "observations.jsonl", output_dir=tmp_path / "run"
    )

    events = [
        json.loads(line)
        for line in (output / "state_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    final = json.loads((output / "final_state.json").read_text(encoding="utf-8"))

    entities = final["entities"]
    person_entities = {eid: e for eid, e in entities.items() if e["entity_type"] == "person"}
    assert len(person_entities) == 3, f"expected 3 stable person IDs, got {person_entities}"

    by_name = {e["canonical_name"]: eid for eid, e in person_entities.items()}
    id_a = by_name["person A"]
    id_b = by_name["person B"]
    id_c = by_name["person C"]

    # No entity was created in the return-to-wide window (2).
    created_windows = {e["window_global_index"] for e in events if e["event_type"] == "entity_created"}
    assert 2 not in created_windows

    # All three persons were matched in windows 1 and 2.
    for window in (1, 2):
        matched = {e["entity_id"] for e in events if e["event_type"] == "entity_matched" and e["window_global_index"] == window}
        assert id_b in matched, f"person B not matched in window {window}"

    # No temporary person entities were introduced.
    assert all(not eid.startswith("temp") for eid in person_entities)

    # Final IDs are the original ones.
    assert {id_a, id_b, id_c} == set(person_entities)
