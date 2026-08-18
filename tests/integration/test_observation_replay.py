from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from qwen_stream_video.config import AppConfig
from qwen_stream_video.exceptions import ObservationReplayError
from qwen_stream_video.state.replay import ObservationReplay, ObservationV1Adapter
from qwen_stream_video.video import VideoWindow


def _copy_fixture(tmp_path: Path) -> Path:
    source = Path(__file__).parents[1] / "golden"
    target = tmp_path / "input"
    target.mkdir()
    (target / "windows.jsonl").write_text((source / "stage2_sequence.jsonl").read_text(), encoding="utf-8")
    (target / "observations.jsonl").write_text((source / "stage2_observations.jsonl").read_text(), encoding="utf-8")
    (target / "run_meta.json").write_text((source / "run_meta.json").read_text(), encoding="utf-8")
    return target


def test_replay_does_not_call_model(tmp_path: Path) -> None:
    """Replay uses only persisted observations; no inference client is invoked."""
    source = _copy_fixture(tmp_path)
    replay = ObservationReplay(AppConfig())
    # No client is passed and the call below must succeed deterministically.
    output = replay.replay(source / "observations.jsonl", output_dir=tmp_path / "no_model")
    assert (output / "state_events.jsonl").is_file()
    assert (output / "final_state.json").is_file()


def test_replay_twice_produces_same_events(tmp_path: Path) -> None:
    source = _copy_fixture(tmp_path)
    replay = ObservationReplay(AppConfig())
    first = replay.replay(source / "observations.jsonl", output_dir=tmp_path / "one")
    second = replay.replay(source / "observations.jsonl", output_dir=tmp_path / "two")
    first_hash = hashlib.sha256((first / "state_events.jsonl").read_bytes()).hexdigest()
    second_hash = hashlib.sha256((second / "state_events.jsonl").read_bytes()).hexdigest()
    assert first_hash == second_hash


def test_replay_twice_produces_same_final_state(tmp_path: Path) -> None:
    source = _copy_fixture(tmp_path)
    replay = ObservationReplay(AppConfig())
    first = replay.replay(source / "observations.jsonl", output_dir=tmp_path / "one")
    second = replay.replay(source / "observations.jsonl", output_dir=tmp_path / "two")
    first_hash = hashlib.sha256((first / "final_state.json").read_bytes()).hexdigest()
    second_hash = hashlib.sha256((second / "final_state.json").read_bytes()).hexdigest()
    assert first_hash == second_hash


def test_replay_rejects_missing_window_metadata(tmp_path: Path) -> None:
    source = _copy_fixture(tmp_path)
    (source / "windows.jsonl").unlink()
    with pytest.raises(ObservationReplayError, match="Replay input is missing"):
        ObservationReplay(AppConfig()).replay(
            source / "observations.jsonl", output_dir=tmp_path / "missing_windows"
        )


def _v1_window_row(global_index: int, start: float, end: float) -> str:
    return json.dumps({
        "global_index": global_index,
        "run_index": global_index,
        "start_seconds": start,
        "end_seconds": end,
        "sampled_frames": [
            {"sample_index": i, "frame_index": i, "timestamp_seconds": start + i}
            for i in range(int(end - start))
        ],
    })


def _v1_observation_row(
    global_index: int,
    start: float,
    end: float,
    action_frame: int | None = None,
    attribute_frame: int | None = None,
    attribute_value: str = "open",
) -> str:
    actions: list[dict[str, object]] = []
    attributes: list[dict[str, object]] = []
    if action_frame is not None:
        actions.append({
            "local_id": "A",
            "actor_local_id": "P",
            "action_type": "open",
            "confidence": 0.9,
            "evidence_frames": [action_frame],
        })
    if attribute_frame is not None:
        attributes.append({
            "entity_local_id": "D",
            "attribute": "door.state",
            "value": attribute_value,
            "confidence": 0.9,
            "evidence_frames": [attribute_frame],
        })
    return json.dumps({
        "schema_version": "1.0",
        "window": {"global_index": global_index},
        "scene": {"camera_change": False, "visibility": "clear", "view_type": "wide"},
        "entities": [
            {"local_id": "P", "entity_type": "person", "name": "worker", "confidence": 0.95, "evidence_frames": [0]},
            {"local_id": "D", "entity_type": "device", "name": "cabinet 4", "confidence": 0.95, "evidence_frames": [0]},
        ],
        "actions": actions,
        "attribute_observations": attributes,
    })


def test_v1_observation_can_be_adapted() -> None:
    """Schema 1.0 observations are adapted: OOV actions become other with raw preserved."""
    raw = {
        "schema_version": "1.0",
        "window": {"global_index": 0},
        "scene": {"camera_change": False, "visibility": "clear", "view_type": "wide"},
        "entities": [
            {"local_id": "P", "entity_type": "person", "name": "worker", "confidence": 0.95, "evidence_frames": [0]}
        ],
        "actions": [
            {"local_id": "A", "actor_local_id": "P", "action_type": "dance", "confidence": 0.9, "evidence_frames": [1]}
        ],
        "attribute_observations": [
            {"entity_local_id": "P", "attribute": "posture", "value": "standing", "confidence": 0.9, "evidence_frames": [1]}
        ],
    }
    window = VideoWindow(
        global_index=0,
        run_index=0,
        start_seconds=0.0,
        commit_start_seconds=0.0,
        end_seconds=3.0,
    )
    adapted = ObservationV1Adapter().adapt(raw, window)
    action = adapted.actions[0]
    assert action.action_type == "other"
    assert action.raw_action_type == "dance"
    assert action.normalization_status == "out_of_vocabulary"
    assert action.action_family == "other"
    # Canonical actions remain canonical.
    raw["actions"][0]["action_type"] = "inspect"
    adapted_canonical = ObservationV1Adapter().adapt(raw, window)
    canonical = adapted_canonical.actions[0]
    assert canonical.action_type == "inspect"
    assert canonical.normalization_status == "canonical"


def test_v1_overlapping_windows_derive_commit_interval(tmp_path: Path) -> None:
    """When windows.jsonl lacks commit_start_seconds, overlapping windows derive correct commits."""
    import json
    source = tmp_path / "v1_input"
    source.mkdir()
    # Three 6-second windows with 3-second stride, no commit_start_seconds.
    (source / "windows.jsonl").write_text(
        "\n".join([
            _v1_window_row(0, 0.0, 6.0),
            _v1_window_row(1, 3.0, 9.0),
            _v1_window_row(2, 6.0, 12.0),
        ]),
        encoding="utf-8",
    )
    # Window 0 commits the action; window 1 repeats it in the context-only overlap;
    # window 2 provides confirming attribute evidence.
    (source / "observations.jsonl").write_text(
        "\n".join([
            _v1_observation_row(0, 0.0, 6.0, action_frame=4, attribute_frame=1, attribute_value="closed"),
            _v1_observation_row(1, 3.0, 9.0, action_frame=1, attribute_frame=5, attribute_value="open"),
            _v1_observation_row(2, 6.0, 12.0, attribute_frame=5, attribute_value="open"),
        ]),
        encoding="utf-8",
    )
    (source / "run_meta.json").write_text(
        json.dumps({"run_id": "v1_test", "observation_schema_version": "1.0"}),
        encoding="utf-8",
    )
    output = tmp_path / "v1_output"
    replay = ObservationReplay(AppConfig())
    result = replay.replay(source / "observations.jsonl", output_dir=output)
    events = [json.loads(line) for line in (result / "state_events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    action_events = [e for e in events if e["event_type"].startswith("action_")]
    transition_events = [e for e in events if e["event_type"] == "attribute_transition"]
    # The same action appearing in the context interval of window 1 must not create a second action.
    assert len([e for e in action_events if e["event_type"] == "action_started"]) == 1
    assert len({e["action_id"] for e in action_events if e.get("action_id")}) == 1
    # The attribute transition must be confirmed exactly once.
    assert len(transition_events) == 1

    # Deterministic replay produces the same files twice.
    result2 = replay.replay(source / "observations.jsonl", output_dir=tmp_path / "v1_output2")
    assert (result / "state_events.jsonl").read_bytes() == (result2 / "state_events.jsonl").read_bytes()
    assert (result / "final_state.json").read_bytes() == (result2 / "final_state.json").read_bytes()


def test_replay_rejects_window_missing_required_fields(tmp_path: Path) -> None:
    source = tmp_path / "bad_input"
    source.mkdir()
    (source / "windows.jsonl").write_text(
        json.dumps({"global_index": 0, "end_seconds": 3.0}),
        encoding="utf-8",
    )
    (source / "observations.jsonl").write_text("{}", encoding="utf-8")
    (source / "run_meta.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ObservationReplayError, match="missing required time fields"):
        ObservationReplay(AppConfig()).replay(
            source / "observations.jsonl", output_dir=tmp_path / "bad_output"
        )
