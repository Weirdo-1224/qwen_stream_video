from __future__ import annotations

import hashlib
import json
from pathlib import Path

from qwen_stream_video.config import AppConfig
from qwen_stream_video.state.replay import ObservationReplay


def _copy_fixture(tmp_path: Path) -> Path:
    source = Path(__file__).parents[1] / "golden"
    target = tmp_path / "input"
    target.mkdir()
    (target / "windows.jsonl").write_text((source / "stage2_sequence.jsonl").read_text(), encoding="utf-8")
    (target / "observations.jsonl").write_text((source / "stage2_observations.jsonl").read_text(), encoding="utf-8")
    (target / "run_meta.json").write_text((source / "run_meta.json").read_text(), encoding="utf-8")
    return target


def test_golden_event_sequence_matches_expected_fixture(tmp_path: Path) -> None:
    source = _copy_fixture(tmp_path)
    output = ObservationReplay(AppConfig()).replay(
        source / "observations.jsonl", output_dir=tmp_path / "golden"
    )
    expected_path = Path(__file__).parents[1] / "golden" / "expected_stage2_events.jsonl"
    expected = [json.loads(line) for line in expected_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    actual = [json.loads(line) for line in (output / "state_events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    fields = (
        "event_id",
        "event_type",
        "window_global_index",
        "scene_id",
        "entity_id",
        "action_id",
        "attribute_key",
        "before",
        "after",
        "reason",
    )
    assert [{key: row.get(key) for key in fields} for row in actual] == [
        {key: row.get(key) for key in fields} for row in expected
    ]


def test_golden_final_state_matches_expected_fixture(tmp_path: Path) -> None:
    source = _copy_fixture(tmp_path)
    output = ObservationReplay(AppConfig()).replay(
        source / "observations.jsonl", output_dir=tmp_path / "golden"
    )
    expected_path = Path(__file__).parents[1] / "golden" / "expected_stage2_final_state.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    actual = json.loads((output / "final_state.json").read_text(encoding="utf-8"))

    # Deterministic key fields.
    assert actual["run_id"] == "replay"
    assert actual["schema_version"] == "2.0"
    assert set(actual["entities"]) == {"device_0001", "person_0001"}
    assert set(actual["actions"]) == {"action_000001"}
    assert actual["entities"]["device_0001"]["attributes"]["door.state"]["value"] == "open"
    assert actual["entities"]["device_0001"]["canonical_name"] == "cabinet 4"
    assert actual["entities"]["person_0001"]["canonical_name"] == "worker"

    # Full deterministic structural comparison.
    assert actual == expected


def test_golden_entity_mapping_and_action_id(tmp_path: Path) -> None:
    source = _copy_fixture(tmp_path)
    output = ObservationReplay(AppConfig()).replay(
        source / "observations.jsonl", output_dir=tmp_path / "golden"
    )
    resolutions = [
        json.loads(line)
        for line in (output / "entity_resolutions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    final_mapping = {}
    for row in resolutions:
        for mapping in row["mappings"]:
            final_mapping[mapping["local_id"]] = (mapping["global_entity_id"], mapping["status"])
    assert final_mapping["P"] == ("person_0001", "matched")
    assert final_mapping["D"] == ("device_0001", "matched")

    events = [
        json.loads(line)
        for line in (output / "state_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    action_started = [e for e in events if e["event_type"] == "action_started"]
    assert len(action_started) == 1
    assert action_started[0]["action_id"] == "action_000001"

    transitions = [e for e in events if e["event_type"] == "attribute_transition"]
    assert len(transitions) == 1
    assert transitions[0]["attribute_key"] == "door.state"
    assert transitions[0]["before"] == "closed"
    assert transitions[0]["after"] == "open"


def test_golden_replay_is_deterministic(tmp_path: Path) -> None:
    source = _copy_fixture(tmp_path)
    replay = ObservationReplay(AppConfig())
    first = replay.replay(source / "observations.jsonl", output_dir=tmp_path / "one")
    second = replay.replay(source / "observations.jsonl", output_dir=tmp_path / "two")
    for filename in ("state_events.jsonl", "final_state.json"):
        first_hash = hashlib.sha256((first / filename).read_bytes()).hexdigest()
        second_hash = hashlib.sha256((second / filename).read_bytes()).hexdigest()
        assert first_hash == second_hash, f"{filename} differs between replays"
