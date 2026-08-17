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


def test_replay_is_model_free_and_deterministic(tmp_path: Path) -> None:
    source = _copy_fixture(tmp_path)
    replay = ObservationReplay(AppConfig())
    first = replay.replay(source / "observations.jsonl", output_dir=tmp_path / "one")
    second = replay.replay(source / "observations.jsonl", output_dir=tmp_path / "two")
    for filename in ("state_events.jsonl", "final_state.json"):
        first_hash = hashlib.sha256((first / filename).read_bytes()).hexdigest()
        second_hash = hashlib.sha256((second / filename).read_bytes()).hexdigest()
        assert first_hash == second_hash
    assert json.loads((first / "final_state.json").read_text())["entities"]


def test_golden_event_sequence_matches_expected_fixture(tmp_path: Path) -> None:
    source = _copy_fixture(tmp_path)
    output = ObservationReplay(AppConfig()).replay(
        source / "observations.jsonl", output_dir=tmp_path / "golden"
    )
    expected_path = Path(__file__).parents[1] / "golden" / "expected_stage2_events.jsonl"
    expected = [json.loads(line) for line in expected_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    actual = [json.loads(line) for line in (output / "state_events.jsonl").read_text().splitlines() if line.strip()]
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
