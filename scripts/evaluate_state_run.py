"""Deterministic structural and heuristic quality report for a state run."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def evaluate(run_dir: str | Path) -> tuple[dict[str, Any], list[str]]:
    root = Path(run_dir)
    errors: list[str] = []
    observations = _jsonl(root / "observations.jsonl")
    resolutions = _jsonl(root / "entity_resolutions.jsonl")
    events = _jsonl(root / "state_events.jsonl")
    snapshots = _jsonl(root / "state_snapshots.jsonl")
    final_path = root / "final_state.json"
    final = json.loads(final_path.read_text(encoding="utf-8")) if final_path.is_file() else None
    if final is None:
        errors.append("final_state.json is missing or invalid")

    action_types = [
        action.get("normalization_status")
        for row in observations
        for action in row.get("actions", [])
    ]
    attribute_types = [
        item.get("normalization_status")
        for row in observations
        for item in row.get("attribute_observations", [])
    ]
    resolution_statuses = [
        mapping.get("status") or mapping.get("resolution_status")
        for row in resolutions
        for mapping in row.get("mappings", [])
    ]
    event_counts = Counter(event.get("event_type") for event in events)
    entity_ids = set((final or {}).get("entities", {}))
    action_ids = set((final or {}).get("actions", {}))
    seen_event_ids: set[str] = set()
    evidence_count = 0
    evidence_total = 0
    for event in events:
        event_id = event.get("event_id")
        if event_id in seen_event_ids:
            errors.append(f"duplicate event_id: {event_id}")
        seen_event_ids.add(event_id)
        evidence_total += 1
        if event.get("evidence"):
            evidence_count += 1
        if event.get("entity_id") and event["entity_id"] not in entity_ids:
            errors.append(f"event references missing entity: {event['entity_id']}")
        if event.get("action_id") and event["action_id"] not in action_ids:
            errors.append(f"event references missing action: {event['action_id']}")
    last_committed_values = [
        snapshot.get("last_committed_window")
        for snapshot in snapshots
        if snapshot.get("last_committed_window") is not None
    ]
    if last_committed_values != sorted(last_committed_values):
        errors.append("last_committed_window is not monotonic")
    if final is not None and snapshots and snapshots[-1] != final:
        errors.append("final_state.json does not equal the last state snapshot")
    duplicate_final_entities = len(entity_ids) != len((final or {}).get("entities", {}))
    if duplicate_final_entities:
        errors.append("duplicate entity IDs in final state")

    ambiguous = sum(status in {"ambiguous", "temporary"} for status in resolution_statuses)
    action_oov = sum(status == "out_of_vocabulary" for status in action_types)
    attribute_oov = sum(status == "out_of_vocabulary" for status in attribute_types)
    report: dict[str, Any] = {
        "window_count": len(_jsonl(root / "windows.jsonl")),
        "successful_observation_count": len(observations),
        "state_update_success_count": len(_jsonl(root / "state_deltas.jsonl")),
        "action_oov_count": action_oov,
        "action_oov_rate": action_oov / len(action_types) if action_types else 0.0,
        "attribute_oov_count": attribute_oov,
        "attribute_oov_rate": attribute_oov / len(attribute_types) if attribute_types else 0.0,
        "entities": {
            "created": resolution_statuses.count("created"),
            "matched": resolution_statuses.count("matched"),
            "ambiguous": ambiguous,
            "temporary": resolution_statuses.count("temporary"),
            "merged": event_counts["entity_merged"],
            "total": len(entity_ids),
        },
        "actions": {
            "started": event_counts["action_started"],
            "ended": event_counts["action_ended"],
            "uncertain": event_counts["action_uncertain"],
        },
        "attributes": {
            "initialized": event_counts["attribute_initialized"],
            "pending": event_counts["attribute_pending"],
            "transition": event_counts["attribute_transition"],
            "conflict": event_counts["attribute_conflict"],
        },
        "no_evidence_event_count": evidence_total - evidence_count,
        "entity_ambiguous_rate": ambiguous / len(resolution_statuses) if resolution_statuses else 0.0,
        "entity_id_switch_candidate_count": 0,
        "duplicate_action_candidate_count": 0,
        "unsupported_transition_count": sum(
            event.get("event_type") == "attribute_transition"
            and "support" not in event.get("reason", "")
            and event.get("reason") != "single_high_policy"
            for event in events
        ),
        "evidence_coverage": evidence_count / evidence_total if evidence_total else 1.0,
        "context_average_characters": 0.0,
        "final_entity_count": len((final or {}).get("entities", {})),
        "final_active_action_count": len((final or {}).get("active_action_ids", [])),
        "structural_errors": errors,
    }
    return report, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    try:
        report, errors = evaluate(args.run_dir)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"state run evaluation failed: {exc}", file=sys.stderr)
        return 1
    if args.as_json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
