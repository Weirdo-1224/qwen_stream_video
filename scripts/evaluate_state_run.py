"""Deterministic structural and heuristic quality report for a state run."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_windows(run_dir: Path) -> dict[int, dict[int, float]]:
    """Map window index -> sample_index -> timestamp_seconds."""
    result: dict[int, dict[int, float]] = {}
    for row in _jsonl(run_dir / "windows.jsonl"):
        index = row.get("global_index")
        if index is None:
            continue
        frames: dict[int, float] = {}
        for frame in row.get("sampled_frames", []):
            sample_index = frame.get("sample_index")
            if sample_index is not None:
                frames[int(sample_index)] = float(frame.get("timestamp_seconds", 0.0))
        result[int(index)] = frames
    return result


def _context_average_characters(run_dir: Path) -> tuple[float | None, list[str]]:
    """Average serialized context length from per-window API metrics.

    Returns the mean number of characters in the ContextBuilder JSON that was
    actually sent to the model for each window.  When no per-window context
    lengths are recorded, returns ``None`` and a warning so that prompt template
    lengths are never reported as context lengths.
    """
    warnings: list[str] = []
    lengths: list[int] = []
    for row in _jsonl(run_dir / "api_metrics.jsonl"):
        value = row.get("context_characters")
        if value is not None:
            lengths.append(int(value))
    if not lengths:
        warnings.append(
            "Per-window context_characters not found in api_metrics.jsonl; "
            "context_average_characters is unavailable"
        )
        return None, warnings
    return sum(lengths) / len(lengths), warnings


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

    # Per-observation reference checks.
    for row in observations:
        window = row.get("window", {}).get("global_index")
        entity_local_ids = {entity.get("local_id") for entity in row.get("entities", [])}
        for action in row.get("actions", []):
            for ref_field, local_field in (
                ("actor_local_id", "actor"),
                ("target_local_id", "target"),
                ("tool_local_id", "tool"),
            ):
                local_id = action.get(ref_field)
                if local_id and local_id not in entity_local_ids:
                    errors.append(
                        f"window {window}: action {action.get('local_id')} "
                        f"{local_field} references missing entity {local_id}"
                    )

    # Event ID uniqueness and reference existence.
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

    # Final-state action/entity ID uniqueness.
    final_entities = (final or {}).get("entities", {})
    final_actions = (final or {}).get("actions", {})
    if len(final_entities) != len({eid for eid in final_entities}):
        errors.append("duplicate entity IDs in final state")
    if len(final_actions) != len({aid for aid in final_actions}):
        errors.append("duplicate action IDs in final state")

    # Snapshot monotonicity and final-state consistency.
    last_committed_values = [
        snapshot.get("last_committed_window")
        for snapshot in snapshots
        if snapshot.get("last_committed_window") is not None
    ]
    if last_committed_values != sorted(last_committed_values):
        errors.append("last_committed_window is not monotonic")
    if final is not None and snapshots and snapshots[-1] != final:
        errors.append("final_state.json does not equal the last state snapshot")

    # Evidence sample index -> timestamp mapping.
    windows_by_index = _load_windows(root)
    for event in events:
        for evidence in event.get("evidence", []):
            window_index = evidence.get("window_global_index")
            sample_indices = evidence.get("sample_indices", [])
            timestamps = evidence.get("timestamps_seconds", [])
            if not sample_indices:
                continue
            if not timestamps:
                errors.append(
                    f"event {event.get('event_id')} has sample_indices but no timestamps"
                )
            frame_map = windows_by_index.get(window_index, {})
            missing = sorted(set(sample_indices) - set(frame_map))
            if missing:
                errors.append(
                    f"event {event.get('event_id')} references missing sample indices {missing} "
                    f"in window {window_index}"
                )

    # Entity ID switch candidates near camera changes.
    camera_change_windows = {
        row.get("window", {}).get("global_index")
        for row in observations
        if row.get("scene", {}).get("camera_change")
    }
    switch_candidate_windows = set()
    for window in camera_change_windows:
        switch_candidate_windows.update({window - 1, window, window + 1})
    temporary_prefix = "temp"
    entity_id_switch_candidate_count = 0
    for row in resolutions:
        window = row.get("window_global_index")
        if window not in switch_candidate_windows:
            continue
        for mapping in row.get("mappings", []):
            status = mapping.get("status") or mapping.get("resolution_status")
            global_id = mapping.get("global_entity_id")
            if status in {"ambiguous", "temporary", "rejected_hint"} or (
                global_id and str(global_id).startswith(temporary_prefix)
            ):
                entity_id_switch_candidate_count += 1

    # Duplicate action candidates: same key in the same or adjacent window.
    duplicate_action_candidate_count = 0
    action_entries = [
        (aid, action)
        for aid, action in (final or {}).get("actions", {}).items()
    ]
    counted_pairs: set[tuple[str, str]] = set()
    for i, (aid1, a1) in enumerate(action_entries):
        key1 = (
            a1.get("actor_id"),
            a1.get("action_type"),
            a1.get("target_id"),
            a1.get("tool_id"),
        )
        windows1 = set(a1.get("observed_windows", []))
        for aid2, a2 in action_entries[i + 1 :]:
            key2 = (
                a2.get("actor_id"),
                a2.get("action_type"),
                a2.get("target_id"),
                a2.get("tool_id"),
            )
            if key1 != key2:
                continue
            windows2 = set(a2.get("observed_windows", []))
            adjacent = any(abs(w1 - w2) <= 1 for w1 in windows1 for w2 in windows2)
            if adjacent:
                pair = tuple(sorted((aid1, aid2)))
                if pair not in counted_pairs:
                    counted_pairs.add(pair)
                    duplicate_action_candidate_count += 1

    ambiguous = sum(status in {"ambiguous", "temporary"} for status in resolution_statuses)
    action_oov = sum(status == "out_of_vocabulary" for status in action_types)
    attribute_oov = sum(status == "out_of_vocabulary" for status in attribute_types)
    context_average, context_warnings = _context_average_characters(root)
    errors.extend(context_warnings)
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
        "entity_id_switch_candidate_count": entity_id_switch_candidate_count,
        "duplicate_action_candidate_count": duplicate_action_candidate_count,
        "unsupported_transition_count": sum(
            event.get("event_type") == "attribute_transition"
            and "support" not in event.get("reason", "")
            and event.get("reason") != "single_high_policy"
            for event in events
        ),
        "evidence_coverage": evidence_count / evidence_total if evidence_total else 1.0,
        "context_average_characters": context_average,
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
