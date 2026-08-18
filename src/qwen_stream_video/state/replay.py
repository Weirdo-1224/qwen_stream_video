"""Offline Observation replay without a model client."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from ..config import AppConfig
from ..domain import ObservationBatch, WindowObservation
from ..exceptions import ObservationReplayError
from ..inference import ObservationNormalizer
from ..storage.state_storage import StateStorage
from ..video import SampledFrame, VideoWindow
from .state_reducer import StateReducer

DEFAULT_ACTIONS_PATH = Path(__file__).resolve().parents[3] / "vocabularies" / "actions.yaml"


class ObservationV1Adapter:
    """Migrate Stage1 fields without inventing identity or evidence.

    Illegal action types are mapped to ``other`` at the adapter layer so that
    replay outputs are self-describing even before normalization.
    """

    def __init__(self, actions_path: str | Path | None = None) -> None:
        path = Path(actions_path or DEFAULT_ACTIONS_PATH)
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        self.actions: dict[str, dict[str, Any]] = data.get("actions", {})

    def adapt(self, data: dict[str, Any], window: VideoWindow) -> ObservationBatch:
        payload = json.loads(json.dumps(data))
        payload["schema_version"] = "2.0"
        payload["window"] = {
            "global_index": window.global_index,
            "start_seconds": window.start_seconds,
            "commit_start_seconds": window.commit_start_seconds,
            "end_seconds": window.end_seconds,
        }
        raw_scene = dict(payload.get("scene", {}))
        if "scene_visibility" not in raw_scene:
            raw_scene["scene_visibility"] = raw_scene.pop("visibility", "unknown")
        if "target_visibility" not in raw_scene:
            raw_scene["target_visibility"] = raw_scene["scene_visibility"]
        raw_scene.setdefault(
            "continuity_hint", "camera_change" if raw_scene.get("camera_change") else "continuous"
        )
        payload["scene"] = raw_scene
        for action in payload.get("actions", []):
            raw = action.get("action_type")
            action["raw_action_type"] = raw
            if raw == "unknown":
                action["action_type"] = "unknown"
                action["action_family"] = self.actions.get("unknown", {}).get("family", "unknown")
                action["normalization_status"] = "unknown"
            elif raw in self.actions:
                action["action_type"] = raw
                action["action_family"] = self.actions[raw].get("family", "other")
                action["normalization_status"] = "canonical"
            else:
                action["action_type"] = "other"
                action["action_family"] = self.actions.get("other", {}).get("family", "other")
                action["normalization_status"] = "out_of_vocabulary"
        for attribute in payload.get("attribute_observations", []):
            raw_attribute = attribute.get("attribute", attribute.get("attribute_key"))
            attribute.setdefault("attribute_key", raw_attribute)
            attribute.setdefault("raw_attribute", raw_attribute)
            attribute.setdefault("raw_value", attribute.get("value"))
            attribute.setdefault("normalization_status", "canonical")
        payload.setdefault("relations", [])
        return ObservationBatch.model_validate(payload)


class ObservationReplay:
    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()
        self.normalizer = ObservationNormalizer()
        self.reducer = StateReducer(self.config)
        self.adapter = ObservationV1Adapter()

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            raise ObservationReplayError(f"Replay input is missing: {path}")
        rows: list[dict[str, Any]] = []
        try:
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if line.strip():
                    rows.append(json.loads(line))
        except (OSError, json.JSONDecodeError) as exc:
            raise ObservationReplayError(f"Cannot read JSONL {path}: {exc}") from exc
        return rows

    @staticmethod
    def _sampled_frames(window_data: dict[str, Any]) -> list[SampledFrame]:
        raw_frames = window_data.get("sampled_frames", [])
        frames: list[SampledFrame] = []
        for row in raw_frames:
            frames.append(
                SampledFrame(
                    run_index=int(window_data.get("run_index", 0)),
                    global_index=int(window_data["global_index"]),
                    sample_index=int(row["sample_index"]),
                    frame_index=int(row.get("frame_index", 0)),
                    timestamp_seconds=float(row["timestamp_seconds"]),
                    encoded_image=row.get("encoded_image"),
                    image=None,
                )
            )
        return frames

    def _build_run_meta(self, source_meta: dict[str, Any]) -> dict[str, Any]:
        """Return deterministic replay metadata without source non-determinism."""
        return {
            "run_id": "replay",
            "state_schema_version": "2.0",
            "state_enabled": True,
            "observation_schema_version": source_meta.get(
                "observation_schema_version", self.config.observation.schema_version
            ),
            "source_run_id": source_meta.get("run_id"),
            "replay": True,
            "experiment_name": self.config.experiment.name,
            "experiment_seed": self.config.experiment.seed,
            "window_count": None,
            "observation_count": None,
        }

    def replay(
        self,
        observations_path: str | Path,
        *,
        output_dir: str | Path,
    ) -> Path:
        observations_path = Path(observations_path)
        input_dir = observations_path.parent
        window_rows = self._read_jsonl(input_dir / "windows.jsonl")
        observation_rows = self._read_jsonl(observations_path)
        run_meta_path = input_dir / "run_meta.json"
        if not run_meta_path.is_file():
            raise ObservationReplayError(f"Replay input is missing: {run_meta_path}")
        windows: dict[int, VideoWindow] = {}
        window_data: dict[int, dict[str, Any]] = {}
        sorted_rows = sorted(window_rows, key=lambda row: row.get("global_index", -1))
        previous_end: float | None = None
        for row in sorted_rows:
            index = row.get("global_index")
            if index is None:
                raise ObservationReplayError("Window metadata is missing global_index")
            if "start_seconds" not in row or "end_seconds" not in row:
                raise ObservationReplayError(
                    f"Window {index} metadata is missing required time fields"
                )
            if row.get("commit_start_seconds") is None:
                start = float(row["start_seconds"])
                if previous_end is None:
                    row["commit_start_seconds"] = start
                else:
                    row["commit_start_seconds"] = max(start, previous_end)
            try:
                window = VideoWindow.model_validate(
                    {key: value for key, value in row.items() if key != "sampled_frames"}
                )
            except Exception as exc:
                raise ObservationReplayError(f"Invalid window metadata: {exc}") from exc
            if previous_end is not None and window.start_seconds < previous_end - 1e-9:
                # Overlapping windows are valid, but the rows must be sorted.
                pass
            previous_end = window.end_seconds
            if window.global_index in windows:
                raise ObservationReplayError(
                    f"Duplicate window global_index: {window.global_index}"
                )
            windows[window.global_index] = window
            window_data[window.global_index] = row
        if not windows:
            raise ObservationReplayError("Replay requires at least one window")
        output = Path(output_dir)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=False)
        for filename in ("observations.jsonl", "windows.jsonl"):
            source = input_dir / filename
            shutil.copyfile(source, output / filename)
        source_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        run_meta = self._build_run_meta(source_meta)
        run_meta["window_count"] = len(window_rows)
        run_meta["observation_count"] = len(observation_rows)
        with (output / "run_meta.json").open("w", encoding="utf-8") as handle:
            json.dump(run_meta, handle, ensure_ascii=False, indent=2, sort_keys=True)
        config_for_storage = self.config.model_copy(
            update={"storage": self.config.storage.model_copy(update={"output_root": str(output.parent)})}
        )
        storage = StateStorage(output, config_for_storage)
        storage.initialize()
        from ..domain import GlobalState

        state = GlobalState(run_id=run_meta["run_id"])
        by_index = {int(row.get("window", {}).get("global_index", -1)): row for row in observation_rows}
        for index in sorted(windows):
            if index not in by_index:
                raise ObservationReplayError(f"Missing observation for window {index}")
            window = windows[index]
            raw = by_index[index]
            if raw.get("schema_version") == "1.0":
                observation = self.adapter.adapt(raw, window)
            elif raw.get("schema_version") == "2.0":
                try:
                    observation = ObservationBatch.model_validate(raw)
                except Exception as exc:
                    raise ObservationReplayError(f"Invalid Schema 2.0 observation in window {index}: {exc}") from exc
                observation.window = WindowObservation(
                    global_index=window.global_index,
                    start_seconds=window.start_seconds,
                    commit_start_seconds=window.commit_start_seconds,
                    end_seconds=window.end_seconds,
                )
            else:
                raise ObservationReplayError(
                    f"Unsupported observation schema {raw.get('schema_version')!r} in window {index}"
                )
            frames = self._sampled_frames(window_data[index])
            all_evidence = [
                item.evidence_frames
                for entity in observation.entities
                for item in [entity]
            ] + [item.evidence_frames for item in observation.actions]
            all_evidence += [item.evidence_frames for item in observation.attribute_observations]
            if any(evidence and not frames for evidence in all_evidence):
                raise ObservationReplayError(
                    f"Window {index} has evidence but windows.jsonl has no sampled frame metadata"
                )
            normalized = self.normalizer.normalize(observation)
            storage.write_normalization_warnings(index, normalized.warnings)
            result = self.reducer.apply_observation(state, normalized.batch, frames, window)
            state = result.state
            storage.write_reduction(result, window_global_index=index, warmup=window.processing_role == "warmup")
        storage.finalize(state)
        return output
