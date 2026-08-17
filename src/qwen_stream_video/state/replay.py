"""Offline Observation replay without a model client."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..domain import ObservationBatch, WindowObservation
from ..exceptions import ObservationReplayError
from ..inference import ObservationNormalizer
from ..storage.state_storage import StateStorage
from ..video import SampledFrame, VideoWindow
from .state_reducer import StateReducer


class ObservationV1Adapter:
    """Migrate Stage1 fields without inventing identity or evidence."""

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
        raw_scene.setdefault("continuity_hint", "camera_change" if raw_scene.get("camera_change") else "continuous")
        payload["scene"] = raw_scene
        for action in payload.get("actions", []):
            action.setdefault("raw_action_type", action.get("action_type"))
            action.setdefault("normalization_status", "canonical")
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
        for row in window_rows:
            try:
                window = VideoWindow.model_validate(
                    {key: value for key, value in row.items() if key != "sampled_frames"}
                )
            except Exception as exc:
                raise ObservationReplayError(f"Invalid window metadata: {exc}") from exc
            windows[window.global_index] = window
            window_data[window.global_index] = row
        if not windows:
            raise ObservationReplayError("Replay requires at least one window")
        output = Path(output_dir)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=False)
        for filename in ("observations.jsonl", "windows.jsonl", "run_meta.json"):
            source = input_dir / filename
            shutil.copyfile(source, output / filename)
        config_for_storage = self.config.model_copy(
            update={"storage": self.config.storage.model_copy(update={"output_root": str(output.parent)})}
        )
        storage = StateStorage(output, config_for_storage)
        storage.initialize()
        source_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        run_id = f"{source_meta.get('run_id', 'replay')}_replay"
        from ..domain import GlobalState

        state = GlobalState(run_id=run_id)
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
