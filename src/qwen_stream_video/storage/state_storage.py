"""Independent persistence for deterministic State Engine results."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..domain import GlobalState
from ..inference import NormalizationWarning
from ..state import StateReductionResult


class StateStorage:
    def __init__(self, run_dir: str | Path, config: AppConfig) -> None:
        self.run_dir = Path(run_dir)
        self.config = config
        self._files: dict[str, Any] = {}
        self._formal_windows = 0
        self._last_snapshot: GlobalState | None = None

    def initialize(self, *, prompt_builder: Any | None = None) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        names = [
            "normalization_warnings.jsonl",
            "entity_resolutions.jsonl",
            "state_events.jsonl",
            "state_deltas.jsonl",
            "state_snapshots.jsonl",
            "state_errors.jsonl",
        ]
        for name in names:
            self._files[name] = (self.run_dir / name).open("w", encoding="utf-8")
        artifacts = self.run_dir / "artifacts"
        for subdir in ("prompts", "schemas", "vocabularies"):
            (artifacts / subdir).mkdir(parents=True, exist_ok=True)
        if prompt_builder is not None:
            (artifacts / "prompts" / "observation_system.txt").write_text(
                prompt_builder.system_prompt, encoding="utf-8"
            )
            (artifacts / "prompts" / "observation_user.txt").write_text(
                prompt_builder.user_prompt_template, encoding="utf-8"
            )
        from ..domain import ObservationBatch

        (artifacts / "schemas" / "observation_v2.schema.json").write_text(
            json.dumps(ObservationBatch.model_json_schema(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        repo_root = Path(__file__).resolve().parents[3]
        for filename in ("actions.yaml", "attributes.yaml", "entity_types.yaml"):
            source = repo_root / "vocabularies" / filename
            if source.is_file():
                shutil.copyfile(source, artifacts / "vocabularies" / filename)

    def _write_jsonl(self, filename: str, value: Any) -> None:
        handle = self._files[filename]
        handle.write(json.dumps(value.model_dump(mode="json") if hasattr(value, "model_dump") else value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()

    def write_normalization_warnings(
        self, window_global_index: int, warnings: list[NormalizationWarning]
    ) -> None:
        for warning in warnings:
            payload = warning.model_dump(mode="json")
            payload["window_global_index"] = window_global_index
            self._write_jsonl("normalization_warnings.jsonl", payload)

    def write_reduction(
        self,
        result: StateReductionResult,
        *,
        window_global_index: int,
        warmup: bool = False,
    ) -> None:
        if result.resolution is not None:
            self._write_jsonl("entity_resolutions.jsonl", result.resolution)
        if result.error:
            self.write_error(
                window_global_index,
                stage="state_reducer",
                error=RuntimeError(result.error),
                observation_succeeded=True,
            )
        if warmup:
            return
        self._formal_windows += 1
        for event in result.events:
            self._write_jsonl("state_events.jsonl", event)
        if result.delta is not None:
            self._write_jsonl("state_deltas.jsonl", result.delta)
        self._last_snapshot = result.state.model_copy(deep=True)
        interval = self.config.state.snapshot_interval_windows
        if self._formal_windows % interval == 0:
            self._write_jsonl("state_snapshots.jsonl", result.state)

    def write_error(
        self,
        window_global_index: int,
        *,
        stage: str,
        error: Exception,
        observation_succeeded: bool,
        state_reference: str | None = None,
    ) -> None:
        self._write_jsonl(
            "state_errors.jsonl",
            {
                "window_global_index": window_global_index,
                "stage": stage,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "observation_succeeded": observation_succeeded,
                "state_affected": stage != "observation",
                "original_state_reference": state_reference,
            },
        )

    def finalize(self, final_state: GlobalState | None) -> None:
        if final_state is not None:
            if self._last_snapshot is None or self._last_snapshot.model_dump(mode="json") != final_state.model_dump(mode="json"):
                self._write_jsonl("state_snapshots.jsonl", final_state)
            tmp = self.run_dir / "final_state.json.tmp"
            final = self.run_dir / "final_state.json"
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(final_state.model_dump(mode="json"), handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, final)
        for handle in self._files.values():
            handle.close()
        self._files.clear()
