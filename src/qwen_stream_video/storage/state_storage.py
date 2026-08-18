"""Independent persistence for deterministic State Engine results."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from ..config import AppConfig
from ..domain import GlobalState
from ..exceptions import StateStorageError
from ..inference import NormalizationWarning
from ..state import StateReductionResult


def _storage_safe(method: Any) -> Any:
    def wrapper(self: StateStorage, *args: Any, **kwargs: Any) -> Any:
        try:
            return method(self, *args, **kwargs)
        except StateStorageError:
            raise
        except Exception as exc:
            raise StateStorageError(
                f"{method.__name__} failed for {self.run_dir}: {exc}"
            ) from exc

    return wrapper


class StateStorage:
    def __init__(self, run_dir: str | Path, config: AppConfig) -> None:
        self.run_dir = Path(run_dir)
        self.config = config
        self._files: dict[str, Any] = {}
        self._formal_windows = 0
        self._last_snapshot: GlobalState | None = None

    @_storage_safe
    def initialize(self, *, prompt_builder: Any | None = None) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        always_open = [
            "normalization_warnings.jsonl",
            "state_errors.jsonl",
            # The final snapshot is required for replay regardless of the
            # periodic-snapshot switch, so the file is always created.
            "state_snapshots.jsonl",
        ]
        for name in always_open:
            self._files[name] = (self.run_dir / name).open("w", encoding="utf-8")
        conditional = [
            (self.config.storage.save_entity_resolutions, "entity_resolutions.jsonl"),
            (self.config.storage.save_state_events, "state_events.jsonl"),
            (self.config.storage.save_state_deltas, "state_deltas.jsonl"),
        ]
        for enabled, name in conditional:
            if enabled:
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
        handle = self._files.get(filename)
        if handle is None:
            return
        handle.write(json.dumps(value.model_dump(mode="json") if hasattr(value, "model_dump") else value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()

    @_storage_safe
    def write_normalization_warnings(
        self, window_global_index: int, warnings: list[NormalizationWarning]
    ) -> None:
        for warning in warnings:
            payload = warning.model_dump(mode="json")
            payload["window_global_index"] = window_global_index
            self._write_jsonl("normalization_warnings.jsonl", payload)

    @_storage_safe
    def write_reduction(
        self,
        result: StateReductionResult,
        *,
        window_global_index: int,
        warmup: bool = False,
    ) -> None:
        if result.resolution is not None and self.config.storage.save_entity_resolutions:
            self._write_jsonl("entity_resolutions.jsonl", result.resolution)
        if result.error:
            # The StateReducer catches exceptions and returns the original state,
            # so the state is not affected by this failed reduction.
            error = result.error if isinstance(result.error, Exception) else RuntimeError(str(result.error))
            self.write_error(
                window_global_index,
                stage="state_reducer",
                error=error,
                observation_succeeded=True,
                state_affected=False,
            )
        if warmup:
            return
        self._formal_windows += 1
        if self.config.storage.save_state_events:
            for event in result.events:
                self._write_jsonl("state_events.jsonl", event)
        if result.delta is not None and self.config.storage.save_state_deltas:
            self._write_jsonl("state_deltas.jsonl", result.delta)
        self._last_snapshot = result.state.model_copy(deep=True)
        interval = self.config.state.snapshot_interval_windows
        if (
            self.config.storage.save_state_snapshots
            and self._formal_windows % interval == 0
        ):
            self._write_jsonl("state_snapshots.jsonl", result.state)

    @_storage_safe
    def write_error(
        self,
        window_global_index: int,
        *,
        stage: str,
        error: Exception,
        observation_succeeded: bool,
        state_affected: bool,
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
                "state_affected": state_affected,
                "original_state_reference": state_reference,
            },
        )

    @_storage_safe
    def finalize(self, final_state: GlobalState | None) -> None:
        if final_state is not None:
            # The final snapshot is required for replay regardless of the
            # periodic snapshot switch, so it is always written.
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
