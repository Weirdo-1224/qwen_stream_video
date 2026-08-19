"""Parsing and validation of model output for incremental observations.

The :class:`ResponseParser` strips Markdown fences, extracts the top-level JSON
object, validates it against the :class:`ObservationBatch` schema and runs
semantic validation.
"""

from __future__ import annotations

import json
import re

from ..domain import ObservationBatch
from ..exceptions import ModelOutputParseError, ModelOutputSchemaError
from ..video import SampledFrame, VideoWindow
from .local_adapter import LocalObservationAdapter
from .validator import ObservationSemanticValidator

CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


class ResponseParser:
    """Parse raw model responses into validated observation batches."""

    def __init__(
        self,
        validator: ObservationSemanticValidator | None = None,
        local_adapter: LocalObservationAdapter | None = None,
    ) -> None:
        """Initialize with an optional semantic validator.

        Args:
            validator: Validator used to check references, IDs, evidence frames
                and action vocabulary. A default validator is created when
                ``None``.
        """
        self.validator = validator or ObservationSemanticValidator()
        self.local_adapter = local_adapter or LocalObservationAdapter()

    def parse(
        self,
        raw_text: str,
        sampled_frames: list[SampledFrame],
        window: VideoWindow | None = None,
    ) -> tuple[ObservationBatch, list[str]]:
        """Parse ``raw_text`` into an observation batch.

        The parser removes Markdown code fences, extracts the top-level JSON
        object, deserializes it into :class:`ObservationBatch` and runs semantic
        validation.

        Args:
            raw_text: Raw model response, possibly wrapped in Markdown fences.
            sampled_frames: Frames sampled from the current window; used to
                validate evidence frame indices.
            window: Optional actual video window; when provided, model-reported
                window fields are overwritten by the real values.

        Returns:
            A tuple of the parsed ``ObservationBatch`` and a list of warnings.

        Raises:
            ModelOutputParseError: If the response contains no JSON or the JSON
                is malformed.
            ModelOutputSchemaError: If the JSON does not match the observation
                schema.
            ModelOutputSemanticError: If the parsed observation violates hard
                semantic constraints.
        """
        json_str = self._extract_json(raw_text)
        data = self._load_json(json_str)
        adapter_result = self.local_adapter.adapt(data, sampled_frames, window)
        adapter_warnings: list[str] = []
        if adapter_result is not None:
            data, adapter_warnings = adapter_result
        data, window_warnings = self._apply_runtime_window(data, window)
        batch = self._validate_schema(data)
        warnings = self.validator.validate(batch, sampled_frames, window=window)
        return batch, adapter_warnings + window_warnings + warnings

    @staticmethod
    def _extract_json(raw_text: str) -> str:
        """Strip Markdown fences and isolate the top-level JSON object."""
        text = raw_text.strip()

        match = CODE_BLOCK_RE.search(text)
        if match:
            text = match.group(1).strip()

        start_idx = text.find("{")
        if start_idx == -1:
            raise ModelOutputParseError("No JSON object found in model output")

        end_idx = text.rfind("}")
        if end_idx == -1 or end_idx < start_idx:
            raise ModelOutputParseError("Unterminated JSON object in model output")

        return text[start_idx : end_idx + 1]

    @staticmethod
    def _load_json(json_str: str) -> dict:
        """Parse the JSON string into a Python dict."""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ModelOutputParseError(
                f"Model output is not valid JSON: {exc}"
            ) from exc

    @staticmethod
    def _apply_runtime_window(
        data: dict,
        window: VideoWindow | None,
    ) -> tuple[dict, list[str]]:
        """Fill the program-owned window when the model omits it."""
        if window is None or "window" in data:
            return data, []
        if data.get("schema_version") != "2.0":
            return data, []
        updated = dict(data)
        updated["window"] = {
            "global_index": window.global_index,
            "start_seconds": window.start_seconds,
            "commit_start_seconds": window.commit_start_seconds,
            "end_seconds": window.end_seconds,
        }
        return updated, ["Injected the program-owned window into Schema 2.0 output"]

    @staticmethod
    def _validate_schema(data: dict) -> ObservationBatch:
        """Validate the parsed data against the observation batch schema."""
        try:
            return ObservationBatch.model_validate(data)
        except Exception as exc:
            raise ModelOutputSchemaError(
                f"Model output does not match the observation schema: {exc}"
            ) from exc
