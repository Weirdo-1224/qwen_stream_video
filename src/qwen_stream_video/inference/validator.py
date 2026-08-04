"""Semantic validation for incremental observation batches.

Pydantic validates field types and basic constraints; this module checks
business-level rules such as ID uniqueness, reference integrity, evidence frame
bounds, and action vocabulary membership.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..domain import ObservationBatch, WindowObservation
from ..exceptions import ModelOutputSemanticError
from ..video import SampledFrame, VideoWindow

DEFAULT_VOCAB_PATH = (
    Path(__file__).resolve().parents[3] / "vocabularies" / "actions.yaml"
)


class ObservationSemanticValidator:
    """Validate that an observation batch makes sense within a single window."""

    def __init__(self, vocab_path: str | Path | None = None) -> None:
        """Load the action vocabulary used during validation.

        Args:
            vocab_path: Path to a YAML file containing an ``actions`` list.
                Defaults to ``vocabularies/actions.yaml`` relative to the repo
                root.
        """
        self.vocab_path = Path(vocab_path) if vocab_path else DEFAULT_VOCAB_PATH
        self.allowed_actions = self._load_actions()

    def _load_actions(self) -> set[str]:
        """Load allowed action strings from the configured vocabulary file."""
        if not self.vocab_path.exists():
            raise FileNotFoundError(
                f"Action vocabulary not found: {self.vocab_path}"
            )
        with self.vocab_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        actions = data.get("actions", [])
        if not isinstance(actions, list):
            raise TypeError(
                "Action vocabulary must contain a list under the 'actions' key"
            )
        return {str(a) for a in actions}

    def validate(
        self,
        batch: ObservationBatch,
        sampled_frames: list[SampledFrame],
        window: VideoWindow | None = None,
    ) -> list[str]:
        """Validate ``batch`` and return a list of non-fatal warnings.

        Hard semantic failures (duplicate IDs, missing references, or out-of-range
        evidence frames) raise :class:`ModelOutputSemanticError`. Invalid action
        types are mapped to ``unknown`` and reported as warnings instead.

        Args:
            batch: The parsed observation batch to validate.
            sampled_frames: Frames sampled from the current window; used to bound
                evidence frame indices.
            window: Optional actual video window. When provided, the model's
                returned window fields are overwritten with the real values.

        Returns:
            A list of warning strings, one per non-fatal issue found.

        Raises:
            ModelOutputSemanticError: If the batch violates a hard semantic
                constraint and cannot be treated as a valid observation.
        """
        warnings: list[str] = []
        frame_count = len(sampled_frames)

        for obs in batch.observations:
            if window is not None:
                self._override_window_fields(obs, window)
            entity_ids = {e.local_id for e in obs.entities}
            self._validate_id_uniqueness(obs)
            self._validate_references(obs, entity_ids)
            self._validate_evidence_frames(obs, frame_count)
            warnings.extend(self._validate_action_vocabulary(obs))

        return warnings

    def _override_window_fields(
        self, obs: WindowObservation, window: VideoWindow
    ) -> None:
        """Replace model-reported window fields with the real window values."""
        obs.window_global_index = window.global_index
        obs.window_run_index = window.run_index
        obs.window_start_seconds = window.start_seconds
        obs.window_end_seconds = window.end_seconds

    def _validate_id_uniqueness(self, obs: WindowObservation) -> None:
        """Ensure entity and action local IDs are unique within the window."""
        entity_ids = [e.local_id for e in obs.entities]
        duplicates = self._find_duplicates(entity_ids)
        if duplicates:
            raise ModelOutputSemanticError(
                f"Duplicate entity local_id(s) in window {obs.window_global_index}: "
                f"{sorted(duplicates)}"
            )

        action_ids = [a.local_id for a in obs.actions]
        duplicates = self._find_duplicates(action_ids)
        if duplicates:
            raise ModelOutputSemanticError(
                f"Duplicate action local_id(s) in window {obs.window_global_index}: "
                f"{sorted(duplicates)}"
            )

    def _find_duplicates(self, items: list[str]) -> set[str]:
        """Return the set of values that appear more than once."""
        seen: set[str] = set()
        duplicates: set[str] = set()
        for item in items:
            if item in seen:
                duplicates.add(item)
            seen.add(item)
        return duplicates

    def _validate_references(
        self, obs: WindowObservation, entity_ids: set[str]
    ) -> None:
        """Ensure every action references existing entities."""
        for action in obs.actions:
            if action.actor_id not in entity_ids:
                raise ModelOutputSemanticError(
                    f"Action {action.local_id} references missing actor "
                    f"{action.actor_id} in window {obs.window_global_index}"
                )
            if action.target_id is not None and action.target_id not in entity_ids:
                raise ModelOutputSemanticError(
                    f"Action {action.local_id} references missing target "
                    f"{action.target_id} in window {obs.window_global_index}"
                )

    def _validate_evidence_frames(
        self, obs: WindowObservation, frame_count: int
    ) -> None:
        """Ensure all evidence frame indices are in range, deduplicated and sorted."""
        for action in obs.actions:
            indices = action.evidence_frame_sample_indices
            for idx in indices:
                if not 0 <= idx < frame_count:
                    raise ModelOutputSemanticError(
                        f"Action {action.local_id} has out-of-range evidence frame "
                        f"index {idx} in window {obs.window_global_index}; "
                        f"valid range is [0, {frame_count - 1}]"
                    )
            # Mutate in place to keep indices clean and deterministic.
            action.evidence_frame_sample_indices = sorted(set(indices))

    def _validate_action_vocabulary(self, obs: WindowObservation) -> list[str]:
        """Map unknown action types to ``unknown`` and return warnings."""
        warnings: list[str] = []
        for action in obs.actions:
            if action.action_type not in self.allowed_actions:
                original = action.action_type
                action.action_type = "unknown"
                warnings.append(
                    f"Action {action.local_id} in window {obs.window_global_index} "
                    f"has unknown action type '{original}'; mapped to 'unknown'. "
                    f"Description: {action.attributes or 'no attributes'}"
                )
        return warnings

    def __getstate__(self) -> dict[str, Any]:
        """Support pickling by serializing the allowed action set as a list."""
        return {
            "vocab_path": str(self.vocab_path),
            "allowed_actions": sorted(self.allowed_actions),
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the validator from pickled state."""
        self.vocab_path = Path(state["vocab_path"])
        self.allowed_actions = set(state["allowed_actions"])
