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
            A tuple of warning strings, one per non-fatal issue found.

        Raises:
            ModelOutputSemanticError: If the batch violates a hard semantic
                constraint and cannot be treated as a valid observation.
        """
        if window is not None:
            self._override_window_fields(batch, window)

        entity_ids = {e.local_id for e in batch.entities}
        self._validate_id_uniqueness(batch)
        self._validate_references(batch, entity_ids)
        self._validate_evidence_frames(batch, len(sampled_frames))
        warnings = self._validate_action_vocabulary(batch)

        return warnings

    def _override_window_fields(
        self, batch: ObservationBatch, window: VideoWindow
    ) -> None:
        """Replace model-reported window fields with the real window values."""
        batch.window = WindowObservation(
            global_index=window.global_index,
            start_seconds=window.start_seconds,
            end_seconds=window.end_seconds,
        )

    def _validate_id_uniqueness(self, batch: ObservationBatch) -> None:
        """Ensure entity and action local IDs are unique within the window."""
        entity_ids = [e.local_id for e in batch.entities]
        duplicates = self._find_duplicates(entity_ids)
        if duplicates:
            raise ModelOutputSemanticError(
                f"Duplicate entity local_id(s) in window {batch.window.global_index}: "
                f"{sorted(duplicates)}"
            )

        action_ids = [a.local_id for a in batch.actions]
        duplicates = self._find_duplicates(action_ids)
        if duplicates:
            raise ModelOutputSemanticError(
                f"Duplicate action local_id(s) in window {batch.window.global_index}: "
                f"{sorted(duplicates)}"
            )

        attribute_entity_ids = [a.entity_local_id for a in batch.attribute_observations]
        duplicates = self._find_duplicates(attribute_entity_ids)
        if duplicates:
            raise ModelOutputSemanticError(
                f"Duplicate attribute entity_local_id(s) in window "
                f"{batch.window.global_index}: {sorted(duplicates)}"
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
        self, batch: ObservationBatch, entity_ids: set[str]
    ) -> None:
        """Ensure every action and attribute references existing entities."""
        for action in batch.actions:
            if action.actor_local_id not in entity_ids:
                raise ModelOutputSemanticError(
                    f"Action {action.local_id} references missing actor "
                    f"{action.actor_local_id} in window {batch.window.global_index}"
                )
            if (
                action.target_local_id is not None
                and action.target_local_id not in entity_ids
            ):
                raise ModelOutputSemanticError(
                    f"Action {action.local_id} references missing target "
                    f"{action.target_local_id} in window {batch.window.global_index}"
                )
            if (
                action.tool_local_id is not None
                and action.tool_local_id not in entity_ids
            ):
                raise ModelOutputSemanticError(
                    f"Action {action.local_id} references missing tool "
                    f"{action.tool_local_id} in window {batch.window.global_index}"
                )

        for attribute in batch.attribute_observations:
            if attribute.entity_local_id not in entity_ids:
                raise ModelOutputSemanticError(
                    f"Attribute {attribute.attribute} references missing entity "
                    f"{attribute.entity_local_id} in window {batch.window.global_index}"
                )

    def _validate_evidence_frames(
        self, batch: ObservationBatch, frame_count: int
    ) -> None:
        """Ensure all evidence frame indices are in range, deduplicated and sorted."""
        for entity in batch.entities:
            self._check_and_clean_evidence_frames(
                entity.evidence_frames, frame_count, "entity", entity.local_id, batch
            )
        for action in batch.actions:
            self._check_and_clean_evidence_frames(
                action.evidence_frames, frame_count, "action", action.local_id, batch
            )
        for attribute in batch.attribute_observations:
            self._check_and_clean_evidence_frames(
                attribute.evidence_frames,
                frame_count,
                "attribute",
                attribute.attribute,
                batch,
            )
        for uncertainty in batch.uncertainties:
            self._check_and_clean_evidence_frames(
                uncertainty.evidence_frames,
                frame_count,
                "uncertainty",
                uncertainty.description[:40],
                batch,
            )

    def _check_and_clean_evidence_frames(
        self,
        indices: list[int],
        frame_count: int,
        item_type: str,
        item_id: str,
        batch: ObservationBatch,
    ) -> None:
        """Validate a list of evidence frame indices and mutate it in place."""
        for idx in indices:
            if not 0 <= idx < frame_count:
                raise ModelOutputSemanticError(
                    f"{item_type.capitalize()} {item_id} has out-of-range evidence frame "
                    f"index {idx} in window {batch.window.global_index}; "
                    f"valid range is [0, {frame_count - 1}]"
                )
        indices[:] = sorted(set(indices))

    def _validate_action_vocabulary(self, batch: ObservationBatch) -> list[str]:
        """Map unknown action types to ``unknown`` and return warnings."""
        warnings: list[str] = []
        for action in batch.actions:
            if action.action_type not in self.allowed_actions:
                original = action.action_type
                action.action_type = "unknown"
                warnings.append(
                    f"Action {action.local_id} in window {batch.window.global_index} "
                    f"has unknown action type '{original}'; mapped to 'unknown'. "
                    f"Description: {action.description or 'no description'}"
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
