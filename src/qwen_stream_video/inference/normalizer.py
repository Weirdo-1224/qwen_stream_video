"""Non-destructive vocabulary normalization for local observations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from ..domain import EntityType, ObservationBatch

DEFAULT_ACTIONS_PATH = Path(__file__).resolve().parents[3] / "vocabularies" / "actions.yaml"
DEFAULT_ATTRIBUTES_PATH = Path(__file__).resolve().parents[3] / "vocabularies" / "attributes.yaml"


class NormalizationWarning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    warning_type: str
    local_id: str | None = None
    field_name: str
    raw_value: str
    normalized_value: str | None = None
    message: str


class NormalizationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch: ObservationBatch
    warnings: list[NormalizationWarning] = Field(default_factory=list)


class ObservationNormalizer:
    """Normalize aliases while retaining every original model value."""

    def __init__(
        self,
        actions_path: str | Path | None = None,
        attributes_path: str | Path | None = None,
    ) -> None:
        self.actions = self._load_mapping(actions_path or DEFAULT_ACTIONS_PATH, "actions")
        self.attributes = self._load_mapping(
            attributes_path or DEFAULT_ATTRIBUTES_PATH, "attributes"
        )
        self.action_aliases = {
            alias: canonical
            for canonical, metadata in self.actions.items()
            for alias in metadata.get("aliases", [])
        }
        self.attribute_aliases = {
            alias: canonical
            for canonical, metadata in self.attributes.items()
            for alias in metadata.get("aliases", [])
        }

    @staticmethod
    def _load_mapping(path: str | Path, key: str) -> dict[str, dict[str, Any]]:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        values = data.get(key, {})
        if not isinstance(values, dict):
            raise TypeError(f"{key} vocabulary must be a mapping")
        return {str(name): (metadata if isinstance(metadata, dict) else {}) for name, metadata in values.items()}

    def normalize(self, batch: ObservationBatch) -> NormalizationResult:
        warnings: list[NormalizationWarning] = []
        for action in batch.actions:
            raw = action.raw_action_type or action.action_type
            action.raw_action_type = raw
            if raw == "unknown":
                action.action_type = "unknown"
                action.normalization_status = "unknown"
                action.action_family = self.actions.get("unknown", {}).get("family", "unknown")
                continue
            canonical = raw if raw in self.actions else self.action_aliases.get(raw)
            if canonical is None:
                action.action_type = "other"
                action.action_family = self.actions.get("other", {}).get("family", "other")
                action.normalization_status = "out_of_vocabulary"
                warnings.append(
                    NormalizationWarning(
                        warning_type="action_out_of_vocabulary",
                        local_id=action.local_id,
                        field_name="action_type",
                        raw_value=raw,
                        normalized_value="other",
                        message=f"Action {raw!r} is outside the controlled vocabulary",
                    )
                )
            else:
                action.action_type = canonical
                action.action_family = self.actions[canonical].get("family", "other")
                action.normalization_status = "canonical" if raw == canonical else "alias_mapped"

        entity_types = {entity.local_id: entity.entity_type for entity in batch.entities}
        for attribute in batch.attribute_observations:
            raw_key = attribute.raw_attribute or attribute.attribute_key or attribute.attribute or ""
            raw_value = attribute.raw_value or attribute.value
            attribute.raw_attribute = raw_key
            attribute.raw_value = raw_value
            canonical = raw_key if raw_key in self.attributes else self.attribute_aliases.get(raw_key)
            if canonical is None:
                attribute.normalization_status = "out_of_vocabulary"
                warnings.append(
                    NormalizationWarning(
                        warning_type="attribute_out_of_vocabulary",
                        local_id=attribute.entity_local_id,
                        field_name="attribute_key",
                        raw_value=raw_key,
                        normalized_value=None,
                        message=f"Attribute {raw_key!r} is outside the controlled vocabulary",
                    )
                )
                # Keep the raw key visible to downstream diagnostics, but it is
                # explicitly non-canonical and cannot enter formal state.
                attribute.attribute_key = raw_key
                attribute.attribute = raw_key
                continue
            attribute.attribute_key = canonical
            attribute.attribute = canonical
            metadata = self.attributes[canonical]
            allowed_types = {str(value) for value in metadata.get("entity_types", [])}
            entity_type = entity_types.get(attribute.entity_local_id, EntityType.UNKNOWN).value
            if entity_type not in allowed_types:
                attribute.normalization_status = "invalid_for_entity_type"
                warnings.append(
                    NormalizationWarning(
                        warning_type="attribute_invalid_for_entity_type",
                        local_id=attribute.entity_local_id,
                        field_name="attribute_key",
                        raw_value=raw_key,
                        normalized_value=canonical,
                        message=f"{canonical} is not valid for entity type {entity_type}",
                    )
                )
            elif raw_value not in {str(value) for value in metadata.get("values", [])}:
                attribute.normalization_status = "out_of_vocabulary"
                warnings.append(
                    NormalizationWarning(
                        warning_type="attribute_value_out_of_vocabulary",
                        local_id=attribute.entity_local_id,
                        field_name="value",
                        raw_value=raw_value,
                        normalized_value=None,
                        message=f"Value {raw_value!r} is not valid for {canonical}",
                    )
                )
            else:
                attribute.normalization_status = "canonical" if raw_key == canonical else "alias_mapped"
        return NormalizationResult(batch=batch, warnings=warnings)
