"""Strict local visual Observation protocol (Schema 2.0).

Observation models deliberately contain no formal global identity, action
lifecycle, or global state.  Those concepts belong to ``state/``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import ActionPhaseObservation, EntityType, ViewType, VisibilityQuality

NormalizationStatus = Literal["canonical", "alias_mapped", "out_of_vocabulary", "unknown"]
AttributeNormalizationStatus = Literal[
    "canonical", "alias_mapped", "out_of_vocabulary", "invalid_for_entity_type"
]
ContinuityHint = Literal["continuous", "reframed", "camera_change", "unknown"]


class WindowObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    global_index: int = Field(ge=0)
    start_seconds: float = Field(ge=0)
    commit_start_seconds: float | None = None
    end_seconds: float = Field(gt=0)

    @model_validator(mode="after")
    def _interval_is_valid(self) -> WindowObservation:
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        if self.commit_start_seconds is not None and not (
            self.start_seconds <= self.commit_start_seconds < self.end_seconds
        ):
            raise ValueError("start_seconds <= commit_start_seconds < end_seconds is required")
        return self


class SceneObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera_change: bool = False
    view_type: ViewType = ViewType.UNKNOWN
    scene_visibility: VisibilityQuality = VisibilityQuality.UNKNOWN
    target_visibility: VisibilityQuality = VisibilityQuality.UNKNOWN
    continuity_hint: ContinuityHint = "unknown"
    description: str = ""
    # Stage1 input compatibility.  New prompts use the two explicit fields.
    visibility: VisibilityQuality | None = None

    @model_validator(mode="after")
    def _legacy_visibility_is_explicit(self) -> SceneObservation:
        if self.visibility is not None:
            if self.scene_visibility == VisibilityQuality.UNKNOWN:
                self.scene_visibility = self.visibility
            if self.target_visibility == VisibilityQuality.UNKNOWN:
                self.target_visibility = self.visibility
        return self


class EntityObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    local_id: str = Field(min_length=1)
    entity_type: EntityType
    name: str = "unknown"
    description: str = ""
    appearance: dict[str, Any] = Field(default_factory=dict)
    spatial_region: str = "unknown"
    candidate_global_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_frames: list[int] = Field(default_factory=list)


class ActionObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    local_id: str = Field(min_length=1)
    actor_local_id: str | None = None
    action_type: str = Field(min_length=1)
    raw_action_type: str | None = None
    action_family: str | None = None
    normalization_status: NormalizationStatus = "canonical"
    target_local_id: str | None = None
    tool_local_id: str | None = None
    phase_observation: ActionPhaseObservation = ActionPhaseObservation.UNKNOWN
    description: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_frames: list[int] = Field(default_factory=list)


class AttributeObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_local_id: str = Field(min_length=1)
    attribute_key: str | None = None
    value: str = Field(min_length=1)
    raw_attribute: str | None = None
    raw_value: str | None = None
    normalization_status: AttributeNormalizationStatus = "canonical"
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_frames: list[int] = Field(default_factory=list)
    # Stage1 spelling retained only as a migration input/output convenience.
    attribute: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _migrate_attribute_name(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        result = dict(value)
        if result.get("attribute_key") is None and result.get("attribute") is not None:
            result["attribute_key"] = result["attribute"]
        if result.get("raw_attribute") is None and result.get("attribute") is not None:
            result["raw_attribute"] = result["attribute"]
        if result.get("raw_value") is None and result.get("value") is not None:
            result["raw_value"] = result["value"]
        return result

    @model_validator(mode="after")
    def _attribute_key_is_present(self) -> AttributeObservation:
        if not self.attribute_key:
            raise ValueError("attribute_key is required")
        if self.attribute is None:
            self.attribute = self.attribute_key
        return self


class RelationObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject_local_id: str = Field(min_length=1)
    relation_type: str = Field(min_length=1)
    object_local_id: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_frames: list[int] = Field(default_factory=list)


class UncertaintyObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uncertainty_type: Literal[
        "identity", "action", "attribute", "visibility", "causality", "other"
    ] = "other"
    description: str = Field(min_length=1)
    related_local_ids: list[str] = Field(default_factory=list)
    evidence_frames: list[int] = Field(default_factory=list)


class TaskConditionedInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_local_ids: list[str] = Field(default_factory=list)


class ObservationBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0", "2.0"] = "2.0"
    window: WindowObservation
    summary: str = ""
    scene: SceneObservation = Field(default_factory=SceneObservation)
    entities: list[EntityObservation] = Field(default_factory=list)
    actions: list[ActionObservation] = Field(default_factory=list)
    attribute_observations: list[AttributeObservation] = Field(default_factory=list)
    relations: list[RelationObservation] = Field(default_factory=list)
    uncertainties: list[UncertaintyObservation] = Field(default_factory=list)
    visual_fact: str | None = None
    task_conditioned_interpretation: TaskConditionedInterpretation | None = None

    @model_validator(mode="after")
    def _v2_requires_commit_interval(self) -> ObservationBatch:
        if self.schema_version == "2.0" and self.window.commit_start_seconds is None:
            raise ValueError("Schema 2.0 requires window.commit_start_seconds")
        return self
