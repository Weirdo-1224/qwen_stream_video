"""Incremental observation schema for a single video window.

The schema describes only the current window. ``candidate_global_id`` is a
hint for cross-window identity tracking, but no global state is maintained
here.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .enums import (
    ActionPhaseObservation,
    EntityType,
    ViewType,
    VisibilityQuality,
)


class WindowObservation(BaseModel):
    """Window coordinates for the observation batch.

    Downstream code overwrites these fields with the real values from the
    corresponding :class:`VideoWindow`.
    """

    model_config = ConfigDict(extra="ignore")

    global_index: int
    start_seconds: float
    end_seconds: float


class SceneObservation(BaseModel):
    """High-level scene description for the current window."""

    model_config = ConfigDict(extra="ignore")

    camera_change: bool = False
    view_type: ViewType = ViewType.UNKNOWN
    visibility: VisibilityQuality = VisibilityQuality.UNKNOWN
    description: str = ""


class EntityObservation(BaseModel):
    """A detected entity in the current window."""

    model_config = ConfigDict(extra="ignore")

    local_id: str = Field(min_length=1)
    entity_type: EntityType
    name: str = "unknown"
    description: str = ""
    appearance: dict[str, str] = Field(default_factory=dict)
    spatial_region: str = "unknown"
    candidate_global_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_frames: list[int] = Field(default_factory=list)


class ActionObservation(BaseModel):
    """A detected action in the current window."""

    model_config = ConfigDict(extra="ignore")

    local_id: str = Field(min_length=1)
    actor_local_id: str = Field(min_length=1)
    action_type: str = Field(min_length=1)
    target_local_id: str | None = None
    tool_local_id: str | None = None
    phase_observation: ActionPhaseObservation = ActionPhaseObservation.UNKNOWN
    description: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_frames: list[int] = Field(default_factory=list)


class AttributeObservation(BaseModel):
    """A change or measurement of an entity attribute in the current window."""

    model_config = ConfigDict(extra="ignore")

    entity_local_id: str = Field(min_length=1)
    attribute: str = Field(min_length=1)
    value: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_frames: list[int] = Field(default_factory=list)


class UncertaintyObservation(BaseModel):
    """Something the model is unsure about in the current window."""

    model_config = ConfigDict(extra="ignore")

    description: str = Field(min_length=1)
    related_local_ids: list[str] = Field(default_factory=list)
    evidence_frames: list[int] = Field(default_factory=list)


class ObservationBatch(BaseModel):
    """A single observation batch describing exactly one video window."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(min_length=1)
    window: WindowObservation
    summary: str = ""
    scene: SceneObservation = Field(default_factory=SceneObservation)
    entities: list[EntityObservation] = Field(default_factory=list)
    actions: list[ActionObservation] = Field(default_factory=list)
    attribute_observations: list[AttributeObservation] = Field(default_factory=list)
    uncertainties: list[UncertaintyObservation] = Field(default_factory=list)
