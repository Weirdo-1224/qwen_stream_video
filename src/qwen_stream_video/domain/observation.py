"""Incremental observation schema for a single video window.

The schema describes only the current window. ``candidate_global_id`` is a
hint for cross-window identity tracking, but no global state is maintained
here.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EntityType(str, Enum):
    """Broad categories of detected entities."""

    PERSON = "person"
    OBJECT = "object"
    EQUIPMENT = "equipment"
    TOOL = "tool"
    LOCATION = "location"
    TEXT = "text"
    OTHER = "other"


class Viewpoint(str, Enum):
    """Camera or observed viewpoint relative to the entity or scene."""

    FRONT = "front"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    CLOSE_UP = "close_up"
    WIDE = "wide"
    OVERHEAD = "overhead"
    OTHER = "other"


class Visibility(str, Enum):
    """How much of an entity is visible in the window."""

    FULLY_VISIBLE = "fully_visible"
    PARTIALLY_OCCLUDED = "partially_occluded"
    MOSTLY_OCCLUDED = "mostly_occluded"
    NOT_VISIBLE = "not_visible"


class ActionPhase(str, Enum):
    """Temporal phase of an action within the current window."""

    START = "start"
    CONTINUE = "continue"
    STOP = "stop"
    HOLD = "hold"


class Attribute(BaseModel):
    """A named attribute attached to an entity or action."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    value: str | float | bool | int
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class Uncertainty(BaseModel):
    """Something the model is unsure about in the current window."""

    model_config = ConfigDict(extra="ignore")

    category: str = Field(min_length=1)
    description: str = Field(min_length=1)
    severity: str | None = Field(default=None, min_length=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class Entity(BaseModel):
    """A detected entity in the current window."""

    model_config = ConfigDict(extra="ignore")

    local_id: str = Field(min_length=1)
    entity_type: EntityType
    label: str = Field(min_length=1)
    candidate_global_id: str | None = Field(default=None, min_length=1)
    viewpoint: Viewpoint | None = None
    visibility: Visibility | None = None
    bounding_box: list[float] | None = Field(default=None)
    attributes: list[Attribute] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("bounding_box")
    @classmethod
    def _bounding_box_is_four(cls, value: list[float] | None) -> list[float] | None:
        if value is None:
            return value
        if len(value) != 4:
            raise ValueError("bounding_box must contain exactly four floats")
        return value


class Action(BaseModel):
    """A detected action in the current window."""

    model_config = ConfigDict(extra="ignore")

    local_id: str = Field(min_length=1)
    actor_id: str = Field(min_length=1)
    action_type: str = Field(min_length=1)
    phase: ActionPhase | None = None
    target_id: str | None = Field(default=None, min_length=1)
    start_time_seconds: float | None = Field(default=None, ge=0.0)
    end_time_seconds: float | None = Field(default=None, ge=0.0)
    evidence_frame_sample_indices: list[int] = Field(default_factory=list)
    attributes: list[Attribute] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("end_time_seconds")
    @classmethod
    def _end_after_start(
        cls,
        value: float | None,
        info: dict[str, Any],
    ) -> float | None:
        start = info.data.get("start_time_seconds")
        if start is not None and value is not None and value <= start:
            raise ValueError("end_time_seconds must be greater than start_time_seconds")
        return value


class SceneObservation(BaseModel):
    """High-level scene description for the current window."""

    model_config = ConfigDict(extra="ignore")

    description: str = Field(min_length=1)
    setting: str | None = Field(default=None, min_length=1)
    lighting: str | None = Field(default=None, min_length=1)
    viewpoint: Viewpoint | None = None


class WindowObservation(BaseModel):
    """All observations for a single temporal window.

    The window fields describe the exact window being observed; downstream code
    may override them with values from the corresponding :class:`VideoWindow`.
    """

    model_config = ConfigDict(extra="ignore")

    schema_version: str = Field(default="1.0", min_length=1)
    window_run_index: int = Field(ge=0)
    window_global_index: int = Field(ge=0)
    window_start_seconds: float = Field(ge=0.0)
    window_end_seconds: float = Field(gt=0.0)
    scene: SceneObservation
    entities: list[Entity] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)
    uncertainties: list[Uncertainty] = Field(default_factory=list)
    summary: str | None = Field(default=None, min_length=1)

    @field_validator("window_end_seconds")
    @classmethod
    def _end_after_start(
        cls,
        value: float,
        info: dict[str, Any],
    ) -> float:
        start = info.data.get("window_start_seconds")
        if start is not None and value <= start:
            raise ValueError("window_end_seconds must be greater than window_start_seconds")
        return value


class ObservationBatch(BaseModel):
    """A batch of per-window observations."""

    model_config = ConfigDict(extra="ignore")

    schema_version: str = Field(default="1.0", min_length=1)
    observations: list[WindowObservation] = Field(default_factory=list)
