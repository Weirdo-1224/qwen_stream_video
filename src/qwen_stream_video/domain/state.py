"""Serializable domain models for deterministic global state."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .enums import (
    ActionLifecycle,
    AttributeConfirmationStatus,
    EntityLifecycleStatus,
    EntityType,
    ViewType,
    VisibilityState,
)


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    window_global_index: int = Field(ge=0)
    local_id: str | None = None
    sample_indices: list[int] = Field(default_factory=list)
    timestamps_seconds: list[float] = Field(default_factory=list)


class TimeInterval(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lower: float = Field(ge=0)
    upper: float = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> TimeInterval:
        if self.lower > self.upper:
            raise ValueError("TimeInterval.lower must be <= upper")
        return self


class SpatialObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_global_index: int = Field(ge=0)
    scene_id: str
    spatial_region: str = "unknown"
    confidence: float = Field(ge=0.0, le=1.0)


class AttributeState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attribute_key: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    status: AttributeConfirmationStatus = AttributeConfirmationStatus.OBSERVED
    first_observed_window: int = Field(ge=0)
    last_observed_window: int = Field(ge=0)
    confirmed_window: int | None = None
    previous_value: str | None = None
    pending_value: str | None = None
    pending_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    pending_support_windows: list[int] = Field(default_factory=list)
    supporting_observations: int = 0
    contradicting_observations: int = 0
    evidence: list[EvidenceReference] = Field(default_factory=list)


class GlobalEntityState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: str
    entity_type: EntityType
    canonical_name: str = "unknown"
    aliases: list[str] = Field(default_factory=list)
    is_temporary: bool = False
    merged_into: str | None = None
    first_seen_window: int = 0
    last_seen_window: int = 0
    first_seen_time: float = 0.0
    last_seen_time: float = 0.0
    current_scene_id: str | None = None
    visibility: VisibilityState = VisibilityState.UNKNOWN
    lifecycle_status: EntityLifecycleStatus = EntityLifecycleStatus.ACTIVE
    appearance_signature: dict[str, str] = Field(default_factory=dict)
    spatial_history: list[SpatialObservation] = Field(default_factory=list)
    attributes: dict[str, AttributeState] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[EvidenceReference] = Field(default_factory=list)
    missing_window_count: int = 0
    appearance_conflicts: dict[str, int] = Field(default_factory=dict)


class GlobalActionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    actor_id: str
    action_type: str
    action_family: str | None = None
    target_id: str | None = None
    tool_id: str | None = None
    lifecycle: ActionLifecycle = ActionLifecycle.CANDIDATE
    start_window: int = 0
    last_observed_window: int = 0
    end_window: int | None = None
    start_time_interval: TimeInterval = Field(
        default_factory=lambda: TimeInterval(lower=0.0, upper=0.0)
    )
    end_time_interval: TimeInterval | None = None
    observed_windows: list[int] = Field(default_factory=list)
    missing_window_count: int = 0
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[EvidenceReference] = Field(default_factory=list)


class SceneState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    view_type: ViewType = ViewType.UNKNOWN
    start_window: int = 0
    last_active_window: int = 0
    continuity: str = "unknown"
    visible_entity_ids: list[str] = Field(default_factory=list)


class GlobalState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = "2.0"
    run_id: str = "replay"
    last_committed_window: int | None = None
    current_scene_id: str | None = None
    scenes: dict[str, SceneState] = Field(default_factory=dict)
    entities: dict[str, GlobalEntityState] = Field(default_factory=dict)
    actions: dict[str, GlobalActionState] = Field(default_factory=dict)
    active_action_ids: list[str] = Field(default_factory=list)
    pending_attribute_keys: list[str] = Field(default_factory=list)
    entity_counters: dict[str, int] = Field(default_factory=dict)
    action_counter: int = 0
    event_counter: int = 0
    scene_counter: int = 0
