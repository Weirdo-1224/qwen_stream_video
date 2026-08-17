"""Formal program-generated state events and per-window deltas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .state import EvidenceReference, TimeInterval


class StateEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    event_type: str
    window_global_index: int = Field(ge=0)
    timestamp_interval: TimeInterval | None = None
    entity_id: str | None = None
    action_id: str | None = None
    scene_id: str | None = None
    attribute_key: str | None = None
    before: Any | None = None
    after: Any | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reason: str = ""
    evidence: list[EvidenceReference] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StateDelta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_global_index: int = Field(ge=0)
    scene_id: str
    entity_updates: list[str] = Field(default_factory=list)
    action_updates: list[str] = Field(default_factory=list)
    attribute_updates: list[str] = Field(default_factory=list)
    emitted_event_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
