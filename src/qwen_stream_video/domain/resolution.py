"""Explainable entity-resolution result models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .enums import EntityResolutionStatus
from .state import EvidenceReference


class MatchScoreBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type_name_score: float = Field(ge=0.0, le=1.0)
    appearance_score: float = Field(ge=0.0, le=1.0)
    spatial_score: float = Field(ge=0.0, le=1.0)
    relation_score: float = Field(ge=0.0, le=1.0)
    recency_score: float = Field(ge=0.0, le=1.0)
    candidate_hint_score: float = Field(ge=0.0, le=1.0)
    total_score: float = Field(ge=0.0, le=1.0)


class EntityResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_global_index: int = Field(ge=0)
    local_id: str
    global_entity_id: str
    status: EntityResolutionStatus
    selected_score: float | None = Field(default=None, ge=0.0, le=1.0)
    second_best_score: float | None = Field(default=None, ge=0.0, le=1.0)
    candidate_scores: dict[str, MatchScoreBreakdown] = Field(default_factory=dict)
    rejected_reasons: list[str] = Field(default_factory=list)
    evidence: list[EvidenceReference] = Field(default_factory=list)


class EntityResolutionBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_global_index: int = Field(ge=0)
    mappings: list[EntityResolution] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
