"""Deterministic global entity registry and lifecycle management."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from ..config import EntityRegistryConfig
from ..domain import (
    EntityLifecycleStatus,
    EntityType,
    EvidenceReference,
    GlobalEntityState,
    GlobalState,
    SpatialObservation,
    VisibilityState,
)
from ..domain.observation import EntityObservation
from ..video import SampledFrame, evidence_timestamps


class EntityMergeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temporary_entity_id: str
    merged_into: str
    migrated_evidence_count: int = 0
    migrated_spatial_count: int = 0
    reason: str = "delayed_match"


class EntityRegistry:
    """Own entity IDs, historical records, and lifecycle state."""

    def __init__(self, config: EntityRegistryConfig | None = None) -> None:
        self.config = config or EntityRegistryConfig()

    def _next_id(self, state: GlobalState, entity_type: EntityType, temporary: bool) -> str:
        key = f"{self.config.temporary_entity_prefix}_{entity_type.value}" if temporary else entity_type.value
        next_number = state.entity_counters.get(key, 0) + 1
        state.entity_counters[key] = next_number
        if temporary:
            return f"{self.config.temporary_entity_prefix}_{entity_type.value}_{next_number:04d}"
        return f"{entity_type.value}_{next_number:04d}"

    def create_entity(
        self,
        state: GlobalState,
        entity_type: EntityType,
        *,
        name: str = "unknown",
        confidence: float = 0.0,
        window_index: int = 0,
        timestamp: float = 0.0,
        scene_id: str | None = None,
        temporary: bool = False,
    ) -> GlobalEntityState:
        entity_id = self._next_id(state, entity_type, temporary)
        entity = GlobalEntityState(
            entity_id=entity_id,
            entity_type=entity_type,
            canonical_name=name or "unknown",
            is_temporary=temporary,
            first_seen_window=window_index,
            last_seen_window=window_index,
            first_seen_time=timestamp,
            last_seen_time=timestamp,
            current_scene_id=scene_id,
            visibility=VisibilityState.VISIBLE,
            confidence=confidence,
        )
        state.entities[entity_id] = entity
        return entity

    def get(self, state: GlobalState, entity_id: str) -> GlobalEntityState | None:
        return state.entities.get(entity_id)

    def find_candidates(
        self,
        state: GlobalState,
        entity_type: EntityType,
        current_scene_id: str | None,
        current_window: int,
        max_missing_windows: int | None = None,
    ) -> list[GlobalEntityState]:
        max_missing = self.config.max_missing_windows if max_missing_windows is None else max_missing_windows
        result = []
        for entity in state.entities.values():
            if entity.entity_type != entity_type or entity.merged_into is not None:
                continue
            if entity.lifecycle_status == EntityLifecycleStatus.INACTIVE:
                continue
            if current_window - entity.last_seen_window > max_missing:
                continue
            if (
                current_scene_id
                and entity.current_scene_id not in {None, current_scene_id}
                and entity.visibility == VisibilityState.NOT_VISIBLE
            ):
                # Historical entities remain eligible when their previous scene
                # is recoverable; scene changes are not identity resets.
                continue
            result.append(entity)
        return sorted(result, key=lambda item: item.entity_id)

    def update_from_observation(
        self,
        state: GlobalState,
        entity_id: str,
        observation: EntityObservation,
        *,
        scene_id: str | None,
        run_id: str,
        sampled_frames: list[SampledFrame] | None = None,
        window_index: int | None = None,
    ) -> GlobalEntityState:
        entity = state.entities[entity_id]
        window = window_index if window_index is not None else entity.last_seen_window
        timestamps = (
            evidence_timestamps(observation.evidence_frames, sampled_frames)
            if sampled_frames is not None and observation.evidence_frames
            else []
        )
        entity.last_seen_window = window
        entity.last_seen_time = max(timestamps) if timestamps else entity.last_seen_time
        entity.current_scene_id = scene_id
        entity.visibility = VisibilityState.VISIBLE
        entity.lifecycle_status = EntityLifecycleStatus.ACTIVE
        entity.missing_window_count = 0
        entity.confidence = max(entity.confidence, observation.confidence)
        if observation.name and observation.name != "unknown" and entity.canonical_name == "unknown":
            entity.canonical_name = observation.name
        for key, value in sorted(observation.appearance.items()):
            text = str(value)
            if observation.confidence >= 0.8:
                previous = entity.appearance_signature.get(key)
                if previous is None:
                    entity.appearance_signature[key] = text
                elif previous != text:
                    entity.appearance_conflicts[key] = entity.appearance_conflicts.get(key, 0) + 1
        entity.spatial_history.append(
            SpatialObservation(
                window_global_index=window,
                scene_id=scene_id or "unknown",
                spatial_region=observation.spatial_region,
                confidence=observation.confidence,
            )
        )
        if observation.evidence_frames:
            entity.evidence.append(
                EvidenceReference(
                    run_id=run_id,
                    window_global_index=window,
                    local_id=observation.local_id,
                    sample_indices=sorted(set(observation.evidence_frames)),
                    timestamps_seconds=timestamps,
                )
            )
        return entity

    def mark_not_observed(
        self,
        state: GlobalState,
        observed_entity_ids: set[str] | str,
        current_window: int,
        *,
        suppress_missing_count: bool = False,
    ) -> list[str]:
        if isinstance(observed_entity_ids, str):
            observed_entity_ids = {observed_entity_ids}
        changed: list[str] = []
        for entity_id in sorted(state.entities):
            entity = state.entities[entity_id]
            if entity_id in observed_entity_ids or entity.lifecycle_status == EntityLifecycleStatus.MERGED:
                continue
            if suppress_missing_count:
                entity.visibility = VisibilityState.NOT_VISIBLE
                changed.append(entity_id)
                continue
            entity.missing_window_count += 1
            entity.visibility = VisibilityState.NOT_VISIBLE
            if entity.missing_window_count > self.config.max_missing_windows:
                entity.lifecycle_status = EntityLifecycleStatus.INACTIVE
            else:
                entity.lifecycle_status = EntityLifecycleStatus.TEMPORARILY_MISSING
            changed.append(entity_id)
        return changed

    def merge_temporary_entity(
        self,
        state: GlobalState,
        temporary_entity_id: str,
        target_entity_id: str,
        *,
        window_index: int | None = None,
    ) -> EntityMergeResult:
        temporary = state.entities.get(temporary_entity_id)
        target = state.entities.get(target_entity_id)
        if temporary is None or target is None:
            raise KeyError("Both temporary and target entities must exist")
        if not temporary.is_temporary:
            raise ValueError("Only temporary entities may be merged")
        if target.is_temporary:
            raise ValueError("A temporary entity cannot be merged into another temporary entity")
        temporary.merged_into = target_entity_id
        temporary.lifecycle_status = EntityLifecycleStatus.MERGED
        migrated_evidence = len(temporary.evidence)
        migrated_spatial = len(temporary.spatial_history)
        target.evidence.extend(temporary.evidence)
        target.spatial_history.extend(temporary.spatial_history)
        target.aliases.append(temporary_entity_id)
        if window_index is not None:
            target.last_seen_window = max(target.last_seen_window, window_index)
        return EntityMergeResult(
            temporary_entity_id=temporary_entity_id,
            merged_into=target_entity_id,
            migrated_evidence_count=migrated_evidence,
            migrated_spatial_count=migrated_spatial,
        )
