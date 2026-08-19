"""Deterministic global entity registry and lifecycle management."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from ..config import EntityRegistryConfig
from ..domain import (
    EntityLifecycleStatus,
    EntityType,
    EvidenceReference,
    GlobalEntityState,
    GlobalState,
    RelationReference,
    SpatialObservation,
    StateEvent,
    VisibilityState,
)
from ..domain.observation import EntityObservation, RelationObservation
from ..exceptions import EntityRegistryError
from ..video import SampledFrame, evidence_timestamps


def _registry_safe(method: Any) -> Any:
    def wrapper(self: EntityRegistry, *args: Any, **kwargs: Any) -> Any:
        try:
            return method(self, *args, **kwargs)
        except EntityRegistryError:
            raise
        except Exception as exc:
            raise EntityRegistryError(
                f"{method.__name__} failed: {exc}"
            ) from exc

    return wrapper


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

    def _event(
        self,
        state: GlobalState,
        event_type: str,
        window_index: int,
        entity_id: str,
        reason: str = "",
        evidence: EvidenceReference | None = None,
        metadata: dict[str, object] | None = None,
    ) -> StateEvent:
        state.event_counter += 1
        return StateEvent(
            event_id=f"event_{state.event_counter:06d}",
            event_type=event_type,
            window_global_index=window_index,
            entity_id=entity_id,
            reason=reason,
            evidence=[evidence] if evidence is not None else [],
            metadata=metadata or {},
        )

    @_registry_safe
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

    @_registry_safe
    def find_candidates(
        self,
        state: GlobalState,
        entity_type: EntityType,
        current_scene_id: str | None,
        current_window: int,
        max_missing_windows: int | None = None,
        *,
        preserve_entities_across_scenes: bool = False,
    ) -> list[GlobalEntityState]:
        max_missing = self.config.max_missing_windows if max_missing_windows is None else max_missing_windows
        result = []
        for entity_id in sorted(state.entities):
            entity = state.entities[entity_id]
            if entity.entity_type != entity_type:
                continue
            if entity.merged_into is not None:
                continue
            if entity.lifecycle_status == EntityLifecycleStatus.INACTIVE:
                continue
            if current_window - entity.last_seen_window > max_missing:
                continue
            if (
                not preserve_entities_across_scenes
                and current_scene_id
                and entity.current_scene_id not in {None, current_scene_id}
                and entity.visibility == VisibilityState.NOT_VISIBLE
            ):
                # Historical entities remain eligible when their previous scene
                # is recoverable; scene changes are not identity resets.
                continue
            result.append(entity)
        return sorted(result, key=lambda item: item.entity_id)

    @_registry_safe
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
    ) -> tuple[GlobalEntityState, list[StateEvent]]:
        entity = state.entities[entity_id]
        window = window_index if window_index is not None else entity.last_seen_window
        timestamps = (
            evidence_timestamps(observation.evidence_frames, sampled_frames)
            if sampled_frames is not None and observation.evidence_frames
            else []
        )
        was_missing = entity.lifecycle_status in {
            EntityLifecycleStatus.TEMPORARILY_MISSING,
            EntityLifecycleStatus.INACTIVE,
        }
        entity.last_seen_window = window
        if timestamps:
            earliest = min(timestamps)
            latest = max(timestamps)
            entity.first_seen_time = min(entity.first_seen_time, earliest)
            entity.last_seen_time = latest
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
        evidence: EvidenceReference | None = None
        if observation.evidence_frames:
            evidence = EvidenceReference(
                run_id=run_id,
                window_global_index=window,
                local_id=observation.local_id,
                sample_indices=sorted(set(observation.evidence_frames)),
                timestamps_seconds=timestamps,
            )
            entity.evidence.append(evidence)
        events: list[StateEvent] = []
        if was_missing:
            events.append(
                self._event(
                    state,
                    "entity_reactivated",
                    window,
                    entity_id,
                    reason="observed_after_missing",
                    evidence=evidence,
                )
            )
        return entity, events

    @_registry_safe
    def record_relations(
        self,
        state: GlobalState,
        relations: list[RelationObservation],
        local_to_global: dict[str, str],
        window_index: int,
    ) -> None:
        """Persist lightweight relation history for resolved entities.

        Only relations whose subject and object both resolve to known global
        entities are recorded.  This is intentionally a minimal, deterministic
        history rather than a full graph.
        """
        for relation in relations:
            subject_id = local_to_global.get(relation.subject_local_id)
            object_id = local_to_global.get(relation.object_local_id)
            if subject_id is None or object_id is None:
                continue
            subject = state.entities.get(subject_id)
            object_ = state.entities.get(object_id)
            if subject is None or object_ is None:
                continue
            subject.relation_history.append(
                RelationReference(
                    relation_type=relation.relation_type,
                    related_entity_id=object_id,
                    window_global_index=window_index,
                    confidence=relation.confidence,
                )
            )
            object_.relation_history.append(
                RelationReference(
                    relation_type=relation.relation_type,
                    related_entity_id=subject_id,
                    window_global_index=window_index,
                    confidence=relation.confidence,
                )
            )

    @_registry_safe
    def mark_not_observed(
        self,
        state: GlobalState,
        observed_entity_ids: set[str] | str,
        current_window: int,
        *,
        suppress_missing_count: bool = False,
    ) -> tuple[list[str], list[StateEvent]]:
        if isinstance(observed_entity_ids, str):
            observed_entity_ids = {observed_entity_ids}
        changed: list[str] = []
        events: list[StateEvent] = []
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
                if entity.lifecycle_status != EntityLifecycleStatus.TEMPORARILY_MISSING:
                    events.append(
                        self._event(
                            state,
                            "entity_temporarily_missing",
                            current_window,
                            entity_id,
                            reason=f"missing_window_count={entity.missing_window_count}",
                        )
                    )
                entity.lifecycle_status = EntityLifecycleStatus.TEMPORARILY_MISSING
            changed.append(entity_id)
        return changed, events

    @_registry_safe
    def merge_temporary_entity(
        self,
        state: GlobalState,
        temporary_entity_id: str,
        target_entity_id: str,
        *,
        window_index: int | None = None,
    ) -> tuple[EntityMergeResult, list[StateEvent]]:
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
        event = self._event(
            state,
            "entity_merged",
            window_index or target.last_seen_window,
            target_entity_id,
            reason=f"{temporary_entity_id} merged into {target_entity_id}",
            metadata={
                "temporary_entity_id": temporary_entity_id,
                "migrated_evidence_count": migrated_evidence,
                "migrated_spatial_count": migrated_spatial,
            },
        )
        return (
            EntityMergeResult(
                temporary_entity_id=temporary_entity_id,
                merged_into=target_entity_id,
                migrated_evidence_count=migrated_evidence,
                migrated_spatial_count=migrated_spatial,
            ),
            [event],
        )

    @_registry_safe
    def record_delayed_merge_support(
        self,
        state: GlobalState,
        temporary_entity_id: str,
        target_entity_id: str,
        window_index: int,
    ) -> int:
        temporary = state.entities[temporary_entity_id]
        if temporary.last_seen_window != window_index - 1:
            # Not consecutive; reset support.
            temporary.delayed_merge_support = {}
        temporary.delayed_merge_support[target_entity_id] = (
            temporary.delayed_merge_support.get(target_entity_id, 0) + 1
        )
        return temporary.delayed_merge_support[target_entity_id]

    @_registry_safe
    def check_delayed_merge(
        self,
        state: GlobalState,
        temporary_entity_id: str,
        target_entity_id: str,
        window_index: int,
    ) -> tuple[EntityMergeResult | None, list[StateEvent]]:
        if not self.config.allow_delayed_merge:
            return None, []
        support = self.record_delayed_merge_support(state, temporary_entity_id, target_entity_id, window_index)
        if support >= self.config.delayed_merge_support_windows:
            return self.merge_temporary_entity(
                state, temporary_entity_id, target_entity_id, window_index=window_index
            )
        return None, []
