"""Deterministic, explainable one-to-one local-to-global entity resolution."""

from __future__ import annotations

import re
from typing import ClassVar

from ..domain import (
    EntityObservation,
    EntityResolution,
    EntityResolutionBatch,
    EntityResolutionStatus,
    EvidenceReference,
    GlobalEntityState,
    GlobalState,
    MatchScoreBreakdown,
    ObservationBatch,
    RelationObservation,
    StateEvent,
)
from ..domain.enums import EntityType
from ..exceptions import EntityResolutionError
from ..video import SampledFrame, evidence_timestamps
from .entity_registry import EntityRegistry
from .scene_tracker import SceneUpdateResult


class EntityResolver:
    """Resolve observations without trusting model-provided IDs."""

    # Fixed base weights for all components except candidate_hint.
    BASE_WEIGHTS: ClassVar[dict[str, float]] = {
        "type_name": 0.30,
        "appearance": 0.25,
        "spatial": 0.15,
        "relation": 0.15,
        "recency": 0.10,
    }

    # Appearance updates below this threshold do not create stable signatures,
    # so conflicts at or above this threshold are considered hard evidence.
    APPEARANCE_CONFIDENCE_THRESHOLD: ClassVar[float] = 0.8

    def __init__(self, registry: EntityRegistry | None = None) -> None:
        self.registry = registry or EntityRegistry()

    def _weights(self) -> dict[str, float]:
        hint_weight = self.registry.config.candidate_hint_weight
        fixed_total = sum(self.BASE_WEIGHTS.values())
        scale = (1.0 - hint_weight) / fixed_total if fixed_total > 0 else 0.0
        weights = {key: value * scale for key, value in self.BASE_WEIGHTS.items()}
        weights["candidate_hint"] = hint_weight
        return weights

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return {token.lower() for token in re.findall(r"[\w]+", value) if token}

    def _name_conflict(self, observed: str, existing: str) -> bool:
        if observed in {"", "unknown"} or existing in {"", "unknown"}:
            return False
        observed_numbers = set(re.findall(r"\d+", observed))
        existing_numbers = set(re.findall(r"\d+", existing))
        return bool(observed_numbers and existing_numbers and observed_numbers != existing_numbers)

    def _appearance_score(
        self,
        observed: dict[str, object],
        existing: dict[str, str],
        confidence: float,
    ) -> float:
        if not observed and not existing:
            return 0.5 + 0.3 * confidence
        if not observed or not existing:
            return 0.6
        keys = sorted(set(observed) | set(existing))
        matches = sum(str(observed.get(key)) == str(existing.get(key)) for key in keys)
        return matches / len(keys)

    def _spatial_score(
        self,
        observation: EntityObservation,
        candidate: GlobalEntityState,
        scene_continuous: bool,
        window: int,
    ) -> float:
        if observation.spatial_region == "unknown":
            # No spatial evidence in the current observation is not evidence against identity.
            return 1.0
        if not candidate.spatial_history:
            # The candidate has no recorded spatial history yet; treat as neutral rather
            # than high-confidence continuity.
            return 0.5
        last_spatial = candidate.spatial_history[-1]
        if last_spatial.spatial_region == "unknown":
            return 1.0
        if last_spatial.spatial_region == observation.spatial_region:
            return 1.0
        # Different region is penalised; a hard reject is handled separately for
        # continuous scenes where the candidate was just observed elsewhere.
        if scene_continuous and candidate.last_seen_window == window - 1:
            return 0.0
        return 0.0

    def _relation_score(
        self,
        observation: EntityObservation,
        candidate: GlobalEntityState,
        relations: list[RelationObservation],
        local_to_global: dict[str, str],
    ) -> float:
        """Score relation consistency between current observation and history.

        Returns a neutral baseline when the local entity has no relations in
        this observation, because absence of relation evidence is not evidence
        against identity.  When relations are present, returns the fraction that
        match the candidate's recent history, bounded to [0, 1].
        """
        relevant = [
            relation
            for relation in relations
            if relation.subject_local_id == observation.local_id
            or relation.object_local_id == observation.local_id
        ]
        if not relevant:
            # Absence of relation evidence in this observation is neutral.
            return 0.5
        historical = {
            (ref.relation_type, ref.related_entity_id)
            for ref in candidate.relation_history[-self.registry.config.max_missing_windows :]
        }
        matches = 0
        for relation in relevant:
            partner_local = (
                relation.object_local_id
                if relation.subject_local_id == observation.local_id
                else relation.subject_local_id
            )
            partner_global = local_to_global.get(partner_local)
            if partner_global is None:
                continue
            if (relation.relation_type, partner_global) in historical:
                matches += 1
        return matches / len(relevant)

    def _hard_reject_reasons(
        self,
        observation: EntityObservation,
        candidate: GlobalEntityState,
        state: GlobalState,
        scene_continuous: bool,
        window: int,
    ) -> list[str]:
        """Return non-empty list of hard-rejection reasons, or [] if candidate is eligible."""
        reasons: list[str] = []
        if candidate.entity_type != observation.entity_type:
            reasons.append("entity_type_mismatch")
        if self._name_conflict(observation.name, candidate.canonical_name):
            reasons.append("distinct_numeric_name")
        if observation.candidate_global_id:
            hint = state.entities.get(observation.candidate_global_id)
            if hint is None:
                # A missing hint is recorded as a warning, not a hard rejection of
                # the candidate itself; the field is checked below for type conflict.
                pass
            elif hint.entity_type != observation.entity_type:
                reasons.append("candidate_hint_type_conflict")
            elif hint.entity_id != candidate.entity_id:
                # Candidate hint points to a different entity of the same type.
                reasons.append("candidate_hint_points_to_other_entity")

        # High-confidence stable appearance conflict.
        if observation.confidence >= self.APPEARANCE_CONFIDENCE_THRESHOLD:
            conflicting_keys = set(candidate.appearance_conflicts) & set(observation.appearance)
            if conflicting_keys:
                reasons.append(f"appearance_conflict_keys={sorted(conflicting_keys)}")

        # Impossible spatial jump in a continuous scene.
        if scene_continuous and candidate.spatial_history:
            last_spatial = candidate.spatial_history[-1]
            if (
                candidate.last_seen_window == window - 1
                and last_spatial.spatial_region != "unknown"
                and observation.spatial_region != "unknown"
                and last_spatial.spatial_region != observation.spatial_region
            ):
                reasons.append(
                    f"impossible_spatial_jump: {last_spatial.spatial_region} -> {observation.spatial_region}"
                )

        return reasons

    def _non_relation_score(
        self,
        observation: EntityObservation,
        candidate: GlobalEntityState,
        state: GlobalState,
        scene_continuous: bool,
        window: int,
    ) -> MatchScoreBreakdown:
        """Compute all score components except relation_score."""
        weights = self._weights()
        if observation.name == "unknown" or candidate.canonical_name == "unknown":
            name_score = 0.5
        elif observation.name == candidate.canonical_name:
            name_score = 1.0
        else:
            name_score = 0.0

        appearance_score = self._appearance_score(
            observation.appearance, candidate.appearance_signature, observation.confidence
        )

        spatial_score = self._spatial_score(observation, candidate, scene_continuous, window)

        current_window = state.last_committed_window if state.last_committed_window is not None else candidate.last_seen_window
        gap = max(0, current_window - candidate.last_seen_window)
        recency_score = max(0.0, 1.0 - gap / (self.registry.config.max_missing_windows + 1))

        hint_score = 1.0 if observation.candidate_global_id == candidate.entity_id else 0.0

        # relation_score placeholder: computed in a second pass after a preliminary
        # local->global mapping is available.
        relation_score = 0.5

        total = (
            weights["type_name"] * name_score
            + weights["appearance"] * appearance_score
            + weights["spatial"] * spatial_score
            + weights["relation"] * relation_score
            + weights["recency"] * recency_score
            + weights["candidate_hint"] * hint_score
        )
        return MatchScoreBreakdown(
            type_name_score=name_score,
            appearance_score=appearance_score,
            spatial_score=spatial_score,
            relation_score=relation_score,
            recency_score=recency_score,
            candidate_hint_score=hint_score,
            total_score=min(1.0, max(0.0, total)),
        )

    def _with_relation_score(
        self,
        observation: EntityObservation,
        candidate: GlobalEntityState,
        breakdown: MatchScoreBreakdown,
        relations: list[RelationObservation],
        local_to_global: dict[str, str],
    ) -> MatchScoreBreakdown:
        """Recompute total score with relation_score from the preliminary mapping."""
        weights = self._weights()
        relation_score = self._relation_score(observation, candidate, relations, local_to_global)
        total = (
            weights["type_name"] * breakdown.type_name_score
            + weights["appearance"] * breakdown.appearance_score
            + weights["spatial"] * breakdown.spatial_score
            + weights["relation"] * relation_score
            + weights["recency"] * breakdown.recency_score
            + weights["candidate_hint"] * breakdown.candidate_hint_score
        )
        return MatchScoreBreakdown(
            type_name_score=breakdown.type_name_score,
            appearance_score=breakdown.appearance_score,
            spatial_score=breakdown.spatial_score,
            relation_score=relation_score,
            recency_score=breakdown.recency_score,
            candidate_hint_score=breakdown.candidate_hint_score,
            total_score=min(1.0, max(0.0, total)),
        )

    def _evidence(
        self,
        state: GlobalState,
        window: int,
        local_id: str,
        frames: list[int],
        sampled: list[SampledFrame],
    ) -> EvidenceReference:
        return EvidenceReference(
            run_id=state.run_id,
            window_global_index=window,
            local_id=local_id,
            sample_indices=sorted(set(frames)),
            timestamps_seconds=evidence_timestamps(frames, sampled) if frames else [],
        )

    def _event(
        self,
        state: GlobalState,
        event_type: str,
        window: int,
        entity_id: str,
        reason: str,
        evidence: EvidenceReference | None = None,
        metadata: dict[str, object] | None = None,
    ) -> StateEvent:
        state.event_counter += 1
        return StateEvent(
            event_id=f"event_{state.event_counter:06d}",
            event_type=event_type,
            window_global_index=window,
            entity_id=entity_id,
            reason=reason,
            evidence=[evidence] if evidence is not None else [],
            metadata=metadata or {},
        )

    def _decide_status(
        self,
        selected_score: float | None,
        second_score: float | None,
    ) -> tuple[EntityResolutionStatus, list[str]]:
        config = self.registry.config
        rejected_reasons: list[str] = []
        if selected_score is None:
            return EntityResolutionStatus.CREATED, rejected_reasons
        close = second_score is not None and selected_score - second_score < config.ambiguous_margin
        if close:
            rejected_reasons.append(
                f"close_score_margin: {selected_score:.4f} - {second_score:.4f} < {config.ambiguous_margin}"
            )
            return EntityResolutionStatus.AMBIGUOUS, rejected_reasons
        if selected_score >= config.confident_match_threshold:
            return EntityResolutionStatus.MATCHED, rejected_reasons
        if selected_score >= config.ambiguous_match_threshold:
            rejected_reasons.append(
                f"below_confident_threshold: {selected_score:.4f} < {config.confident_match_threshold}"
            )
            return EntityResolutionStatus.AMBIGUOUS, rejected_reasons
        rejected_reasons.append(
            f"below_ambiguous_threshold: {selected_score:.4f} < {config.ambiguous_match_threshold}"
        )
        return EntityResolutionStatus.CREATED, rejected_reasons

    def _timestamp_from_evidence(
        self,
        observation: EntityObservation,
        sampled_frames: list[SampledFrame],
        window_start_seconds: float,
    ) -> float:
        if observation.evidence_frames and sampled_frames:
            try:
                return min(evidence_timestamps(observation.evidence_frames, sampled_frames))
            except ValueError:
                return window_start_seconds
        return window_start_seconds

    def _try_merge_recent_temporary(
        self,
        state: GlobalState,
        registry: EntityRegistry,
        window: int,
        formal_id: str,
        entity_type: EntityType,
    ) -> list[StateEvent]:
        """Merge a recent temporary entity into ``formal_id`` when support is enough.

        Delegates to the registry's controlled delayed-merge interface so the
        ``allow_delayed_merge`` configuration is respected consistently.
        """
        candidates = [
            entity
            for entity in state.entities.values()
            if entity.is_temporary
            and entity.entity_type == entity_type
            and window - entity.last_seen_window <= registry.config.max_missing_windows
            and entity.merged_into is None
        ]
        if not candidates:
            return []
        candidates.sort(
            key=lambda e: (-e.delayed_merge_support.get(formal_id, 0), e.entity_id)
        )
        temporary = candidates[0]
        merge_result, merge_events = registry.check_delayed_merge(
            state, temporary.entity_id, formal_id, window
        )
        # Ensure the temporary entity is still considered "recent" for bookkeeping,
        # but do not let a merged entity retain active support.
        if merge_result is None:
            temporary.last_seen_window = window
        return merge_events

    def resolve(
        self,
        state: GlobalState,
        registry: EntityRegistry,
        scene_result: SceneUpdateResult,
        observation: ObservationBatch,
        sampled_frames: list[SampledFrame],
    ) -> EntityResolutionBatch:
        try:
            return self._resolve(state, registry, scene_result, observation, sampled_frames)
        except EntityResolutionError:
            raise
        except Exception as exc:
            raise EntityResolutionError(
                f"Entity resolution failed for window {observation.window.global_index}: {exc}"
            ) from exc

    def _resolve(
        self,
        state: GlobalState,
        registry: EntityRegistry,
        scene_result: SceneUpdateResult,
        observation: ObservationBatch,
        sampled_frames: list[SampledFrame],
    ) -> EntityResolutionBatch:
        self.registry = registry
        window = observation.window.global_index
        mappings: list[EntityResolution] = []
        warnings: list[str] = []
        events: list[StateEvent] = []
        candidates_by_local: dict[str, list[tuple[GlobalEntityState, MatchScoreBreakdown]]] = {}
        rejected_by_local: dict[str, list[str]] = {}
        used_global_ids: set[str] = set()
        local_entities = sorted(observation.entities, key=lambda entity: entity.local_id)
        preserve_across_scenes = getattr(
            scene_result, "preserve_entities_across_scenes", True
        )

        # First pass: score all candidates (without relation_score) and collect
        # hard-rejection reasons.  find_candidates already filters by entity_type.
        for local in local_entities:
            candidates = registry.find_candidates(
                state,
                local.entity_type,
                scene_result.scene_id,
                window,
                preserve_entities_across_scenes=preserve_across_scenes,
            )
            scene_continuous = (
                not scene_result.camera_change
                and observation.scene.view_type not in {"closeup", "detail"}
                and scene_result.continuity != "camera_change"
            )
            scored: list[tuple[GlobalEntityState, MatchScoreBreakdown]] = []
            rejected: list[str] = []
            for candidate in candidates:
                reasons = self._hard_reject_reasons(local, candidate, state, scene_continuous, window)
                if reasons:
                    rejected.append(f"{candidate.entity_id}: {','.join(reasons)}")
                    continue
                scored.append((candidate, self._non_relation_score(local, candidate, state, scene_continuous, window)))

            # Document hard rejections for candidate_global_id targets that were
            # filtered out by entity type (e.g. a person observation pointing to
            # a device ID).  These entities are not candidates but the hint
            # conflict should still be traceable.
            if local.candidate_global_id:
                hint = state.entities.get(local.candidate_global_id)
                if hint is not None and hint.entity_type != local.entity_type:
                    reason = f"{local.candidate_global_id}: candidate_hint_type_conflict"
                    if reason not in rejected:
                        rejected.append(reason)
                elif hint is not None and any(hint.entity_id == c.entity_id for c in candidates):
                    pass  # Already evaluated by _hard_reject_reasons.
                elif hint is not None:
                    reason = f"{local.candidate_global_id}: candidate_hint_not_eligible"
                    if reason not in rejected:
                        rejected.append(reason)

            # Sort by total score descending, then by entity_id for determinism.
            scored.sort(key=lambda pair: (-pair[1].total_score, pair[0].entity_id))
            candidates_by_local[local.local_id] = scored
            rejected_by_local[local.local_id] = rejected

        # Build a preliminary local->global mapping using only non-relation scores
        # so that relation endpoints can be resolved for the real relation score.
        preliminary_pairs = [
            (score.total_score, local_id, candidate.entity_id, candidate, score)
            for local_id, scored in candidates_by_local.items()
            for candidate, score in scored
        ]
        preliminary_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
        preliminary_mapping: dict[str, str] = {}
        used_in_preliminary: set[str] = set()
        for _, local_id, global_id, candidate, _ in preliminary_pairs:
            if local_id in preliminary_mapping or global_id in used_in_preliminary:
                continue
            preliminary_mapping[local_id] = global_id
            used_in_preliminary.add(global_id)

        # Second pass: recompute scores with relation_score.
        final_candidates_by_local: dict[str, list[tuple[GlobalEntityState, MatchScoreBreakdown]]] = {}
        for local in local_entities:
            scored = [
                (candidate, self._with_relation_score(local, candidate, score, observation.relations, preliminary_mapping))
                for candidate, score in candidates_by_local[local.local_id]
            ]
            scored.sort(key=lambda pair: (-pair[1].total_score, pair[0].entity_id))
            final_candidates_by_local[local.local_id] = scored

        # Greedy one-to-one matching using the final scores.
        pairs = [
            (score.total_score, local_id, candidate.entity_id, candidate, score)
            for local_id, scored in final_candidates_by_local.items()
            for candidate, score in scored
        ]
        pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
        selected: dict[str, tuple[GlobalEntityState, MatchScoreBreakdown]] = {}
        for _, local_id, global_id, candidate, score in pairs:
            if local_id in selected or global_id in used_global_ids:
                continue
            selected[local_id] = (candidate, score)
            used_global_ids.add(global_id)

        for local in local_entities:
            scored = final_candidates_by_local[local.local_id]
            selected_pair = selected.get(local.local_id)
            selected_score = selected_pair[1].total_score if selected_pair else None
            second_score = scored[1][1].total_score if len(scored) > 1 else None
            best_formal_pair: tuple[GlobalEntityState, MatchScoreBreakdown] | None = None
            for alt, alt_score in scored:
                if alt.is_temporary:
                    continue
                if best_formal_pair is None or alt_score.total_score > best_formal_pair[1].total_score:
                    best_formal_pair = (alt, alt_score)
            best_formal_second_score = None
            if best_formal_pair is not None:
                best_formal_second_score = next(
                    (
                        s.total_score
                        for c, s in scored
                        if c.entity_id != best_formal_pair[0].entity_id and not c.is_temporary
                    ),
                    None,
                )

            if local.candidate_global_id and local.candidate_global_id not in state.entities:
                entity = registry.create_entity(
                    state,
                    local.entity_type,
                    name=local.name,
                    confidence=local.confidence,
                    window_index=window,
                    timestamp=self._timestamp_from_evidence(local, sampled_frames, observation.window.start_seconds),
                    scene_id=scene_result.scene_id,
                    temporary=False,
                )
                global_id = entity.entity_id
                status = EntityResolutionStatus.REJECTED_HINT
                rejected_reasons = [f"{local.candidate_global_id}: candidate_hint_not_in_state"]
            elif selected_pair is not None and selected_score is not None:
                candidate, score = selected_pair
                status, rejected_reasons = self._decide_status(selected_score, second_score)

                if candidate.is_temporary and candidate.last_seen_window == window - 1:
                    # Re-identified a recent temporary entity.  Keep it as the current global ID
                    # and attempt delayed merge against the best formal candidate.
                    global_id = candidate.entity_id
                    status = EntityResolutionStatus.AMBIGUOUS
                    if (
                        best_formal_pair is not None
                        and best_formal_pair[1].total_score >= registry.config.confident_match_threshold
                        and (
                            best_formal_second_score is None
                            or best_formal_pair[1].total_score - best_formal_second_score
                            >= registry.config.ambiguous_margin
                        )
                    ):
                        merge_result, merge_events = registry.check_delayed_merge(
                            state, candidate.entity_id, best_formal_pair[0].entity_id, window
                        )
                        if merge_result is not None:
                            global_id = best_formal_pair[0].entity_id
                            status = EntityResolutionStatus.MATCHED
                            events.extend(merge_events)
                elif status == EntityResolutionStatus.MATCHED:
                    global_id = candidate.entity_id
                    events.extend(
                        self._try_merge_recent_temporary(
                            state, registry, window, candidate.entity_id, local.entity_type
                        )
                    )
                elif status == EntityResolutionStatus.AMBIGUOUS:
                    temp = registry.create_entity(
                        state,
                        local.entity_type,
                        name=local.name,
                        confidence=local.confidence,
                        window_index=window,
                        timestamp=self._timestamp_from_evidence(local, sampled_frames, observation.window.start_seconds),
                        scene_id=scene_result.scene_id,
                        temporary=True,
                    )
                    global_id = temp.entity_id
                    if selected_pair is not None:
                        target_id = selected_pair[0].entity_id
                        if not selected_pair[0].is_temporary:
                            # Only seed support when the formal candidate is already confident.
                            if (
                                selected_pair[1].total_score >= registry.config.confident_match_threshold
                                and (
                                    best_formal_second_score is None
                                    or selected_pair[1].total_score - best_formal_second_score
                                    >= registry.config.ambiguous_margin
                                )
                            ):
                                temp.delayed_merge_support[target_id] = 1
                        elif selected_pair[0].last_seen_window == window - 1:
                            # Carry over support from the previous temporary entity.
                            for formal_id, count in selected_pair[0].delayed_merge_support.items():
                                temp.delayed_merge_support[formal_id] = count
                else:
                    entity = registry.create_entity(
                        state,
                        local.entity_type,
                        name=local.name,
                        confidence=local.confidence,
                        window_index=window,
                        timestamp=self._timestamp_from_evidence(local, sampled_frames, observation.window.start_seconds),
                        scene_id=scene_result.scene_id,
                        temporary=False,
                    )
                    global_id = entity.entity_id
            else:
                entity = registry.create_entity(
                    state,
                    local.entity_type,
                    name=local.name,
                    confidence=local.confidence,
                    window_index=window,
                    timestamp=self._timestamp_from_evidence(local, sampled_frames, observation.window.start_seconds),
                    scene_id=scene_result.scene_id,
                    temporary=False,
                )
                global_id = entity.entity_id
                status = EntityResolutionStatus.CREATED
                rejected_reasons = []

            rejected_reasons = rejected_by_local.get(local.local_id, []) + rejected_reasons

            if status == EntityResolutionStatus.AMBIGUOUS and not global_id.startswith(
                self.registry.config.temporary_entity_prefix
            ):
                # Ensure ambiguous resolutions always point to a temporary entity.
                temp = registry.create_entity(
                    state,
                    local.entity_type,
                    name=local.name,
                    confidence=local.confidence,
                    window_index=window,
                    timestamp=self._timestamp_from_evidence(local, sampled_frames, observation.window.start_seconds),
                    scene_id=scene_result.scene_id,
                    temporary=True,
                )
                global_id = temp.entity_id

            mapping = EntityResolution(
                window_global_index=window,
                local_id=local.local_id,
                global_entity_id=global_id,
                status=status,
                selected_score=selected_score,
                second_best_score=second_score,
                candidate_scores={candidate.entity_id: score for candidate, score in scored},
                rejected_reasons=rejected_reasons,
                evidence=[self._evidence(state, window, local.local_id, local.evidence_frames, sampled_frames)],
            )
            mappings.append(mapping)

            if status == EntityResolutionStatus.MATCHED:
                events.append(
                    self._event(
                        state,
                        "entity_matched",
                        window,
                        global_id,
                        reason=f"confident_match score={selected_score:.4f}",
                        evidence=mapping.evidence[0] if mapping.evidence else None,
                    )
                )
            elif status == EntityResolutionStatus.CREATED:
                events.append(
                    self._event(
                        state,
                        "entity_created",
                        window,
                        global_id,
                        reason=f"new_formal_entity score={selected_score}",
                        evidence=mapping.evidence[0] if mapping.evidence else None,
                    )
                )
            elif status == EntityResolutionStatus.AMBIGUOUS:
                events.append(
                    self._event(
                        state,
                        "entity_ambiguous",
                        window,
                        global_id,
                        reason=f"temporary_entity score={selected_score:.4f}",
                        evidence=mapping.evidence[0] if mapping.evidence else None,
                        metadata={"best_formal_candidate": selected_pair[0].entity_id if selected_pair else None},
                    )
                )
            elif status == EntityResolutionStatus.REJECTED_HINT:
                events.append(
                    self._event(
                        state,
                        "entity_created",
                        window,
                        global_id,
                        reason="rejected_candidate_hint_created_formal_entity",
                        evidence=mapping.evidence[0] if mapping.evidence else None,
                        metadata={"rejected_hint": local.candidate_global_id},
                    )
                )

            _, update_events = registry.update_from_observation(
                state,
                global_id,
                local,
                scene_id=scene_result.scene_id,
                run_id=state.run_id,
                sampled_frames=sampled_frames,
                window_index=window,
            )
            events.extend(update_events)

            scene = state.scenes[scene_result.scene_id]
            if global_id not in scene.visible_entity_ids:
                scene.visible_entity_ids.append(global_id)
                scene.visible_entity_ids.sort()

        # Persist lightweight relation history for resolved entities.
        local_to_global_final = {mapping.local_id: mapping.global_entity_id for mapping in mappings}
        registry.record_relations(state, observation.relations, local_to_global_final, window)

        return EntityResolutionBatch(
            window_global_index=window, mappings=mappings, warnings=warnings, events=events
        )
