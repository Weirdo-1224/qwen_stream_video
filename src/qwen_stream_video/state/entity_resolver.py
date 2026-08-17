"""Deterministic, explainable one-to-one local-to-global entity resolution."""

from __future__ import annotations

import re
from typing import ClassVar

from ..domain import (
    EntityResolution,
    EntityResolutionBatch,
    EntityResolutionStatus,
    EvidenceReference,
    GlobalEntityState,
    GlobalState,
    MatchScoreBreakdown,
    ObservationBatch,
)
from ..video import SampledFrame, evidence_timestamps
from .entity_registry import EntityRegistry
from .scene_tracker import SceneUpdateResult


class EntityResolver:
    """Resolve observations without trusting model-provided IDs."""

    WEIGHTS: ClassVar[dict[str, float]] = {
        "type_name": 0.30,
        "appearance": 0.25,
        "spatial": 0.15,
        "relation": 0.15,
        "recency": 0.10,
        "candidate_hint": 0.05,
    }

    def __init__(self, registry: EntityRegistry | None = None) -> None:
        self.registry = registry or EntityRegistry()

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return {token.lower() for token in re.findall(r"[\w]+", value) if token}

    def _name_conflict(self, observed: str, existing: str) -> bool:
        if observed in {"", "unknown"} or existing in {"", "unknown"}:
            return False
        observed_numbers = set(re.findall(r"\d+", observed))
        existing_numbers = set(re.findall(r"\d+", existing))
        return bool(observed_numbers and existing_numbers and observed_numbers != existing_numbers)

    def _appearance_score(self, observed: dict[str, object], existing: dict[str, str]) -> float:
        if not observed or not existing:
            return 0.8
        keys = set(observed) | set(existing)
        matches = sum(str(observed.get(key)) == str(existing.get(key)) for key in keys)
        return matches / len(keys)

    def _score(
        self,
        observation,
        candidate: GlobalEntityState,
        state: GlobalState,
        scene_result: SceneUpdateResult,
    ) -> MatchScoreBreakdown:
        name_score = 1.0 if observation.name == candidate.canonical_name else 0.5
        if observation.name == "unknown" or candidate.canonical_name == "unknown":
            name_score = 0.5
        appearance_score = self._appearance_score(
            observation.appearance, candidate.appearance_signature
        )
        if observation.spatial_region == "unknown" or not candidate.spatial_history:
            spatial_score = 0.8
        else:
            spatial_score = 1.0 if observation.spatial_region == candidate.spatial_history[-1].spatial_region else 0.25
        relation_score = 0.8
        current_window = state.last_committed_window if state.last_committed_window is not None else candidate.last_seen_window
        gap = max(0, current_window - candidate.last_seen_window)
        recency_score = max(0.0, 1.0 - gap / (self.registry.config.max_missing_windows + 1))
        hint_score = 1.0 if observation.candidate_global_id == candidate.entity_id else 0.0
        total = (
            self.WEIGHTS["type_name"] * name_score
            + self.WEIGHTS["appearance"] * appearance_score
            + self.WEIGHTS["spatial"] * spatial_score
            + self.WEIGHTS["relation"] * relation_score
            + self.WEIGHTS["recency"] * recency_score
            + self.WEIGHTS["candidate_hint"] * hint_score
        )
        # Keep an explicit bounded result even if future weight changes occur.
        return MatchScoreBreakdown(
            type_name_score=name_score,
            appearance_score=appearance_score,
            spatial_score=spatial_score,
            relation_score=relation_score,
            recency_score=recency_score,
            candidate_hint_score=hint_score,
            total_score=min(1.0, max(0.0, total)),
        )

    def _evidence(self, state: GlobalState, window: int, local_id: str, frames: list[int], sampled: list[SampledFrame]) -> EvidenceReference:
        return EvidenceReference(
            run_id=state.run_id,
            window_global_index=window,
            local_id=local_id,
            sample_indices=sorted(set(frames)),
            timestamps_seconds=evidence_timestamps(frames, sampled) if frames else [],
        )

    def resolve(
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
        candidates_by_local: dict[str, list[tuple[GlobalEntityState, MatchScoreBreakdown]]] = {}
        used_global_ids: set[str] = set()
        local_entities = sorted(observation.entities, key=lambda entity: entity.local_id)

        for local in local_entities:
            candidates = registry.find_candidates(
                state,
                local.entity_type,
                scene_result.scene_id,
                window,
            )
            scored: list[tuple[GlobalEntityState, MatchScoreBreakdown]] = []
            for candidate in candidates:
                reasons: list[str] = []
                if self._name_conflict(local.name, candidate.canonical_name):
                    reasons.append("distinct_numeric_name")
                if local.candidate_global_id:
                    hint = state.entities.get(local.candidate_global_id)
                    if hint is None:
                        warnings.append(
                            f"candidate_global_id {local.candidate_global_id} for {local.local_id} is not in state"
                        )
                    elif hint.entity_type != local.entity_type:
                        reasons.append("candidate_hint_type_conflict")
                if reasons:
                    continue
                scored.append((candidate, self._score(local, candidate, state, scene_result)))
            scored.sort(key=lambda pair: (-pair[1].total_score, pair[0].entity_id))
            candidates_by_local[local.local_id] = scored

        pairs = [
            (score.total_score, local_id, candidate.entity_id, candidate, score)
            for local_id, scored in candidates_by_local.items()
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
            scored = candidates_by_local[local.local_id]
            selected_pair = selected.get(local.local_id)
            selected_score = selected_pair[1].total_score if selected_pair else None
            second_score = scored[1][1].total_score if len(scored) > 1 else None
            status: EntityResolutionStatus
            rejected_reasons: list[str] = []
            if selected_pair and selected_score is not None:
                candidate, score = selected_pair
                close = second_score is not None and selected_score - second_score < registry.config.ambiguous_margin
                if close or selected_score < registry.config.confident_match_threshold:
                    temp = registry.create_entity(
                        state,
                        local.entity_type,
                        name=local.name,
                        confidence=local.confidence,
                        window_index=window,
                        scene_id=scene_result.scene_id,
                        temporary=True,
                    )
                    global_id = temp.entity_id
                    status = EntityResolutionStatus.AMBIGUOUS
                    rejected_reasons.append("close_score_margin" if close else "below_confident_threshold")
                else:
                    global_id = candidate.entity_id
                    status = EntityResolutionStatus.MATCHED
            else:
                temp = registry.create_entity(
                    state,
                    local.entity_type,
                    name=local.name,
                    confidence=local.confidence,
                    window_index=window,
                    scene_id=scene_result.scene_id,
                    temporary=False,
                )
                global_id = temp.entity_id
                status = EntityResolutionStatus.CREATED
                if local.candidate_global_id and local.candidate_global_id not in state.entities:
                    status = EntityResolutionStatus.REJECTED_HINT
                    rejected_reasons.append("candidate_hint_not_in_state")
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
            registry.update_from_observation(
                state,
                global_id,
                local,
                scene_id=scene_result.scene_id,
                run_id=state.run_id,
                sampled_frames=sampled_frames,
                window_index=window,
            )
            scene = state.scenes[scene_result.scene_id]
            if global_id not in scene.visible_entity_ids:
                scene.visible_entity_ids.append(global_id)
                scene.visible_entity_ids.sort()
        return EntityResolutionBatch(window_global_index=window, mappings=mappings, warnings=warnings)
