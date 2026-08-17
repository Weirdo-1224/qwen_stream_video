"""Deterministic confirmation of attribute states and transitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..config import TransitionEngineConfig
from ..domain import (
    AttributeConfirmationStatus,
    AttributeState,
    EntityResolutionBatch,
    EvidenceReference,
    GlobalState,
    ObservationBatch,
    StateEvent,
)
from ..video import (
    SampledFrame,
    VideoWindow,
    evidence_intersects_commit_interval,
    evidence_timestamps,
)
from .action_tracker import ActionUpdateResult

DEFAULT_ATTRIBUTES_PATH = Path(__file__).resolve().parents[3] / "vocabularies" / "attributes.yaml"


class TransitionUpdateResult:
    """Small result object kept intentionally independent of storage models."""

    def __init__(self) -> None:
        self.events: list[StateEvent] = []
        self.attribute_keys: list[str] = []
        self.warnings: list[str] = []


class TransitionEngine:
    def __init__(
        self,
        config: TransitionEngineConfig | None = None,
        attributes_path: str | Path | None = None,
    ) -> None:
        self.config = config or TransitionEngineConfig()
        path = Path(attributes_path or DEFAULT_ATTRIBUTES_PATH)
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        self.attributes: dict[str, dict[str, Any]] = data.get("attributes", {})

    def _event(
        self,
        state: GlobalState,
        event_type: str,
        window: int,
        *,
        entity_id: str,
        attribute_key: str,
        before: Any = None,
        after: Any = None,
        confidence: float | None = None,
        reason: str = "",
        evidence: list[EvidenceReference] | None = None,
    ) -> StateEvent:
        state.event_counter += 1
        return StateEvent(
            event_id=f"event_{state.event_counter:06d}",
            event_type=event_type,
            window_global_index=window,
            entity_id=entity_id,
            attribute_key=attribute_key,
            before=before,
            after=after,
            confidence=confidence,
            reason=reason,
            evidence=evidence or [],
        )

    def _evidence(
        self,
        state: GlobalState,
        window: int,
        local_id: str,
        frames: list[int],
        sampled_frames: list[SampledFrame],
    ) -> EvidenceReference:
        return EvidenceReference(
            run_id=state.run_id,
            window_global_index=window,
            local_id=local_id,
            sample_indices=sorted(set(frames)),
            timestamps_seconds=evidence_timestamps(frames, sampled_frames) if frames else [],
        )

    def _action_supports(
        self,
        state: GlobalState,
        action_result: ActionUpdateResult,
        entity_id: str,
        attribute_key: str,
        value: str,
    ) -> bool:
        metadata = self.attributes.get(attribute_key, {})
        supported = set(metadata.get("supporting_actions", {}).get(value, []))
        if not supported:
            return False
        for action_id in action_result.action_ids:
            action = state.actions.get(action_id)
            if action and action.action_type in supported and (
                action.target_id == entity_id or action.actor_id == entity_id
            ):
                return True
        return False

    def update(
        self,
        state: GlobalState,
        observation: ObservationBatch,
        resolutions: EntityResolutionBatch,
        action_result: ActionUpdateResult,
        sampled_frames: list[SampledFrame],
        window: VideoWindow,
    ) -> TransitionUpdateResult:
        result = TransitionUpdateResult()
        resolution_by_local = {mapping.local_id: mapping for mapping in resolutions.mappings}
        window_index = observation.window.global_index
        for item in sorted(
            observation.attribute_observations,
            key=lambda value: (value.entity_local_id, value.attribute_key or "", value.value),
        ):
            mapping = resolution_by_local.get(item.entity_local_id)
            if mapping is None:
                result.warnings.append(f"Attribute {item.attribute_key} has unresolved entity")
                continue
            if mapping.status.value in {"ambiguous", "temporary", "rejected_hint"}:
                result.warnings.append(f"Attribute {item.attribute_key} has ambiguous entity resolution")
                continue
            if item.normalization_status in {"out_of_vocabulary", "invalid_for_entity_type"}:
                result.warnings.append(
                    f"Attribute {item.attribute_key} excluded because it is {item.normalization_status}"
                )
                continue
            if not item.evidence_frames:
                result.warnings.append(f"Attribute {item.attribute_key} has no evidence")
                continue
            in_commit = evidence_intersects_commit_interval(item.evidence_frames, sampled_frames, window)
            if not in_commit:
                # Context evidence may support an existing pending value, but
                # cannot establish a new formal state or transition.
                entity = state.entities.get(mapping.global_entity_id)
                existing = entity.attributes.get(item.attribute_key or "") if entity else None
                if existing is None or existing.status != AttributeConfirmationStatus.PENDING:
                    result.warnings.append(
                        f"Context-only attribute {item.attribute_key} cannot create a formal state"
                    )
                    continue
            entity = state.entities.get(mapping.global_entity_id)
            key = item.attribute_key or item.attribute or ""
            if entity is None or not key:
                result.warnings.append(f"Attribute {key} has no target entity")
                continue
            evidence = self._evidence(
                state, window_index, item.entity_local_id, item.evidence_frames, sampled_frames
            )
            current = entity.attributes.get(key)
            if current is None:
                if item.confidence >= self.config.high_confidence_threshold:
                    current = AttributeState(
                        attribute_key=key,
                        value=item.value,
                        confidence=item.confidence,
                        status=AttributeConfirmationStatus.CONFIRMED,
                        first_observed_window=window_index,
                        last_observed_window=window_index,
                        confirmed_window=window_index,
                        supporting_observations=1,
                        evidence=[evidence],
                    )
                    entity.attributes[key] = current
                    result.events.append(
                        self._event(
                            state,
                            "attribute_initialized",
                            window_index,
                            entity_id=entity.entity_id,
                            attribute_key=key,
                            after=item.value,
                            confidence=item.confidence,
                            reason="initial_high_confidence_observation",
                            evidence=[evidence],
                        )
                    )
                elif item.confidence >= self.config.medium_confidence_threshold:
                    current = AttributeState(
                        attribute_key=key,
                        value=item.value,
                        confidence=item.confidence,
                        status=AttributeConfirmationStatus.PENDING,
                        first_observed_window=window_index,
                        last_observed_window=window_index,
                        pending_value=item.value,
                        pending_confidence=item.confidence,
                        pending_support_windows=[window_index],
                        evidence=[evidence],
                    )
                    entity.attributes[key] = current
                    result.events.append(
                        self._event(
                            state,
                            "attribute_pending",
                            window_index,
                            entity_id=entity.entity_id,
                            attribute_key=key,
                            after=item.value,
                            confidence=item.confidence,
                            reason="medium_confidence_initial_observation",
                            evidence=[evidence],
                        )
                    )
                else:
                    result.warnings.append(f"Low-confidence attribute {key} recorded as observed only")
                if current is not None:
                    result.attribute_keys.append(f"{entity.entity_id}:{key}")
                continue

            current.last_observed_window = window_index
            current.evidence.append(evidence)
            current.confidence = max(current.confidence, item.confidence)
            if item.value == current.value and current.status == AttributeConfirmationStatus.CONFIRMED:
                current.supporting_observations += 1
                current.pending_value = None
                current.pending_support_windows = []
                result.attribute_keys.append(f"{entity.entity_id}:{key}")
                continue

            if current.status == AttributeConfirmationStatus.CONFLICTED and item.value != current.value:
                result.warnings.append(f"Conflicting value for {key} remains conflicted")
                continue

            support_action = self._action_supports(
                state, action_result, entity.entity_id, key, item.value
            )
            can_single_high = self.attributes.get(key, {}).get("confirmation_policy") == "single_high"
            can_confirm = (
                in_commit
                and item.confidence >= self.config.high_confidence_threshold
                and (support_action or not self.config.require_action_support_for_transition or can_single_high)
            )
            if current.status == AttributeConfirmationStatus.CONFIRMED and item.value != current.value and can_confirm:
                before = current.value
                current.previous_value = before
                current.value = item.value
                current.status = AttributeConfirmationStatus.CONFIRMED
                current.confirmed_window = window_index
                current.pending_value = None
                current.pending_support_windows = []
                current.supporting_observations += 1
                result.events.append(
                    self._event(
                        state,
                        "attribute_transition",
                        window_index,
                        entity_id=entity.entity_id,
                        attribute_key=key,
                        before=before,
                        after=item.value,
                        confidence=item.confidence,
                        reason="high_confidence_with_action_support" if support_action else "single_high_policy",
                        evidence=[evidence],
                    )
                )
            elif current.status == AttributeConfirmationStatus.CONFIRMED and item.value != current.value:
                if current.pending_value == item.value:
                    current.pending_support_windows.append(window_index)
                else:
                    current.pending_value = item.value
                    current.pending_confidence = item.confidence
                    current.pending_support_windows = [window_index]
                if len(current.pending_support_windows) >= self.config.confirm_support_windows and in_commit:
                    before = current.value
                    current.previous_value = before
                    current.value = item.value
                    current.status = AttributeConfirmationStatus.CONFIRMED
                    current.confirmed_window = window_index
                    current.pending_value = None
                    current.pending_support_windows = []
                    result.events.append(
                        self._event(
                            state,
                            "attribute_transition",
                            window_index,
                            entity_id=entity.entity_id,
                            attribute_key=key,
                            before=before,
                            after=item.value,
                            confidence=item.confidence,
                            reason="consecutive_window_support",
                            evidence=[evidence],
                        )
                    )
                else:
                    result.events.append(
                        self._event(
                            state,
                            "attribute_pending",
                            window_index,
                            entity_id=entity.entity_id,
                            attribute_key=key,
                            before=current.value,
                            after=item.value,
                            confidence=item.confidence,
                            reason="awaiting_confirmation_support",
                            evidence=[evidence],
                        )
                    )
            elif current.status == AttributeConfirmationStatus.PENDING:
                if item.value == current.pending_value:
                    current.pending_support_windows.append(window_index)
                    if len(current.pending_support_windows) >= self.config.confirm_support_windows and in_commit:
                        current.value = item.value
                        current.status = AttributeConfirmationStatus.CONFIRMED
                        current.confirmed_window = window_index
                        current.pending_value = None
                        current.pending_support_windows = []
                        result.events.append(
                            self._event(
                                state,
                                "attribute_initialized",
                                window_index,
                                entity_id=entity.entity_id,
                                attribute_key=key,
                                after=item.value,
                                confidence=item.confidence,
                                reason="pending_value_confirmed",
                                evidence=[evidence],
                            )
                        )
                elif item.value == current.value:
                    current.pending_value = None
                    current.pending_support_windows = []
                    current.status = AttributeConfirmationStatus.CONFIRMED
                else:
                    current.status = AttributeConfirmationStatus.CONFLICTED
                    current.contradicting_observations += 1
                    result.events.append(
                        self._event(
                            state,
                            "attribute_conflict",
                            window_index,
                            entity_id=entity.entity_id,
                            attribute_key=key,
                            before=current.pending_value,
                            after=item.value,
                            confidence=item.confidence,
                            reason="conflicting_pending_value",
                            evidence=[evidence],
                        )
                    )
            result.attribute_keys.append(f"{entity.entity_id}:{key}")
        state.pending_attribute_keys = sorted(
            f"{entity.entity_id}:{key}"
            for entity in state.entities.values()
            for key, value in entity.attributes.items()
            if value.status == AttributeConfirmationStatus.PENDING
        )
        return result
