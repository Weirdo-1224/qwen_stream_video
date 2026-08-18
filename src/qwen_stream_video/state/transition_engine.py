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
    GlobalEntityState,
    GlobalState,
    ObservationBatch,
    StateEvent,
)
from ..exceptions import TransitionError
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
        *,
        require_evidence_frames: bool = True,
    ) -> None:
        self.config = config or TransitionEngineConfig()
        self.require_evidence_frames = require_evidence_frames
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
                action.target_id == entity_id
                or action.actor_id == entity_id
                or action.tool_id == entity_id
            ):
                return True
        return False

    def _support_evidence(
        self,
        current: AttributeState,
        evidence: EvidenceReference,
        support_windows: list[int],
    ) -> list[EvidenceReference]:
        """Collect EvidenceReference objects for the current and supporting windows.

        Keeps the original chronological order and deduplicates by object
        identity so the current window is not duplicated.
        """
        if not support_windows:
            return [evidence]
        support_set = set(support_windows)
        refs = [ref for ref in current.evidence if ref.window_global_index in support_set]
        if evidence not in refs:
            refs.append(evidence)
        return refs

    def _is_reactivated(
        self,
        entity: GlobalEntityState,
        current: AttributeState,
        window_index: int,
    ) -> bool:
        """Detect an attribute observed again after an absence gap.

        This avoids generating a false ``unknown -> value`` transition when an
        entity or attribute simply was not seen in the previous window.
        """
        return (
            entity.first_seen_window < window_index
            and current.last_observed_window < window_index - 1
        )

    def _add_support_window(
        self,
        current: AttributeState,
        window_index: int,
        camera_change: bool,
    ) -> bool:
        """Append ``window_index`` to the pending support chain if it is valid.

        Support windows must be unique and consecutive within the same scene.
        Non-consecutive or cross-scene observations restart the chain so that
        they are not counted as consecutive support.
        """
        if window_index in current.pending_support_windows:
            return True
        if camera_change or not current.pending_support_windows:
            current.pending_support_windows = [window_index]
            return True
        last = current.pending_support_windows[-1]
        if window_index == last + 1:
            current.pending_support_windows.append(window_index)
            return True
        # Non-consecutive observation: restart the support chain.
        current.pending_support_windows = [window_index]
        return True

    def _expire_pending(
        self,
        state: GlobalState,
        entity_id: str,
        attribute_key: str,
        current: AttributeState,
        window_index: int,
        result: TransitionUpdateResult,
        evidence: EvidenceReference | None = None,
    ) -> bool:
        """Expire stale pending support when too many windows passed without support.

        Returns ``True`` when the attribute was actually expired.  After expiration
        the pending fields are cleared and the status is restored to the most
        reasonable non-pending state: ``CONFIRMED`` if the attribute had already
        been confirmed, otherwise ``OBSERVED``.
        """
        if not current.pending_support_windows or not current.pending_value:
            return False
        last_support = current.pending_support_windows[-1]
        if window_index - last_support <= self.config.max_pending_gap_windows:
            return False
        pending_value = current.pending_value
        current.pending_value = None
        current.pending_confidence = None
        current.pending_support_windows = []
        if current.status == AttributeConfirmationStatus.PENDING:
            current.status = (
                AttributeConfirmationStatus.CONFIRMED
                if current.confirmed_window is not None
                else AttributeConfirmationStatus.OBSERVED
            )
        result.events.append(
            self._event(
                state,
                "attribute_pending_expired",
                window_index,
                entity_id=entity_id,
                attribute_key=attribute_key,
                before=pending_value,
                after=current.value,
                reason="pending_support_gap_exceeded",
                evidence=[evidence] if evidence is not None else [],
            )
        )
        return True

    def _expire_pending_if_needed(
        self,
        state: GlobalState,
        entity_id: str,
        attribute_key: str,
        current: AttributeState,
        window_index: int,
        evidence: EvidenceReference,
        result: TransitionUpdateResult,
    ) -> bool:
        """Convenience wrapper for expiration during normal attribute processing."""
        return self._expire_pending(
            state, entity_id, attribute_key, current, window_index, result, evidence
        )

    def update(
        self,
        state: GlobalState,
        observation: ObservationBatch,
        resolutions: EntityResolutionBatch,
        action_result: ActionUpdateResult,
        sampled_frames: list[SampledFrame],
        window: VideoWindow,
    ) -> TransitionUpdateResult:
        try:
            return self._update(state, observation, resolutions, action_result, sampled_frames, window)
        except TransitionError:
            raise
        except Exception as exc:
            raise TransitionError(
                f"Transition engine failed for window {window.global_index}: {exc}"
            ) from exc

    def _update(
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
        camera_change = bool(observation.scene.camera_change)

        # Determine which (entity, attribute) pairs are explicitly observed this
        # window so that gaps are not incorrectly treated as support.
        observed_attribute_keys: set[str] = set()
        for item in observation.attribute_observations:
            mapping = resolution_by_local.get(item.entity_local_id)
            if mapping is None:
                continue
            if mapping.status.value in {"ambiguous", "temporary", "rejected_hint"}:
                continue
            if item.normalization_status in {"out_of_vocabulary", "invalid_for_entity_type"}:
                continue
            if not item.evidence_frames and self.require_evidence_frames:
                continue
            observed_attribute_keys.add(f"{mapping.global_entity_id}:{item.attribute_key or item.attribute or ''}")

        # Every successful state window must check all pending attributes.  Those
        # without a continuing observation in this window expire once the gap exceeds
        # the configured threshold.  A pending candidate may exist while the status
        # is still ``CONFIRMED`` (a transition candidate) or ``PENDING`` (an
        # initial candidate).
        for entity in sorted(state.entities.values(), key=lambda e: e.entity_id):
            for key, current in sorted(entity.attributes.items()):
                if not current.pending_value:
                    continue
                attribute_key = f"{entity.entity_id}:{key}"
                if attribute_key in observed_attribute_keys:
                    continue
                self._expire_pending(
                    state, entity.entity_id, key, current, window_index, result
                )

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
                if self.require_evidence_frames:
                    result.warnings.append(f"Attribute {item.attribute_key} has no evidence")
                    continue
                in_commit = True
            else:
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
            low_confidence = item.confidence < self.config.medium_confidence_threshold

            if current is None:
                if low_confidence:
                    current = AttributeState(
                        attribute_key=key,
                        value=item.value,
                        confidence=item.confidence,
                        status=AttributeConfirmationStatus.OBSERVED,
                        first_observed_window=window_index,
                        last_observed_window=window_index,
                        evidence=[evidence],
                    )
                    entity.attributes[key] = current
                    result.events.append(
                        self._event(
                            state,
                            "attribute_observed",
                            window_index,
                            entity_id=entity.entity_id,
                            attribute_key=key,
                            after=item.value,
                            confidence=item.confidence,
                            reason="low_confidence_observation",
                            evidence=[evidence],
                        )
                    )
                elif item.confidence >= self.config.high_confidence_threshold:
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
                else:
                    # Medium confidence initial observation.
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
                result.attribute_keys.append(f"{entity.entity_id}:{key}")
                continue

            # Existing attribute state: refresh basic bookkeeping.
            current.last_observed_window = window_index
            current.evidence.append(evidence)
            current.confidence = max(current.confidence, item.confidence)

            if low_confidence:
                result.events.append(
                    self._event(
                        state,
                        "attribute_observed",
                        window_index,
                        entity_id=entity.entity_id,
                        attribute_key=key,
                        after=item.value,
                        confidence=item.confidence,
                        reason="low_confidence_observation",
                        evidence=[evidence],
                    )
                )
                result.attribute_keys.append(f"{entity.entity_id}:{key}")
                continue

            # Clear stale pending support before deciding on this observation.
            self._expire_pending_if_needed(
                state, entity.entity_id, key, current, window_index, evidence, result
            )

            # An attribute reappearing after a gap with the same confirmed value
            # is just a visibility recovery, not a formal state transition.
            reactivated = self._is_reactivated(entity, current, window_index)
            if (
                reactivated
                and item.value == current.value
                and current.status == AttributeConfirmationStatus.CONFIRMED
            ):
                current.supporting_observations += 1
                current.pending_value = None
                current.pending_support_windows = []
                result.attribute_keys.append(f"{entity.entity_id}:{key}")
                continue

            if item.value == current.value and current.status == AttributeConfirmationStatus.CONFIRMED:
                current.supporting_observations += 1
                current.pending_value = None
                current.pending_support_windows = []
                result.attribute_keys.append(f"{entity.entity_id}:{key}")
                continue

            if current.status == AttributeConfirmationStatus.CONFLICTED and item.value != current.value:
                result.warnings.append(f"Conflicting value for {key} remains conflicted")
                result.attribute_keys.append(f"{entity.entity_id}:{key}")
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

            if current.status == AttributeConfirmationStatus.CONFIRMED and item.value != current.value:
                if can_confirm:
                    before = current.value
                    current.previous_value = before
                    current.value = item.value
                    current.status = AttributeConfirmationStatus.CONFIRMED
                    current.confirmed_window = window_index
                    current.pending_value = None
                    current.pending_support_windows = []
                    current.supporting_observations = max(1, current.supporting_observations + 1)
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
                elif current.pending_value is not None:
                    if item.value == current.pending_value:
                        self._add_support_window(current, window_index, camera_change)
                        if (
                            len(current.pending_support_windows) >= self.config.confirm_support_windows
                            and in_commit
                        ):
                            before = current.value
                            support_windows = list(current.pending_support_windows)
                            current.previous_value = before
                            current.value = item.value
                            current.status = AttributeConfirmationStatus.CONFIRMED
                            current.confirmed_window = window_index
                            current.pending_value = None
                            current.pending_support_windows = []
                            current.supporting_observations = max(1, current.supporting_observations + 1)
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
                                    evidence=self._support_evidence(current, evidence, support_windows),
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
                    elif item.value == current.value:
                        # Observation of the confirmed value cancels the pending candidate.
                        current.pending_value = None
                        current.pending_confidence = None
                        current.pending_support_windows = []
                        current.supporting_observations += 1
                    else:
                        # Third conflicting value: keep the confirmed value and the
                        # pending candidate, but move the attribute to CONFLICTED.
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
                else:
                    # First diverging observation: start a pending transition.
                    current.pending_value = item.value
                    current.pending_confidence = item.confidence
                    current.pending_support_windows = [window_index]
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
                    self._add_support_window(current, window_index, camera_change)
                    if (
                        len(current.pending_support_windows) >= self.config.confirm_support_windows
                        and in_commit
                    ):
                        support_windows = list(current.pending_support_windows)
                        current.status = AttributeConfirmationStatus.CONFIRMED
                        current.confirmed_window = window_index
                        current.pending_value = None
                        current.pending_confidence = None
                        current.pending_support_windows = []
                        current.supporting_observations = max(1, current.supporting_observations + 1)
                        result.events.append(
                            self._event(
                                state,
                                "attribute_confirmed",
                                window_index,
                                entity_id=entity.entity_id,
                                attribute_key=key,
                                after=item.value,
                                confidence=item.confidence,
                                reason="pending_value_confirmed",
                                evidence=self._support_evidence(current, evidence, support_windows),
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
                                after=item.value,
                                confidence=item.confidence,
                                reason="awaiting_confirmation_support",
                                evidence=[evidence],
                            )
                        )
                elif item.value == current.value:
                    # Same value as the pending candidate is effectively the same candidate.
                    current.pending_value = None
                    current.pending_confidence = None
                    current.pending_support_windows = []
                    current.status = AttributeConfirmationStatus.CONFIRMED
                else:
                    # New value while already pending -> conflict.
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

            elif current.status == AttributeConfirmationStatus.OBSERVED:
                if item.value == current.value:
                    if item.confidence >= self.config.high_confidence_threshold:
                        current.status = AttributeConfirmationStatus.CONFIRMED
                        current.confirmed_window = window_index
                        current.supporting_observations = max(1, current.supporting_observations + 1)
                        result.events.append(
                            self._event(
                                state,
                                "attribute_initialized",
                                window_index,
                                entity_id=entity.entity_id,
                                attribute_key=key,
                                after=item.value,
                                confidence=item.confidence,
                                reason="observed_value_confirmed",
                                evidence=[evidence],
                            )
                        )
                    elif item.confidence >= self.config.medium_confidence_threshold:
                        current.status = AttributeConfirmationStatus.PENDING
                        current.pending_value = item.value
                        current.pending_confidence = item.confidence
                        current.pending_support_windows = [window_index]
                        result.events.append(
                            self._event(
                                state,
                                "attribute_pending",
                                window_index,
                                entity_id=entity.entity_id,
                                attribute_key=key,
                                after=item.value,
                                confidence=item.confidence,
                                reason="observed_value_pending",
                                evidence=[evidence],
                            )
                        )
                else:
                    if item.confidence >= self.config.high_confidence_threshold:
                        before = current.value
                        current.previous_value = before
                        current.value = item.value
                        current.status = AttributeConfirmationStatus.CONFIRMED
                        current.confirmed_window = window_index
                        current.pending_value = None
                        current.pending_confidence = None
                        current.pending_support_windows = []
                        current.supporting_observations = max(1, current.supporting_observations + 1)
                        result.events.append(
                            self._event(
                                state,
                                "attribute_initialized",
                                window_index,
                                entity_id=entity.entity_id,
                                attribute_key=key,
                                after=item.value,
                                confidence=item.confidence,
                                reason="high_confidence_replaces_observed",
                                evidence=[evidence],
                            )
                        )
                    elif item.confidence >= self.config.medium_confidence_threshold:
                        current.value = item.value
                        current.status = AttributeConfirmationStatus.PENDING
                        current.pending_value = item.value
                        current.pending_confidence = item.confidence
                        current.pending_support_windows = [window_index]
                        result.events.append(
                            self._event(
                                state,
                                "attribute_pending",
                                window_index,
                                entity_id=entity.entity_id,
                                attribute_key=key,
                                after=item.value,
                                confidence=item.confidence,
                                reason="observed_value_pending",
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
