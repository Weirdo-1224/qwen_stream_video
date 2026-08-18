"""Deterministic action identity and lifecycle tracking."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..config import ActionTrackerConfig
from ..domain import (
    ActionLifecycle,
    EntityResolutionBatch,
    EvidenceReference,
    GlobalActionState,
    GlobalState,
    ObservationBatch,
    StateEvent,
    TimeInterval,
)
from ..exceptions import ActionTrackingError
from ..video import (
    SampledFrame,
    VideoWindow,
    evidence_intersects_commit_interval,
    evidence_timestamps,
)
from .scene_tracker import SceneUpdateResult


class ActionUpdateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_global_index: int
    action_ids: list[str] = Field(default_factory=list)
    events: list[StateEvent] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ActionTracker:
    def __init__(self, config: ActionTrackerConfig | None = None) -> None:
        self.config = config or ActionTrackerConfig()

    @staticmethod
    def _mapping(resolutions: EntityResolutionBatch) -> dict[str, str]:
        return {mapping.local_id: mapping.global_entity_id for mapping in resolutions.mappings}

    def _event(
        self,
        state: GlobalState,
        event_type: str,
        window: int,
        action: GlobalActionState,
        evidence: EvidenceReference | None,
        reason: str = "",
    ) -> StateEvent:
        state.event_counter += 1
        return StateEvent(
            event_id=f"event_{state.event_counter:06d}",
            event_type=event_type,
            window_global_index=window,
            action_id=action.action_id,
            reason=reason,
            confidence=action.confidence,
            evidence=[evidence] if evidence else [],
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

    @staticmethod
    def _first_evidence_timestamp(
        evidence_frames: list[int], sampled_frames: list[SampledFrame]
    ) -> float | None:
        if not evidence_frames:
            return None
        timestamps = evidence_timestamps(evidence_frames, sampled_frames)
        return min(timestamps) if timestamps else None

    @staticmethod
    def _previous_sample_timestamp(
        evidence_frames: list[int], sampled_frames: list[SampledFrame], window_start: float
    ) -> float:
        if not evidence_frames or not sampled_frames:
            return window_start
        first_evidence_index = min(evidence_frames)
        candidates = [
            frame.timestamp_seconds
            for frame in sampled_frames
            if frame.sample_index < first_evidence_index
        ]
        return max(candidates) if candidates else window_start

    @staticmethod
    def _key(actor_id: str, action_type: str, target_id: str | None, tool_id: str | None) -> tuple[str, str, str | None, str | None]:
        return actor_id, action_type, target_id, tool_id

    def _find_existing(
        self,
        state: GlobalState,
        key: tuple[str, str, str | None, str | None],
        window: int,
        current_time: float,
        scene_continuous: bool,
    ) -> GlobalActionState | None:
        continuable_lifecycles = {
            ActionLifecycle.STARTED,
            ActionLifecycle.ONGOING,
            ActionLifecycle.UNCERTAIN,
            ActionLifecycle.POSSIBLE_ENDED,
        }
        matches = [
            action
            for action in state.actions.values()
            if self._key(action.actor_id, action.action_type, action.target_id, action.tool_id) == key
            and action.lifecycle in continuable_lifecycles
        ]
        matches.sort(key=lambda item: (-item.last_observed_window, item.action_id))
        for action in matches:
            window_gap = window - action.last_observed_window
            time_gap = current_time - action.last_observed_time
            if (
                scene_continuous
                and window_gap <= self.config.continue_max_gap_windows + 1
                and time_gap < self.config.repeat_action_min_gap_seconds
            ):
                return action
        return None

    def update(
        self,
        state: GlobalState,
        observation: ObservationBatch,
        resolutions: EntityResolutionBatch,
        sampled_frames: list[SampledFrame],
        scene_result: SceneUpdateResult,
        *,
        commit_only: bool = False,
    ) -> ActionUpdateResult:
        try:
            return self._update(state, observation, resolutions, sampled_frames, scene_result, commit_only=commit_only)
        except ActionTrackingError:
            raise
        except Exception as exc:
            raise ActionTrackingError(
                f"Action tracking failed for window {observation.window.global_index}: {exc}"
            ) from exc

    def _update(
        self,
        state: GlobalState,
        observation: ObservationBatch,
        resolutions: EntityResolutionBatch,
        sampled_frames: list[SampledFrame],
        scene_result: SceneUpdateResult,
        *,
        commit_only: bool = False,
    ) -> ActionUpdateResult:
        window = observation.window.global_index
        mapping = self._mapping(resolutions)
        result = ActionUpdateResult(window_global_index=window)
        scene_continuous = not scene_result.camera_change
        for local_action in sorted(observation.actions, key=lambda item: item.local_id):
            actor_id = mapping.get(local_action.actor_local_id or "")
            if actor_id is None:
                result.warnings.append(f"Action {local_action.local_id} has unresolved actor")
                continue
            target_id = mapping.get(local_action.target_local_id) if local_action.target_local_id else None
            tool_id = mapping.get(local_action.tool_local_id) if local_action.tool_local_id else None
            uncertain_reference = (
                (local_action.target_local_id is not None and target_id is None)
                or (local_action.tool_local_id is not None and tool_id is None)
            )
            evidence = self._evidence(
                state, window, local_action.local_id, local_action.evidence_frames, sampled_frames
            )
            in_commit = evidence_intersects_commit_interval(
                local_action.evidence_frames, sampled_frames, observation_to_window(observation)
            ) if local_action.evidence_frames else False
            current_time = max(evidence.timestamps_seconds) if evidence.timestamps_seconds else observation.window.end_seconds
            key = self._key(actor_id, local_action.action_type, target_id, tool_id)
            existing = self._find_existing(state, key, window, current_time, scene_continuous)
            if existing is not None:
                existing.last_observed_window = window
                existing.last_observed_time = current_time
                existing.missing_window_count = 0
                existing.lifecycle = (
                    ActionLifecycle.UNCERTAIN
                    if uncertain_reference or scene_result.camera_change
                    else ActionLifecycle.ONGOING
                )
                existing.confidence = max(existing.confidence, local_action.confidence)
                if window not in existing.observed_windows:
                    existing.observed_windows.append(window)
                existing.evidence.append(evidence)
                if existing.action_id not in state.active_action_ids:
                    state.active_action_ids.append(existing.action_id)
                result.action_ids.append(existing.action_id)
                reason = (
                    "unresolved_reference"
                    if existing.lifecycle == ActionLifecycle.UNCERTAIN and uncertain_reference
                    else "camera_change_or_missing_reference"
                    if existing.lifecycle == ActionLifecycle.UNCERTAIN
                    else "same_action_key"
                )
                result.events.append(
                    self._event(
                        state,
                        "action_uncertain" if existing.lifecycle == ActionLifecycle.UNCERTAIN else "action_continued",
                        window,
                        existing,
                        evidence,
                        reason,
                    )
                )
                continue
            if not in_commit or commit_only:
                result.warnings.append(
                    f"Context-only action {local_action.local_id} cannot create a new global action"
                )
                continue
            state.action_counter += 1
            action_id = f"action_{state.action_counter:06d}"
            first_timestamp = self._first_evidence_timestamp(
                local_action.evidence_frames, sampled_frames
            )
            start_lower = self._previous_sample_timestamp(
                local_action.evidence_frames, sampled_frames, observation.window.start_seconds
            )
            start_upper = first_timestamp if first_timestamp is not None else observation.window.end_seconds
            lifecycle = (
                ActionLifecycle.INSTANT
                if local_action.action_type in self.config.instant_actions
                else ActionLifecycle.UNCERTAIN if uncertain_reference else ActionLifecycle.STARTED
            )
            action = GlobalActionState(
                action_id=action_id,
                actor_id=actor_id,
                action_type=local_action.action_type,
                action_family=local_action.action_family,
                target_id=target_id,
                tool_id=tool_id,
                lifecycle=lifecycle,
                start_window=window,
                last_observed_window=window,
                last_observed_time=current_time,
                start_time_interval=TimeInterval(lower=start_lower, upper=start_upper),
                observed_windows=[window],
                confidence=local_action.confidence,
                evidence=[evidence],
            )
            state.actions[action_id] = action
            if lifecycle != ActionLifecycle.INSTANT:
                state.active_action_ids.append(action_id)
            result.action_ids.append(action_id)
            event_type = "action_instant" if lifecycle == ActionLifecycle.INSTANT else "action_uncertain" if lifecycle == ActionLifecycle.UNCERTAIN else "action_started"
            result.events.append(self._event(state, event_type, window, action, evidence, "commit_evidence"))
        result.action_ids = sorted(set(result.action_ids))
        state.active_action_ids = sorted(set(state.active_action_ids))
        return result

    def mark_missing(
        self,
        state: GlobalState,
        current_window: VideoWindow,
        *,
        camera_change: bool = False,
        observed_action_ids: set[str] | None = None,
    ) -> list[StateEvent]:
        try:
            return self._mark_missing(state, current_window, camera_change=camera_change, observed_action_ids=observed_action_ids)
        except ActionTrackingError:
            raise
        except Exception as exc:
            raise ActionTrackingError(
                f"Action missing-mark failed for window {current_window.global_index}: {exc}"
            ) from exc

    def _mark_missing(
        self,
        state: GlobalState,
        current_window: VideoWindow,
        *,
        camera_change: bool = False,
        observed_action_ids: set[str] | None = None,
    ) -> list[StateEvent]:
        events: list[StateEvent] = []
        observed_action_ids = observed_action_ids or set()
        active_ids = sorted(state.active_action_ids)
        new_active: list[str] = []
        for action_id in active_ids:
            if action_id in observed_action_ids:
                new_active.append(action_id)
                continue
            action = state.actions.get(action_id)
            if action is None or action.lifecycle in {ActionLifecycle.ENDED, ActionLifecycle.INSTANT}:
                continue
            last_evidence = action.evidence[-1] if action.evidence else EvidenceReference(
                run_id=state.run_id,
                window_global_index=current_window.global_index,
            )
            if camera_change:
                action.lifecycle = ActionLifecycle.UNCERTAIN
                state.event_counter += 1
                events.append(
                    StateEvent(
                        event_id=f"event_{state.event_counter:06d}",
                        event_type="action_uncertain",
                        window_global_index=current_window.global_index,
                        action_id=action_id,
                        reason="camera_change_or_occlusion",
                        evidence=[last_evidence],
                    )
                )
                new_active.append(action_id)
                continue
            action.missing_window_count += 1
            if action.missing_window_count < self.config.end_missing_windows:
                action.lifecycle = ActionLifecycle.POSSIBLE_ENDED
                event_type = "action_possible_ended"
                new_active.append(action_id)
            else:
                action.lifecycle = ActionLifecycle.ENDED
                action.end_window = current_window.global_index
                action.end_time_interval = TimeInterval(
                    lower=action.last_observed_time,
                    upper=current_window.end_seconds,
                )
                event_type = "action_ended"
            state.event_counter += 1
            events.append(
                StateEvent(
                    event_id=f"event_{state.event_counter:06d}",
                    event_type=event_type,
                    window_global_index=current_window.global_index,
                    action_id=action_id,
                    reason="missing_observation_window",
                    evidence=[last_evidence],
                )
            )
        state.active_action_ids = new_active
        return events


def observation_to_window(observation: ObservationBatch):
    """Create a minimal VideoWindow for evidence interval checks."""
    from ..video import VideoWindow

    return VideoWindow(
        global_index=observation.window.global_index,
        run_index=observation.window.global_index,
        start_seconds=observation.window.start_seconds,
        commit_start_seconds=observation.window.commit_start_seconds,
        end_seconds=observation.window.end_seconds,
    )
