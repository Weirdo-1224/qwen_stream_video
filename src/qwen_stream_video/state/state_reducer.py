"""The single atomic coordinator allowed to commit GlobalState changes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..config import AppConfig
from ..domain import (
    EntityResolutionBatch,
    GlobalState,
    ObservationBatch,
    StateDelta,
    StateEvent,
)
from ..exceptions import StateEngineError
from ..video import SampledFrame, VideoWindow
from .action_tracker import ActionTracker, ActionUpdateResult
from .entity_registry import EntityRegistry
from .entity_resolver import EntityResolver
from .scene_tracker import SceneTracker, SceneUpdateResult
from .transition_engine import TransitionEngine


class StateReductionResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    state: GlobalState
    resolution: EntityResolutionBatch | None = None
    scene_result: SceneUpdateResult | None = None
    action_result: ActionUpdateResult | None = None
    transition_result: Any | None = None
    events: list[StateEvent] = Field(default_factory=list)
    delta: StateDelta | None = None
    warnings: list[str] = Field(default_factory=list)
    error: Exception | str | None = None


class StateReducer:
    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        scene_tracker: SceneTracker | None = None,
        registry: EntityRegistry | None = None,
        resolver: EntityResolver | None = None,
        action_tracker: ActionTracker | None = None,
        transition_engine: TransitionEngine | None = None,
    ) -> None:
        self.config = config or AppConfig()
        self.registry = registry or EntityRegistry(self.config.entity_registry)
        self.scene_tracker = scene_tracker or SceneTracker(self.config.scene_tracker)
        self.resolver = resolver or EntityResolver(self.registry)
        self.action_tracker = action_tracker or ActionTracker(self.config.action_tracker)
        self.transition_engine = transition_engine or TransitionEngine(
            self.config.transition_engine,
            require_evidence_frames=self.config.observation.require_evidence_frames,
        )

    def apply_observation(
        self,
        state: GlobalState,
        observation: ObservationBatch,
        sampled_frames: list[SampledFrame],
        window: VideoWindow,
    ) -> StateReductionResult:
        """Apply one window to a deep copy and commit only after all stages pass."""
        working = state.model_copy(deep=True)
        try:
            scene_result = self.scene_tracker.update(working, observation)
            resolution = self.resolver.resolve(
                working, self.registry, scene_result, observation, sampled_frames
            )
            action_result = self.action_tracker.update(
                working,
                observation,
                resolution,
                sampled_frames,
                scene_result,
            )
            transition_result = self.transition_engine.update(
                working,
                observation,
                resolution,
                action_result,
                sampled_frames,
                window,
            )
            observed_ids = {
                mapping.global_entity_id
                for mapping in resolution.mappings
                if mapping.global_entity_id in working.entities
            }
            visibility_updates, visibility_events = self.registry.mark_not_observed(
                working,
                observed_ids,
                window.global_index,
                suppress_missing_count=(
                    scene_result.camera_change
                    or observation.scene.view_type.value in {"closeup", "detail"}
                ),
            )
            missing_events = self.action_tracker.mark_missing(
                working,
                window,
                camera_change=scene_result.camera_change,
                observed_action_ids=set(action_result.action_ids),
            )
            events = list(scene_result.events)
            events.extend(resolution.events)
            events.extend(action_result.events)
            events.extend(transition_result.events)
            events.extend(visibility_events)
            events.extend(missing_events)
            warnings = list(resolution.warnings)
            warnings.extend(action_result.warnings)
            warnings.extend(transition_result.warnings)
            delta = StateDelta(
                window_global_index=window.global_index,
                scene_id=scene_result.scene_id,
                entity_updates=sorted(set(observed_ids | set(visibility_updates))),
                action_updates=sorted(set(action_result.action_ids)),
                attribute_updates=sorted(set(transition_result.attribute_keys)),
                emitted_event_ids=[event.event_id for event in events],
                warnings=warnings,
            )
            if window.processing_role == "commit":
                working.last_committed_window = window.global_index
            return StateReductionResult(
                state=working,
                resolution=resolution,
                scene_result=scene_result,
                action_result=action_result,
                transition_result=transition_result,
                events=events,
                delta=delta,
                warnings=warnings,
            )
        except StateEngineError as exc:
            event = StateEvent(
                event_id=f"state_error_{window.global_index:06d}",
                event_type="state_update_error",
                window_global_index=window.global_index,
                reason=f"{type(exc).__name__}: {exc}",
                metadata={"state_unchanged": True, "error_type": type(exc).__name__},
            )
            if self.config.state.fail_on_state_error:
                raise
            return StateReductionResult(
                state=state,
                events=[event],
                warnings=[str(exc)],
                error=exc,
            )
        except Exception as exc:
            wrapped = StateEngineError(
                f"State update failed for window {window.global_index}: {exc}"
            )
            event = StateEvent(
                event_id=f"state_error_{window.global_index:06d}",
                event_type="state_update_error",
                window_global_index=window.global_index,
                reason=f"{type(wrapped).__name__}: {wrapped}",
                metadata={
                    "state_unchanged": True,
                    "error_type": type(wrapped).__name__,
                    "original_error_type": type(exc).__name__,
                },
            )
            if self.config.state.fail_on_state_error:
                raise wrapped from exc
            return StateReductionResult(
                state=state,
                events=[event],
                warnings=[str(wrapped)],
                error=wrapped,
            )

    def apply_observation_gap(
        self,
        state: GlobalState,
        window: VideoWindow,
        *,
        reason: str = "observation_failed",
    ) -> StateReductionResult:
        """Record a gap without inferring disappearance or state transitions."""
        working = state.model_copy(deep=True)
        working.event_counter += 1
        event = StateEvent(
            event_id=f"event_{working.event_counter:06d}",
            event_type="observation_gap",
            window_global_index=window.global_index,
            scene_id=working.current_scene_id,
            reason=reason,
        )
        return StateReductionResult(state=working, events=[event], warnings=[reason])
