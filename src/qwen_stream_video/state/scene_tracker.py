"""Scene continuity tracking based only on validated visual observations."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..config import SceneTrackerConfig
from ..domain import (
    EntityLifecycleStatus,
    GlobalState,
    ObservationBatch,
    SceneState,
    StateEvent,
    ViewType,
    VisibilityState,
)
from ..exceptions import SceneTrackingError


class SceneUpdateResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str
    scene_changed: bool
    previous_scene_id: str | None = None
    continuity: str
    events: list[StateEvent] = Field(default_factory=list)
    camera_change: bool = False
    visibility_changed_entity_ids: list[str] = Field(default_factory=list)
    preserve_entities_across_scenes: bool = True


class SceneTracker:
    """Maintain scene IDs and visibility context without matching entities."""

    def __init__(self, config: SceneTrackerConfig | None = None) -> None:
        self.config = config or SceneTrackerConfig()

    def _event(
        self,
        state: GlobalState,
        event_type: str,
        window_index: int,
        *,
        scene_id: str | None = None,
        reason: str = "",
        metadata: dict[str, object] | None = None,
    ) -> StateEvent:
        state.event_counter += 1
        return StateEvent(
            event_id=f"event_{state.event_counter:06d}",
            event_type=event_type,
            window_global_index=window_index,
            scene_id=scene_id,
            reason=reason,
            metadata=metadata or {},
        )

    def update(self, state: GlobalState, observation: ObservationBatch) -> SceneUpdateResult:
        try:
            return self._update(state, observation)
        except SceneTrackingError:
            raise
        except Exception as exc:
            raise SceneTrackingError(
                f"Scene tracking failed for window {observation.window.global_index}: {exc}"
            ) from exc

    def _update(self, state: GlobalState, observation: ObservationBatch) -> SceneUpdateResult:
        window_index = observation.window.global_index
        scene_observation = observation.scene
        previous_id = state.current_scene_id
        events: list[StateEvent] = []
        visibility_changed: list[str] = []
        changed = False

        if previous_id is None:
            state.scene_counter += 1
            scene_id = f"scene_{state.scene_counter:04d}"
            state.current_scene_id = scene_id
            state.scenes[scene_id] = SceneState(
                scene_id=scene_id,
                view_type=scene_observation.view_type,
                start_window=window_index,
                last_active_window=window_index,
                continuity="continuous",
            )
            events.append(
                self._event(
                    state,
                    "scene_started",
                    window_index,
                    scene_id=scene_id,
                    reason="first_observation_window",
                )
            )
            changed = True
        elif (
            self.config.enabled
            and scene_observation.camera_change
            and self.config.camera_change_starts_new_scene
        ):
            state.scene_counter += 1
            scene_id = f"scene_{state.scene_counter:04d}"
            state.current_scene_id = scene_id
            state.scenes[scene_id] = SceneState(
                scene_id=scene_id,
                view_type=scene_observation.view_type,
                start_window=window_index,
                last_active_window=window_index,
                continuity="camera_change",
            )
            events.append(
                self._event(
                    state,
                    "scene_changed",
                    window_index,
                    scene_id=scene_id,
                    reason="camera_change",
                    metadata={"previous_scene_id": previous_id},
                )
            )
            changed = True
        else:
            scene_id = previous_id
            scene = state.scenes[scene_id]
            scene.last_active_window = window_index
            scene.view_type = scene_observation.view_type
            if scene_observation.continuity_hint == "reframed":
                scene.continuity = "reframed"
                events.append(
                    self._event(
                        state,
                        "scene_reframed",
                        window_index,
                        scene_id=scene_id,
                        reason="continuity_hint_reframed",
                    )
                )
            else:
                scene.continuity = scene_observation.continuity_hint

        scene = state.scenes[scene_id]
        if scene_observation.view_type in {ViewType.CLOSEUP, ViewType.DETAIL}:
            new_visibility = (
                VisibilityState.PARTIAL
                if scene_observation.target_visibility.value == "partial"
                else VisibilityState.NOT_VISIBLE
            )
            for entity_id in sorted(scene.visible_entity_ids):
                entity = state.entities.get(entity_id)
                if entity is None or entity.lifecycle_status == EntityLifecycleStatus.MERGED:
                    continue
                entity.visibility = new_visibility
                visibility_changed.append(entity_id)
                events.append(
                    self._event(
                        state,
                        "scene_visibility_changed",
                        window_index,
                        scene_id=scene_id,
                        reason="closeup_or_detail_view",
                        metadata={"entity_id": entity_id, "visibility": new_visibility.value},
                    )
                )
            # Scene visible IDs remain historical context; resolver decides which
            # entities are visible in the current observation.
        if scene_observation.camera_change or scene_observation.continuity_hint == "reframed":
            events.append(
                self._event(
                    state,
                    "scene_visibility_changed",
                    window_index,
                    scene_id=scene_id,
                    reason="camera_or_framing_change",
                )
            )

        return SceneUpdateResult(
            scene_id=scene_id,
            scene_changed=changed,
            previous_scene_id=previous_id,
            continuity=state.scenes[scene_id].continuity,
            events=events,
            camera_change=scene_observation.camera_change,
            visibility_changed_entity_ids=sorted(visibility_changed),
            preserve_entities_across_scenes=self.config.preserve_entities_across_scenes,
        )
