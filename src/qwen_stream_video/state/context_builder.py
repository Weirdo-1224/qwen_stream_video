"""Bounded structured context sent to the local observation model."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..config import ContextConfig
from ..domain import (
    ActionLifecycle,
    AttributeConfirmationStatus,
    EntityLifecycleStatus,
    GlobalState,
    ObservationBatch,
)
from ..video import VideoWindow


class ObservationContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene: dict[str, Any] = Field(default_factory=dict)
    candidate_entities: list[dict[str, Any]] = Field(default_factory=list)
    active_actions: list[dict[str, Any]] = Field(default_factory=list)
    pending_attributes: list[dict[str, Any]] = Field(default_factory=list)
    recent_scene_changes: list[dict[str, Any]] = Field(default_factory=list)
    truncated: bool = False

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class ContextBuilder:
    def __init__(self, config: ContextConfig | None = None) -> None:
        self.config = config or ContextConfig()

    def _base(self, state: GlobalState, current_window: VideoWindow) -> ObservationContext:
        current_scene = state.scenes.get(state.current_scene_id or "")
        current_index = current_window.global_index
        scene = {
            "scene_id": current_scene.scene_id if current_scene else None,
            "view_type": current_scene.view_type.value if current_scene else "unknown",
            "continuity": current_scene.continuity if current_scene else "unknown",
        }
        recent_scene_changes: list[dict[str, Any]] = []
        for candidate_scene in sorted(
            state.scenes.values(),
            key=lambda s: (-s.start_window, s.scene_id),
        ):
            if candidate_scene.start_window > current_index:
                continue
            if candidate_scene.continuity in {"camera_change", "reframed"} and (
                current_index - candidate_scene.start_window <= self.config.recent_window_count
            ):
                recent_scene_changes.append(
                    {
                        "scene_id": candidate_scene.scene_id,
                        "continuity": candidate_scene.continuity,
                        "start_window": candidate_scene.start_window,
                        "view_type": candidate_scene.view_type.value,
                    }
                )
                break
        related_entity_ids = {
            reference
            for action in state.actions.values()
            if action.lifecycle
            in {
                ActionLifecycle.STARTED,
                ActionLifecycle.ONGOING,
                ActionLifecycle.UNCERTAIN,
                ActionLifecycle.POSSIBLE_ENDED,
            }
            for reference in (action.actor_id, action.target_id, action.tool_id)
            if reference
        }
        entities = []
        for entity in sorted(
            state.entities.values(),
            key=lambda item: (
                -int(item.entity_id in related_entity_ids),
                -item.last_seen_window,
                -int(item.lifecycle_status == EntityLifecycleStatus.ACTIVE),
                item.entity_id,
            ),
        ):
            if entity.lifecycle_status in {EntityLifecycleStatus.INACTIVE, EntityLifecycleStatus.MERGED}:
                continue
            if (
                current_index - entity.last_seen_window > self.config.recent_window_count
                and entity.entity_id not in related_entity_ids
            ):
                continue
            entities.append(
                {
                    "entity_id": entity.entity_id,
                    "entity_type": entity.entity_type.value,
                    "canonical_name": entity.canonical_name,
                    "appearance": dict(sorted(entity.appearance_signature.items())),
                    "last_seen_windows_ago": max(0, current_index - entity.last_seen_window),
                    "visibility": entity.visibility.value,
                }
            )
        entities = entities[: self.config.max_entities]

        actions = []
        for action in sorted(
            state.actions.values(),
            key=lambda item: (
                int(item.lifecycle == ActionLifecycle.POSSIBLE_ENDED),
                -item.last_observed_window,
                item.action_id,
            ),
        ):
            if action.lifecycle not in {
                ActionLifecycle.STARTED,
                ActionLifecycle.ONGOING,
                ActionLifecycle.UNCERTAIN,
                ActionLifecycle.POSSIBLE_ENDED,
            }:
                continue
            actions.append(
                {
                    "action_id": action.action_id,
                    "actor_id": action.actor_id,
                    "action_type": action.action_type,
                    "target_id": action.target_id,
                    "tool_id": action.tool_id,
                    "lifecycle": action.lifecycle.value,
                }
            )
        actions = actions[: self.config.max_active_actions]

        pending = []
        for entity in sorted(state.entities.values(), key=lambda item: item.entity_id):
            for key, attribute in sorted(entity.attributes.items()):
                if attribute.status != AttributeConfirmationStatus.PENDING:
                    continue
                pending.append(
                    (
                        attribute.pending_confidence or 0.0,
                        entity.entity_id,
                        key,
                        {
                            "entity_id": entity.entity_id,
                            "attribute_key": key,
                            "candidate_value": attribute.pending_value,
                            "confidence": attribute.pending_confidence,
                        },
                    )
                )
        pending.sort(key=lambda item: -item[0])
        pending = [item[3] for item in pending[: self.config.max_pending_attributes]]
        return ObservationContext(
            scene=scene,
            candidate_entities=entities,
            active_actions=actions,
            pending_attributes=pending,
            recent_scene_changes=recent_scene_changes,
        )

    def build(self, state: GlobalState, current_window: VideoWindow) -> ObservationContext:
        context = self._base(state, current_window)
        if len(context.to_json()) <= self.config.max_serialized_characters:
            return context

        # Prune whole objects in a deterministic priority order.  Never slice
        # the serialized string: the result must remain valid JSON.
        while len(context.to_json()) > self.config.max_serialized_characters and context.candidate_entities:
            context.candidate_entities.pop()
            context.truncated = True
        while len(context.to_json()) > self.config.max_serialized_characters and context.active_actions:
            context.active_actions.pop()
            context.truncated = True
        while len(context.to_json()) > self.config.max_serialized_characters and context.pending_attributes:
            context.pending_attributes.pop()
            context.truncated = True
        if len(context.to_json()) > self.config.max_serialized_characters:
            # The fixed scene object itself can still be compacted without
            # invalidating the JSON structure.
            context.scene = {"scene_id": context.scene.get("scene_id"), "view_type": context.scene.get("view_type")}
            context.truncated = True
        return context

    def candidate_entity_ids(self, state: GlobalState, current_window: VideoWindow) -> set[str]:
        """Return the set of global entity IDs exposed as candidates in the prompt.

        The IDs reflect the actually truncated context, so they match what the
        model is allowed to reference.
        """
        return {entity["entity_id"] for entity in self.build(state, current_window).candidate_entities}

    def sanitize_candidate_global_ids(
        self,
        state: GlobalState,
        current_window: VideoWindow,
        observation: ObservationBatch,
        allow_candidate_global_ids: bool,
    ) -> list[Any]:
        """Validate and sanitize model-provided candidate_global_id values.

        When ``allow_candidate_global_ids`` is false, or the referenced ID is
        not in the current candidate set, the raw value is preserved in a
        normalization warning and the field is set to ``None`` before the
        observation is persisted.
        """
        from ..inference.normalizer import NormalizationWarning

        if not allow_candidate_global_ids:
            warnings: list[Any] = []
            for entity in observation.entities:
                if entity.candidate_global_id is not None:
                    warnings.append(
                        NormalizationWarning(
                            warning_type="candidate_global_id_disabled",
                            local_id=entity.local_id,
                            field_name="candidate_global_id",
                            raw_value=entity.candidate_global_id,
                            normalized_value=None,
                            message="candidate_global_ids are disabled by configuration",
                        )
                    )
                    entity.candidate_global_id = None
            return warnings

        candidate_map = {
            entity["entity_id"]: entity["entity_type"]
            for entity in self.build(state, current_window).candidate_entities
        }
        warnings = []
        for entity in observation.entities:
            candidate_id = entity.candidate_global_id
            if candidate_id is None:
                continue
            if candidate_id not in candidate_map:
                warnings.append(
                    NormalizationWarning(
                        warning_type="candidate_global_id_not_in_context",
                        local_id=entity.local_id,
                        field_name="candidate_global_id",
                        raw_value=candidate_id,
                        normalized_value=None,
                        message=f"candidate_global_id {candidate_id!r} is not in the current candidate list",
                    )
                )
                entity.candidate_global_id = None
                continue
            if candidate_map[candidate_id] != entity.entity_type.value:
                warnings.append(
                    NormalizationWarning(
                        warning_type="candidate_global_id_type_mismatch",
                        local_id=entity.local_id,
                        field_name="candidate_global_id",
                        raw_value=candidate_id,
                        normalized_value=None,
                        message=(
                            f"candidate_global_id {candidate_id!r} has entity type "
                            f"{candidate_map[candidate_id]!r} but observation declares {entity.entity_type.value!r}"
                        ),
                    )
                )
                entity.candidate_global_id = None
        return warnings
