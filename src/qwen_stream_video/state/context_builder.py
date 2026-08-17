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
        scene = {
            "scene_id": current_scene.scene_id if current_scene else None,
            "view_type": current_scene.view_type.value if current_scene else "unknown",
            "continuity": current_scene.continuity if current_scene else "unknown",
        }
        current_index = current_window.global_index
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
            key=lambda item: (-int(item.entity_id in related_entity_ids), -item.last_seen_window, item.entity_id),
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
        for action in sorted(state.actions.values(), key=lambda item: (-item.last_observed_window, item.action_id)):
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
                    {
                        "entity_id": entity.entity_id,
                        "attribute_key": key,
                        "candidate_value": attribute.pending_value,
                        "confidence": attribute.pending_confidence,
                    }
                )
        pending = pending[: self.config.max_pending_attributes]
        return ObservationContext(
            scene=scene,
            candidate_entities=entities,
            active_actions=actions,
            pending_attributes=pending,
            recent_scene_changes=[],
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
