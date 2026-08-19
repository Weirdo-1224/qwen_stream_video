"""Adapters for the compact JSON dialect emitted by local Qwen checkpoints."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from ..domain import (
    ActionPhaseObservation,
    EntityType,
    ViewType,
    VisibilityQuality,
)
from ..video import SampledFrame, VideoWindow


class LocalObservationAdapter:
    """Convert the observed local-Qwen dialect into Observation Schema 2.0.

    The local model sometimes emits ``{"observation": {...}}`` instead of the
    project protocol.  This adapter only migrates explicit fields and keeps
    unsupported values in entity appearance or warnings; it never assigns a
    formal global ID or invents evidence.
    """

    def adapt(
        self,
        data: Mapping[str, Any],
        sampled_frames: list[SampledFrame],
        window: VideoWindow | None,
    ) -> tuple[dict[str, Any], list[str]] | None:
        payload = data.get("observation")
        if not isinstance(payload, Mapping):
            if data.get("schema_version") == "2.0":
                return self._adapt_direct_v2(data, sampled_frames)
            return None

        warnings = [
            "Adapted local Qwen compact observation format to Observation Schema 2.0"
        ]
        if window is None:
            raise ValueError(
                "Local compact observation format requires the program-provided video window"
            )

        evidence = self._evidence(payload.get("evidence_frames"), sampled_frames, warnings)
        entities = self._entities(payload, evidence, warnings)
        entity_ids = {entity["local_id"] for entity in entities}
        actions = self._actions(payload, evidence, entity_ids, sampled_frames, warnings)
        scene = self._scene(payload.get("scene"), warnings)
        summary = self._summary(payload)

        adapted: dict[str, Any] = {
            "schema_version": "2.0",
            "window": {
                "global_index": window.global_index,
                "start_seconds": window.start_seconds,
                "commit_start_seconds": window.commit_start_seconds,
                "end_seconds": window.end_seconds,
            },
            "summary": summary,
            "scene": scene,
            "entities": entities,
            "actions": actions,
            "attribute_observations": [],
            "relations": [],
            "uncertainties": [],
        }
        if payload.get("attributes"):
            warnings.append(
                "Local compact attributes were preserved in entity appearance; "
                "they were not promoted to formal attributes without an entity and canonical key"
            )
        return adapted, warnings

    @staticmethod
    def _adapt_direct_v2(
        data: Mapping[str, Any],
        sampled_frames: list[SampledFrame],
    ) -> tuple[dict[str, Any], list[str]] | None:
        """Normalize local shorthand fields in an otherwise V2-shaped object."""
        raw_actions = data.get("actions")
        raw_attributes = data.get("attribute_observations")
        raw_entities = data.get("entities")
        valid_frames = {frame.sample_index for frame in sampled_frames}
        entity_needs_adaptation = any(
            isinstance(item, Mapping)
            and (
                LocalObservationAdapter._is_formal_id(item.get("local_id"))
                or any(
                    frame not in valid_frames
                    for frame in item.get("evidence_frames", [])
                    if isinstance(frame, int)
                )
            )
            for item in raw_entities or []
        )
        needs_action = any(
            isinstance(item, Mapping)
            and any(
                key in item
                for key in (
                    "action",
                    "phase",
                    "actor_id",
                    "target_id",
                    "tool_id",
                    "start_frame",
                    "end_frame",
                )
                )
            for item in raw_actions or []
        ) or entity_needs_adaptation
        needs_attribute = any(
            isinstance(item, Mapping)
            and ("local_id" in item or not isinstance(item.get("value"), str))
            for item in raw_attributes or []
        ) or entity_needs_adaptation
        if not needs_action and not needs_attribute and not entity_needs_adaptation:
            return None

        updated = dict(data)
        warnings = ["Adapted local shorthand fields in Schema 2.0 output"]
        entities = raw_entities if isinstance(raw_entities, list) else []
        id_map: dict[str, str] = {}
        normalized_entities: list[dict[str, Any]] = []
        for index, item in enumerate(entities, start=1):
            if not isinstance(item, Mapping):
                continue
            entity = dict(item)
            old_local_id = entity.get("local_id")
            local_id = old_local_id if isinstance(old_local_id, str) else f"entity_{index:04d}"
            entity_type = str(entity.get("entity_type", EntityType.UNKNOWN.value))
            if LocalObservationAdapter._is_formal_id(local_id):
                entity.setdefault("candidate_global_id", local_id)
                local_id = f"{entity_type}_local_{index:04d}"
                warnings.append(f"Moved model global-like entity ID {old_local_id!r} to candidate_global_id")
            if isinstance(old_local_id, str):
                id_map[old_local_id] = local_id
            entity["local_id"] = local_id
            entity["evidence_frames"] = LocalObservationAdapter._normalize_evidence(
                entity.get("evidence_frames", []), valid_frames, warnings
            )
            normalized_entities.append(entity)
        if entity_needs_adaptation:
            updated["entities"] = normalized_entities
        else:
            normalized_entities = [dict(item) for item in entities if isinstance(item, Mapping)]
        entity_ids = {
            item.get("local_id")
            for item in normalized_entities
            if isinstance(item, Mapping) and isinstance(item.get("local_id"), str)
        }
        global_to_local = {
            item.get("candidate_global_id"): item.get("local_id")
            for item in normalized_entities
            if isinstance(item, Mapping)
            and isinstance(item.get("candidate_global_id"), str)
            and isinstance(item.get("local_id"), str)
        }
        global_to_local.update({key: value for key, value in id_map.items()})
        if needs_action:
            actions: list[dict[str, Any]] = []
            for index, item in enumerate(raw_actions or [], start=1):
                if not isinstance(item, Mapping):
                    continue
                action = dict(item)
                raw_action = action.pop("action", action.get("action_type", "unknown"))
                action["action_type"] = str(raw_action)
                action["raw_action_type"] = str(raw_action)
                raw_phase = action.pop("phase", action.get("phase_observation", "unknown"))
                if raw_phase not in {phase.value for phase in ActionPhaseObservation}:
                    warnings.append(f"Unknown local action phase {raw_phase!r}; mapped to unknown")
                    raw_phase = ActionPhaseObservation.UNKNOWN.value
                action["phase_observation"] = raw_phase
                action.setdefault("confidence", 0.0)
                actor_id = action.pop("actor_id", None)
                target_id = action.pop("target_id", None)
                tool_id = action.pop("tool_id", None)
                action["actor_local_id"] = id_map.get(
                    action.get("actor_local_id"), action.get("actor_local_id")
                ) or global_to_local.get(actor_id)
                action["target_local_id"] = id_map.get(
                    action.get("target_local_id"), action.get("target_local_id")
                ) or global_to_local.get(target_id)
                action["tool_local_id"] = id_map.get(
                    action.get("tool_local_id"), action.get("tool_local_id")
                ) or global_to_local.get(tool_id)
                if action.get("actor_local_id") not in entity_ids:
                    candidate = action.get("local_id")
                    action["actor_local_id"] = candidate if candidate in entity_ids else None
                for reference_field in ("target_local_id", "tool_local_id"):
                    reference = action.get(reference_field)
                    if reference is not None and reference not in entity_ids:
                        warnings.append(
                            f"Dropped unresolved action {reference_field} {reference!r}"
                        )
                        action[reference_field] = None
                start_frame = action.pop("start_frame", None)
                end_frame = action.pop("end_frame", None)
                if not action.get("evidence_frames") and isinstance(start_frame, int):
                    end = end_frame if isinstance(end_frame, int) else start_frame
                    valid = {frame.sample_index for frame in sampled_frames}
                    action["evidence_frames"] = [
                        index for index in range(start_frame, end + 1) if index in valid
                    ]
                action["evidence_frames"] = LocalObservationAdapter._normalize_evidence(
                    action.get("evidence_frames", []), valid_frames, warnings
                )
                if (
                    action.get("local_id") in entity_ids
                    or action.get("local_id") in id_map
                    or not action.get("local_id")
                ):
                    action["local_id"] = f"local_action_{index:04d}"
                actions.append(action)
            updated["actions"] = actions
        if needs_attribute:
            attributes: list[dict[str, Any]] = []
            for item in raw_attributes or []:
                if not isinstance(item, Mapping):
                    continue
                attribute = dict(item)
                local_id = attribute.pop("local_id", attribute.get("entity_local_id"))
                attribute["entity_local_id"] = id_map.get(local_id, local_id) or "unknown"
                value = attribute.get("value", "unknown")
                attribute["value"] = str(value)
                attribute.setdefault("raw_value", str(value))
                attribute.setdefault("raw_attribute", attribute.get("attribute_key", "unknown"))
                attribute.setdefault("confidence", 0.0)
                attribute["evidence_frames"] = LocalObservationAdapter._normalize_evidence(
                    attribute.get("evidence_frames", []), valid_frames, warnings
                )
                attributes.append(attribute)
            updated["attribute_observations"] = attributes
        if entity_needs_adaptation:
            relations: list[dict[str, Any]] = []
            for item in data.get("relations", []) or []:
                if not isinstance(item, Mapping):
                    continue
                relation = dict(item)
                for field in ("subject_local_id", "object_local_id"):
                    reference = relation.get(field)
                    mapped = id_map.get(reference, reference)
                    if mapped not in entity_ids:
                        warnings.append(
                            f"Dropped unresolved relation {field} {reference!r}"
                        )
                        relation = None
                        break
                    relation[field] = mapped
                if relation is not None:
                    relations.append(relation)
            updated["relations"] = relations
        return updated, warnings

    @staticmethod
    def _is_formal_id(value: Any) -> bool:
        return isinstance(value, str) and bool(
            re.fullmatch(
                r"(?:person|device|component|tool|ppe|sign|document|environment|equipment)_\d+",
                value,
            )
        )

    @staticmethod
    def _normalize_evidence(
        value: Any,
        valid_frames: set[int],
        warnings: list[str],
    ) -> list[int]:
        if not isinstance(value, list):
            return []
        result: list[int] = []
        for item in value:
            index: int | None = None
            if isinstance(item, int):
                index = item
            elif isinstance(item, str) and item.startswith("F") and item[1:].isdigit():
                index = int(item[1:])
            if index is not None and index in valid_frames:
                result.append(index)
            else:
                warnings.append(f"Dropped invalid local evidence frame {item!r}")
        return sorted(set(result))

    @staticmethod
    def _summary(payload: Mapping[str, Any]) -> str:
        for key in ("summary", "description", "visual_fact"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @staticmethod
    def _scene(value: Any, warnings: list[str]) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}
        result: dict[str, Any] = {}
        for key in ("camera_change", "description"):
            if key in value:
                result[key] = value[key]
        view_type = value.get("view_type")
        if isinstance(view_type, str) and view_type in {item.value for item in ViewType}:
            result["view_type"] = view_type
        elif view_type is not None:
            warnings.append(f"Unknown local scene view_type {view_type!r}; mapped to unknown")
        continuity = value.get("continuity", value.get("continuity_hint"))
        if isinstance(continuity, str) and continuity in {
            "continuous", "reframed", "camera_change", "unknown"
        }:
            result["continuity_hint"] = continuity
        elif continuity is not None:
            warnings.append(f"Unknown local scene continuity {continuity!r}; mapped to unknown")
        visibility = value.get("visibility")
        if isinstance(visibility, str) and visibility in {item.value for item in VisibilityQuality}:
            result["scene_visibility"] = visibility
            result["target_visibility"] = visibility
        return result

    def _entities(
        self,
        payload: Mapping[str, Any],
        evidence: list[int],
        warnings: list[str],
    ) -> list[dict[str, Any]]:
        local_id = payload.get("local_id")
        if not isinstance(local_id, str) or not local_id.strip() or local_id == "unknown":
            return []
        attributes = payload.get("attributes")
        appearance = dict(attributes) if isinstance(attributes, Mapping) else {}
        entity_type = self._infer_entity_type(local_id, appearance)
        if entity_type == EntityType.UNKNOWN.value:
            warnings.append(f"Could not infer entity type for local entity {local_id!r}")
        return [
            {
                "local_id": local_id,
                "entity_type": entity_type,
                "name": local_id,
                "description": self._appearance_description(appearance),
                "appearance": appearance,
                "spatial_region": "unknown",
                "candidate_global_id": self._candidate_id(payload.get("candidate_global_id")),
                "confidence": 0.0,
                "evidence_frames": evidence,
            }
        ]

    @staticmethod
    def _infer_entity_type(local_id: str, appearance: Mapping[str, Any]) -> str:
        lowered = local_id.lower()
        if lowered.startswith(("person", "worker", "operator")):
            return EntityType.PERSON.value
        if lowered.startswith(("tool", "device_tool")):
            return EntityType.TOOL.value
        if lowered.startswith(("device", "cabinet", "panel")):
            return EntityType.DEVICE.value
        role = str(appearance.get("role", "")).lower()
        if role in {"operator", "worker", "person"}:
            return EntityType.PERSON.value
        return EntityType.UNKNOWN.value

    @staticmethod
    def _appearance_description(appearance: Mapping[str, Any]) -> str:
        values = [str(value) for value in appearance.values() if isinstance(value, (str, int, float))]
        return "; ".join(values[:6])

    def _actions(
        self,
        payload: Mapping[str, Any],
        default_evidence: list[int],
        entity_ids: set[str],
        sampled_frames: list[SampledFrame],
        warnings: list[str],
    ) -> list[dict[str, Any]]:
        raw_actions = payload.get("actions")
        if not isinstance(raw_actions, list):
            return []
        result: list[dict[str, Any]] = []
        implicit_actor = min(entity_ids) if len(entity_ids) == 1 else None
        for index, raw in enumerate(raw_actions, start=1):
            if not isinstance(raw, Mapping):
                warnings.append(f"Ignored non-object local action at index {index}")
                continue
            action_type = raw.get("action_type", raw.get("action", "unknown"))
            if not isinstance(action_type, str) or not action_type.strip():
                action_type = "unknown"
            action_id = raw.get("local_id")
            if not isinstance(action_id, str) or not action_id.strip():
                action_id = f"local_action_{index:04d}"
            actor = raw.get("actor_local_id")
            if not isinstance(actor, str) or actor not in entity_ids:
                actor = implicit_actor
            action_evidence = self._evidence(
                raw.get("evidence_frames"), sampled_frames, warnings
            ) or default_evidence
            phase = raw.get("phase_observation", raw.get("phase", "unknown"))
            if phase not in {item.value for item in ActionPhaseObservation}:
                warnings.append(f"Unknown local action phase {phase!r}; mapped to unknown")
                phase = ActionPhaseObservation.UNKNOWN.value
            result.append(
                {
                    "local_id": action_id,
                    "actor_local_id": actor,
                    "action_type": action_type,
                    "raw_action_type": action_type,
                    "phase_observation": phase,
                    "description": str(raw.get("description", "")),
                    "confidence": 0.0,
                    "evidence_frames": action_evidence,
                }
            )
        return result

    @staticmethod
    def _candidate_id(value: Any) -> str | None:
        return value.strip() if isinstance(value, str) and value.strip() else None

    @staticmethod
    def _evidence(
        value: Any,
        sampled_frames: list[SampledFrame],
        warnings: list[str],
    ) -> list[int]:
        if not isinstance(value, list):
            return []
        valid_indices = {frame.sample_index for frame in sampled_frames}
        result: list[int] = []
        for item in value:
            if isinstance(item, int) and item in valid_indices:
                result.append(item)
                continue
            if isinstance(item, str) and item.startswith("F") and item[1:].isdigit():
                index = int(item[1:])
                if index in valid_indices:
                    result.append(index)
                    continue
            warnings.append(f"Dropped unsupported local evidence frame {item!r}")
        return sorted(set(result))
