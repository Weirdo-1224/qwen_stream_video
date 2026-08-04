"""Enumerations used by the incremental observation schema."""

from __future__ import annotations

from enum import Enum


class EntityType(str, Enum):
    """Broad categories of detected entities."""

    PERSON = "person"
    DEVICE = "device"
    COMPONENT = "component"
    TOOL = "tool"
    PPE = "ppe"
    SIGN = "sign"
    ENVIRONMENT = "environment"
    UNKNOWN = "unknown"


class ViewType(str, Enum):
    """Camera framing / viewpoint of the observed scene or entity."""

    WIDE = "wide"
    MEDIUM = "medium"
    CLOSEUP = "closeup"
    DETAIL = "detail"
    UNKNOWN = "unknown"


class VisibilityQuality(str, Enum):
    """Quality of visibility in the current window."""

    CLEAR = "clear"
    PARTIAL = "partial"
    POOR = "poor"
    UNKNOWN = "unknown"


class ActionPhaseObservation(str, Enum):
    """Temporal phase of an action as observed in the current window."""

    STARTING = "starting"
    ONGOING = "ongoing"
    POSSIBLY_COMPLETED = "possibly_completed"
    INSTANT = "instant"
    UNKNOWN = "unknown"
