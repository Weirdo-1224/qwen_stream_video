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
    DOCUMENT = "document"
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


class VisibilityState(str, Enum):
    VISIBLE = "visible"
    PARTIAL = "partial"
    OCCLUDED = "occluded"
    NOT_VISIBLE = "not_visible"
    UNKNOWN = "unknown"


class EntityLifecycleStatus(str, Enum):
    ACTIVE = "active"
    TEMPORARILY_MISSING = "temporarily_missing"
    INACTIVE = "inactive"
    MERGED = "merged"


class EntityResolutionStatus(str, Enum):
    MATCHED = "matched"
    CREATED = "created"
    AMBIGUOUS = "ambiguous"
    TEMPORARY = "temporary"
    REJECTED_HINT = "rejected_hint"


class ActionLifecycle(str, Enum):
    CANDIDATE = "candidate"
    STARTED = "started"
    ONGOING = "ongoing"
    POSSIBLE_ENDED = "possible_ended"
    ENDED = "ended"
    INSTANT = "instant"
    UNCERTAIN = "uncertain"
    INTERRUPTED = "interrupted"


class AttributeConfirmationStatus(str, Enum):
    OBSERVED = "observed"
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CONFLICTED = "conflicted"
    REJECTED = "rejected"
