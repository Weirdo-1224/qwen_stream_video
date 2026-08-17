from qwen_stream_video.domain import (
    ActionObservation,
    EntityObservation,
    EntityType,
    ObservationBatch,
    WindowObservation,
)
from qwen_stream_video.inference import ObservationNormalizer


def _batch(action: str, attribute: str = "door_status") -> ObservationBatch:
    return ObservationBatch(
        window=WindowObservation(global_index=0, start_seconds=0, commit_start_seconds=0, end_seconds=1),
        entities=[EntityObservation(local_id="D", entity_type=EntityType.DEVICE, confidence=0.9)],
        actions=[ActionObservation(local_id="A", actor_local_id="D", action_type=action, confidence=0.9)],
        attribute_observations=[
            {"entity_local_id": "D", "attribute": attribute, "value": "closed", "confidence": 0.8}
        ],
    )


def test_non_destructive_action_and_attribute_normalization() -> None:
    result = ObservationNormalizer().normalize(_batch("hand_over"))
    assert result.batch.actions[0].action_type == "hand_over"
    assert result.batch.actions[0].raw_action_type == "hand_over"
    assert result.batch.attribute_observations[0].attribute_key == "door.state"
    assert result.batch.attribute_observations[0].raw_attribute == "door_status"


def test_oov_action_is_other_and_unknown_stays_unknown() -> None:
    normalizer = ObservationNormalizer()
    oov = normalizer.normalize(_batch("dance")).batch.actions[0]
    unknown = normalizer.normalize(_batch("unknown")).batch.actions[0]
    assert (oov.action_type, oov.normalization_status, oov.raw_action_type) == ("other", "out_of_vocabulary", "dance")
    assert (unknown.action_type, unknown.normalization_status) == ("unknown", "unknown")
