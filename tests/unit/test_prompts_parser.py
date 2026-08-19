"""Unit tests for prompt construction and response parsing."""

from __future__ import annotations

import json

import numpy as np
import pytest

from qwen_stream_video.domain import (
    ActionObservation,
    ActionPhaseObservation,
    AttributeObservation,
    EntityObservation,
    EntityType,
    ObservationBatch,
    SceneObservation,
    ViewType,
    VisibilityQuality,
    WindowObservation,
)
from qwen_stream_video.exceptions import (
    ModelOutputParseError,
    ModelOutputSchemaError,
    ModelOutputSemanticError,
)
from qwen_stream_video.inference import PromptBuilder, ResponseParser
from qwen_stream_video.video import SampledFrame, VideoWindow


@pytest.fixture
def video_window() -> VideoWindow:
    return VideoWindow(
        global_index=3,
        run_index=1,
        start_seconds=9.0,
        end_seconds=15.0,
        window_type="regular",
    )


@pytest.fixture
def sampled_frames(video_window: VideoWindow) -> list[SampledFrame]:
    return [
        SampledFrame(
            run_index=video_window.run_index,
            global_index=video_window.global_index,
            sample_index=i,
            frame_index=i,
            timestamp_seconds=9.0 + i,
            image=np.zeros((10, 10, 3), dtype="uint8"),
        )
        for i in range(4)
    ]


@pytest.fixture
def prompt_builder() -> PromptBuilder:
    return PromptBuilder()


@pytest.fixture
def response_parser() -> ResponseParser:
    return ResponseParser()


@pytest.fixture
def valid_batch(video_window: VideoWindow) -> ObservationBatch:
    return ObservationBatch(
        schema_version="1.0",
        window=WindowObservation(
            global_index=video_window.global_index,
            start_seconds=video_window.start_seconds,
            end_seconds=video_window.end_seconds,
        ),
        summary="A technician touches the breaker.",
        scene=SceneObservation(
            camera_change=False,
            view_type=ViewType.MEDIUM,
            visibility=VisibilityQuality.CLEAR,
            description="Indoor substation.",
        ),
        entities=[
            EntityObservation(
                local_id="E1",
                entity_type=EntityType.PERSON,
                name="technician",
                confidence=0.9,
                evidence_frames=[0],
            ),
            EntityObservation(
                local_id="E2",
                entity_type=EntityType.DEVICE,
                name="breaker",
                confidence=0.8,
                evidence_frames=[1],
            ),
        ],
        actions=[
            ActionObservation(
                local_id="A1",
                actor_local_id="E1",
                action_type="touch",
                target_local_id="E2",
                phase_observation=ActionPhaseObservation.ONGOING,
                description="Technician touches the breaker.",
                confidence=0.85,
                evidence_frames=[0, 1],
            ),
        ],
        attribute_observations=[
            AttributeObservation(
                entity_local_id="E2",
                attribute="state",
                value="closed",
                confidence=0.8,
                evidence_frames=[1],
            )
        ],
    )


def test_build_user_prompt_contains_window_and_frame_info(
    prompt_builder: PromptBuilder,
    video_window: VideoWindow,
    sampled_frames: list[SampledFrame],
) -> None:
    prompt = prompt_builder.build_user_prompt(
        video_window,
        sampled_frames,
        video_context={
            "video_name": "test.mp4",
            "video_category": "breaker",
            "task_background": "training",
        },
        previous_summary="Previous summary.",
        previous_entities=[
            {
                "candidate_global_id": "person_1",
                "entity_type": "person",
                "description": "A technician.",
            }
        ],
    )
    assert "test.mp4" not in prompt
    assert "Context Interval" in prompt
    assert "Commit Interval" in prompt
    assert "9.000" in prompt
    assert "15.000" in prompt
    assert "F3" in prompt
    assert "F0 = 9.000" in prompt
    assert "Previous summary." not in prompt
    assert "person_1" in prompt
    assert "A technician." in prompt


def test_build_user_prompt_without_previous_summary(
    prompt_builder: PromptBuilder,
    video_window: VideoWindow,
    sampled_frames: list[SampledFrame],
) -> None:
    prompt = prompt_builder.build_user_prompt(video_window, sampled_frames)
    assert "Schema 2.0" in prompt
    assert "candidate_global_id" in prompt


def test_build_prompts_include_authoritative_schema_contract(
    prompt_builder: PromptBuilder,
    video_window: VideoWindow,
    sampled_frames: list[SampledFrame],
) -> None:
    prompt = prompt_builder.build_user_prompt(video_window, sampled_frames)

    assert '"$defs"' in prompt
    assert '"additionalProperties":false' in prompt
    assert "entity_type 只能使用 person、device、component、tool、ppe、sign、document、environment、unknown" in prompt
    assert '"attribute_key"' in prompt
    assert '"relation_type"' in prompt
    assert '"actor_local_id"' in prompt
    assert '"uncertainty_type"' in prompt
    assert '"continuity_hint"' in prompt
    assert '"additionalProperties":false' in prompt_builder.system_prompt


def test_parse_valid_json_returns_batch_and_warnings(
    response_parser: ResponseParser,
    valid_batch: ObservationBatch,
    sampled_frames: list[SampledFrame],
    video_window: VideoWindow,
) -> None:
    raw = valid_batch.model_dump_json()
    batch, warnings = response_parser.parse(raw, sampled_frames, window=video_window)
    assert batch.window.global_index == video_window.global_index
    assert batch.window.start_seconds == video_window.start_seconds
    assert batch.window.end_seconds == video_window.end_seconds
    assert batch.summary == "A technician touches the breaker."
    assert warnings == []


def test_parse_json_in_markdown_code_block(
    response_parser: ResponseParser,
    valid_batch: ObservationBatch,
    sampled_frames: list[SampledFrame],
) -> None:
    raw = f"```json\n{valid_batch.model_dump_json()}\n```"
    batch, warnings = response_parser.parse(raw, sampled_frames)
    assert batch.summary == "A technician touches the breaker."
    assert warnings == []


def test_parse_json_with_surrounding_text(
    response_parser: ResponseParser,
    valid_batch: ObservationBatch,
    sampled_frames: list[SampledFrame],
) -> None:
    raw = f"Here is the result:\n{valid_batch.model_dump_json()}\nEnd of output."
    batch, warnings = response_parser.parse(raw, sampled_frames)
    assert batch.summary == "A technician touches the breaker."
    assert warnings == []


def test_parse_invalid_json_raises(
    response_parser: ResponseParser,
    sampled_frames: list[SampledFrame],
) -> None:
    raw = "{ invalid json"
    with pytest.raises(ModelOutputParseError):
        response_parser.parse(raw, sampled_frames)


def test_parse_missing_json_raises(
    response_parser: ResponseParser,
    sampled_frames: list[SampledFrame],
) -> None:
    raw = "No JSON here."
    with pytest.raises(ModelOutputParseError, match="No JSON object"):
        response_parser.parse(raw, sampled_frames)


def test_parse_invalid_schema_raises(
    response_parser: ResponseParser,
    sampled_frames: list[SampledFrame],
) -> None:
    raw = json.dumps({"schema_version": "1.0", "window": {"invalid": "data"}})
    with pytest.raises(ModelOutputSchemaError):
        response_parser.parse(raw, sampled_frames)


def test_parse_invalid_top_level_list_raises(
    response_parser: ResponseParser,
    sampled_frames: list[SampledFrame],
) -> None:
    raw = json.dumps([{"schema_version": "1.0"}])
    with pytest.raises(ModelOutputSchemaError):
        response_parser.parse(raw, sampled_frames)


def test_parse_semantic_error_duplicate_entity_id_raises(
    response_parser: ResponseParser,
    sampled_frames: list[SampledFrame],
) -> None:
    batch = ObservationBatch(
        schema_version="1.0",
        window=WindowObservation(global_index=0, start_seconds=0.0, end_seconds=1.0),
        summary="test",
        scene=SceneObservation(description="test"),
        entities=[
            EntityObservation(
                local_id="E1",
                entity_type=EntityType.PERSON,
                name="person1",
                confidence=0.9,
            ),
            EntityObservation(
                local_id="E1",
                entity_type=EntityType.DEVICE,
                name="device1",
                confidence=0.8,
            ),
        ],
    )
    raw = batch.model_dump_json()
    with pytest.raises(ModelOutputSemanticError, match="Duplicate entity local_id"):
        response_parser.parse(raw, sampled_frames)


def test_parse_returns_warning_for_unknown_action(
    response_parser: ResponseParser,
    sampled_frames: list[SampledFrame],
) -> None:
    batch = ObservationBatch(
        schema_version="1.0",
        window=WindowObservation(global_index=0, start_seconds=0.0, end_seconds=1.0),
        summary="test",
        scene=SceneObservation(description="test"),
        entities=[
            EntityObservation(
                local_id="E1",
                entity_type=EntityType.PERSON,
                name="person",
                confidence=0.9,
            ),
        ],
        actions=[
            ActionObservation(
                local_id="A1",
                actor_local_id="E1",
                action_type="dance",
                phase_observation=ActionPhaseObservation.STARTING,
                confidence=0.8,
            ),
        ],
    )
    raw = batch.model_dump_json()
    batch, warnings = response_parser.parse(raw, sampled_frames)
    assert len(warnings) == 1
    assert "mapped to 'unknown'" in warnings[0]
    assert batch.actions[0].action_type == "dance"


def test_parse_forbidden_eval_not_used(
    response_parser: ResponseParser,
    sampled_frames: list[SampledFrame],
) -> None:
    raw = "__import__('os').system('echo pwned')"
    with pytest.raises(ModelOutputParseError):
        response_parser.parse(raw, sampled_frames)


def test_parse_local_qwen_compact_observation_format(
    response_parser: ResponseParser,
    sampled_frames: list[SampledFrame],
    video_window: VideoWindow,
) -> None:
    raw = json.dumps(
        {
            "observation": {
                "local_id": "person_0",
                "candidate_global_id": None,
                "phase": "other",
                "evidence_frames": ["F0", "F1"],
                "attributes": {
                    "headwear": "red hard hat",
                    "location": "in front of a panel",
                },
                "actions": [
                    {
                        "action": "operate",
                        "phase": "commit",
                        "evidence_frames": ["F1"],
                    }
                ],
                "scene": {"view_type": "wide", "continuity": "continuous"},
            }
        }
    )
    batch, warnings = response_parser.parse(raw, sampled_frames, window=video_window)

    assert batch.schema_version == "2.0"
    assert batch.window.global_index == video_window.global_index
    assert batch.entities[0].local_id == "person_0"
    assert batch.entities[0].entity_type == EntityType.PERSON
    assert batch.entities[0].appearance["headwear"] == "red hard hat"
    assert batch.actions[0].action_type == "operate"
    assert batch.actions[0].actor_local_id == "person_0"
    assert batch.actions[0].evidence_frames == [1]
    assert any("compact observation format" in warning for warning in warnings)


def test_parse_schema_v2_without_window_uses_runtime_window(
    response_parser: ResponseParser,
    sampled_frames: list[SampledFrame],
    video_window: VideoWindow,
) -> None:
    raw = json.dumps(
        {
            "schema_version": "2.0",
            "scene": {},
            "entities": [],
            "actions": [],
            "attribute_observations": [],
            "relations": [],
            "uncertainties": [],
        }
    )
    batch, warnings = response_parser.parse(raw, sampled_frames, window=video_window)
    assert batch.window.global_index == video_window.global_index
    assert any("program-owned window" in warning for warning in warnings)


def test_parse_local_shorthand_fields_inside_schema_v2(
    response_parser: ResponseParser,
    sampled_frames: list[SampledFrame],
    video_window: VideoWindow,
) -> None:
    raw = json.dumps(
        {
            "schema_version": "2.0",
            "scene": {},
            "entities": [
                {
                    "local_id": "person_0",
                    "entity_type": "person",
                    "name": "person",
                    "candidate_global_id": "person_0001",
                    "confidence": 0.98,
                    "evidence_frames": [0],
                }
            ],
            "actions": [
                {
                    "local_id": "person_0",
                    "actor_id": "person_0001",
                    "target_id": "equipment_0001",
                    "tool_id": None,
                    "action": "operate",
                    "phase": "commit",
                    "start_frame": 1,
                    "end_frame": 2,
                }
            ],
            "attribute_observations": [
                {
                    "local_id": "person_0",
                    "attribute_key": "wearing_hard_hat",
                    "value": True,
                    "evidence_frames": [0],
                }
            ],
            "relations": [],
            "uncertainties": [],
        }
    )
    batch, warnings = response_parser.parse(raw, sampled_frames, window=video_window)
    assert batch.actions[0].local_id == "local_action_0001"
    assert batch.actions[0].actor_local_id == "person_local_0001"
    assert batch.actions[0].action_type == "operate"
    assert batch.actions[0].evidence_frames == [1, 2]
    assert batch.attribute_observations[0].entity_local_id == "person_local_0001"
    assert batch.attribute_observations[0].value == "True"
    assert any("shorthand fields" in warning for warning in warnings)


def test_local_qwen_adapter_does_not_invent_window_without_runtime_window(
    response_parser: ResponseParser,
    sampled_frames: list[SampledFrame],
) -> None:
    raw = json.dumps({"observation": {"local_id": "person_0"}})
    with pytest.raises(ValueError, match="program-provided video window"):
        response_parser.parse(raw, sampled_frames)
