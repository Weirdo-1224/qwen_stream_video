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
