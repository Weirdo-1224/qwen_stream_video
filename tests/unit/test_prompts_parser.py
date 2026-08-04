"""Unit tests for prompt construction and response parsing."""

from __future__ import annotations

import json

import numpy as np
import pytest

from qwen_stream_video.domain import (
    Action,
    ActionPhase,
    Entity,
    EntityType,
    ObservationBatch,
    SceneObservation,
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
            timestamp=9.0 + i,
            frame_index=i,
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
    obs = WindowObservation(
        schema_version="1.0",
        window_run_index=video_window.run_index,
        window_global_index=video_window.global_index,
        window_start_seconds=video_window.start_seconds,
        window_end_seconds=video_window.end_seconds,
        scene=SceneObservation(description="A technician operates a breaker."),
        entities=[
            Entity(
                local_id="E1",
                entity_type=EntityType.PERSON,
                label="technician",
                confidence=0.9,
            ),
            Entity(
                local_id="E2",
                entity_type=EntityType.EQUIPMENT,
                label="breaker",
                confidence=0.8,
            ),
        ],
        actions=[
            Action(
                local_id="A1",
                actor_id="E1",
                action_type="touch",
                phase=ActionPhase.CONTINUE,
                target_id="E2",
                evidence_frame_sample_indices=[0, 1],
                confidence=0.85,
            ),
        ],
        uncertainties=[],
        summary="Technician touches the breaker.",
    )
    return ObservationBatch(observations=[obs])


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
    )
    assert "test.mp4" in prompt
    assert "breaker" in prompt
    assert "training" in prompt
    assert "9.000" in prompt
    assert "15.000" in prompt
    assert "4" in prompt
    assert "9.000, 10.000, 11.000, 12.000" in prompt
    assert "Previous summary." in prompt


def test_build_user_prompt_without_previous_summary(
    prompt_builder: PromptBuilder,
    video_window: VideoWindow,
    sampled_frames: list[SampledFrame],
) -> None:
    prompt = prompt_builder.build_user_prompt(video_window, sampled_frames)
    assert "无（当前窗口是首个窗口）" in prompt


def test_parse_valid_json_returns_batch_and_warnings(
    response_parser: ResponseParser,
    valid_batch: ObservationBatch,
    sampled_frames: list[SampledFrame],
    video_window: VideoWindow,
) -> None:
    raw = valid_batch.model_dump_json()
    batch, warnings = response_parser.parse(raw, sampled_frames, window=video_window)
    assert len(batch.observations) == 1
    assert batch.observations[0].window_global_index == video_window.global_index
    assert batch.observations[0].window_run_index == video_window.run_index
    assert batch.observations[0].window_start_seconds == video_window.start_seconds
    assert batch.observations[0].window_end_seconds == video_window.end_seconds
    assert warnings == []


def test_parse_json_in_markdown_code_block(
    response_parser: ResponseParser,
    valid_batch: ObservationBatch,
    sampled_frames: list[SampledFrame],
) -> None:
    raw = f"```json\n{valid_batch.model_dump_json()}\n```"
    batch, warnings = response_parser.parse(raw, sampled_frames)
    assert len(batch.observations) == 1
    assert warnings == []


def test_parse_json_with_surrounding_text(
    response_parser: ResponseParser,
    valid_batch: ObservationBatch,
    sampled_frames: list[SampledFrame],
) -> None:
    raw = f"Here is the result:\n{valid_batch.model_dump_json()}\nEnd of output."
    batch, warnings = response_parser.parse(raw, sampled_frames)
    assert len(batch.observations) == 1
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
    raw = json.dumps({"schema_version": "1.0", "observations": [{"invalid": "data"}]})
    with pytest.raises(ModelOutputSchemaError):
        response_parser.parse(raw, sampled_frames)


def test_parse_invalid_top_level_raises(
    response_parser: ResponseParser,
    sampled_frames: list[SampledFrame],
) -> None:
    raw = json.dumps({"schema_version": "1.0", "observations": "not-a-list"})
    with pytest.raises(ModelOutputSchemaError):
        response_parser.parse(raw, sampled_frames)


def test_parse_semantic_error_duplicate_entity_id_raises(
    response_parser: ResponseParser,
    sampled_frames: list[SampledFrame],
) -> None:
    obs = WindowObservation(
        schema_version="1.0",
        window_run_index=0,
        window_global_index=0,
        window_start_seconds=0.0,
        window_end_seconds=1.0,
        scene=SceneObservation(description="test"),
        entities=[
            Entity(
                local_id="E1",
                entity_type=EntityType.PERSON,
                label="person1",
                confidence=0.9,
            ),
            Entity(
                local_id="E1",
                entity_type=EntityType.EQUIPMENT,
                label="device1",
                confidence=0.8,
            ),
        ],
    )
    raw = ObservationBatch(observations=[obs]).model_dump_json()
    with pytest.raises(ModelOutputSemanticError, match="Duplicate entity local_id"):
        response_parser.parse(raw, sampled_frames)


def test_parse_returns_warning_for_unknown_action(
    response_parser: ResponseParser,
    sampled_frames: list[SampledFrame],
) -> None:
    obs = WindowObservation(
        schema_version="1.0",
        window_run_index=0,
        window_global_index=0,
        window_start_seconds=0.0,
        window_end_seconds=1.0,
        scene=SceneObservation(description="test"),
        entities=[
            Entity(
                local_id="E1",
                entity_type=EntityType.PERSON,
                label="person",
                confidence=0.9,
            ),
        ],
        actions=[
            Action(
                local_id="A1",
                actor_id="E1",
                action_type="dance",
                phase=ActionPhase.START,
                confidence=0.8,
            ),
        ],
    )
    raw = ObservationBatch(observations=[obs]).model_dump_json()
    batch, warnings = response_parser.parse(raw, sampled_frames)
    assert len(warnings) == 1
    assert "mapped to 'unknown'" in warnings[0]
    assert batch.observations[0].actions[0].action_type == "unknown"
