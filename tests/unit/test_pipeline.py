"""Unit tests for the streaming video pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from qwen_stream_video.config import AppConfig
from qwen_stream_video.inference import FakeQwenClient
from qwen_stream_video.pipeline import StreamingVideoPipeline


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    return AppConfig.model_validate(
        {
            "experiment": {"name": "test_pipeline", "seed": 7},
            "video": {"window_seconds": 2.0, "stride_seconds": 1.0},
            "sampling": {
                "sample_fps": 1.0,
                "min_frames": 2,
                "max_frames": 4,
                "max_image_side": 64,
            },
            "storage": {"output_root": str(tmp_path), "save_raw_responses": True},
            "model": {"api_key": "secret", "name": "fake-model"},
            "video_context": {
                "video_name": "test.mp4",
                "video_category": "demo",
                "task_background": "unit test",
            },
        }
    )


@pytest.fixture
def valid_response_text() -> str:
    """Return a minimal valid single-window observation response."""
    return json.dumps(
        {
            "schema_version": "1.0",
            "window": {
                "global_index": 0,
                "start_seconds": 0.0,
                "end_seconds": 1.0,
            },
            "summary": "Test summary.",
            "scene": {
                "camera_change": False,
                "view_type": "unknown",
                "visibility": "unknown",
                "description": "A test scene.",
            },
            "entities": [
                {
                    "local_id": "E1",
                    "entity_type": "person",
                    "name": "operator",
                    "candidate_global_id": "person_1",
                    "confidence": 0.9,
                    "evidence_frames": [0],
                }
            ],
            "actions": [
                {
                    "local_id": "A1",
                    "actor_local_id": "E1",
                    "action_type": "observe",
                    "phase_observation": "ongoing",
                    "description": "Operator observes.",
                    "confidence": 0.8,
                    "evidence_frames": [0],
                }
            ],
            "attribute_observations": [],
            "uncertainties": [],
        }
    )


def _make_test_video(path: Path, duration: float = 5.0, fps: float = 5.0) -> Path:
    """Create a tiny synthetic video for pipeline testing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 64, 64
    total_frames = int(duration * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    for i in range(total_frames):
        frame = np.full((height, width, 3), (i % 256, 128, 128), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_validate_only_returns_none_and_prints_report(
    app_config: AppConfig, capsys: pytest.CaptureFixture
) -> None:
    video_path = _make_test_video(Path(app_config.storage.output_root) / "validate.mp4", duration=5.0)
    pipeline = StreamingVideoPipeline(app_config, video_path)
    result = pipeline.run(validate_only=True)
    assert result is None
    captured = capsys.readouterr()
    assert "视频:" in captured.out
    assert "窗口数:" in captured.out


def test_dry_run_creates_storage_without_calling_client(
    app_config: AppConfig,
) -> None:
    video_path = _make_test_video(Path(app_config.storage.output_root) / "dry.mp4", duration=5.0)
    fake_client = FakeQwenClient(response_text="{}")

    pipeline = StreamingVideoPipeline(app_config, video_path, client=fake_client)
    storage = pipeline.run(dry_run=True)
    assert storage is not None
    assert storage.run_dir.exists()
    assert len(fake_client.calls) == 0
    windows = _read_jsonl(storage.run_dir / "windows.jsonl")
    assert len(windows) > 0
    assert len(_read_jsonl(storage.run_dir / "observations.jsonl")) == 0
    assert len(_read_jsonl(storage.run_dir / "api_metrics.jsonl")) == 0
    assert len(_read_jsonl(storage.run_dir / "errors.jsonl")) == 0


def test_normal_run_with_fake_client_writes_observations(
    app_config: AppConfig,
    valid_response_text: str,
) -> None:
    video_path = _make_test_video(Path(app_config.storage.output_root) / "normal.mp4", duration=5.0)
    fake_client = FakeQwenClient(response_text=valid_response_text)
    pipeline = StreamingVideoPipeline(app_config, video_path, client=fake_client)
    storage = pipeline.run()
    assert storage is not None

    observations = _read_jsonl(storage.run_dir / "observations.jsonl")
    assert len(observations) > 0
    assert observations[0]["window"]["global_index"] == 0
    assert observations[0]["summary"] == "Test summary."

    metrics = _read_jsonl(storage.run_dir / "api_metrics.jsonl")
    assert len(metrics) == len(observations)

    errors = _read_jsonl(storage.run_dir / "errors.jsonl")
    assert len(errors) == 0


def test_window_selection_limits_processed_windows(
    app_config: AppConfig,
) -> None:
    video_path = _make_test_video(Path(app_config.storage.output_root) / "select.mp4", duration=5.0)
    pipeline = StreamingVideoPipeline(app_config, video_path)
    storage = pipeline.run(dry_run=True, max_windows=2)
    assert storage is not None
    windows = _read_jsonl(storage.run_dir / "windows.jsonl")
    assert len(windows) == 2


def test_pipeline_continues_after_single_window_failure(
    app_config: AppConfig,
) -> None:
    """A failing window records an error and the pipeline continues."""
    video_path = _make_test_video(Path(app_config.storage.output_root) / "fail.mp4", duration=5.0)

    class FailingClient:
        def infer(self, system_prompt: str, user_prompt: str, images: list[str]) -> Any:
            raise RuntimeError("simulated inference failure")

    pipeline = StreamingVideoPipeline(app_config, video_path, client=FailingClient())
    storage = pipeline.run()
    assert storage is not None

    errors = _read_jsonl(storage.run_dir / "errors.jsonl")
    assert len(errors) > 0
    assert errors[0]["error_type"] == "RuntimeError"
    assert "simulated inference failure" in errors[0]["error_message"]

    metadata = json.loads((storage.run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert metadata["final_stats"]["error_count"] > 0


def test_keyboard_interrupt_propagates(
    app_config: AppConfig,
) -> None:
    """The pipeline must not swallow KeyboardInterrupt."""
    video_path = _make_test_video(Path(app_config.storage.output_root) / "interrupt.mp4", duration=5.0)

    class InterruptingClient:
        def infer(self, system_prompt: str, user_prompt: str, images: list[str]) -> Any:
            raise KeyboardInterrupt

    pipeline = StreamingVideoPipeline(app_config, video_path, client=InterruptingClient())
    with pytest.raises(KeyboardInterrupt):
        pipeline.run()


def test_previous_summary_carried_between_windows(
    app_config: AppConfig,
    valid_response_text: str,
) -> None:
    video_path = _make_test_video(Path(app_config.storage.output_root) / "state.mp4", duration=5.0)
    fake_client = FakeQwenClient(response_text=valid_response_text)
    pipeline = StreamingVideoPipeline(app_config, video_path, client=fake_client)
    pipeline.run()
    assert pipeline._previous_summary == "Test summary."
    assert len(pipeline._previous_entities) == 1
    assert pipeline._previous_entities[0]["candidate_global_id"] == "person_1"


def test_no_state_does_not_carry_summary(
    app_config: AppConfig,
    valid_response_text: str,
) -> None:
    video_path = _make_test_video(Path(app_config.storage.output_root) / "no_state.mp4", duration=5.0)
    fake_client = FakeQwenClient(response_text=valid_response_text)
    pipeline = StreamingVideoPipeline(app_config, video_path, client=fake_client)
    pipeline.run(carry_previous_state=False)
    assert pipeline._previous_summary is None
    assert pipeline._previous_entities == []


def test_missing_client_raises_on_normal_run(
    app_config: AppConfig,
) -> None:
    video_path = _make_test_video(Path(app_config.storage.output_root) / "missing_client.mp4", duration=2.0)
    pipeline = StreamingVideoPipeline(app_config, video_path, client=None)
    storage = pipeline.run()
    assert storage is not None
    errors = _read_jsonl(storage.run_dir / "errors.jsonl")
    assert len(errors) > 0
    assert errors[0]["error_type"] == "RuntimeError"
