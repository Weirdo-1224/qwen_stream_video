"""Unit tests for the Pydantic configuration system."""

from __future__ import annotations

from pathlib import Path

import pytest

from qwen_stream_video.config import (
    AppConfig,
    ExperimentConfig,
    ModelConfig,
    ObservationConfig,
    RuntimeConfig,
    SamplingConfig,
    StorageConfig,
    VideoConfig,
    load_config,
    summarize_config,
)
from qwen_stream_video.exceptions import ConfigurationError

DEFAULTS = AppConfig()


def _write_yaml(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_default_config_is_valid() -> None:
    config = load_config()
    assert config.experiment.name == "incremental_observation_v1"
    assert config.video.window_seconds == 6.0
    assert config.sampling.jpeg_quality == 80
    assert config.model.provider == "dashscope"
    assert config.model.name == "qwen3-vl-plus"
    assert config.model.source == "default"
    assert config.observation.schema_version == "1.0"
    assert config.runtime.max_windows is None
    assert config.storage.output_root == "outputs"


def test_load_valid_config(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
experiment:
  name: test_exp
  seed: 7
video:
  window_seconds: 4.0
  stride_seconds: 2.0
sampling:
  sample_fps: 2.0
  min_frames: 2
  max_frames: 6
  max_image_side: 512
  jpeg_quality: 90
model:
  provider: dashscope
  name: qwen-vl-test
  max_tokens: 800
  timeout_seconds: 60
  network_retries: 1
observation:
  require_evidence_frames: false
runtime:
  realtime: true
  max_windows: 10
storage:
  output_root: test_outputs
  save_raw_responses: false
""",
    )

    config = load_config(config_path=path)
    assert config.experiment == ExperimentConfig(name="test_exp", seed=7)
    assert config.video == VideoConfig(window_seconds=4.0, stride_seconds=2.0)
    assert config.sampling == SamplingConfig(
        sample_fps=2.0,
        min_frames=2,
        max_frames=6,
        max_image_side=512,
        jpeg_quality=90,
    )
    assert config.model == ModelConfig(
        provider="dashscope",
        name="qwen-vl-test",
        api_key=None,
        base_url=None,
        temperature=0.0,
        max_tokens=800,
        timeout_seconds=60,
        network_retries=1,
        source="yaml",
    )
    assert config.observation == ObservationConfig(
        schema_version="1.0",
        require_evidence_frames=False,
        use_candidate_global_ids=True,
    )
    assert config.runtime == RuntimeConfig(
        realtime=True,
        carry_previous_state=True,
        save_sampled_frames=False,
        max_windows=10,
    )
    assert config.storage == StorageConfig(
        output_root="test_outputs",
        save_raw_responses=False,
        save_sampled_frames=False,
    )


def test_invalid_window_seconds(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "video:\n  window_seconds: -1.0\n")
    with pytest.raises(ConfigurationError):
        load_config(config_path=path)


def test_invalid_stride_seconds(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "video:\n  stride_seconds: 0.0\n")
    with pytest.raises(ConfigurationError):
        load_config(config_path=path)


def test_invalid_frame_limits(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        "sampling:\n  min_frames: 10\n  max_frames: 4\n",
    )
    with pytest.raises(ConfigurationError):
        load_config(config_path=path)


def test_min_frames_below_one(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "sampling:\n  min_frames: 0\n")
    with pytest.raises(ConfigurationError):
        load_config(config_path=path)


def test_jpeg_quality_out_of_range(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "sampling:\n  jpeg_quality: 101\n")
    with pytest.raises(ConfigurationError):
        load_config(config_path=path)


def test_environment_model_override(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "model:\n  name: yaml-model\n")
    env = {
        "DASHSCOPE_API_KEY": "sk-test",
        "DASHSCOPE_BASE_URL": "https://test.example.com/v1",
        "QWEN_MODEL": "env-model",
    }

    config = load_config(config_path=path, env=env)
    assert config.model.name == "env-model"
    assert config.model.api_key == "sk-test"
    assert config.model.base_url == "https://test.example.com/v1"
    assert config.model.source == "environment"


def test_cli_override_has_highest_priority(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "model:\n  name: yaml-model\n  temperature: 0.5\n")
    env = {"QWEN_MODEL": "env-model"}
    cli_overrides = {"model.name": "cli-model", "video.window_seconds": 10.0}

    config = load_config(config_path=path, env=env, cli_overrides=cli_overrides)
    assert config.model.name == "cli-model"
    assert config.model.temperature == 0.5
    assert config.video.window_seconds == 10.0
    assert config.model.source == "cli"


def test_missing_config_file() -> None:
    with pytest.raises(ConfigurationError):
        load_config(config_path="nonexistent-config.yaml")


def test_invalid_yaml(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "video:\n  window_seconds: [")
    with pytest.raises(ConfigurationError):
        load_config(config_path=path)


def test_config_summary_hides_api_key() -> None:
    config = load_config(
        env={
            "DASHSCOPE_API_KEY": "super-secret-key",
            "QWEN_MODEL": "qwen-vl",
        },
    )
    summary = summarize_config(config, video_path="videos/demo.mp4", source="test")
    assert "super-secret-key" not in summary
    assert "API Key: 已配置" in summary
    assert "videos/demo.mp4" in summary


def test_config_summary_shows_unconfigured_key() -> None:
    config = load_config()
    summary = summarize_config(config)
    assert "API Key: 未配置" in summary


def test_video_metadata_populates_video_context(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
video_metadata:
  video_name: demo
  video_category: breaker
  task_background: training
""",
    )
    config = load_config(config_path=path)
    assert config.video_context.video_name == "demo"
    assert config.video_context.video_category == "breaker"
    assert config.video_context.task_background == "training"


def test_local_transformers_provider_requires_model_path(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path,
        """
model:
  provider: local_transformers
  local_model_path: ""
""",
    )
    with pytest.raises(ConfigurationError):
        load_config(config_path=path)


def test_local_transformers_provider_accepts_valid_path() -> None:
    config = AppConfig.model_validate(
        {
            "model": {
                "provider": "local_transformers",
                "local_model_path": "/home/Datasets/Hf_model/Qwen3-VL-8B-Instruct",
                "device": "auto",
                "torch_dtype": "bfloat16",
            }
        }
    )
    assert config.model.provider == "local_transformers"
    assert config.model.local_model_path == "/home/Datasets/Hf_model/Qwen3-VL-8B-Instruct"
    assert config.model.api_key is None


def test_unknown_yaml_keys_are_ignored(tmp_path: Path) -> None:
    path = _write_yaml(tmp_path, "unknown_section:\n  value: demo\n")
    config = load_config(config_path=path)
    assert config.video.window_seconds == DEFAULTS.video.window_seconds
