"""Application configuration with strict Pydantic validation.

Configuration resolution order:

1. Code defaults
2. YAML config file
3. Environment variables
4. CLI overrides

Supported environment variables:

* ``DASHSCOPE_API_KEY`` -> ``model.api_key``
* ``DASHSCOPE_BASE_URL`` -> ``model.base_url``
* ``QWEN_MODEL`` -> ``model.name``
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .exceptions import ConfigurationError


class ExperimentConfig(BaseModel):
    """High-level experiment metadata."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(default="incremental_observation_v1", min_length=1)
    seed: int = Field(default=42, ge=0)


class VideoConfig(BaseModel):
    """Temporal windowing parameters."""

    model_config = ConfigDict(extra="ignore")

    window_seconds: float = Field(default=6.0, gt=0)
    stride_seconds: float = Field(default=3.0, gt=0)


class SamplingConfig(BaseModel):
    """Frame sampling and image encoding parameters."""

    model_config = ConfigDict(extra="ignore")

    sample_fps: float = Field(default=1.0, gt=0)
    min_frames: int = Field(default=4, ge=1)
    max_frames: int = Field(default=12, ge=1)
    max_image_side: int = Field(default=768, gt=0)
    jpeg_quality: int = Field(default=80, ge=1, le=100)

    @field_validator("max_frames")
    @classmethod
    def _max_frames_not_smaller_than_min(cls, value: int, info: Any) -> int:
        data = info.data
        min_frames = data.get("min_frames")
        if min_frames is not None and value < min_frames:
            raise ValueError("max_frames must be greater than or equal to min_frames")
        return value


class ModelConfig(BaseModel):
    """Model and API connection settings."""

    model_config = ConfigDict(extra="ignore")

    provider: str = Field(default="dashscope", min_length=1)
    name: str = Field(default="qwen3-vl-plus", min_length=1)
    api_key: str | None = Field(default=None)
    base_url: str | None = Field(default=None)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1200, ge=1)
    timeout_seconds: float = Field(default=120.0, gt=0)
    network_retries: int = Field(default=2, ge=0)
    source: str = Field(default="default", min_length=1)


class ObservationConfig(BaseModel):
    """Observation schema behaviour."""

    model_config = ConfigDict(extra="ignore")

    schema_version: str = Field(default="1.0", min_length=1)
    require_evidence_frames: bool = True
    use_candidate_global_ids: bool = True


class RuntimeConfig(BaseModel):
    """Runtime execution switches."""

    model_config = ConfigDict(extra="ignore")

    realtime: bool = False
    carry_previous_state: bool = True
    save_sampled_frames: bool = False
    max_windows: int | None = Field(default=None, ge=1)


class StorageConfig(BaseModel):
    """Output directory and artefact persistence settings."""

    model_config = ConfigDict(extra="ignore")

    output_root: str = Field(default="outputs", min_length=1)
    save_raw_responses: bool = True
    save_sampled_frames: bool = False


class VideoContextConfig(BaseModel):
    """Optional human-readable context about the video being analysed."""

    model_config = ConfigDict(extra="ignore")

    video_name: str | None = None
    video_category: str | None = None
    task_background: str | None = None


class AppConfig(BaseModel):
    """Top-level application configuration."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    observation: ObservationConfig = Field(default_factory=ObservationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    video_context: VideoContextConfig = Field(
        default_factory=VideoContextConfig, alias="video_metadata"
    )


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``updates`` into ``base`` in place."""
    for key, value in dict(updates).items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _apply_env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    """Translate known environment variables into a nested config update dict."""
    updates: dict[str, Any] = {}

    api_key = env.get("DASHSCOPE_API_KEY")
    if api_key is not None and api_key != "":
        updates.setdefault("model", {})["api_key"] = api_key

    base_url = env.get("DASHSCOPE_BASE_URL")
    if base_url is not None and base_url != "":
        updates.setdefault("model", {})["base_url"] = base_url

    model_name = env.get("QWEN_MODEL")
    if model_name is not None and model_name != "":
        updates.setdefault("model", {})["name"] = model_name

    return updates


def _apply_dotted_overrides(base: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    """Apply dot-separated keys such as ``video.window_seconds`` to a nested dict."""
    for key, value in overrides.items():
        parts = key.split(".")
        target = base
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value


def load_config(
    config_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> AppConfig:
    """Build the resolved application configuration.

    Resolution order is defaults -> YAML -> environment variables -> CLI overrides.
    Validation failures are raised as :class:`ConfigurationError`.
    """
    # Start from code defaults so every key has a value.
    merged = AppConfig().model_dump()
    model_source = "default"

    if config_path is not None:
        path = Path(config_path)
        if not path.is_file():
            raise ConfigurationError(f"Configuration file not found: {path}")
        try:
            with path.open("r", encoding="utf-8") as handle:
                yaml_data = yaml.safe_load(handle)
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc
        except OSError as exc:
            raise ConfigurationError(f"Cannot read configuration file {path}: {exc}") from exc

        if yaml_data:
            if not isinstance(yaml_data, dict):
                raise ConfigurationError(f"Configuration file {path} must contain a mapping")
            merged = _deep_update(merged, yaml_data)
            if isinstance(yaml_data.get("model"), dict):
                model_source = "yaml"

    if env is None:
        env = os.environ
    env_overrides = _apply_env_overrides(env)
    if env_overrides.get("model"):
        model_source = "environment"
    merged = _deep_update(merged, env_overrides)

    if cli_overrides:
        merged = _deep_update(merged, {})
        _apply_dotted_overrides(merged, cli_overrides)
        if any(key.startswith("model.") for key in cli_overrides):
            model_source = "cli"

    merged.setdefault("model", {})["source"] = model_source

    try:
        return AppConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigurationError(f"Configuration validation failed:\n{exc}") from exc


def summarize_config(
    config: AppConfig,
    video_path: str | Path | None = None,
    source: str | None = None,
) -> str:
    """Return a human-readable configuration summary that hides secrets."""
    lines = ["最终配置摘要:", "=" * 40]

    if source:
        lines.append(f"配置来源: {source}")
    else:
        lines.append("配置来源: 代码默认值")

    if video_path is not None:
        lines.append(f"视频路径: {video_path}")

    lines.extend([
        f"实验名称: {config.experiment.name}",
        f"模型: {config.model.provider}/{config.model.name}",
        f"API Key: {'已配置' if config.model.api_key else '未配置'}",
        f"窗口大小: {config.video.window_seconds}s",
        f"步长: {config.video.stride_seconds}s",
        f"采样率: {config.sampling.sample_fps} fps",
        f"每窗口帧数范围: [{config.sampling.min_frames}, {config.sampling.max_frames}]",
        f"输出目录: {config.storage.output_root}",
        f"实时模式: {config.runtime.realtime}",
    ])

    if config.video_context.video_name:
        lines.append(f"视频名称: {config.video_context.video_name}")
    if config.video_context.video_category:
        lines.append(f"视频类别: {config.video_context.video_category}")

    return "\n".join(lines)
