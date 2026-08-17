"""Strict, layered application configuration for Stage 2."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .exceptions import ConfigurationError


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="incremental_state_v2", min_length=1)
    seed: int = Field(default=42, ge=0)


class VideoConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_seconds: float = Field(default=6.0, gt=0)
    stride_seconds: float = Field(default=3.0, gt=0)
    warmup_windows: int = Field(default=0, ge=0)


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_fps: float = Field(default=1.0, gt=0)
    min_frames: int = Field(default=4, ge=1)
    max_frames: int = Field(default=12, ge=1)
    max_image_side: int = Field(default=768, gt=0)
    jpeg_quality: int = Field(default=80, ge=1, le=100)

    @model_validator(mode="after")
    def _frame_limits_are_ordered(self) -> SamplingConfig:
        if self.max_frames < self.min_frames:
            raise ValueError("max_frames must be greater than or equal to min_frames")
        return self


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["dashscope", "local_transformers"] = "dashscope"
    name: str = Field(default="qwen3-vl-plus", min_length=1)
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=3000, ge=1)
    timeout_seconds: float = Field(default=120.0, gt=0)
    network_retries: int = Field(default=2, ge=0)
    structured_json: bool = True
    source: str = Field(default="default", min_length=1)
    local_model_path: str | None = Field(default=None, min_length=1)
    device: Literal["auto", "cuda", "cpu"] = "auto"
    torch_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    max_model_len: int | None = Field(default=None, ge=1)
    trust_remote_code: bool = True

    @model_validator(mode="after")
    def _check_local_model_path(self) -> ModelConfig:
        if self.provider == "local_transformers" and not self.local_model_path:
            raise ValueError("local_model_path is required when provider is local_transformers")
        return self


class ObservationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0", "2.0"] = "2.0"
    require_evidence_frames: bool = True
    use_candidate_global_ids: bool = True
    allow_candidate_global_ids: bool = True
    context_policy: Literal["visual_only", "weak_context", "task_conditioned"] = "visual_only"

    @model_validator(mode="before")
    @classmethod
    def _normalize_candidate_aliases(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        result = dict(value)
        old = result.get("use_candidate_global_ids")
        new = result.get("allow_candidate_global_ids")
        if old is not None and new is not None and old != new:
            raise ValueError("candidate global ID flags must match")
        if old is not None and new is None:
            result["allow_candidate_global_ids"] = old
        if new is not None and old is None:
            result["use_candidate_global_ids"] = new
        return result


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    realtime: bool = False
    carry_previous_state: bool = True  # Stage1 compatibility switch
    save_sampled_frames: bool = False
    max_windows: int | None = Field(default=None, ge=1)


class StorageConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_root: str = Field(default="outputs", min_length=1)
    save_raw_responses: bool = True
    save_sampled_frames: bool = False
    save_entity_resolutions: bool = True
    save_state_events: bool = True
    save_state_deltas: bool = True
    save_state_snapshots: bool = True


class StateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    snapshot_interval_windows: int = Field(default=10, ge=1)
    fail_on_state_error: bool = False


class SceneTrackerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    camera_change_starts_new_scene: bool = True
    preserve_entities_across_scenes: bool = True


class EntityRegistryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    confident_match_threshold: float = Field(default=0.78, ge=0.0, le=1.0)
    ambiguous_match_threshold: float = Field(default=0.58, ge=0.0, le=1.0)
    max_missing_windows: int = Field(default=10, ge=0)
    temporary_entity_prefix: str = Field(default="temp", min_length=1)
    candidate_hint_weight: float = Field(default=0.05, ge=0.0, le=1.0)
    allow_delayed_merge: bool = True
    delayed_merge_support_windows: int = Field(default=2, ge=1)
    ambiguous_margin: float = Field(default=0.08, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _thresholds_are_ordered(self) -> EntityRegistryConfig:
        if self.ambiguous_match_threshold >= self.confident_match_threshold:
            raise ValueError(
                "ambiguous_match_threshold must be lower than confident_match_threshold"
            )
        return self


class ActionTrackerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    continue_max_gap_windows: int = Field(default=1, ge=0)
    end_missing_windows: int = Field(default=2, ge=1)
    repeat_action_min_gap_seconds: float = Field(default=5.0, ge=0.0)
    instant_actions: list[str] = Field(
        default_factory=lambda: ["press", "switch", "pick_up", "put_down", "hand_over", "receive"]
    )


class TransitionEngineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    high_confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    medium_confidence_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    confirm_support_windows: int = Field(default=2, ge=1)
    max_pending_gap_windows: int = Field(default=1, ge=0)
    require_action_support_for_transition: bool = True

    @model_validator(mode="after")
    def _confidence_thresholds_are_ordered(self) -> TransitionEngineConfig:
        if self.medium_confidence_threshold >= self.high_confidence_threshold:
            raise ValueError(
                "medium_confidence_threshold must be lower than high_confidence_threshold"
            )
        return self


class ContextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_entities: int = Field(default=15, ge=1)
    recent_window_count: int = Field(default=5, ge=1)
    max_active_actions: int = Field(default=8, ge=1)
    max_pending_attributes: int = Field(default=8, ge=1)
    max_serialized_characters: int = Field(default=6000, ge=100)


class VideoContextConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_name: str | None = None
    video_category: str | None = None
    task_background: str | None = None


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    video: VideoConfig = Field(default_factory=VideoConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    observation: ObservationConfig = Field(default_factory=ObservationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    state: StateConfig = Field(default_factory=StateConfig)
    scene_tracker: SceneTrackerConfig = Field(default_factory=SceneTrackerConfig)
    entity_registry: EntityRegistryConfig = Field(default_factory=EntityRegistryConfig)
    action_tracker: ActionTrackerConfig = Field(default_factory=ActionTrackerConfig)
    transition_engine: TransitionEngineConfig = Field(default_factory=TransitionEngineConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    video_context: VideoContextConfig = Field(
        default_factory=VideoContextConfig, alias="video_metadata"
    )


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in dict(updates).items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _apply_env_overrides(env: Mapping[str, str]) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if env.get("DASHSCOPE_API_KEY"):
        updates.setdefault("model", {})["api_key"] = env["DASHSCOPE_API_KEY"]
    if env.get("DASHSCOPE_BASE_URL"):
        updates.setdefault("model", {})["base_url"] = env["DASHSCOPE_BASE_URL"]
    if env.get("QWEN_MODEL"):
        updates.setdefault("model", {})["name"] = env["QWEN_MODEL"]
    return updates


def _apply_dotted_overrides(base: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    for key, value in overrides.items():
        target = base
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value


def load_config(
    config_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
) -> AppConfig:
    """Resolve defaults, YAML, environment variables, then CLI overrides."""
    merged = AppConfig().model_dump(by_alias=False)
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
            if not isinstance(yaml_data, Mapping):
                raise ConfigurationError(f"Configuration file {path} must contain a mapping")
            yaml_data = dict(yaml_data)
            if "video_metadata" in yaml_data and "video_context" not in yaml_data:
                yaml_data["video_context"] = yaml_data.pop("video_metadata")
            _deep_update(merged, yaml_data)
            if isinstance(yaml_data.get("model"), Mapping):
                model_source = "yaml"
    env = os.environ if env is None else env
    env_updates = _apply_env_overrides(env)
    if env_updates.get("model"):
        model_source = "environment"
    _deep_update(merged, env_updates)
    if cli_overrides:
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
    """Return a secret-safe startup summary."""
    lines = ["最终配置摘要:", "=" * 40]
    lines.append(f"配置来源: {source or config.model.source}")
    if video_path is not None:
        lines.append(f"视频路径: {video_path}")
    lines.extend(
        [
            f"实验名称: {config.experiment.name}",
            f"模型: {config.model.provider}/{config.model.name}",
            f"API Key: {'已配置' if config.model.api_key else '未配置'}",
            f"窗口大小: {config.video.window_seconds}s",
            f"步长: {config.video.stride_seconds}s",
            f"采样率: {config.sampling.sample_fps} fps",
            f"输出目录: {config.storage.output_root}",
            f"Observation Schema: {config.observation.schema_version}",
            "State Schema: 2.0",
            f"状态维护: {'启用' if config.state.enabled else '禁用'}",
            f"Context Policy: {config.observation.context_policy}",
            f"Warmup 窗口数: {config.video.warmup_windows}",
            f"实体匹配阈值: {config.entity_registry.confident_match_threshold}",
            f"属性确认阈值: {config.transition_engine.high_confidence_threshold}",
            f"快照间隔: {config.state.snapshot_interval_windows}",
        ]
    )
    if config.video_context.video_name:
        lines.append(f"视频名称: {config.video_context.video_name}")
    if config.video_context.video_category:
        lines.append(f"视频类别: {config.video_context.video_category}")
    if config.video_context.task_background:
        lines.append(f"任务背景: {config.video_context.task_background}")
    return "\n".join(lines)
