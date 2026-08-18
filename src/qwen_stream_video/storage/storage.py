"""Persistent run storage for incremental video observations.

Each run gets a unique, timestamped output directory. The storage layer writes
configuration, video metadata, windows, validated observations, per-window API
metrics, errors, and optional raw model responses and sampled frames.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

from ..config import AppConfig
from ..domain import ObservationBatch
from ..inference import PromptBuilder, RawInferenceResult
from ..video import SampledFrame, VideoMetadata, VideoWindow


def _sanitize_experiment_name(name: str) -> str:
    """Return a filesystem-safe version of the experiment name."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return safe[:50] or "experiment"


def _generate_unique_run_id(experiment_name: str, output_root: str | Path) -> str:
    """Generate a unique run id of the form YYYYMMDD_HHMMSS_<experiment>_<hash>."""
    output_root = Path(output_root)
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_name = _sanitize_experiment_name(experiment_name)
    for _ in range(100):
        short_hash = secrets.token_hex(3)
        run_id = f"{timestamp}_{safe_name}_{short_hash}"
        run_dir = output_root / run_id
        if not run_dir.exists():
            return run_id
    raise RuntimeError("Unable to generate a unique run id after 100 attempts")


def _file_sha256(path: str | Path) -> str:
    """Return the SHA256 hex digest of a file's contents."""
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _video_hash(video_metadata: VideoMetadata) -> str:
    """Return a deterministic hash identifying the video metadata."""
    content = "|".join(
        [
            video_metadata.path,
            str(video_metadata.fps),
            str(video_metadata.frame_count),
            str(video_metadata.duration_seconds),
            str(video_metadata.width),
            str(video_metadata.height),
        ]
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _config_sha256(config: AppConfig) -> str:
    """Return the SHA256 hex digest of the resolved, redacted configuration."""
    config_dict = config.model_dump()
    if config_dict.get("model", {}).get("api_key"):
        config_dict["model"]["api_key"] = "***REDACTED***"
    return hashlib.sha256(
        json.dumps(config_dict, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _prompt_sha256(text: str) -> str:
    """Return the SHA256 hex digest of a prompt template."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _package_version(package_name: str) -> str | None:
    """Return the installed version of a package, or None if not available."""
    try:
        return version(package_name)
    except PackageNotFoundError:  # pragma: no cover - defensive
        return None


def _git_commit() -> str | None:
    """Return the current git commit hash, or None if not in a git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):  # pragma: no cover
        return None


class RunStorage:
    """Manage output artefacts for a single analysis run."""

    def __init__(
        self,
        config: AppConfig,
        video_metadata: VideoMetadata,
        run_id: str | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        """Initialize storage and create the unique run directory.

        Args:
            config: Resolved application configuration.
            video_metadata: Metadata for the video being analysed.
            run_id: Optional explicit run id. When omitted, a unique id is
                generated automatically.
            prompt_builder: Prompt builder used for the run; used to record
                prompt template hashes in the run metadata.

        Raises:
            FileExistsError: If the requested run directory already exists.
        """
        self.config = config
        self.video_metadata = video_metadata
        self.prompt_builder = prompt_builder
        self.run_id = run_id or _generate_unique_run_id(
            config.experiment.name, config.storage.output_root
        )
        self.run_dir = Path(config.storage.output_root) / self.run_id
        self._ensure_unique_run_dir()

        self._raw_dir = self.run_dir / "raw_responses"
        self._frame_dir = self.run_dir / "sampled_frames"
        self._metadata_path = self.run_dir / "run_meta.json"
        self._metrics_path = self.run_dir / "api_metrics.jsonl"
        self._errors_path = self.run_dir / "errors.jsonl"
        self._observations_path = self.run_dir / "observations.jsonl"
        self._windows_path = self.run_dir / "windows.jsonl"

        self._window_count = 0
        self._observation_count = 0
        self._error_count = 0
        self._metrics_file = None
        self._errors_file = None
        self._observations_file = None
        self._start_time: datetime | None = None

    def _ensure_unique_run_dir(self) -> None:
        """Create the run directory, failing if it already exists."""
        if self.run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {self.run_dir}")
        self.run_dir.mkdir(parents=True)

    def initialize(self) -> None:
        """Write static run artefacts and open append-only JSONL files."""
        self._start_time = datetime.now(tz=timezone.utc)
        self._raw_dir.mkdir(exist_ok=True)
        self._frame_dir.mkdir(exist_ok=True)
        self._write_config()
        self._write_initial_metadata()
        self._metrics_file = self._metrics_path.open("w", encoding="utf-8")
        self._errors_file = self._errors_path.open("w", encoding="utf-8")
        self._observations_file = self._observations_path.open("w", encoding="utf-8")

    def write_windows(self, windows: list[VideoWindow]) -> None:
        """Write all video windows to ``windows.jsonl``."""
        with self._windows_path.open("w", encoding="utf-8") as handle:
            for window in windows:
                handle.write(window.model_dump_json() + "\n")

    def write_window_result(
        self,
        window: VideoWindow,
        sampled_frames: list[SampledFrame],
        raw_result: RawInferenceResult | None = None,
        observation: ObservationBatch | None = None,
        error: Exception | None = None,
        *,
        save_frames: bool | None = None,
        context_characters: int | None = None,
    ) -> None:
        """Persist one window's results.

        Validated observations are appended to ``observations.jsonl``. API
        metrics are appended to ``api_metrics.jsonl``. Errors are appended to
        ``errors.jsonl`` and reference the saved raw response when available.
        Raw responses and sampled frames are saved according to configuration
        or the ``save_frames`` override.

        Args:
            window: The video window being processed.
            sampled_frames: Frames sampled from the window.
            raw_result: Optional raw inference response.
            observation: Optional validated observation for the window.
            error: Optional exception raised while processing the window.
            save_frames: Override the configured sampled-frame persistence.
            context_characters: Length of the serialized context sent to the model.
        """
        self._window_count += 1
        raw_response_path = None

        if raw_result is not None and self.config.storage.save_raw_responses:
            raw_response_path = self._save_raw_response(window, raw_result)
        if raw_result is not None:
            self._write_metrics(window, raw_result, raw_response_path, context_characters)

        if observation is not None:
            self._write_observation(observation)

        if error is not None:
            self._write_error(window, error, raw_response_path)

        if self._should_save_frames(save_frames):
            self._save_sampled_frames(window, sampled_frames)

    def finalize(self, stats: dict[str, Any] | None = None) -> None:
        """Close JSONL files and update metadata with final statistics."""
        if self._metrics_file is not None:
            self._metrics_file.close()
            self._metrics_file = None
        if self._errors_file is not None:
            self._errors_file.close()
            self._errors_file = None
        if self._observations_file is not None:
            self._observations_file.close()
            self._observations_file = None

        final_stats = {
            "window_count": self._window_count,
            "observation_count": self._observation_count,
            "error_count": self._error_count,
            "processed_windows": self._window_count,
            "successful_windows": self._observation_count,
            "failed_windows": self._error_count,
        }
        if stats:
            final_stats.update(stats)
        self._write_final_metadata(final_stats)

    def update_run_meta(self, updates: dict[str, Any]) -> None:
        """Merge deterministic run-scope metadata before finalization."""
        if not self._metadata_path.exists():
            return
        with self._metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata.update(updates)
        with self._metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

    def _write_config(self) -> None:
        """Write the resolved configuration with secrets redacted."""
        config_dict = self.config.model_dump()
        if config_dict.get("model", {}).get("api_key"):
            config_dict["model"]["api_key"] = "***REDACTED***"
        with (self.run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config_dict, handle, sort_keys=True, allow_unicode=True)

    def _build_initial_metadata(self) -> dict[str, Any]:
        """Assemble the initial metadata record."""
        video = self.video_metadata
        model = self.config.model
        video_path = Path(video.path)
        try:
            video_sha256 = _file_sha256(video_path)
        except OSError:
            video_sha256 = ""

        prompt_builder = self.prompt_builder or PromptBuilder()
        system_prompt_sha256 = _prompt_sha256(prompt_builder.system_prompt)
        user_prompt_sha256 = _prompt_sha256(prompt_builder.user_prompt_template)

        return {
            "run_id": self.run_id,
            "experiment_name": self.config.experiment.name,
            "experiment_seed": self.config.experiment.seed,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "end_time": None,
            "video_path": video.path,
            "video_sha256": video_sha256,
            "video_hash": _video_hash(video),
            "video_duration_seconds": video.duration_seconds,
            "video_fps": video.fps,
            "video_frame_count": video.frame_count,
            "video_width": video.width,
            "video_height": video.height,
            "model_provider": model.provider,
            "model_name": model.name,
            "resolved_model": model.name,
            "model_source": model.source,
            "observation_schema_version": self.config.observation.schema_version,
            "state_schema_version": "2.0" if self.config.state.enabled else None,
            "state_enabled": self.config.state.enabled,
            "config_sha256": _config_sha256(self.config),
            "system_prompt_sha256": system_prompt_sha256,
            "user_prompt_sha256": user_prompt_sha256,
            "python_version": sys.version.split()[0],
            "package_versions": {
                "qwen_stream_video": _package_version("qwen-stream-video"),
                "pydantic": _package_version("pydantic"),
                "openai": _package_version("openai"),
                "numpy": _package_version("numpy"),
                "opencv_python_headless": _package_version("opencv-python-headless"),
            },
            "git_commit": _git_commit(),
        }

    def _write_initial_metadata(self) -> None:
        """Write the initial metadata file before any windows are processed."""
        metadata = self._build_initial_metadata()
        metadata["final_stats"] = {
            "window_count": 0,
            "observation_count": 0,
            "error_count": 0,
            "processed_windows": 0,
            "successful_windows": 0,
            "failed_windows": 0,
        }
        with self._metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

    def _write_final_metadata(self, stats: dict[str, Any]) -> None:
        """Update the metadata file with final statistics."""
        if self._metadata_path.exists():
            with self._metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        else:
            metadata = self._build_initial_metadata()
        metadata["end_time"] = datetime.now(tz=timezone.utc).isoformat()
        metadata["final_stats"] = stats
        with self._metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

    def _save_raw_response(
        self, window: VideoWindow, raw_result: RawInferenceResult
    ) -> str:
        """Save the raw model response and return its relative path."""
        filename = f"window_{window.run_index:04d}_{window.global_index:04d}.txt"
        path = self._raw_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            handle.write(raw_result.raw_text)
        return f"raw_responses/{filename}"

    def _save_sampled_frames(
        self, window: VideoWindow, sampled_frames: list[SampledFrame]
    ) -> None:
        """Save sampled frames as JPEG files under ``sampled_frames/``."""
        import cv2

        window_dir = self._frame_dir / f"window_{window.run_index:04d}_{window.global_index:04d}"
        window_dir.mkdir(exist_ok=True)
        for frame in sampled_frames:
            if frame.image is None:
                continue
            filename = f"frame_{frame.sample_index:03d}_{frame.timestamp_seconds:.3f}.jpg"
            path = window_dir / filename
            cv2.imwrite(str(path), frame.image)

    def _should_save_frames(self, override: bool | None) -> bool:
        """Return whether sampled frames should be persisted for this window."""
        if override is not None:
            return override
        return self.config.storage.save_sampled_frames or self.config.runtime.save_sampled_frames

    def _write_metrics(
        self,
        window: VideoWindow,
        raw_result: RawInferenceResult,
        raw_response_path: str | None,
        context_characters: int | None = None,
    ) -> None:
        """Append a metrics record to ``api_metrics.jsonl``."""
        record: dict[str, Any] = {
            "window_run_index": window.run_index,
            "window_global_index": window.global_index,
            "window_start_seconds": window.start_seconds,
            "window_end_seconds": window.end_seconds,
            "resolved_model": raw_result.resolved_model,
            "latency_seconds": raw_result.latency_seconds,
            "request_id": raw_result.request_id,
            "input_tokens": raw_result.input_tokens,
            "output_tokens": raw_result.output_tokens,
            "attempt_count": raw_result.attempt_count,
            "raw_response_path": raw_response_path,
        }
        if context_characters is not None:
            record["context_characters"] = context_characters
        self._metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._metrics_file.flush()

    def _write_observation(self, observation: ObservationBatch) -> None:
        """Append a validated observation batch to ``observations.jsonl``."""
        self._observations_file.write(observation.model_dump_json() + "\n")
        self._observations_file.flush()
        self._observation_count += 1

    def _write_error(
        self,
        window: VideoWindow,
        error: Exception,
        raw_response_path: str | None,
    ) -> None:
        """Append an error record to ``errors.jsonl``."""
        record = {
            "window_run_index": window.run_index,
            "window_global_index": window.global_index,
            "window_start_seconds": window.start_seconds,
            "window_end_seconds": window.end_seconds,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "raw_response_path": raw_response_path,
        }
        self._errors_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._errors_file.flush()
        self._error_count += 1
