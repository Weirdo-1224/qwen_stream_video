"""Qwen API client for raw vision-language inference.

The :class:`QwenClient` isolates OpenAI-compatible Qwen/DashScope requests and
returns a :class:`RawInferenceResult` containing the raw text plus request
metrics. It only retries transient network errors, HTTP 429 and 5xx responses.

A :class:`LocalTransformersClient` is provided for local Qwen3-VL inference via
Hugging Face transformers.

A :class:`FakeQwenClient` is provided for offline tests and local development.
"""

from __future__ import annotations

import base64
import time
import uuid
from io import BytesIO
from typing import Any

import openai
from openai import APIConnectionError, APIStatusError, APITimeoutError
from pydantic import BaseModel, ConfigDict, Field

from ..config import ModelConfig
from ..exceptions import InferenceNetworkError, InferenceRateLimitError, InferenceServerError

try:
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor

    try:
        # transformers >= 5.0 renamed AutoModelForVision2Seq to AutoModelForImageTextToText
        from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
    except ImportError:
        from transformers import AutoModelForVision2Seq

    _HAS_LOCAL_DEPS = True
except ImportError:  # pragma: no cover - optional local dependencies
    Image = None
    process_vision_info = None
    AutoModelForVision2Seq = None
    AutoProcessor = None
    _HAS_LOCAL_DEPS = False


class RawInferenceResult(BaseModel):
    """Raw response and request metrics from a vision-language model call."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    raw_text: str
    resolved_model: str
    latency_seconds: float
    request_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    attempt_count: int


class QwenClient:
    """Thin wrapper around the OpenAI SDK for Qwen vision requests."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the client from model configuration.

        Args:
            config: Resolved model configuration including API key, base URL,
                timeout and retry settings.
        """
        self.config = config
        self._client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )

    def infer(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[str],
    ) -> RawInferenceResult:
        """Call the vision model and return the raw response with metrics.

        Retries are attempted only for network failures, HTTP 429 and HTTP 5xx.
        The total number of attempts is ``network_retries + 1``.

        Args:
            system_prompt: System instructions for the model.
            user_prompt: User text prompt describing the current window.
            images: List of image Data URLs to send as vision input.

        Returns:
            A :class:`RawInferenceResult` with the model's raw text output.

        Raises:
            InferenceNetworkError: After exhausting retries for timeout or
                connection failures.
            InferenceRateLimitError: After exhausting retries for HTTP 429.
            InferenceServerError: After exhausting retries for HTTP 5xx.
            APIStatusError: For non-retryable HTTP client errors (4xx other than
                429), raised on the first attempt.
        """
        messages = self._build_messages(system_prompt, user_prompt, images)
        max_attempts = self.config.network_retries + 1
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            start = time.perf_counter()
            try:
                request_kwargs: dict[str, Any] = {
                    "model": self.config.name,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                }
                if self.config.structured_json:
                    request_kwargs["response_format"] = {"type": "json_object"}
                response = self._client.chat.completions.create(
                    **request_kwargs,
                )
                latency_seconds = time.perf_counter() - start
                raw_text = response.choices[0].message.content or ""
                usage = response.usage
                return RawInferenceResult(
                    raw_text=raw_text,
                    resolved_model=self.config.name,
                    latency_seconds=latency_seconds,
                    request_id=response.id,
                    input_tokens=usage.prompt_tokens if usage else None,
                    output_tokens=usage.completion_tokens if usage else None,
                    attempt_count=attempt,
                )
            except APITimeoutError as exc:
                last_error = InferenceNetworkError(
                    f"Inference request timed out (attempt {attempt}/{max_attempts})"
                )
                last_error.__cause__ = exc
            except APIConnectionError as exc:
                last_error = InferenceNetworkError(
                    f"Inference connection failed (attempt {attempt}/{max_attempts})"
                )
                last_error.__cause__ = exc
            except APIStatusError as exc:
                if exc.status_code == 429:
                    last_error = InferenceRateLimitError(
                        f"Inference rate limited (attempt {attempt}/{max_attempts})"
                    )
                    last_error.__cause__ = exc
                elif exc.status_code >= 500:
                    last_error = InferenceServerError(
                        f"Inference server error {exc.status_code} "
                        f"(attempt {attempt}/{max_attempts})"
                    )
                    last_error.__cause__ = exc
                else:
                    # Non-retryable client error (e.g. 400, 401, 403).
                    raise

        # All retryable attempts exhausted.
        assert last_error is not None
        raise last_error

    @staticmethod
    def _build_messages(
        system_prompt: str,
        user_prompt: str,
        images: list[str],
    ) -> list[dict[str, Any]]:
        """Build the OpenAI chat-messages payload with text and images."""
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for data_url in images:
            content.append({"type": "image_url", "image_url": {"url": data_url}})

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]


class LocalTransformersClient:
    """Local Qwen3-VL inference via Hugging Face transformers."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the client without loading the model yet.

        Args:
            config: Resolved model configuration including the local model path,
                device, dtype, and quantization settings.
        """
        if not _HAS_LOCAL_DEPS:
            raise RuntimeError(
                "Local inference dependencies are missing. "
                'Install them with: pip install -e ".[local]"'
            )
        if config.provider != "local_transformers":
            raise ValueError(
                f"LocalTransformersClient requires provider=local_transformers, "
                f"got {config.provider}"
            )
        self.config = config
        self._processor: AutoProcessor | None = None
        self._model: AutoModelForVision2Seq | None = None

    def _load(self) -> None:
        """Lazy-load the processor and model on first inference."""
        if self._model is not None:
            return

        model_path = self.config.local_model_path
        if not model_path:
            raise ValueError("local_model_path is not configured")

        torch_dtype = self._torch_dtype()
        device_map = self._device_map()

        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "trust_remote_code": self.config.trust_remote_code,
        }
        # Only pass quantization flags when enabled; some local model classes
        # (e.g. Qwen3-VL) do not accept them as ``False``.
        if self.config.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        if self.config.load_in_4bit:
            model_kwargs["load_in_4bit"] = True

        self._processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        self._model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            **model_kwargs,
        )

    def _torch_dtype(self) -> Any:
        """Return the torch dtype object matching the configured dtype name."""
        import torch

        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return mapping[self.config.torch_dtype]

    def _device_map(self) -> str:
        """Return the device_map argument for transformers model loading."""
        if self.config.device in ("auto", "cuda"):
            return "auto"
        return "cpu"

    @staticmethod
    def _decode_data_url(data_url: str) -> Image.Image:
        """Decode a base64 data URL into a PIL RGB image."""
        prefix = "data:image/"
        if not data_url.startswith(prefix):
            raise ValueError(f"Unsupported image data URL: {data_url[:40]}...")

        try:
            meta, b64 = data_url.split(",", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid image data URL: {data_url[:40]}...") from exc

        if "base64" not in meta:
            raise ValueError(f"Only base64 image data URLs are supported: {meta}")

        raw = base64.b64decode(b64)
        return Image.open(BytesIO(raw)).convert("RGB")

    def _build_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[Image.Image],
    ) -> list[dict[str, Any]]:
        """Build a Qwen3-VL chat message list with system + user images/text."""
        content: list[dict[str, Any]] = []
        for image in images:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": user_prompt})

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

    def infer(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[str],
    ) -> RawInferenceResult:
        """Run local Qwen3-VL inference and return raw text + metrics."""
        self._load()

        pil_images = [self._decode_data_url(url) for url in images]
        messages = self._build_messages(system_prompt, user_prompt, pil_images)

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)

        start = time.perf_counter()
        generated_ids = self._model.generate(
            **inputs,
            max_new_tokens=self.config.max_tokens,
            do_sample=self.config.temperature > 0,
            temperature=(
                self.config.temperature if self.config.temperature > 0 else None
            ),
        )
        latency_seconds = time.perf_counter() - start

        input_token_count = int(inputs.input_ids.shape[1])
        generated_ids_trimmed = generated_ids[:, input_token_count:]
        output_token_count = int(generated_ids.shape[1] - input_token_count)

        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return RawInferenceResult(
            raw_text=output_text,
            resolved_model=self.config.name,
            latency_seconds=latency_seconds,
            request_id=uuid.uuid4().hex,
            input_tokens=input_token_count,
            output_tokens=output_token_count,
            attempt_count=1,
        )


class FakeQwenClient(BaseModel):
    """Offline stand-in for :class:`QwenClient`.

    Returns a fixed response on every call and records inputs for assertions in
    tests.
    """

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    response_text: str = "{}"
    latency_seconds: float = 0.0
    request_id: str = "fake-request-id"
    resolved_model: str = "fake-qwen-model"
    input_tokens: int = 0
    output_tokens: int = 0
    calls: list[dict[str, Any]] = Field(default_factory=list)

    def infer(
        self,
        system_prompt: str,
        user_prompt: str,
        images: list[str],
    ) -> RawInferenceResult:
        """Return a canned response and record the call arguments."""
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "images": images,
            }
        )
        return RawInferenceResult(
            raw_text=self.response_text,
            resolved_model=self.resolved_model,
            latency_seconds=self.latency_seconds,
            request_id=self.request_id,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            attempt_count=1,
        )
