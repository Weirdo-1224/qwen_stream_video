"""Qwen API client for raw vision-language inference.

The :class:`QwenClient` isolates OpenAI-compatible Qwen/DashScope requests and
returns a :class:`RawInferenceResult` containing the raw text plus request
metrics.  It only retries transient network errors, HTTP 429 and 5xx responses.

A :class:`FakeQwenClient` is provided for offline tests and local development.
"""

from __future__ import annotations

import time
from typing import Any

import openai
from openai import APIConnectionError, APIStatusError, APITimeoutError
from pydantic import BaseModel, ConfigDict, Field

from ..config import ModelConfig
from ..exceptions import InferenceNetworkError, InferenceRateLimitError, InferenceServerError


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
                response = self._client.chat.completions.create(
                    model=self.config.name,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
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
