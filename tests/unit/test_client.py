"""Unit tests for the Qwen API client."""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import pytest

from qwen_stream_video.config import ModelConfig
from qwen_stream_video.exceptions import (
    InferenceNetworkError,
    InferenceRateLimitError,
    InferenceServerError,
)
from qwen_stream_video.inference import FakeQwenClient, QwenClient, RawInferenceResult
from qwen_stream_video.inference.client import LocalTransformersClient


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        provider="dashscope",
        name="qwen3-vl-plus",
        api_key="sk-test",
        base_url="https://example.com/v1",
        network_retries=2,
        timeout_seconds=30.0,
    )


@pytest.fixture
def mock_openai():
    with patch("qwen_stream_video.inference.client.openai.OpenAI") as patched:
        yield patched


VALID_FAKE_RESPONSE = """{
  "schema_version": "1.0",
  "window": {"global_index": 0, "start_seconds": 0.0, "end_seconds": 1.0},
  "summary": "Test summary.",
  "scene": {"camera_change": false, "view_type": "unknown", "visibility": "unknown", "description": "test"},
  "entities": [],
  "actions": [],
  "attribute_observations": [],
  "uncertainties": []
}"""


def _make_success_response(raw_text: str) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = raw_text
    response.id = "req-123"
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    response.usage.total_tokens = 15
    return response


def _make_request() -> MagicMock:
    return MagicMock()


def test_fake_qwen_client_returns_canned_response_and_records_call() -> None:
    client = FakeQwenClient(
        response_text=VALID_FAKE_RESPONSE,
        latency_seconds=0.1,
        request_id="fake-req",
        resolved_model="fake-model",
        input_tokens=3,
        output_tokens=2,
    )
    result = client.infer("sys", "user", ["data:image/jpeg;base64,abc"])

    assert result.raw_text == VALID_FAKE_RESPONSE
    assert result.resolved_model == "fake-model"
    assert result.latency_seconds == 0.1
    assert result.request_id == "fake-req"
    assert result.input_tokens == 3
    assert result.output_tokens == 2
    assert result.attempt_count == 1
    assert len(client.calls) == 1
    assert client.calls[0]["system_prompt"] == "sys"
    assert client.calls[0]["user_prompt"] == "user"
    assert client.calls[0]["images"] == ["data:image/jpeg;base64,abc"]


def test_qwen_client_success(mock_openai: MagicMock, model_config: ModelConfig) -> None:
    mock_client = mock_openai.return_value
    mock_client.chat.completions.create.return_value = _make_success_response('{"ok": true}')

    client = QwenClient(model_config)
    result = client.infer("system text", "user text", ["data:image/png;base64,xxx"])

    assert isinstance(result, RawInferenceResult)
    assert result.raw_text == '{"ok": true}'
    assert result.resolved_model == "qwen3-vl-plus"
    assert result.request_id == "req-123"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.attempt_count == 1
    assert result.latency_seconds >= 0.0

    call_args = mock_client.chat.completions.create.call_args
    assert call_args.kwargs["model"] == "qwen3-vl-plus"
    assert call_args.kwargs["temperature"] == 0.0
    assert call_args.kwargs["max_tokens"] == 1200
    messages = call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "system text"}
    assert messages[1]["role"] == "user"
    assert messages[1]["content"][0] == {"type": "text", "text": "user text"}
    assert messages[1]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,xxx"},
    }


def test_qwen_client_retries_timeout_then_succeeds(
    mock_openai: MagicMock, model_config: ModelConfig
) -> None:
    from openai import APITimeoutError

    mock_client = mock_openai.return_value
    mock_client.chat.completions.create.side_effect = [
        APITimeoutError(request=_make_request()),
        _make_success_response('{"retry": "ok"}'),
    ]

    client = QwenClient(model_config)
    result = client.infer("sys", "user", [])

    assert result.raw_text == '{"retry": "ok"}'
    assert result.attempt_count == 2
    assert mock_client.chat.completions.create.call_count == 2


def test_qwen_client_retries_connection_error_exhausted(
    mock_openai: MagicMock, model_config: ModelConfig
) -> None:
    from openai import APIConnectionError

    mock_client = mock_openai.return_value
    mock_client.chat.completions.create.side_effect = APIConnectionError(
        message="connection failed", request=_make_request()
    )

    client = QwenClient(model_config)
    with pytest.raises(InferenceNetworkError):
        client.infer("sys", "user", [])

    assert mock_client.chat.completions.create.call_count == model_config.network_retries + 1


def test_qwen_client_retries_rate_limit_exhausted(
    mock_openai: MagicMock, model_config: ModelConfig
) -> None:
    from openai import APIStatusError

    mock_client = mock_openai.return_value
    response_429 = MagicMock()
    response_429.status_code = 429
    mock_client.chat.completions.create.side_effect = APIStatusError(
        message="rate limit", response=response_429, body=None
    )

    client = QwenClient(model_config)
    with pytest.raises(InferenceRateLimitError):
        client.infer("sys", "user", [])

    assert mock_client.chat.completions.create.call_count == model_config.network_retries + 1


def test_qwen_client_retries_server_error_exhausted(
    mock_openai: MagicMock, model_config: ModelConfig
) -> None:
    from openai import APIStatusError

    mock_client = mock_openai.return_value
    response_503 = MagicMock()
    response_503.status_code = 503
    mock_client.chat.completions.create.side_effect = APIStatusError(
        message="server down", response=response_503, body=None
    )

    client = QwenClient(model_config)
    with pytest.raises(InferenceServerError):
        client.infer("sys", "user", [])

    assert mock_client.chat.completions.create.call_count == model_config.network_retries + 1


def test_qwen_client_non_retryable_client_error_not_retried(
    mock_openai: MagicMock, model_config: ModelConfig
) -> None:
    from openai import APIStatusError

    mock_client = mock_openai.return_value
    response_400 = MagicMock()
    response_400.status_code = 400
    mock_client.chat.completions.create.side_effect = APIStatusError(
        message="bad request", response=response_400, body=None
    )

    client = QwenClient(model_config)
    with pytest.raises(APIStatusError):
        client.infer("sys", "user", [])

    assert mock_client.chat.completions.create.call_count == 1


def test_qwen_client_zero_retries_fail_immediately(
    mock_openai: MagicMock, model_config: ModelConfig
) -> None:
    from openai import APITimeoutError

    model_config.network_retries = 0
    mock_client = mock_openai.return_value
    mock_client.chat.completions.create.side_effect = APITimeoutError(
        request=_make_request()
    )

    client = QwenClient(model_config)
    with pytest.raises(InferenceNetworkError):
        client.infer("sys", "user", [])

    assert mock_client.chat.completions.create.call_count == 1


def _make_b64_image() -> str:
    return "data:image/jpeg;base64," + base64.b64encode(b"fake").decode("ascii")


@patch("qwen_stream_video.inference.client._HAS_LOCAL_DEPS", False)
def test_local_transformers_client_requires_dependencies() -> None:
    config = ModelConfig(
        provider="local_transformers",
        local_model_path="/some/path",
    )
    with pytest.raises(RuntimeError, match="Local inference dependencies are missing"):
        LocalTransformersClient(config)


@patch("qwen_stream_video.inference.client.uuid.uuid4")
@patch("qwen_stream_video.inference.client.process_vision_info")
@patch("qwen_stream_video.inference.client.AutoModelForVision2Seq")
@patch("qwen_stream_video.inference.client.AutoProcessor")
@patch("qwen_stream_video.inference.client.Image")
def test_local_transformers_client_runs_inference(
    mock_image: MagicMock,
    mock_auto_processor: MagicMock,
    mock_auto_model: MagicMock,
    mock_process_vision: MagicMock,
    mock_uuid: MagicMock,
) -> None:
    mock_uuid.return_value.hex = "req-abc-123"

    mock_processor = MagicMock()
    mock_auto_processor.from_pretrained.return_value = mock_processor
    mock_processor.apply_chat_template.return_value = "<chat_text>"
    mock_processor_inputs = MagicMock()
    mock_processor_inputs.to.return_value = mock_processor_inputs
    mock_processor_inputs.input_ids.shape = [1, 10]
    mock_processor.return_value = mock_processor_inputs
    mock_processor.batch_decode.return_value = ['{"summary": "test"}']

    mock_generated_ids = MagicMock()
    mock_generated_ids.shape = [1, 20]
    mock_generated_ids.__getitem__.return_value = MagicMock()
    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.generate.return_value = mock_generated_ids
    mock_auto_model.from_pretrained.return_value = mock_model

    mock_process_vision.return_value = ([MagicMock()], None)

    mock_pil_image = MagicMock()
    mock_image.open.return_value.convert.return_value = mock_pil_image

    config = ModelConfig(
        provider="local_transformers",
        name="Qwen3-VL-8B-Instruct",
        local_model_path="/home/Datasets/Hf_model/Qwen3-VL-8B-Instruct",
        device="cpu",
        torch_dtype="float32",
        max_tokens=100,
        temperature=0,
    )
    client = LocalTransformersClient(config)
    result = client.infer("system text", "user text", [_make_b64_image()])

    assert isinstance(result, RawInferenceResult)
    assert result.raw_text == '{"summary": "test"}'
    assert result.resolved_model == "Qwen3-VL-8B-Instruct"
    assert result.request_id == "req-abc-123"
    assert result.attempt_count == 1
    assert result.input_tokens == 10
    assert result.output_tokens == 10

    mock_auto_model.from_pretrained.assert_called_once()
    call_kwargs = mock_auto_model.from_pretrained.call_args.kwargs
    assert str(call_kwargs["torch_dtype"]) == "torch.float32"
    assert call_kwargs["device_map"] == "cpu"
