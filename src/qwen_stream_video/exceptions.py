"""Project exception hierarchy."""


class QwenStreamVideoError(Exception):
    """Base class for expected qwen-stream-video errors."""


class ConfigurationError(QwenStreamVideoError):
    """Raised when application configuration is invalid."""


class VideoOpenError(QwenStreamVideoError):
    """Raised when a video cannot be opened."""


class VideoMetadataError(QwenStreamVideoError):
    """Raised when video metadata is invalid."""


class FrameReadError(QwenStreamVideoError):
    """Raised when a frame cannot be read."""


class InferenceNetworkError(QwenStreamVideoError):
    """Raised for retryable inference network failures."""


class InferenceRateLimitError(QwenStreamVideoError):
    """Raised when the inference provider rate-limits a request."""


class InferenceServerError(QwenStreamVideoError):
    """Raised for retryable inference server failures."""


class ModelOutputParseError(QwenStreamVideoError):
    """Raised when model output cannot be parsed as JSON."""


class ModelOutputSchemaError(QwenStreamVideoError):
    """Raised when parsed model output does not match the schema."""


class ModelOutputSemanticError(QwenStreamVideoError):
    """Raised when model output violates semantic constraints."""
