# 本地 Qwen3-VL 模型集成实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在 `qwen-stream-video` 中新增本地 `Qwen3-VL-8B-Instruct` 推理客户端，并通过 `model.provider` 配置在 DashScope API 与本地模型之间切换。

**架构：** 新增 `LocalTransformersClient` 实现与现有 `QwenClient` 相同的 `infer(system_prompt, user_prompt, images) -> RawInferenceResult` 接口；`cli._build_client` 根据配置决定实例化哪个客户端；pipeline、parser、storage 完全无感。

**技术栈：** Python 3.10、Pydantic v2、transformers、qwen_vl_utils、torch、Pillow、pytest。

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `src/qwen_stream_video/config.py` | 扩展 `ModelConfig`：新增 `provider` 枚举、本地模型路径/设备/dtype/量化等字段；添加验证规则。 |
| `src/qwen_stream_video/inference/client.py` | 新增 `LocalTransformersClient`：加载 Qwen3-VL、解码 base64 图像、构造 chat messages、生成文本并包装成 `RawInferenceResult`。 |
| `src/qwen_stream_video/inference/__init__.py` | 导出 `LocalTransformersClient`。 |
| `src/qwen_stream_video/cli.py` | 修改 `_build_client`：按 `provider` 路由到 `LocalTransformersClient` 或 `QwenClient`。 |
| `pyproject.toml` | 新增可选依赖组 `[local]`。 |
| `config.yaml` / `configs/base.yaml` | 补充本地模型默认参数（注释即可）。 |
| `tests/unit/test_client.py` | 新增 `LocalTransformersClient` 单元测试，用 mock 替代 transformers。 |
| `tests/unit/test_config.py` | 新增本地模型配置验证测试。 |
| `tests/unit/test_cli.py` | 新增 CLI 客户端路由测试。 |
| `README.md` | 补充本地模型启用说明。 |

---

## 任务 1：扩展 `ModelConfig` 支持本地 provider

**文件：**
- 修改：`src/qwen_stream_video/config.py`
- 测试：`tests/unit/test_config.py`

### 步骤 1：编写失败的测试

在 `tests/unit/test_config.py` 末尾新增：

```python
from pathlib import Path


def test_local_transformers_provider_requires_model_path() -> None:
    with pytest.raises(ConfigurationError):
        AppConfig.model_validate(
            {
                "model": {
                    "provider": "local_transformers",
                    "local_model_path": "",
                }
            }
        )


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
```

运行：

```bash
pytest tests/unit/test_config.py::test_local_transformers_provider_requires_model_path -v
```

预期：FAIL（`provider` 只能是 `dashscope` 或非法字符串，且没有 `local_model_path` 字段）。

### 步骤 2：修改 `ModelConfig`

在 `src/qwen_stream_video/config.py` 中：

1. 顶部引入 `Literal`：

```python
from typing import Any, Literal
```

2. 修改 `ModelConfig` 类：

```python
class ModelConfig(BaseModel):
    """Model and API connection settings."""

    model_config = ConfigDict(extra="ignore")

    provider: Literal["dashscope", "local_transformers"] = Field(default="dashscope")
    name: str = Field(default="qwen3-vl-plus", min_length=1)
    api_key: str | None = Field(default=None)
    base_url: str | None = Field(default=None)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1200, ge=1)
    timeout_seconds: float = Field(default=120.0, gt=0)
    network_retries: int = Field(default=2, ge=0)
    source: str = Field(default="default", min_length=1)

    # Local-only settings
    local_model_path: str | None = Field(default=None, min_length=1)
    device: Literal["auto", "cuda", "cpu"] = Field(default="auto")
    torch_dtype: Literal["bfloat16", "float16", "float32"] = Field(default="bfloat16")
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    max_model_len: int | None = Field(default=None, ge=1)
    trust_remote_code: bool = True

    @model_validator(mode="after")
    def _check_local_model_path(self) -> "ModelConfig":
        if self.provider == "local_transformers" and not self.local_model_path:
            raise ValueError("local_model_path is required when provider is local_transformers")
        return self
```

运行：

```bash
pytest tests/unit/test_config.py -v
```

预期：新增两个测试通过；原有测试继续通过。

### 步骤 3：Commit

```bash
git add src/qwen_stream_video/config.py tests/unit/test_config.py
git commit -m "feat(config): add local_transformers provider and local model settings"
```

---

## 任务 2：新增可选本地依赖

**文件：**
- 修改：`pyproject.toml`

### 步骤 1：添加 `[local]` 可选依赖组

在 `pyproject.toml` 中 `[project.optional-dependencies]` 下添加：

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.6",
]
local = [
    "torch>=2.0",
    "transformers>=4.57.0",
    "qwen_vl_utils",
    "accelerate",
    "sentencepiece",
    "protobuf",
    "pillow",
]
```

### 步骤 2：验证 TOML 语法

运行：

```bash
python -c "import tomllib, pathlib; tomllib.load(pathlib.Path('pyproject.toml').open('rb'))"
```

预期：无报错。

### 步骤 3：Commit

```bash
git add pyproject.toml
git commit -m "build: add optional local inference dependencies"
```

---

## 任务 3：实现 `LocalTransformersClient`

**文件：**
- 修改：`src/qwen_stream_video/inference/client.py`
- 测试：`tests/unit/test_client.py`

### 步骤 1：编写失败的测试

在 `tests/unit/test_client.py` 中新增一个 fixture 和测试：

```python
import base64
import uuid
from unittest.mock import MagicMock, patch
from io import BytesIO
from PIL import Image

import pytest
import numpy as np

from qwen_stream_video.config import ModelConfig
from qwen_stream_video.inference import LocalTransformersClient, RawInferenceResult


@pytest.fixture
def local_model_config() -> ModelConfig:
    return ModelConfig(
        provider="local_transformers",
        name="Qwen3-VL-8B-Instruct",
        local_model_path="/home/Datasets/Hf_model/Qwen3-VL-8B-Instruct",
        device="cpu",
        torch_dtype="float32",
        max_tokens=100,
        temperature=0,
    )


def _make_b64_image() -> str:
    img = Image.new("RGB", (16, 16), color=(128, 64, 32))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def test_local_transformers_client_decodes_image_and_returns_result(
    local_model_config: ModelConfig,
) -> None:
    # Arrange: mock transformers components and qwen_vl_utils
    mock_processor = MagicMock()
    mock_model = MagicMock()
    mock_model.device = "cpu"

    mock_generated_ids = MagicMock()
    mock_generated_ids.shape = [1, 20]
    mock_generated_ids.__len__ = lambda self: 20
    mock_model.generate.return_value = [mock_generated_ids]

    mock_input_ids = MagicMock()
    mock_input_ids.shape = [1, 10]
    mock_processor_inputs = MagicMock()
    mock_processor_inputs.input_ids = mock_input_ids
    mock_processor.return_value = mock_processor_inputs

    mock_processor.apply_chat_template.return_value = "<chat_text>"
    mock_processor.batch_decode.return_value = ['{"summary": "test"}']

    with (
        patch(
            "qwen_stream_video.inference.client.AutoProcessor"
        ) as mock_auto_processor,
        patch(
            "qwen_stream_video.inference.client.AutoModelForVision2Seq"
        ) as mock_auto_model,
        patch(
            "qwen_stream_video.inference.client.process_vision_info"
        ) as mock_process_vision,
        patch(
            "qwen_stream_video.inference.client.uuid"
        ) as mock_uuid,
    ):
        mock_auto_processor.from_pretrained.return_value = mock_processor
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_process_vision.return_value = ([MagicMock()], None)
        mock_uuid.uuid4.return_value.hex = "req-abc-123"

        client = LocalTransformersClient(local_model_config)
        result = client.infer(
            "system text",
            "user text",
            [_make_b64_image()],
        )

    assert isinstance(result, RawInferenceResult)
    assert result.raw_text == '{"summary": "test"}'
    assert result.resolved_model == "Qwen3-VL-8B-Instruct"
    assert result.request_id == "req-abc-123"
    assert result.attempt_count == 1
    assert result.input_tokens == 10
    assert result.output_tokens == 10

    # Verify model was loaded with correct path and dtype
    mock_auto_model.from_pretrained.assert_called_once()
    call_kwargs = mock_auto_model.from_pretrained.call_args.kwargs
    assert call_kwargs["torch_dtype"].__name__ == "float32"
    assert call_kwargs["device_map"] == "cpu"

    # Verify processor was called with image and text
    messages = mock_processor.apply_chat_template.call_args.kwargs
    assert messages is not None
```

运行：

```bash
pytest tests/unit/test_client.py::test_local_transformers_client_decodes_image_and_returns_result -v
```

预期：FAIL（`LocalTransformersClient` 未定义）。

### 步骤 2：实现 `LocalTransformersClient`

在 `src/qwen_stream_video/inference/client.py` 中，保持现有 `QwenClient` 不变，在文件顶部添加可选依赖导入：

```python
import base64
import uuid
from io import BytesIO
from typing import Any

try:
    from PIL import Image
    from qwen_vl_utils import process_vision_info
    from transformers import AutoModelForVision2Seq, AutoProcessor

    _HAS_LOCAL_DEPS = True
except ImportError:  # pragma: no cover - optional local dependencies
    _HAS_LOCAL_DEPS = False
```

在 `QwenClient` 之后、
`FakeQwenClient` 之前（或之后）新增 `LocalTransformersClient`：

```python
class LocalTransformersClient:
    """Process-vision Qwen3-VL local inference via Hugging Face transformers."""

    def __init__(self, config: ModelConfig) -> None:
        """Initialize the client without loading the model yet.

        Args:
            config: Resolved model configuration including the local model path,
                device, dtype, and quantization settings.
        """
        if not _HAS_LOCAL_DEPS:
            raise RuntimeError(
                "Local inference dependencies are missing. "
                "Install them with: pip install -e \".[local]\""
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

        self._processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        self._model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=self.config.trust_remote_code,
            load_in_8bit=self.config.load_in_8bit,
            load_in_4bit=self.config.load_in_4bit,
        )

    def _torch_dtype(self) -> Any:
        import torch

        mapping = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return mapping[self.config.torch_dtype]

    def _device_map(self) -> str:
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

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
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
            input_tokens=int(inputs.input_ids.shape[1]),
            output_tokens=int(generated_ids_trimmed[0].shape[0]),
            attempt_count=1,
        )
```

运行：

```bash
pytest tests/unit/test_client.py::test_local_transformers_client_decodes_image_and_returns_result -v
```

预期：PASS。

### 步骤 3：处理缺失依赖的测试路径

在 `tests/unit/test_client.py` 中新增一个测试：

```python
@patch("qwen_stream_video.inference.client._HAS_LOCAL_DEPS", False)
def test_local_transformers_client_raises_when_dependencies_missing() -> None:
    config = ModelConfig(
        provider="local_transformers",
        local_model_path="/some/path",
    )
    with pytest.raises(RuntimeError, match="Local inference dependencies are missing"):
        LocalTransformersClient(config)
```

运行：

```bash
pytest tests/unit/test_client.py -v
```

预期：所有测试通过。

### 步骤 4：Commit

```bash
git add src/qwen_stream_video/inference/client.py tests/unit/test_client.py
git commit -m "feat(inference): add LocalTransformersClient for Qwen3-VL local inference"
```

---

## 任务 4：导出 `LocalTransformersClient`

**文件：**
- 修改：`src/qwen_stream_video/inference/__init__.py`

### 步骤 1：更新导出

```python
from .client import FakeQwenClient, LocalTransformersClient, QwenClient, RawInferenceResult
from .parser import ResponseParser
from .prompts import PromptBuilder
from .validator import ObservationSemanticValidator

__all__ = [
    "FakeQwenClient",
    "LocalTransformersClient",
    "ObservationSemanticValidator",
    "PromptBuilder",
    "QwenClient",
    "RawInferenceResult",
    "ResponseParser",
]
```

### 步骤 2：验证导入

运行：

```bash
python -c "from qwen_stream_video.inference import LocalTransformersClient; print(LocalTransformersClient)"
```

预期：成功打印类对象。

### 步骤 3：Commit

```bash
git add src/qwen_stream_video/inference/__init__.py
git commit -m "chore(inference): export LocalTransformersClient"
```

---

## 任务 5：CLI 按 provider 路由客户端

**文件：**
- 修改：`src/qwen_stream_video/cli.py`
- 测试：`tests/unit/test_cli.py`

### 步骤 1：编写失败的测试

在 `tests/unit/test_cli.py` 中新增测试（先查看现有 fixture 结构）：

```python
from unittest.mock import patch

from qwen_stream_video.cli import _build_client
from qwen_stream_video.config import ModelConfig


def test_build_client_uses_local_transformers_for_local_provider() -> None:
    config = ModelConfig(
        provider="local_transformers",
        name="Qwen3-VL-8B-Instruct",
        local_model_path="/home/Datasets/Hf_model/Qwen3-VL-8B-Instruct",
    )

    class FakeArgs:
        dry_run = False
        validate_only = False

    with patch(
        "qwen_stream_video.cli.LocalTransformersClient"
    ) as mock_local_client:
        mock_local_client.return_value = "local-client-instance"
        client = _build_client(FakeArgs(), config)

    assert client == "local-client-instance"
    mock_local_client.assert_called_once_with(config)


def test_build_client_uses_qwen_client_for_dashscope_provider() -> None:
    config = ModelConfig(
        provider="dashscope",
        name="qwen3-vl-plus",
        api_key="sk-test",
    )

    class FakeArgs:
        dry_run = False
        validate_only = False

    with patch("qwen_stream_video.cli.QwenClient") as mock_qwen_client:
        mock_qwen_client.return_value = "qwen-client-instance"
        client = _build_client(FakeArgs(), config)

    assert client == "qwen-client-instance"
    mock_qwen_client.assert_called_once_with(config)
```

运行：

```bash
pytest tests/unit/test_cli.py::test_build_client_uses_local_transformers_for_local_provider -v
```

预期：FAIL（`_build_client` 尚未按 provider 分支）。

### 步骤 2：修改 `_build_client`

在 `src/qwen_stream_video/cli.py` 中更新导入：

```python
from .inference import FakeQwenClient, LocalTransformersClient, PromptBuilder, QwenClient, ResponseParser
```

并修改 `_build_client` 函数：

```python
def _build_client(args: argparse.Namespace, config: Any) -> Any:
    """Create a real or fake inference client depending on the run mode."""
    if args.dry_run or args.validate_only:
        return FakeQwenClient(response_text=DEFAULT_FAKE_RESPONSE)
    if config.model.provider == "local_transformers":
        return LocalTransformersClient(config.model)
    if not config.model.api_key:
        print(
            "错误: 未配置 API Key。请设置 DASHSCOPE_API_KEY 环境变量或在配置中提供 model.api_key。",
            file=sys.stderr,
        )
        return None
    return QwenClient(config.model)
```

运行：

```bash
pytest tests/unit/test_cli.py -v
```

预期：新增两个测试通过，原有测试继续通过。

### 步骤 3：Commit

```bash
git add src/qwen_stream_video/cli.py tests/unit/test_cli.py
git commit -m "feat(cli): route client selection by model provider"
```

---

## 任务 6：补充默认配置模板

**文件：**
- 修改：`config.yaml` 和 `configs/base.yaml`

### 步骤 1：修改 `config.yaml`

在 `model` 区块下添加本地参数（保留默认值与注释）：

```yaml
model:
  provider: dashscope
  name: qwen3-vl-plus
  api_key: null
  base_url: null
  temperature: 0
  max_tokens: 3000
  timeout_seconds: 120
  network_retries: 2

  # 本地模型参数，仅当 provider == local_transformers 时生效
  local_model_path: /home/Datasets/Hf_model/Qwen3-VL-8B-Instruct
  device: auto          # auto | cuda | cpu
  torch_dtype: bfloat16 # bfloat16 | float16 | float32
  load_in_8bit: false
  load_in_4bit: false
  max_model_len: 32768
  trust_remote_code: true
```

### 步骤 2：修改 `configs/base.yaml`

做相同修改，但 `max_tokens` 保持 `1200`。

### 步骤 3：验证配置打印

运行：

```bash
python run.py --print-config
```

预期：正常输出配置摘要，不报错。

### 步骤 4：Commit

```bash
git add config.yaml configs/base.yaml
git commit -m "chore(config): add local model defaults to yaml templates"
```

---

## 任务 7：README 补充本地模型说明

**文件：**
- 修改：`README.md`

### 步骤 1：在“配置 API”节后添加“本地模型部署”小节

```markdown
## 本地模型部署（可选）

本项目也支持使用本地 `Qwen3-VL-8B-Instruct` 模型替代 DashScope API。

### 1. 安装本地推理依赖

```bash
pip install -e ".[local]"
```

### 2. 修改配置

在 `config.yaml` 中切换 provider：

```yaml
model:
  provider: local_transformers
  name: Qwen3-VL-8B-Instruct
  local_model_path: /home/Datasets/Hf_model/Qwen3-VL-8B-Instruct
  device: auto
  torch_dtype: bfloat16
```

注意：原目录 `/home/Datasets/Hf_model/Qwen3-8B` 是纯文本模型，不能读图；本地视觉模型请使用 `Qwen3-VL-8B-Instruct`。

### 3. 运行

```bash
python run.py --video videos/demo.mp4
```

本地模型首次加载会消耗一定时间，请确保有足够显存（BF16 约 16–20 GB）。
```

### 步骤 2：Commit

```bash
git add README.md
git commit -m "docs(readme): add local model deployment instructions"
```

---

## 任务 8：全量回归测试

**文件：**
- 运行：`pytest tests/ -q`

### 步骤 1：运行测试

```bash
pytest tests/ -q
```

### 步骤 2：运行 ruff 检查

```bash
ruff check .
```

### 步骤 3：Commit（如有格式修复）

如果 ruff 报告问题，修复后提交：

```bash
ruff check --fix .
git add .
git commit -m "style: fix lint issues"
```

---

## 自检清单

| 规格需求 | 实现任务 |
|---|---|
| 配置支持 `provider` 切换 | 任务 1 |
| 本地模型路径/设备/dtype 等参数 | 任务 1 |
| 本地客户端实现相同 `infer` 接口 | 任务 3 |
| base64 图像解码为 PIL | 任务 3 |
| Qwen3-VL chat template 构造 | 任务 3 |
| 返回 `RawInferenceResult` 含 token/延迟 | 任务 3 |
| CLI 按 provider 路由客户端 | 任务 5 |
| 导出本地客户端 | 任务 4 |
| 可选依赖组 `[local]` | 任务 2 |
| YAML 配置模板 | 任务 6 |
| 测试覆盖 | 任务 1、3、5 |
| 文档说明 | 任务 7 |

## 已知风险与验证边界

- 真实 8B 模型加载未在当前计划范围内测试；单元测试使用 mock 验证接口与数据流。
- 若本地环境未安装 `torch`/`transformers`/`qwen_vl_utils`，`_build_client` 会在首次推理时抛出 `RuntimeError` 并记录到 `errors.jsonl`。
- 运行全量测试前确保未安装 `[local]` 依赖时也不会破坏现有 DashScope 路径。
