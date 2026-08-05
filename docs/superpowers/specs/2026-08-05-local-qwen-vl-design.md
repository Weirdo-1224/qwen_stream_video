# 本地 Qwen3-VL 模型集成设计

日期：2026-08-05
主题：在现有 DashScope API 调用基础上，新增本地 `Qwen3-VL-8B-Instruct` 推理支持，并可通过配置在 API 与本地模型之间切换。

## 1. 背景与目标

当前项目 `qwen-stream-video` 通过 OpenAI 兼容接口调用 DashScope 的 `qwen3-vl-plus` 视觉模型，对本地视频按滑动窗口做结构化观察。目标是在不改动视频窗口、抽帧、提示词、解析、存储等核心流程的前提下，增加一条本地模型推理路径，实现：

- 通过配置切换 `provider` 即可选择 `dashscope` 或 `local_transformers`。
- 本地模型使用 `/home/Datasets/Hf_model/Qwen3-VL-8B-Instruct`（原路径 `Qwen3-8B` 为纯文本模型，不能读图，故使用同目录下视觉模型）。
- 输出产物（`observations.jsonl`、`api_metrics.jsonl`、`errors.jsonl` 等）与现有 API 路径完全一致。

## 2. 非目标

- 不支持把 `Qwen3-8B` 纯文本模型直接作为视觉模型使用。
- 不实现本地模型自动启动/守护进程（如 vLLM server 生命周期管理），只提供进程内推理客户端。
- 不改动提示词模板、Observation Schema、解析器语义校验。
- 不改动现有测试对 `FakeQwenClient` 的依赖。

## 3. 总体设计

在现有 `inference/client.py` 中新增 `LocalTransformersClient`，实现与 `QwenClient` 完全相同的接口：

```python
def infer(
    self,
    system_prompt: str,
    user_prompt: str,
    images: list[str],
) -> RawInferenceResult
```

`StreamingVideoPipeline` 不需要知道底层是 API 还是本地模型；`cli._build_client` 根据 `config.model.provider` 决定实例化哪个客户端。

```text
run.py
  └── cli._build_client
        ├── provider == "dashscope"      → QwenClient
        └── provider == "local_transformers" → LocalTransformersClient
```

## 4. 配置变更

### 4.1 `config.py` 的 `ModelConfig`

新增字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `provider` | `Literal["dashscope", "local_transformers"]` | 模型提供方 |
| `local_model_path` | `str \| None` | 本地模型目录路径 |
| `device` | `Literal["auto", "cuda", "cpu"]` | 设备选择，默认 `auto` |
| `torch_dtype` | `Literal["bfloat16", "float16", "float32"]` | 默认 `bfloat16` |
| `load_in_8bit` | `bool` | 默认 `false` |
| `load_in_4bit` | `bool` | 默认 `false` |
| `max_model_len` | `int \| None` | 模型最大上下文长度 |
| `trust_remote_code` | `bool` | 默认 `true`（Qwen3-VL 需要） |

`api_key` 在 `provider == "local_transformers"` 时可选；`base_url` 在本地模式下不使用。

### 4.2 默认 `config.yaml` 示例

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

  # 本地模型参数，仅在 provider == local_transformers 时生效
  local_model_path: /home/Datasets/Hf_model/Qwen3-VL-8B-Instruct
  device: auto
  torch_dtype: bfloat16
  load_in_8bit: false
  load_in_4bit: false
  max_model_len: 32768
  trust_remote_code: true
```

### 4.3 环境变量

保持现有 `DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL`、`QWEN_MODEL` 不变。本地模型路径优先通过 `config.yaml` 或 CLI 指定，避免把绝对路径写入共享环境。

## 5. 本地推理客户端

文件：`src/qwen_stream_video/inference/client.py`（新增类，与 `QwenClient` 同文件）。

### 5.1 初始化

```python
class LocalTransformersClient:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self._processor = None
        self._model = None
```

为避免在 CLI 初始化阶段就加载大模型（例如 `--print-config` 不应触发加载），采用**首次调用时懒加载**策略。

### 5.2 首次推理时加载

```python
def _load(self) -> None:
    if self._model is not None:
        return

    from transformers import AutoModelForVision2Seq, AutoProcessor

    model_path = self.config.local_model_path
    self._processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=self.config.trust_remote_code
    )
    self._model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=self._torch_dtype(),
        device_map=self._device_map(),
        trust_remote_code=self.config.trust_remote_code,
        load_in_8bit=self.config.load_in_8bit,
        load_in_4bit=self.config.load_in_4bit,
    )
```

### 5.3 图像转换

当前传入的 `images` 是 `data:image/jpeg;base64,...` 字符串。本地模型需要 `PIL.Image`：

```python
def _decode_data_url(data_url: str) -> Image.Image:
    prefix = "data:image/jpeg;base64,"
    if not data_url.startswith(prefix):
        raise ValueError(f"Unsupported image format: {data_url[:30]}...")
    raw = base64.b64decode(data_url[len(prefix):])
    return Image.open(BytesIO(raw)).convert("RGB")
```

### 5.4 Qwen3-VL 消息拼接

```python
def _build_messages(
    self,
    system_prompt: str,
    user_prompt: str,
    images: list[Image.Image],
) -> list[dict]:
    content: list[dict] = []
    for image in images:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": user_prompt})

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
```

### 5.5 生成与结果包装

```python
def infer(self, system_prompt, user_prompt, images) -> RawInferenceResult:
    self._load()
    pil_images = [self._decode_data_url(url) for url in images]
    messages = self._build_messages(system_prompt, user_prompt, pil_images)

    text = self._processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
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
        temperature=self.config.temperature if self.config.temperature > 0 else None,
        do_sample=self.config.temperature > 0,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = self._processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    latency = time.perf_counter() - start

    return RawInferenceResult(
        raw_text=output_text,
        resolved_model=self.config.name,
        latency_seconds=latency,
        request_id=uuid.uuid4().hex,
        input_tokens=inputs.input_ids.shape[1],
        output_tokens=generated_ids_trimmed[0].shape[0],
        attempt_count=1,
    )
```

### 5.6 错误处理

- 模型加载失败（路径不存在、文件损坏）→ `ConfigurationError` 或 `InferenceServerError`。
- 显存不足 → 捕获 `torch.cuda.OutOfMemoryError`（或 `RuntimeError`），抛出 `InferenceServerError`，pipeline 会记录到 `errors.jsonl` 并继续处理后续窗口。
- 不实现自动重试，因为本地进程内错误大多为确定性错误。

## 6. CLI 入口修改

`cli._build_client` 逻辑改为：

```python
def _build_client(args, config):
    if args.dry_run or args.validate_only:
        return FakeQwenClient(response_text=DEFAULT_FAKE_RESPONSE)

    if config.model.provider == "local_transformers":
        return LocalTransformersClient(config.model)

    if not config.model.api_key:
        print("错误: 未配置 API Key...", file=sys.stderr)
        return None
    return QwenClient(config.model)
```

## 7. 输出产物

`RawInferenceResult` 结构不变，因此：

- `observations.jsonl`：与 API 路径完全一致。
- `api_metrics.jsonl`：继续使用，记录本地模型的 `latency_seconds`、`input_tokens`、`output_tokens`、`attempt_count`。
- `raw_responses/`：保存本地模型原始文本输出。
- `errors.jsonl`：记录本地推理异常。

## 8. 依赖与安装

在 `pyproject.toml` 新增可选依赖组：

```toml
[project.optional-dependencies]
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

使用本地模型时安装：

```bash
pip install -e ".[local]"
```

`requirements.txt` 保持基础依赖不变，避免强制所有用户安装 torch。

## 9. 测试策略

- 单元测试：为 `LocalTransformersClient` 增加测试，使用 `unittest.mock` 模拟 `transformers.AutoModelForVision2Seq` 和 `AutoProcessor`，验证 `infer()` 能正确解码 base64、构造 messages、返回 `RawInferenceResult`。
- 集成测试：不引入真正的 8B 模型加载（太慢且不稳定），保持 `tests/unit/test_pipeline.py` 使用 `FakeQwenClient` 覆盖 pipeline。
- 配置测试：新增 `test_config.py` 用例，验证 `provider == local_transformers` 时 `local_model_path` 必填且路径可校验。

## 10. 风险与限制

- `Qwen3-VL-8B-Instruct` BF16 推理需要约 16–20 GB 显存；CPU 推理极慢，仅适合验证。
- 当前环境未安装 `torch`/`transformers`/`qwen_vl_utils`，首次启用需安装可选依赖。
- `LocalTransformersClient` 首次加载会显著增加第一个窗口的延迟。
- 如果后续想同时跑 API 和本地做对比，需要进一步扩展为多 provider 模式，不在本次范围。
