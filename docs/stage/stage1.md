# qwen_stream_video v2 第一阶段开发任务

你需要在以下仓库中完成第一阶段重构：

```text
https://github.com/Weirdo-1224/qwen_stream_video.git
```

## 一、任务目标

将当前以 `run.py` 为核心的单文件视频分析原型，重构为标准 Python 工程，并将模型输出从“每个窗口完整状态描述”修改为“当前窗口增量观察”。

本阶段重点完成：

1. 工程目录重构；
2. 配置系统重构；
3. 视频窗口和抽帧模块拆分；
4. 千问 API 调用模块拆分；
5. 引入 Pydantic 数据模型；
6. 定义增量观察协议；
7. 增加严格的输出校验；
8. 保存原始模型响应和运行元数据；
9. 修复已知的窗口编号和实时模式问题；
10. 增加基础单元测试。

本阶段不实现：

* EntityRegistry；
* ActionTracker；
* TransitionEngine；
* RTSP 或摄像头输入；
* 向量数据库；
* 多 Agent；
* 违规判断；
* 自适应抽帧；
* Web 前端。

不要提前实现后续阶段的功能。

---

# 二、修改原则

## 2.1 保持现有功能可用

重构后必须继续支持类似命令：

```bash
python run.py \
  --video videos/demo.mp4 \
  --config configs/base.yaml
```

原有用户不能因为目录重构而无法运行项目。

根目录的 `run.py` 只保留兼容入口：

```python
from qwen_stream_video.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

## 2.2 不要大规模重写后再统一调试

按照以下顺序逐步修改：

```text
工程结构
→ 配置加载
→ 视频模块
→ 模型调用模块
→ 数据模型
→ 新提示词
→ 输出存储
→ 测试
```

每完成一个部分，都要保证代码能够导入并运行。

## 2.3 不允许虚假实现

禁止：

* 使用空函数占位但声称已实现；
* 通过硬编码构造模型输出；
* 删除异常处理来让测试通过；
* 跳过模型响应校验；
* 在测试中只写 `assert True`；
* 静默吞掉异常；
* 为了兼容旧代码复制两套重复实现。

---

# 三、目标目录结构

将项目调整为：

```text
qwen_stream_video/
├── pyproject.toml
├── run.py
├── README.md
├── .env.example
│
├── configs/
│   └── base.yaml
│
├── prompts/
│   ├── observation_system.txt
│   ├── observation_user.txt
│   └── json_repair.txt
│
├── vocabularies/
│   ├── actions.yaml
│   ├── entity_types.yaml
│   └── attributes.yaml
│
├── src/
│   └── qwen_stream_video/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── pipeline.py
│       ├── exceptions.py
│       │
│       ├── domain/
│       │   ├── __init__.py
│       │   ├── enums.py
│       │   ├── observation.py
│       │   └── runtime.py
│       │
│       ├── video/
│       │   ├── __init__.py
│       │   ├── metadata.py
│       │   ├── window.py
│       │   ├── sampler.py
│       │   └── frame_encoder.py
│       │
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── qwen_client.py
│       │   ├── prompt_builder.py
│       │   ├── parser.py
│       │   └── validator.py
│       │
│       └── storage/
│           ├── __init__.py
│           └── run_store.py
│
├── scripts/
│   ├── analyze_results.py
│   └── make_test_video.py
│
└── tests/
    ├── unit/
    │   ├── test_config.py
    │   ├── test_windows.py
    │   ├── test_observation_schema.py
    │   └── test_validator.py
    └── fixtures/
```

不要求目录完全机械一致，但职责必须清晰，不允许继续把主要逻辑留在 `run.py`。

---

# 四、依赖和工程配置

## 4.1 新增 pyproject.toml

使用标准 `src` 布局。

项目至少包含以下依赖：

```text
openai
opencv-python
pydantic
pydantic-settings
PyYAML
python-dotenv
numpy
```

开发依赖：

```text
pytest
pytest-cov
ruff
```

配置命令入口：

```toml
[project.scripts]
qwen-stream-video = "qwen_stream_video.cli:main"
```

确保以下两种方式都能运行：

```bash
python run.py --help
```

```bash
qwen-stream-video --help
```

## 4.2 Python 版本

要求：

```text
Python >= 3.10
```

所有公共函数和类必须有类型标注。

---

# 五、配置系统

新增：

```text
src/qwen_stream_video/config.py
```

使用 Pydantic 定义配置模型，不允许在业务代码中到处直接访问无类型字典。

配置结构：

```yaml
experiment:
  name: incremental_observation_v1
  seed: 42

video:
  window_seconds: 6.0
  stride_seconds: 3.0

sampling:
  sample_fps: 1.0
  min_frames: 4
  max_frames: 12
  max_image_side: 768
  jpeg_quality: 80

model:
  provider: dashscope
  name: qwen3-vl-plus
  temperature: 0
  max_tokens: 1200
  timeout_seconds: 120
  network_retries: 2

observation:
  schema_version: "1.0"
  require_evidence_frames: true
  use_candidate_global_ids: true

runtime:
  realtime: false

storage:
  output_root: outputs
  save_raw_responses: true
  save_sampled_frames: false
```

至少定义：

```python
class ExperimentConfig(BaseModel): ...
class VideoConfig(BaseModel): ...
class SamplingConfig(BaseModel): ...
class ModelConfig(BaseModel): ...
class ObservationConfig(BaseModel): ...
class RuntimeConfig(BaseModel): ...
class StorageConfig(BaseModel): ...
class AppConfig(BaseModel): ...
```

## 5.1 配置优先级

优先级必须明确：

```text
命令行参数
> 环境变量
> YAML 配置
> 代码默认值
```

环境变量至少支持：

```text
DASHSCOPE_API_KEY
DASHSCOPE_BASE_URL
QWEN_MODEL
```

启动时打印最终生效配置摘要，尤其是：

```text
最终模型名称
配置来源
视频路径
窗口大小
步长
采样率
输出目录
```

API Key 只能显示是否已配置，不能打印实际内容。

## 5.2 配置校验

需要校验：

```text
window_seconds > 0
stride_seconds > 0
sample_fps > 0
min_frames >= 1
max_frames >= min_frames
jpeg_quality 位于 1 到 100
max_image_side > 0
network_retries >= 0
```

配置错误时，应输出明确错误信息并退出，不要在运行中途失败。

---

# 六、视频模块

## 6.1 VideoMetadata

新增：

```python
class VideoMetadata(BaseModel):
    path: str
    fps: float
    frame_count: int
    duration_seconds: float
    width: int
    height: int
```

读取视频时检查：

* 文件是否存在；
* OpenCV 是否成功打开；
* FPS 是否有效；
* 总帧数是否有效；
* 视频时长是否有效。

异常统一抛出：

```python
VideoOpenError
VideoMetadataError
FrameReadError
```

## 6.2 VideoWindow

定义：

```python
class VideoWindow(BaseModel):
    global_index: int
    run_index: int
    start_seconds: float
    end_seconds: float
    window_type: Literal["regular", "tail_completion"]
```

必须保留：

* `global_index`：窗口在完整视频中的真实编号；
* `run_index`：本次运行选择范围内的编号。

例如从全局窗口 100 开始：

```json
{
  "global_index": 100,
  "run_index": 0
}
```

不允许重新编号后丢失全局位置。

## 6.3 窗口生成

将窗口生成逻辑移动到：

```text
video/window.py
```

提供类似接口：

```python
def build_video_windows(
    metadata: VideoMetadata,
    window_seconds: float,
    stride_seconds: float,
) -> list[VideoWindow]:
    ...
```

需要处理：

1. 普通视频；
2. 视频短于一个窗口；
3. 末尾不足完整窗口；
4. 末尾补齐窗口；
5. 窗口不能超过视频实际结束时间；
6. 窗口必须按照时间排序；
7. `end_seconds > start_seconds`。

## 6.4 修复实时模式时间原点

不要使用：

```python
target_wall_time = wall_start + window.end_seconds
```

应使用：

```python
video_origin = selected_windows[0].start_seconds

target_wall_time = (
    wall_start
    + window.end_seconds
    - video_origin
)
```

确保：

```bash
--start-time 480 --realtime
```

不会等待 486 秒，而只等待第一个窗口实际需要的时间。

将该逻辑提取为可测试函数：

```python
def calculate_realtime_target(
    wall_start: float,
    video_origin: float,
    window_end: float,
) -> float:
    ...
```

---

# 七、抽帧模块

## 7.1 SampledFrame

定义：

```python
class SampledFrame(BaseModel):
    sample_index: int
    frame_index: int
    timestamp_seconds: float
    encoded_image: str | None = None
```

模型证据帧使用 `sample_index`，而不是视频原始帧号。

## 7.2 抽帧约束

必须保证：

```text
window.start_seconds
<= sampled_timestamp
< window.end_seconds
```

禁止采到右边界以及未来窗口帧。

抽帧函数：

```python
def sample_window_frames(
    capture: cv2.VideoCapture,
    metadata: VideoMetadata,
    window: VideoWindow,
    config: SamplingConfig,
) -> list[SampledFrame]:
    ...
```

抽帧数量应根据：

```text
窗口时长 × sample_fps
```

计算，并限制在：

```text
[min_frames, max_frames]
```

当视频太短或重复帧号导致不足时：

* 尽可能返回有效帧；
* 不允许伪造重复图像；
* 应记录实际帧数；
* 少于模型最低要求时抛出明确异常。

## 7.3 图像编码

单独实现：

```python
def encode_frame_to_data_url(
    frame: np.ndarray,
    max_image_side: int,
    jpeg_quality: int,
) -> str:
    ...
```

功能：

* 保持长宽比缩放；
* 不放大小图；
* JPEG 编码；
* Base64；
* 返回标准 Data URL。

---

# 八、增量观察数据协议

本阶段最重要的修改是：模型不再输出完整状态，而只输出当前窗口观察。

## 8.1 枚举

新增：

```text
domain/enums.py
```

至少包含：

```python
class EntityType(str, Enum):
    PERSON = "person"
    DEVICE = "device"
    COMPONENT = "component"
    TOOL = "tool"
    PPE = "ppe"
    SIGN = "sign"
    ENVIRONMENT = "environment"
    UNKNOWN = "unknown"


class ViewType(str, Enum):
    WIDE = "wide"
    MEDIUM = "medium"
    CLOSEUP = "closeup"
    DETAIL = "detail"
    UNKNOWN = "unknown"


class VisibilityQuality(str, Enum):
    CLEAR = "clear"
    PARTIAL = "partial"
    POOR = "poor"
    UNKNOWN = "unknown"


class ActionPhaseObservation(str, Enum):
    STARTING = "starting"
    ONGOING = "ongoing"
    POSSIBLY_COMPLETED = "possibly_completed"
    INSTANT = "instant"
    UNKNOWN = "unknown"
```

动作类型可以先采用字符串加词表校验，不需要把所有动作写死为 Python Enum。

## 8.2 ObservationBatch

在：

```text
domain/observation.py
```

定义以下模型。

### WindowObservation

```python
class WindowObservation(BaseModel):
    global_index: int
    start_seconds: float
    end_seconds: float
```

### SceneObservation

```python
class SceneObservation(BaseModel):
    camera_change: bool = False
    view_type: ViewType = ViewType.UNKNOWN
    visibility: VisibilityQuality = VisibilityQuality.UNKNOWN
    description: str = ""
```

### EntityObservation

```python
class EntityObservation(BaseModel):
    local_id: str
    entity_type: EntityType
    name: str = "unknown"
    description: str = ""
    appearance: dict[str, str] = Field(default_factory=dict)
    spatial_region: str = "unknown"
    candidate_global_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_frames: list[int]
```

### ActionObservation

```python
class ActionObservation(BaseModel):
    local_id: str
    actor_local_id: str
    action_type: str
    target_local_id: str | None = None
    tool_local_id: str | None = None
    phase_observation: ActionPhaseObservation
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_frames: list[int]
```

### AttributeObservation

```python
class AttributeObservation(BaseModel):
    entity_local_id: str
    attribute: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_frames: list[int]
```

### UncertaintyObservation

```python
class UncertaintyObservation(BaseModel):
    description: str
    related_local_ids: list[str] = Field(default_factory=list)
    evidence_frames: list[int] = Field(default_factory=list)
```

### ObservationBatch

```python
class ObservationBatch(BaseModel):
    schema_version: str
    window: WindowObservation
    summary: str
    scene: SceneObservation
    entities: list[EntityObservation]
    actions: list[ActionObservation]
    attribute_observations: list[AttributeObservation]
    uncertainties: list[UncertaintyObservation]
```

所有 list 和 dict 都必须使用 `default_factory`，禁止使用可变默认参数。

---

# 九、业务语义校验

Pydantic 只能验证字段类型，还要新增：

```text
inference/validator.py
```

实现：

```python
class ObservationSemanticValidator:
    def validate(
        self,
        batch: ObservationBatch,
        sampled_frames: list[SampledFrame],
    ) -> list[str]:
        ...
```

至少检查：

## 9.1 ID 唯一性

同一个窗口中：

* 实体 `local_id` 不能重复；
* 动作 `local_id` 不能重复。

## 9.2 引用完整性

* `actor_local_id` 必须引用当前窗口实体；
* `target_local_id` 非空时必须引用当前窗口实体；
* `tool_local_id` 非空时必须引用当前窗口实体；
* `entity_local_id` 必须引用当前窗口实体。

## 9.3 证据帧合法性

所有：

```text
evidence_frames
```

必须满足：

```text
0 <= frame_index < sampled_frame_count
```

证据帧去重并排序。

## 9.4 动作词表

从：

```text
vocabularies/actions.yaml
```

读取允许动作。

第一版建议包含：

```yaml
actions:
  - observe
  - inspect
  - approach
  - leave
  - hold
  - pick_up
  - place
  - touch
  - press
  - rotate
  - switch
  - open
  - close
  - insert
  - remove
  - connect
  - disconnect
  - adjust
  - measure
  - record
  - point
  - unknown
```

非法动作统一映射为：

```text
unknown
```

但必须保留原始描述，并生成校验警告。

## 9.5 时间字段覆盖

模型返回的窗口时间不能作为真实时间来源。

程序必须使用当前 `VideoWindow` 覆盖：

```python
batch.window = WindowObservation(
    global_index=window.global_index,
    start_seconds=window.start_seconds,
    end_seconds=window.end_seconds,
)
```

---

# 十、提示词修改

## 10.1 observation_system.txt

系统提示词需要明确：

1. 你是流式视频观察器；
2. 只能分析当前窗口；
3. 不能使用未来信息；
4. 不能推断完整视频结论；
5. 只输出当前窗口增量观察；
6. 不负责维护完整全局状态；
7. 不负责分配正式全局实体 ID；
8. `local_id` 只在当前窗口有效；
9. `candidate_global_id` 只是候选；
10. 动作开始不等于动作完成；
11. 未看到实体不等于实体消失；
12. 看不清时使用 `unknown`；
13. 每个事实都尽量引用证据帧；
14. 只输出 JSON，不输出解释文字。

## 10.2 observation_user.txt

动态用户提示词包含：

```text
视频背景
当前窗口范围
抽样帧编号及对应时间
可选的上一窗口候选上下文
输出协议说明
```

帧信息示例：

```text
F0 = 12.500 秒
F1 = 13.500 秒
F2 = 14.500 秒
F3 = 15.500 秒
F4 = 16.500 秒
F5 = 17.500 秒
```

要求模型在：

```json
{
  "evidence_frames": [2, 3, 4]
}
```

中引用 `F2、F3、F4` 对应的整数索引。

## 10.3 本阶段的历史上下文

由于 EntityRegistry 尚未实现，本阶段只允许传入上一窗口的简化观察摘要：

```json
{
  "previous_window_summary": "...",
  "previous_entities": [
    {
      "candidate_global_id": "person_1",
      "type": "person",
      "description": "..."
    }
  ]
}
```

这只是过渡兼容设计。

不得实现复杂实体匹配，也不要声称该 ID 已经稳定。

---

# 十一、Qwen 调用层

## 11.1 QwenClient

定义：

```python
class RawInferenceResult(BaseModel):
    raw_text: str
    resolved_model: str
    latency_seconds: float
    request_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    attempt_count: int
```

`QwenClient` 只负责：

* 发送请求；
* 网络重试；
* 收集延迟；
* 收集 token；
* 返回原始文本。

不负责：

* JSON 解析；
* Schema 校验；
* 状态更新；
* 文件写入。

## 11.2 错误分类

在：

```text
exceptions.py
```

增加：

```python
class QwenStreamVideoError(Exception): ...
class ConfigurationError(QwenStreamVideoError): ...
class VideoOpenError(QwenStreamVideoError): ...
class FrameReadError(QwenStreamVideoError): ...
class InferenceNetworkError(QwenStreamVideoError): ...
class InferenceRateLimitError(QwenStreamVideoError): ...
class InferenceServerError(QwenStreamVideoError): ...
class ModelOutputParseError(QwenStreamVideoError): ...
class ModelOutputSchemaError(QwenStreamVideoError): ...
class ModelOutputSemanticError(QwenStreamVideoError): ...
```

## 11.3 重试范围

仅以下错误允许重发视频请求：

* 网络超时；
* 连接中断；
* HTTP 429；
* HTTP 5xx。

以下错误不能重新发送完整视频：

* JSON 格式错误；
* Schema 错误；
* 引用错误；
* 非法枚举。

本阶段 JSON 修复可以先采用本地轻量修复，不需要再次调用模型。

---

# 十二、响应解析

新增：

```text
inference/parser.py
```

处理流程：

```text
原始文本
→ 去除 Markdown 代码块
→ 提取最外层 JSON 对象
→ json.loads
→ ObservationBatch.model_validate
→ 业务语义校验
```

不得使用危险的：

```python
eval()
```

解析失败时：

1. 保存原始响应；
2. 写入错误日志；
3. 当前窗口标记失败；
4. 继续处理下一个窗口；
5. 不生成伪造 ObservationBatch。

---

# 十三、运行存储结构

每次运行创建：

```text
outputs/<run_id>/
├── run_meta.json
├── resolved_config.yaml
├── windows.jsonl
├── observations.jsonl
├── api_metrics.jsonl
├── errors.jsonl
├── raw_responses/
└── sampled_frames/
```

## 13.1 run_id

格式：

```text
YYYYMMDD_HHMMSS_<experiment_name>_<short_hash>
```

确保不会覆盖已有输出。

## 13.2 run_meta.json

至少包含：

```json
{
  "run_id": "...",
  "experiment_name": "...",
  "video_path": "...",
  "video_sha256": "...",
  "video_metadata": {},
  "resolved_model": "...",
  "model_source": "yaml 或 environment",
  "config_sha256": "...",
  "system_prompt_sha256": "...",
  "user_prompt_sha256": "...",
  "git_commit": "...",
  "python_version": "...",
  "opencv_version": "...",
  "start_time": "...",
  "end_time": null
}
```

运行结束后补充：

```text
end_time
processed_windows
successful_windows
failed_windows
```

## 13.3 windows.jsonl

保存窗口和采样信息。

## 13.4 observations.jsonl

只保存通过 Schema 和业务语义校验的观察。

## 13.5 api_metrics.jsonl

保存：

```text
窗口编号
模型
请求延迟
重试次数
输入 token
输出 token
状态
```

## 13.6 errors.jsonl

保存：

```text
窗口编号
错误阶段
错误类型
错误消息
是否可重试
原始响应路径
```

## 13.7 raw_responses

无论响应成功或失败，只要配置：

```yaml
save_raw_responses: true
```

就保存每个窗口的原始文本：

```text
raw_responses/window_000012.txt
```

---

# 十四、Pipeline

新增：

```text
pipeline.py
```

主流程应清晰表达为：

```python
class StreamingVideoPipeline:
    def run(self, video_path: Path) -> RunSummary:
        metadata = read_video_metadata(video_path)
        windows = build_video_windows(...)

        selected_windows = select_windows(...)

        for window in selected_windows:
            sampled_frames = sample_window_frames(...)
            prompt = prompt_builder.build(...)
            raw_result = qwen_client.observe(...)
            parsed_batch = parser.parse(...)
            validated_batch = validator.validate(...)
            store.write_window(...)
            store.write_observation(...)
```

不要在一个函数中写数百行逻辑。

每个窗口应独立捕获错误：

```python
try:
    process_window(...)
except QwenStreamVideoError as exc:
    store.write_error(...)
    continue
```

但 `KeyboardInterrupt` 不应被吞掉。

---

# 十五、命令行参数

保留并整理现有参数，至少支持：

```text
--video
--config
--output-dir
--start-time
--end-time
--start-window
--end-window
--realtime
--no-state
--save-frames
--dry-run
```

新增：

```text
--validate-only
```

含义：

* 不调用模型；
* 只检查视频、配置、提示词和输出目录；
* 构造窗口；
* 输出预计窗口数和预计抽样帧数。

新增：

```text
--print-config
```

输出最终解析配置后退出，隐藏 API Key。

---

# 十六、测试要求

使用 pytest。

## 16.1 test_config.py

至少包含：

```text
test_load_valid_config
test_invalid_window_seconds
test_invalid_frame_limits
test_environment_model_override
test_cli_override_has_highest_priority
```

## 16.2 test_windows.py

至少包含：

```text
test_regular_windows
test_short_video
test_tail_completion_window
test_windows_do_not_exceed_duration
test_global_index_preserved_after_selection
test_realtime_target_from_nonzero_start
```

重点验证：

```python
video_origin = 480.0
window_end = 486.0
wall_start = 1000.0
```

结果必须是：

```text
1006.0
```

而不是：

```text
1486.0
```

## 16.3 test_observation_schema.py

至少包含：

```text
test_valid_observation_batch
test_confidence_above_one_fails
test_confidence_below_zero_fails
test_mutable_defaults_are_isolated
test_invalid_enum_fails
```

## 16.4 test_validator.py

至少包含：

```text
test_duplicate_entity_local_id
test_duplicate_action_local_id
test_missing_actor_reference
test_missing_target_reference
test_invalid_evidence_frame
test_valid_references_pass
```

## 16.5 不调用真实 API

单元测试不能调用真实千问 API。

需要为 `QwenClient` 提供 Mock 或 Fake 实现。

---

# 十七、README 更新

README 必须新增：

1. 项目定位；
2. 当前属于本地视频流式模拟，不是真实 RTSP 流；
3. 安装方式；
4. 环境变量配置；
5. 基础运行命令；
6. `--dry-run`；
7. 输出目录结构；
8. 增量 Observation Schema；
9. 当前未实现的功能；
10. 常见错误排查；
11. 测试命令。

明确说明：

```text
模型输出的 candidate_global_id 不是正式稳定实体 ID。
实体注册和全局状态维护将在下一阶段实现。
```

---

# 十八、安全要求

`.env.example` 只能包含：

```env
DASHSCOPE_API_KEY=your_dashscope_api_key_here
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3-vl-plus
```

不得写入真实 Key。

检查 `.gitignore` 至少包含：

```text
.env
outputs/
__pycache__/
.pytest_cache/
.ruff_cache/
*.pyc
```

不得把：

* API Key；
* 原始私有视频；
* 大量输出；
* Base64 响应；

提交进 Git。

---

# 十九、验收标准

本阶段完成后必须满足以下要求。

## 工程验收

* `run.py` 不再包含主要业务逻辑；
* 核心代码位于 `src/qwen_stream_video/`；
* 可以通过 `pip install -e .` 安装；
* `python run.py --help` 正常；
* `qwen-stream-video --help` 正常；
* 配置有严格类型；
* 公共函数具有类型标注。

## 功能验收

* 能处理本地 MP4；
* 能生成因果滑动窗口；
* 不采样窗口右边界及未来帧；
* 能调用千问视觉接口；
* 模型输出为增量 ObservationBatch；
* 响应经过 Pydantic 校验；
* 响应经过语义引用校验；
* 单窗口错误不会终止整个任务；
* 所有原始响应可追踪；
* 输出不会覆盖旧运行结果。

## 正确性验收

* 非零 `start_time` 下实时等待时间正确；
* 截取部分窗口后保留全局窗口编号；
* API 环境变量覆盖会记录在元数据中；
* JSON 格式错误不会重新发送完整视频请求；
* 无效 Observation 不会写入 `observations.jsonl`；
* 任何失败窗口都不会伪造成功结果。

## 测试验收

执行：

```bash
pytest -q
```

必须全部通过。

执行：

```bash
ruff check .
```

不得存在严重错误。

---

# 二十、最终交付内容

完成后输出：

1. 修改后的完整代码；
2. 新目录树；
3. 主要架构说明；
4. 新旧运行方式；
5. 配置说明；
6. Observation Schema 示例；
7. 输出目录示例；
8. 测试结果；
9. 尚未实现的后续功能；
10. 本阶段修改涉及的文件列表。

不要只输出设计文档，必须实际修改仓库代码。

不要实现 EntityRegistry、ActionTracker 和 TransitionEngine。本阶段通过验收后，再进入第二阶段。
