# qwen_stream_video 详细修改方案

## 一、修改目标

当前仓库已经完成了最小闭环：

```text
本地视频
→ 因果滑动窗口
→ 均匀抽帧
→ 千问视觉 API
→ 当前窗口结构化结果
→ 压缩后传递给下一窗口
→ JSONL 输出
```

仓库目前仍由一个约 669 行的 `run.py` 承担参数解析、视频读取、抽帧、API 调用、JSON 解析、状态压缩和结果存储等职责；模型输出的是每个窗口的完整局部描述，程序只做基础字段补全。

本次修改后的核心目标是：

```text
当前窗口帧
→ 模型生成增量视觉观察
→ 程序匹配全局实体
→ 程序维护动作生命周期
→ 程序执行状态转移
→ 输出观察、事件和状态快照
```

模型只负责回答：

> 当前窗口中看到了什么？

程序负责回答：

> 它和历史中的哪个实体对应？
> 是否是同一个动作？
> 哪个状态真正发生了变化？
> 当前全局状态是什么？

---

# 二、修改范围与阶段边界

建议不要一次性直接做 RTSP、检测器、数据库、Web 服务和复杂 Agent。

整个项目分成四个阶段。

## 阶段一：结构化重构

目标：

* 拆分单文件；
* 引入严格数据模型；
* 改成增量观察协议；
* 完善错误处理；
* 保证实验可复现。

完成后仍然使用本地完整视频，不改变当前主要运行方式。

## 阶段二：状态维护系统

目标：

* 实现全局实体注册；
* 实现动作生命周期；
* 实现属性状态更新；
* 实现镜头切换和遮挡处理；
* 输出全局状态快照。

这是最重要的研究功能。

## 阶段三：实时调度模拟

目标：

* 修正当前 `--realtime`；
* 引入生产者—消费者结构；
* 增加窗口队列和积压控制；
* 模拟模型慢于视频流的情况；
* 实现 latest-window-only。

此阶段仍可以使用本地视频模拟实时流。

## 阶段四：真实视频流接入

目标：

* 摄像头；
* RTSP；
* 环形帧缓冲区；
* 丢帧、断流和重连；
* 实时状态输出接口。

阶段四不应阻塞阶段一和阶段二。

---

# 三、推荐的项目目录

建议将项目改造成标准 Python 包：

```text
qwen_stream_video/
├── pyproject.toml
├── README.md
├── .env.example
├── configs/
│   ├── base.yaml
│   ├── substation.yaml
│   └── experiments/
│       ├── no_state.yaml
│       ├── full_state.yaml
│       └── incremental_state.yaml
├── prompts/
│   ├── observation_system.txt
│   ├── observation_user.txt
│   └── json_repair.txt
├── vocabularies/
│   ├── actions.yaml
│   ├── entity_types.yaml
│   └── attributes.yaml
├── src/
│   └── qwen_stream_video/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── pipeline.py
│       ├── exceptions.py
│       │
│       ├── domain/
│       │   ├── observation.py
│       │   ├── state.py
│       │   ├── event.py
│       │   └── enums.py
│       │
│       ├── video/
│       │   ├── source.py
│       │   ├── file_source.py
│       │   ├── window.py
│       │   ├── sampler.py
│       │   ├── frame_cache.py
│       │   └── scene_detector.py
│       │
│       ├── inference/
│       │   ├── base.py
│       │   ├── qwen_client.py
│       │   ├── prompt_builder.py
│       │   ├── parser.py
│       │   ├── validator.py
│       │   └── repair.py
│       │
│       ├── state/
│       │   ├── entity_registry.py
│       │   ├── action_tracker.py
│       │   ├── transition_engine.py
│       │   ├── global_state.py
│       │   └── context_builder.py
│       │
│       ├── runtime/
│       │   ├── scheduler.py
│       │   ├── queue.py
│       │   └── metrics.py
│       │
│       └── storage/
│           ├── run_store.py
│           ├── jsonl_writer.py
│           └── snapshot_writer.py
├── scripts/
│   ├── analyze_results.py
│   ├── compare_runs.py
│   └── make_test_video.py
└── tests/
    ├── unit/
    ├── integration/
    ├── fixtures/
    └── golden/
```

原来的 `run.py` 最终只保留兼容入口：

```python
from qwen_stream_video.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

---

# 四、重新定义模型输出协议

## 4.1 当前协议的问题

当前提示词要求模型输出：

```text
scene
entities
actions
state_changes
observed_results
uncertainties
```

并让模型自行创建和沿用 `person_1`、`device_1` 等 ID。

这样会让模型同时承担：

* 实体发现；
* 全局 ID 分配；
* 跨窗口实体匹配；
* 动作去重；
* 状态历史维护；
* 状态变化判断。

建议取消这种设计。

## 4.2 新的模型输出：ObservationBatch

模型只输出当前窗口的局部观察：

```json
{
  "window": {
    "start_seconds": 12.0,
    "end_seconds": 18.0
  },
  "summary": "工作人员在柜门附近进行操作。",
  "scene_observation": {
    "camera_change": false,
    "view_type": "wide",
    "visibility": "clear"
  },
  "entities": [
    {
      "local_id": "local_entity_1",
      "entity_type": "person",
      "name": "worker",
      "appearance": {
        "clothing": "blue_uniform",
        "helmet": "present"
      },
      "spatial_region": "center_left",
      "candidate_global_id": "person_001",
      "confidence": 0.91,
      "evidence_frames": [1, 2, 3]
    }
  ],
  "actions": [
    {
      "local_id": "local_action_1",
      "actor_local_id": "local_entity_1",
      "action_type": "open",
      "target_local_id": "local_entity_2",
      "tool_local_id": null,
      "phase_observation": "ongoing",
      "description": "工作人员持续拉动柜门",
      "confidence": 0.86,
      "evidence_frames": [2, 3, 4]
    }
  ],
  "attribute_observations": [
    {
      "entity_local_id": "local_entity_2",
      "attribute": "door_state",
      "value": "partially_open",
      "confidence": 0.83,
      "evidence_frames": [3, 4, 5]
    }
  ],
  "uncertainties": []
}
```

### 重要原则

`local_id` 只在当前窗口内有效。

模型可以通过 `candidate_global_id` 建议它认为对应的历史实体，但这个字段只是候选建议，程序必须重新验证。

模型不能直接创建正式的：

```text
person_001
device_003
action_015
event_028
```

正式全局 ID 只能由程序分配。

---

# 五、Pydantic 数据模型

新增：

```text
src/qwen_stream_video/domain/observation.py
```

核心模型建议包括：

```python
class WindowDescriptor(BaseModel):
    index: int
    start_seconds: float
    end_seconds: float
    frame_ids: list[str]


class FrameEvidence(BaseModel):
    frame_indices: list[int]
    timestamps_seconds: list[float]


class EntityObservation(BaseModel):
    local_id: str
    entity_type: EntityType
    name: str = "unknown"
    appearance: dict[str, str] = {}
    spatial_region: SpatialRegion = SpatialRegion.UNKNOWN
    candidate_global_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_frames: list[int]


class ActionObservation(BaseModel):
    local_id: str
    actor_local_id: str
    action_type: ActionType
    target_local_id: str | None = None
    tool_local_id: str | None = None
    phase_observation: ActionPhaseObservation
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_frames: list[int]


class AttributeObservation(BaseModel):
    entity_local_id: str
    attribute: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_frames: list[int]


class ObservationBatch(BaseModel):
    window: WindowDescriptor
    summary: str
    scene_observation: SceneObservation
    entities: list[EntityObservation]
    actions: list[ActionObservation]
    attribute_observations: list[AttributeObservation]
    uncertainties: list[UncertaintyObservation]
```

## 5.1 三层校验

模型返回后依次进行：

```text
JSON 语法校验
→ Pydantic Schema 校验
→ 业务语义校验
```

业务语义校验包括：

* `actor_local_id` 必须在当前实体列表中；
* `target_local_id` 必须存在或为 `null`；
* `evidence_frames` 不能越界；
* `confidence` 必须位于 `[0,1]`；
* 动作类型必须属于受控词表；
* 属性值必须属于相应枚举；
* 同一窗口内 `local_id` 不得重复；
* 当前窗口时间由程序覆盖，禁止使用模型时间；
* `before == after` 时不得生成状态变化。

当前代码只检查顶层字段是否存在，并将错误类型重置为空列表，不能保证内部引用和字段语义正确。

---

# 六、实体注册系统 EntityRegistry

## 6.1 全局实体结构

新增：

```python
class GlobalEntityState(BaseModel):
    entity_id: str
    entity_type: EntityType
    canonical_name: str
    aliases: list[str]

    first_seen_window: int
    last_seen_window: int
    first_seen_time: float
    last_seen_time: float

    current_scene_id: str | None
    visibility: VisibilityState

    stable_attributes: dict[str, AttributeState]
    appearance_signature: dict[str, str]
    spatial_history: list[SpatialObservation]

    confidence: float
    status: EntityLifecycleStatus
```

其中：

```text
visibility:
visible | partial | occluded | not_visible

status:
active | temporarily_missing | inactive | merged
```

必须明确：

```text
当前窗口没有看到实体
≠
实体已经不存在
```

## 6.2 实体匹配流程

对于每一个局部实体，按以下流程匹配全局实体。

### 第一步：候选过滤

只选择：

* 类型相同；
* 最近若干窗口出现过；
* 当前场景或相关场景中存在；
* 未被标记为永久失效；

的实体。

例如：

```python
candidates = registry.find_candidates(
    entity_type=observation.entity_type,
    current_scene_id=current_scene_id,
    max_missing_windows=10,
)
```

### 第二步：硬约束排除

满足以下情况直接排除：

* `person` 不得匹配 `device`；
* 明确不同颜色工装且置信度较高；
* 一个窗口中两个同时可见局部实体不得映射到同一全局实体；
* 明确不同设备类别；
* 当前空间位置明显冲突且镜头未切换。

### 第三步：计算匹配分数

第一版不需要训练 ReID 模型，可以采用可解释规则：

```text
总分 =
0.30 × 名称/类型相似度
+ 0.25 × 外观属性相似度
+ 0.15 × 空间连续性
+ 0.15 × 当前关系一致性
+ 0.10 × 最近出现程度
+ 0.05 × 模型候选 ID 建议
```

示例：

```python
match_score = (
    0.30 * type_name_score
    + 0.25 * appearance_score
    + 0.15 * spatial_score
    + 0.15 * relation_score
    + 0.10 * recency_score
    + 0.05 * candidate_hint_score
)
```

### 第四步：阈值决策

建议初始参数：

```yaml
entity_registry:
  confident_match_threshold: 0.78
  ambiguous_match_threshold: 0.58
  max_missing_windows: 10
```

规则：

```text
score ≥ 0.78
→ 直接匹配已有实体

0.58 ≤ score < 0.78
→ 标记 ambiguous，创建临时实体或等待后续证据

score < 0.58
→ 创建新实体
```

### 第五步：避免错误覆盖

模糊匹配时绝不能为了保持 ID 连续性强行合并。

宁可暂时生成：

```text
temp_person_003
```

也不要错误覆盖：

```text
person_001
```

后续可以通过更多窗口证据执行：

```text
temp_person_003 → person_001
```

并生成实体合并事件。

## 6.3 全局 ID 规则

统一由程序生成：

```text
person_0001
device_0001
component_0001
tool_0001
sign_0001
```

ID 一旦分配不得复用。

---

# 七、动作生命周期 ActionTracker

当前提示词让模型直接判断 `started`、`ongoing`、`ended` 和 `instant`，但窗口重叠和抽帧遗漏会导致动作碎片化。

建议模型只报告“本窗口看见动作处于什么状态”，程序维护正式生命周期。

## 7.1 全局动作结构

```python
class GlobalActionState(BaseModel):
    action_id: str

    actor_id: str
    action_type: ActionType
    target_id: str | None
    tool_id: str | None

    lifecycle: ActionLifecycle

    start_window: int
    last_observed_window: int
    end_window: int | None

    start_time_lower: float
    start_time_upper: float
    end_time_lower: float | None
    end_time_upper: float | None

    observed_windows: list[int]
    missing_window_count: int
    confidence: float
    evidence: list[EvidenceReference]
```

生命周期：

```text
candidate
→ started
→ ongoing
→ possible_ended
→ ended

也可以进入：
uncertain
interrupted
```

## 7.2 动作键

基础动作键：

```python
action_key = (
    actor_id,
    action_type,
    target_id,
    tool_id,
)
```

但不能仅凭动作键永久合并，因为同一个人可能多次执行同一个动作。

还需要加入时间间隔：

```yaml
action_tracker:
  continue_max_gap_windows: 1
  end_missing_windows: 2
  repeat_action_min_gap_seconds: 5.0
```

## 7.3 生命周期规则

```text
历史中没有匹配动作，本窗口观察到
→ started

历史中动作 started/ongoing，本窗口继续观察
→ ongoing

历史中动作 ongoing，本窗口第一次未观察到
→ possible_ended

连续两个窗口未观察到
→ ended

模型明确观察到动作完整发生且持续很短
→ instant

镜头切换或严重遮挡时动作消失
→ 保持 uncertain，不立即结束
```

## 7.4 时间区间而不是伪精确时间点

由于默认采样约 1 FPS，动作发生时间通常只能定位在两个采样帧之间。

不要保存：

```json
{
  "start_time": 13.427
}
```

应该保存：

```json
{
  "start_time_interval": {
    "lower": 13.0,
    "upper": 14.0
  }
}
```

---

# 八、状态转移引擎 TransitionEngine

模型只报告属性观察：

```json
{
  "entity_local_id": "local_device_1",
  "attribute": "door_state",
  "value": "open",
  "confidence": 0.82
}
```

由程序决定是否更新全局状态。

## 8.1 属性状态结构

```python
class AttributeState(BaseModel):
    value: str
    confidence: float

    confirmed: bool
    first_observed_window: int
    last_observed_window: int

    supporting_observations: int
    contradicting_observations: int

    previous_value: str | None
    pending_value: str | None
```

## 8.2 更新规则

推荐第一版采用确定性规则：

```text
高置信度观察，confidence ≥ 0.80
→ 可以直接确认

中等置信度，0.55 ≤ confidence < 0.80
→ 连续两个窗口相同才确认

低置信度，confidence < 0.55
→ 仅记录，不更新状态
```

## 8.3 冲突处理

例如当前状态：

```text
door_state = closed
```

新窗口观察：

```text
door_state = open，confidence = 0.61
```

不能立即更新，应生成：

```text
pending_value = open
supporting_observations = 1
```

下一个窗口再次观察为 `open` 后再提交：

```text
closed → open
```

如果下一个窗口又观察为 `closed`：

```text
取消 pending_value
记录一次冲突观察
```

## 8.4 状态变化事件

正式事件由程序产生：

```json
{
  "event_id": "event_00021",
  "event_type": "attribute_transition",
  "entity_id": "device_0003",
  "attribute": "door_state",
  "before": "closed",
  "after": "open",
  "effective_window": 17,
  "confidence": 0.88,
  "evidence": [
    {
      "window_index": 16,
      "frame_indices": [4, 5]
    },
    {
      "window_index": 17,
      "frame_indices": [2, 3]
    }
  ]
}
```

这样可以区分：

```text
模型观察
程序确认的状态变化
当前全局状态
```

---

# 九、受控词表设计

新增：

```text
vocabularies/actions.yaml
vocabularies/attributes.yaml
```

## 9.1 动作类型

第一版建议控制在 20～30 个：

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
  - cut
  - strip
  - measure
  - record
  - point
  - unknown
```

自然语言描述单独保留：

```json
{
  "action_type": "adjust",
  "description": "工作人员右手在端子排附近调整白色导线"
}
```

## 9.2 属性词表

```yaml
attributes:
  door_state:
    values:
      - unknown
      - closed
      - partially_open
      - open

  connection_state:
    values:
      - unknown
      - connected
      - partially_disconnected
      - disconnected

  switch_state:
    values:
      - unknown
      - on
      - off
      - work
      - test
      - intermediate

  holding_state:
    values:
      - unknown
      - held
      - released

  visibility:
    values:
      - visible
      - partial
      - occluded
      - not_visible
```

属性必须采用：

```text
entity_type + attribute_name
```

进行约束，避免给人员输出 `door_state`。

---

# 十、镜头切换与场景管理

## 10.1 SceneState

```python
class SceneState(BaseModel):
    scene_id: str
    view_type: ViewType
    start_window: int
    last_active_window: int
    continuity: SceneContinuity
    visible_entity_ids: list[str]
```

`view_type`：

```text
wide
medium
closeup
detail
unknown
```

## 10.2 场景切换规则

模型输出：

```json
{
  "camera_change": true,
  "view_type": "closeup",
  "visibility": "partial"
}
```

程序执行：

1. 创建或恢复相应场景；
2. 原场景实体设置为 `not_visible`；
3. 不删除原实体；
4. 近景中的局部部件允许先使用临时 ID；
5. 返回全景时重新激活已有实体；
6. 镜头切换附近降低实体匹配置信度；
7. 镜头切换期间动作消失时不立即判定结束。

仓库已有改进文档也提出了“不因镜头切换删除实体、近景使用临时 ID、返回全景后重新激活”的处理方向。

## 10.3 简单场景切换检测

第一版可以组合：

```text
模型判断
+ 帧直方图差异
+ SSIM/感知哈希变化
```

暂不需要引入复杂视觉模型。

程序检测结果作为辅助信号，不直接取代模型判断。

---

# 十一、上一状态上下文 ContextBuilder

不能再把完整 GlobalState 发给模型，否则视频越长，上下文越大。

新增：

```text
state/context_builder.py
```

只发送与当前窗口有关的信息：

```json
{
  "scene": {
    "scene_id": "scene_002",
    "view_type": "closeup"
  },
  "candidate_entities": [
    {
      "entity_id": "person_0001",
      "type": "person",
      "name": "worker",
      "appearance": {
        "clothing": "blue_uniform"
      },
      "last_seen_windows_ago": 1
    }
  ],
  "active_actions": [
    {
      "action_id": "action_0012",
      "actor_id": "person_0001",
      "action_type": "adjust",
      "target_id": "component_0004"
    }
  ],
  "pending_attributes": [
    {
      "entity_id": "device_0002",
      "attribute": "door_state",
      "candidate_value": "open"
    }
  ]
}
```

## 11.1 上下文选择规则

只保留：

* 当前场景中最近出现的实体；
* 最近 3～5 个窗口出现的实体；
* 活跃动作；
* 待确认状态；
* 与活跃实体存在关系的设备和工具；
* 最近一次镜头切换信息。

不发送：

* 所有历史事件；
* 已结束很久的动作；
* 与当前场景无关的完整实体；
* 每个实体的所有历史属性。

配置示例：

```yaml
context:
  max_entities: 15
  recent_window_count: 5
  max_active_actions: 8
  max_pending_attributes: 8
  max_serialized_characters: 6000
```

---

# 十二、帧编号与证据映射

当前请求把图像列表作为视频传入，但模型没有明确的帧编号语义。建议在用户提示词中加入：

```text
当前窗口抽取了 6 帧：

F0 = 00:00:12.500
F1 = 00:00:13.500
F2 = 00:00:14.500
F3 = 00:00:15.500
F4 = 00:00:16.500
F5 = 00:00:17.500
```

模型只能输出：

```json
{
  "evidence_frames": [2, 3, 4]
}
```

程序再转换成真实时间：

```python
timestamps = [
    sampled_frames[index].timestamp_seconds
    for index in evidence_frames
]
```

这样能够防止模型编造精确时间。仓库的改进文档也已经提出帧编号与程序侧时间映射。

---

# 十三、Qwen 调用层重构

将当前 `QwenWindowAnalyzer` 拆成：

```text
QwenClient
PromptBuilder
ResponseParser
SchemaValidator
JsonRepairService
```

## 13.1 QwenClient

只负责：

* 构造 API 请求；
* 调用模型；
* 捕获 HTTP 信息；
* 返回原始文本和元数据。

不负责 JSON 解析和状态更新。

返回：

```python
class RawInferenceResult(BaseModel):
    raw_text: str
    request_id: str | None
    resolved_model: str
    latency_seconds: float
    usage: dict[str, Any] | None
    attempt_count: int
```

## 13.2 错误分类

新增异常：

```python
class VideoReadError(Exception): ...
class InferenceNetworkError(Exception): ...
class InferenceRateLimitError(Exception): ...
class InferenceServerError(Exception): ...
class ModelOutputParseError(Exception): ...
class ModelOutputSchemaError(Exception): ...
class ModelOutputSemanticError(Exception): ...
```

当前代码将 API、网络和 JSON 解析错误放在同一重试循环里，因此 JSON 损坏也会重新发送整组视频帧。

修改为：

```text
网络超时、429、5xx
→ 重发视觉请求

JSON 语法错误
→ 本地 json-repair

仍然失败
→ 只发送原始文本进行 JSON 修复，不再发送视频

Schema 错误
→ 尝试一次文本格式修复

业务语义错误
→ 记录错误，不更新状态
```

## 13.3 原始响应永久保存

所有窗口都保存：

```text
raw_responses/window_000012.txt
```

不能只在出现 warning 时将原始响应写入 `windows.jsonl`。

这样可以进行：

* Prompt 回归分析；
* JSON 修复策略验证；
* 模型版本比较；
* 错误复盘。

---

# 十四、窗口与索引修正

## 14.1 保留全局窗口编号

当前 `--start-window` 切片后重新 `enumerate(windows)`，输出编号会从 0 开始。

建议定义：

```python
class VideoWindow(BaseModel):
    global_index: int
    selected_index: int
    start_seconds: float
    end_seconds: float
```

例如从第 100 个窗口开始运行：

```json
{
  "global_window_index": 100,
  "run_window_index": 0
}
```

这样多个区间运行后才能正确合并。

## 14.2 修复实时模式时间原点

当前实时等待使用：

```python
target_wall_time = wall_start + end_seconds
```

当 `--start-time` 不为 0 时，会把视频绝对时间错误地当成等待时长。

改为：

```python
realtime_origin_video_time = selected_windows[0].start_seconds

target_wall_time = (
    wall_start
    + window.end_seconds
    - realtime_origin_video_time
)
```

## 14.3 记录窗口生成原因

末尾补齐窗口可能与前一个窗口高度重叠，应记录：

```json
{
  "window_type": "regular"
}
```

或者：

```json
{
  "window_type": "tail_completion"
}
```

方便分析重复事件。

---

# 十五、视频抽帧模块改造

## 15.1 抽帧结果结构

```python
class SampledFrame(BaseModel):
    sample_index: int
    timestamp_seconds: float
    frame_index: int
    image_path: str | None
    encoding_cache_key: str
```

## 15.2 帧缓存

由于相邻窗口重叠，相同帧可能被重复读取、缩放和 JPEG 编码。

增加：

```python
class FrameCache:
    decoded_frames: LRUCache[int, np.ndarray]
    encoded_frames: LRUCache[tuple[int, int, int], str]
```

编码缓存键：

```text
frame_index
+ max_image_side
+ jpeg_quality
```

## 15.3 自适应采样放到后续阶段

第一版保持均匀采样作为基线。

第二版再加入：

```text
基础帧：1 FPS
+
高变化区域补充帧
```

变化信号可以采用：

* 帧差；
* SSIM；
* 光流幅值；
* 场景切换分数。

建议配置：

```yaml
sampling:
  mode: uniform
  base_fps: 1.0
  min_frames: 4
  max_frames: 12

  adaptive:
    enabled: false
    motion_threshold: 0.25
    boosted_fps: 3.0
```

不要在第一阶段立即加入复杂自适应采样，否则无法判断性能变化来自状态系统还是采样策略。

---

# 十六、运行管线 Pipeline

核心执行流程改为：

```python
for window in scheduler:
    sampled_frames = sampler.sample(window)

    prompt_context = context_builder.build(
        window=window,
        global_state=global_state,
    )

    raw_result = observer.observe(
        window=window,
        frames=sampled_frames,
        context=prompt_context,
    )

    observation_batch = parser.parse_and_validate(raw_result)

    entity_mapping = entity_registry.resolve(
        observation_batch.entities,
        global_state,
    )

    action_events = action_tracker.update(
        observation_batch.actions,
        entity_mapping,
    )

    state_events = transition_engine.apply(
        observation_batch.attribute_observations,
        entity_mapping,
    )

    global_state.commit(
        entity_mapping,
        action_events,
        state_events,
    )

    run_store.write_all(...)
```

## 16.1 事务式状态更新

每个窗口必须遵循：

```text
解析成功
+ Schema 成功
+ 语义校验成功
+ 状态更新成功
→ commit
```

任何一步失败：

```text
→ rollback
→ 当前窗口不更新 GlobalState
```

即使使用内存状态，也建议实现：

```python
candidate_state = global_state.copy(deep=True)

try:
    update(candidate_state)
except Exception:
    discard(candidate_state)
else:
    global_state = candidate_state
```

这样错误窗口不会污染后续所有窗口。

---

# 十七、输出文件重新设计

每次运行生成：

```text
outputs/<run_id>/
├── run_meta.json
├── resolved_config.yaml
├── prompt_snapshot/
│   ├── observation_system.txt
│   └── observation_user.txt
├── windows.jsonl
├── observations.jsonl
├── events.jsonl
├── state_snapshots.jsonl
├── api_metrics.jsonl
├── errors.jsonl
├── raw_responses/
└── sampled_frames/
```

## 17.1 windows.jsonl

只记录窗口和采样：

```json
{
  "global_window_index": 12,
  "run_window_index": 0,
  "start_seconds": 36.0,
  "end_seconds": 42.0,
  "sampled_frame_indices": [912, 937, 962, 987, 1012, 1037],
  "sampled_timestamps": [36.5, 37.5, 38.5, 39.5, 40.5, 41.5],
  "window_type": "regular"
}
```

## 17.2 observations.jsonl

保存模型经过 Schema 校验后的增量观察。

## 17.3 events.jsonl

保存程序确认的事件：

```text
entity_created
entity_reidentified
entity_visibility_changed
action_started
action_ended
attribute_transition
scene_changed
observation_gap
```

## 17.4 state_snapshots.jsonl

每隔一定窗口保存全量状态：

```yaml
storage:
  snapshot_interval_windows: 10
```

最后一个窗口必须保存最终状态。

## 17.5 api_metrics.jsonl

记录：

```json
{
  "window_index": 12,
  "model": "qwen3-vl-plus",
  "request_id": "...",
  "latency_seconds": 2.31,
  "attempt_count": 1,
  "input_tokens": 4321,
  "output_tokens": 276,
  "parse_repaired": false,
  "status": "ok"
}
```

## 17.6 run_meta.json

增加：

* Git commit SHA；
* 配置文件哈希；
* Prompt 哈希；
* 视频 SHA256；
* 最终解析模型；
* 环境变量覆盖来源；
* Python 版本；
* OpenCV 版本；
* 依赖版本；
* 开始和结束时间；
* 实验名称。

当前配置允许环境变量 `QWEN_MODEL` 覆盖 YAML 模型，但运行元数据只保存原始配置，没有明确记录覆盖后的最终模型。

---

# 十八、配置文件调整

建议配置结构：

```yaml
experiment:
  name: incremental_state_v1
  seed: 42

video:
  window_seconds: 6.0
  stride_seconds: 3.0

sampling:
  mode: uniform
  sample_fps: 1.0
  min_frames: 4
  max_frames: 12
  max_image_side: 768
  jpeg_quality: 80

model:
  provider: dashscope
  name: qwen3-vl-plus
  temperature: 0
  max_tokens: 1000
  timeout_seconds: 120
  network_retries: 2
  enable_thinking: false

observation:
  schema_version: "1.0"
  use_candidate_entity_hints: true
  require_evidence_frames: true

entity_registry:
  confident_match_threshold: 0.78
  ambiguous_match_threshold: 0.58
  max_missing_windows: 10

action_tracker:
  end_missing_windows: 2
  continue_max_gap_windows: 1
  repeat_action_min_gap_seconds: 5.0

transition:
  high_confidence_threshold: 0.80
  medium_confidence_threshold: 0.55
  medium_confirmation_windows: 2

context:
  recent_window_count: 5
  max_entities: 15
  max_active_actions: 8
  max_serialized_characters: 6000

runtime:
  mode: offline_fast
  queue_size: 2
  backpressure_policy: latest_window_only

storage:
  save_raw_responses: true
  save_sampled_frames: false
  snapshot_interval_windows: 10
```

配置读取建议使用 Pydantic Settings，启动时输出最终生效配置。

---

# 十九、真正的实时调度设计

这一部分放在状态系统稳定之后。

## 19.1 统一 VideoSource 接口

```python
class VideoSource(Protocol):
    def open(self) -> None: ...
    def read(self) -> FramePacket | None: ...
    def close(self) -> None: ...
```

实现：

```text
FileVideoSource
CameraVideoSource
RtspVideoSource
```

帧结构：

```python
class FramePacket(BaseModel):
    sequence_id: int
    source_timestamp: float
    arrival_timestamp: float
    image: Any
```

## 19.2 环形缓冲区

```python
class FrameRingBuffer:
    max_duration_seconds: float
    frames: deque[FramePacket]
```

窗口组装器只读取：

```text
window_start ≤ frame.source_timestamp < window_end
```

## 19.3 生产者—消费者

```text
视频读取线程
→ FrameRingBuffer
→ WindowAssembler
→ BoundedWindowQueue
→ QwenInferenceWorker
→ StateEngine
```

## 19.4 回压策略

默认：

```yaml
runtime:
  queue_size: 2
  backpressure_policy: latest_window_only
```

当队列已满：

1. 保留当前正在处理的窗口；
2. 丢弃尚未处理的旧窗口；
3. 保留最新完整窗口；
4. 生成 `observation_gap` 事件；
5. 增加状态不确定度；
6. 记录丢弃数量和时间范围。

不能静默丢弃，否则后续状态变化可能被误判。

## 19.5 实时指标

每个窗口记录：

```text
capture_lag
queue_wait
preprocess_latency
api_latency
state_update_latency
end_to_end_latency
realtime_factor
queue_depth
dropped_window_count
```

其中：

```text
Realtime Factor =
总处理耗时 / 对应视频时长
```

`RTF <= 1` 才表示平均处理速度跟得上视频产生速度。

---

# 二十、提示词修改方案

## 20.1 系统提示词职责

系统提示词只保留：

* 当前窗口因果约束；
* 不得预测未来；
* 不得补全不可见事实；
* 只输出增量观察；
* 只使用受控枚举；
* 必须引用证据帧；
* 全局 ID 只是候选引用；
* 不负责确认状态变化；
* 不负责判断动作最终结束；
* 输出合法 JSON。

删除让模型维护完整全局状态的要求。

## 20.2 用户提示词内容

动态注入：

```text
视频基本背景
当前窗口范围
帧编号与时间
候选历史实体
当前活跃动作
待确认状态
输出 Schema 简要说明
```

不要把完整长 Schema 每个窗口重复发送，可以：

* 系统提示词中给完整 Schema；
* 用户提示词只给字段约束；
* 或使用结构化输出能力。

## 20.3 Prompt 版本管理

每个 Prompt 文件顶部增加：

```text
prompt_version: observation-v1.0
schema_version: 1.0
```

运行时保存 Prompt 哈希。

---

# 二十一、测试方案

## 21.1 单元测试

### 视频窗口

```text
test_build_windows_normal
test_build_windows_short_video
test_build_windows_tail_completion
test_no_sample_after_window_end
test_global_window_index_preserved
test_realtime_origin_with_start_time
```

### Schema

```text
test_valid_observation_batch
test_invalid_confidence
test_duplicate_local_entity_id
test_invalid_action_reference
test_evidence_frame_out_of_range
test_invalid_attribute_value
```

### EntityRegistry

```text
test_same_person_reuses_global_id
test_two_visible_people_not_merged
test_entity_not_deleted_when_occluded
test_ambiguous_entity_uses_temp_id
test_scene_change_does_not_overwrite_id
```

### ActionTracker

```text
test_action_started
test_action_continues_across_overlap
test_action_not_ended_after_one_missing_window
test_action_ends_after_two_missing_windows
test_repeated_action_creates_new_action_id
```

### TransitionEngine

```text
test_high_confidence_transition_commits
test_medium_transition_requires_confirmation
test_low_confidence_does_not_update
test_conflicting_observation_cancels_pending
test_failed_window_does_not_mutate_state
```

## 21.2 集成测试

使用 Mock Observer，预设窗口输出：

```text
窗口 1：人员出现
窗口 2：人员靠近柜门
窗口 3：开始开门
窗口 4：柜门部分打开
窗口 5：柜门完全打开
窗口 6：镜头切换近景
窗口 7：人员暂时不可见
窗口 8：返回全景
```

验证：

* 人员 ID 是否稳定；
* 柜门实体是否稳定；
* 开门动作是否只创建一次；
* `closed → partially_open → open` 是否正确；
* 镜头切换是否没有删除实体；
* 暂时不可见是否没有结束所有动作。

## 21.3 回归测试

保存一组模型原始响应作为 Golden Fixtures：

```text
tests/golden/raw_responses/
```

每次修改解析器、Schema、Registry 后，离线重放这些响应，不重新调用 API。

这样可以低成本验证代码修改。

---

# 二十二、评测指标

## 22.1 结构质量

```text
JSON 解析成功率
Schema 通过率
内部引用有效率
枚举合法率
证据帧有效率
错误窗口状态污染率
```

## 22.2 跨窗口一致性

```text
实体 ID Switch
重复实体创建数
错误实体合并数
动作碎片化数
动作重复创建数
无依据状态翻转数
```

## 22.3 状态抽取能力

```text
实体识别 Precision / Recall / F1
动作识别 Precision / Recall / F1
属性状态准确率
状态转移 Precision / Recall / F1
动作开始检测延迟
动作结束检测延迟
状态变化检测延迟
```

## 22.4 系统性能

```text
平均 API 延迟
P50 / P95 API 延迟
每窗口输入 Token
每窗口输出 Token
每分钟视频成本
Real-Time Factor
峰值队列长度
丢弃窗口数
```

## 22.5 第一阶段建议验收目标

| 指标            |   目标 |
| ------------- | ---: |
| JSON 解析成功率    | ≥99% |
| Schema 通过率    | ≥98% |
| 实体引用有效率       | 100% |
| 枚举合法率         | 100% |
| 失败窗口状态污染数     |    0 |
| 全局 ID 冲突数     |    0 |
| 重复实体描述下降      | ≥60% |
| 无变化窗口输出 Token | ≤200 |
| 单元测试通过率       | 100% |

这些是工程验收目标，不代表视觉模型最终准确率。

---

# 二十三、实验对照设计

为了后续研究和论文验证，建议保留三种模式。

## Baseline A：无状态窗口分析

```text
当前窗口
→ 模型完整描述
```

对应当前的 `--no-state`。

## Baseline B：模型状态传递

```text
当前窗口
+ 上一窗口压缩状态
→ 模型完整描述
```

对应当前默认实现。

## Proposed：程序状态维护

```text
当前窗口
+ 精简候选状态
→ 模型增量观察
→ EntityRegistry
→ ActionTracker
→ TransitionEngine
→ GlobalState
```

比较：

* 模型输出 Token；
* 实体 ID 稳定性；
* 动作重复率；
* 状态变化准确率；
* 长视频状态漂移；
* 延迟；
* 成本。

这个对照能够直接支撑后续论文中的核心论点：

> 把状态维护从通用 VLM 中剥离到显式程序状态系统，是否能提高流式视频理解的一致性、可解释性和效率。

---

# 二十四、具体实施顺序

## Commit 1：工程基线

修改：

* 增加 `pyproject.toml`；
* 增加 `src/`；
* 将 CLI 和视频函数从 `run.py` 拆出；
* 增加 Pydantic 配置；
* 保持现有功能完全可运行；
* 增加最基础测试。

验收：

```text
新旧命令输出结果基本一致
```

## Commit 2：Observation Schema

修改：

* 新增增量 Schema；
* 新增受控枚举；
* 修改提示词；
* 增加 Pydantic 校验；
* 增加帧编号和证据映射；
* 保存所有原始响应。

验收：

```text
模型不再输出完整世界状态
```

## Commit 3：解析和错误链路

修改：

* 网络错误分类；
* JSON repair；
* Schema repair；
* 失败窗口禁止更新状态；
* 增加 `errors.jsonl`。

验收：

```text
格式损坏不会重新发送视频请求
```

## Commit 4：EntityRegistry

修改：

* 全局实体结构；
* 候选过滤；
* 匹配打分；
* 临时 ID；
* 实体合并事件；
* 可见性状态。

验收：

```text
相邻窗口同一人员和设备使用稳定 ID
```

## Commit 5：ActionTracker

修改：

* 动作键；
* 生命周期；
* 缺失窗口容忍；
* 重复动作分割；
* 动作证据累计。

验收：

```text
重叠窗口不会重复创建同一个动作
```

## Commit 6：TransitionEngine

修改：

* 属性状态；
* pending 状态；
* 多窗口确认；
* 冲突处理；
* 状态变化事件。

验收：

```text
低置信度单窗口观察不会导致全局状态翻转
```

## Commit 7：输出和实验评测

修改：

* 拆分输出文件；
* 增加运行哈希；
* 增加指标分析；
* 增加不同模式比较脚本。

验收：

```text
可以一条命令比较三种模式
```

## Commit 8：实时调度模拟

修改：

* 修复时间原点；
* WindowQueue；
* latest-window-only；
* observation gap；
* 实时性能指标。

验收：

```text
API 速度低于视频速度时不会无限积压
```

## Commit 9：真实流接入

修改：

* CameraVideoSource；
* RtspVideoSource；
* FrameRingBuffer；
* 断流重连。

验收：

```text
相同 Pipeline 同时支持文件、摄像头和 RTSP
```

---

# 二十五、暂时不要做的内容

第一轮不建议加入：

* Milvus 或向量数据库；
* 多 Agent 编排；
* Kafka；
* 分布式推理；
* 复杂前端；
* 专门训练 ReID 模型；
* 规则库违规判断；
* 自动故障诊断；
* 完整作业流程规划；
* 大规模数据库建模。

这些都不是当前最核心的问题。

现阶段最重要的是验证：

```text
局部视觉观察
→ 稳定实体
→ 连续动作
→ 可靠状态变化
```

只有这一层稳定后，才能继续做：

```text
规则校验
异常判断
主动寻证
风险决策
```

---

# 二十六、最终建议

建议将下一版正式定义为：

> **qwen_stream_video v2：面向长程视频的增量观察与显式状态维护框架**

v2 的最低闭环应该是：

```text
本地视频因果滑窗
→ 千问输出增量观察
→ Pydantic 严格校验
→ 全局实体注册
→ 动作生命周期管理
→ 属性状态转移
→ 事件日志
→ 周期性状态快照
→ 三种模式对照评测
```

不要在这一版立即追求完整 RTSP 生产系统。

从研究价值看，最值得优先实现的三个模块依次是：

```text
1. EntityRegistry
2. ActionTracker
3. TransitionEngine
```

这三个模块决定了项目究竟是一个“逐窗口视频描述工具”，还是一个真正的“Streaming Video State Extraction Agent”。
