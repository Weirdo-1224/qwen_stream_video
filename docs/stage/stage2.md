# qwen_stream_video v2 第二阶段开发任务

你需要在以下仓库中完成第二阶段开发：

```text
https://github.com/Weirdo-1224/qwen_stream_video.git
```

第一阶段已经完成标准 Python 工程化、视频因果滑窗、帧采样、千问视觉调用、增量 ObservationBatch、Pydantic 校验、运行产物保存和基础测试。

第二阶段不再以“生成更多窗口描述”为目标，而是将窗口级局部观察转换为可持续维护、可追踪、可复核的全局状态。

---

# 一、任务目标

本阶段需要将当前链路：

```text
当前窗口帧
→ 千问生成局部 ObservationBatch
→ 保存 observations.jsonl
```

扩展为：

```text
当前窗口帧
→ 千问生成局部视觉观察
→ 程序解析场景连续性
→ 程序匹配全局实体
→ 程序维护动作生命周期
→ 程序确认属性状态与状态转移
→ 输出事件、状态增量和全局状态快照
```

本阶段重点完成：

1. 修复第一阶段运行结果中暴露出的非破坏性归一化问题；
2. 将 Observation 协议升级为 `2.0`；
3. 增加重叠窗口的 Context Interval / Commit Interval；
4. 实现 SceneTracker；
5. 实现 EntityRegistry 和 EntityResolver；
6. 实现 ActionTracker；
7. 实现 TransitionEngine；
8. 实现 GlobalState 和 StateReducer；
9. 实现 ContextBuilder，替换简单的上一窗口摘要；
10. 输出实体解析、事件、状态增量和状态快照；
11. 支持从已有 Observation JSONL 离线重放状态维护；
12. 增加单元测试、集成测试和 Golden Regression；
13. 增加状态质量分析脚本；
14. 保持第一阶段本地视频运行方式可用。

本阶段完成后，系统必须能够明确区分：

```text
模型当前看到了什么
程序认为它对应哪个历史实体
程序认为哪个动作正在持续
程序确认了什么状态变化
系统当前维护的全局状态是什么
```

本阶段不实现：

* RTSP 或摄像头输入；
* 生产者—消费者实时调度；
* 多窗口并行模型推理；
* latest-window-only 积压控制；
* 目标检测器、分割器或 ReID 神经网络；
* 向量数据库；
* 多 Agent；
* 作业规程推理；
* 违规判断或报警；
* Web 前端；
* 复杂图数据库；
* 自动训练或微调视觉模型。

不要提前实现第三阶段和第四阶段功能。

---

# 二、第二阶段核心原则

## 2.1 模型只负责局部观察

千问视觉模型只回答：

> 当前窗口中直接看到了哪些实体、动作、属性和不确定性？

模型不负责：

* 分配正式全局实体 ID；
* 决定两个窗口中的实体一定是同一个实体；
* 决定动作已经正式开始或结束；
* 根据历史直接生成状态转移；
* 删除当前窗口未出现的历史实体；
* 根据任务标题补充不可见的专业语义；
* 输出完整全局状态。

正式全局 ID、动作生命周期、状态转移和全局状态只能由程序生成。

## 2.2 全局状态必须由确定性代码维护

同一份按时间排序的 Observation 输入，在相同配置下重复运行，必须得到一致的：

```text
entity_resolutions
state_events
state_deltas
state_snapshots
```

禁止在 State Engine 中再次调用大模型决定实体匹配或状态转移。

## 2.3 不得破坏原始模型信息

禁止将模型已经输出的有效信息直接覆盖为 `unknown`。

例如模型输出：

```json
{
  "action_type": "hand_over"
}
```

如果词表暂时不支持，不能直接改成：

```json
{
  "action_type": "unknown"
}
```

必须保留：

```json
{
  "action_type": "other",
  "raw_action_type": "hand_over",
  "normalization_status": "out_of_vocabulary"
}
```

其中：

* `unknown` 表示视觉上无法判断动作类型；
* `other` 表示模型给出了具体动作，但当前受控词表没有对应规范类型。

## 2.4 未看到不等于消失

必须始终遵守：

```text
当前窗口未观察到某实体
≠
实体已从世界中消失
```

镜头切换、特写、遮挡、取景范围变化时，只能更新实体可见性，不能立即删除实体。

## 2.5 新出现不等于状态变化

必须区分：

```text
newly_visible
became_occluded
camera_reframed
entity_reidentified
actual_state_change
```

例如某指示灯在上一窗口未进入画面、当前窗口首次进入画面，不能直接生成“指示灯由不存在变为亮起”的状态转移。

## 2.6 宁可暂时不匹配，也不要错误合并

实体匹配不确定时，应创建临时实体或输出 ambiguous 解析结果。

禁止为了保持 ID 连续性，强制把低分候选合并到已有实体。

## 2.7 所有正式结论必须可追踪

每个正式实体匹配、动作更新和属性状态转移必须能够追踪到：

```text
run_id
window_global_index
local_id
sample_index
frame timestamp
原始 Observation
匹配或状态更新理由
```

## 2.8 不允许虚假实现

禁止：

* 使用空类和空函数占位但声称完成；
* 硬编码某个视频中的人员或开关柜 ID；
* 在 EntityResolver 中通过实体数组顺序直接沿用 ID；
* 通过只比较 `candidate_global_id` 完成实体匹配；
* 在 ActionTracker 中把每个窗口动作都创建为新动作；
* 为了让测试通过而跳过冲突处理；
* 静默吞掉状态更新错误；
* 在测试中只写 `assert True`；
* 调用真实千问 API 执行单元测试；
* 修改失败 Observation 伪造成成功状态。

---

# 三、第一阶段兼容要求

## 3.1 保持原有运行方式

以下命令必须继续可用：

```bash
python run.py \
  --video videos/demo.mp4 \
  --config configs/base.yaml
```

以及：

```bash
qwen-stream-video \
  --video videos/demo.mp4 \
  --config configs/base.yaml
```

根目录 `run.py` 继续只保留兼容入口。

## 3.2 保留第一阶段输出

以下文件继续生成：

```text
run_meta.json
resolved_config.yaml
windows.jsonl
observations.jsonl
api_metrics.jsonl
errors.jsonl
raw_responses/
sampled_frames/
```

其中 `observations.jsonl` 仍然表示模型局部观察，不得改成全局状态快照。

第二阶段新增独立文件保存程序侧状态结果。

## 3.3 Observation Schema 版本

第二阶段默认：

```yaml
observation:
  schema_version: "2.0"
```

程序必须显式拒绝未知 Schema 版本，不允许静默按当前模型解析。

为了能够重放第一阶段产物，应提供：

```text
ObservationV1Adapter
```

将合法的 `1.0` Observation 转换成内部统一的 `2.0` 结构。

适配器只负责字段迁移和非破坏性归一化，不得伪造缺失证据或全局状态。

## 3.4 状态功能开关

保留当前禁用状态继承的能力，并增加明确配置：

```yaml
state:
  enabled: true
```

当 `state.enabled: false` 时：

* 仍然生成窗口 Observation；
* 不运行 SceneTracker、EntityResolver、ActionTracker 和 TransitionEngine；
* 不生成状态文件；
* 不影响第一阶段行为。

---

# 四、目标目录结构

在第一阶段目录基础上扩展为：

```text
qwen_stream_video/
├── pyproject.toml
├── run.py
├── README.md
├── .env.example
│
├── configs/
│   ├── base.yaml
│   └── experiments/
│       ├── observation_only.yaml
│       └── state_tracking.yaml
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
│       │   ├── state.py
│       │   ├── event.py
│       │   └── resolution.py
│       │
│       ├── video/
│       │   ├── __init__.py
│       │   ├── metadata.py
│       │   ├── window.py
│       │   ├── sampling.py
│       │   └── frame_encoder.py
│       │
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── client.py
│       │   ├── parser.py
│       │   ├── prompts.py
│       │   ├── validator.py
│       │   └── normalizer.py
│       │
│       ├── state/
│       │   ├── __init__.py
│       │   ├── global_state.py
│       │   ├── scene_tracker.py
│       │   ├── entity_registry.py
│       │   ├── entity_resolver.py
│       │   ├── action_tracker.py
│       │   ├── transition_engine.py
│       │   ├── state_reducer.py
│       │   ├── context_builder.py
│       │   └── replay.py
│       │
│       └── storage/
│           ├── __init__.py
│           ├── storage.py
│           └── state_storage.py
│
├── scripts/
│   ├── analyze_results.py
│   ├── evaluate_state_run.py
│   └── make_test_video.py
│
└── tests/
    ├── unit/
    │   ├── test_observation_v2.py
    │   ├── test_normalizer.py
    │   ├── test_commit_interval.py
    │   ├── test_scene_tracker.py
    │   ├── test_entity_registry.py
    │   ├── test_entity_resolver.py
    │   ├── test_action_tracker.py
    │   ├── test_transition_engine.py
    │   ├── test_context_builder.py
    │   └── test_state_storage.py
    │
    ├── integration/
    │   ├── test_state_pipeline.py
    │   ├── test_observation_replay.py
    │   └── test_camera_change_sequence.py
    │
    ├── golden/
    │   ├── stage2_sequence.jsonl
    │   └── expected_stage2_events.jsonl
    │
    └── fixtures/
```

不要求文件名完全机械一致，但以下职责必须独立：

```text
模型观察
场景管理
实体解析
动作跟踪
属性状态更新
状态存储
```

禁止把第二阶段全部逻辑继续堆入 `pipeline.py`。

---

# 五、配置系统扩展

在现有 Pydantic 配置模型中新增以下配置。

```yaml
experiment:
  name: incremental_state_v2
  seed: 42

video:
  window_seconds: 6.0
  stride_seconds: 3.0
  warmup_windows: 3

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
  max_tokens: 3000
  timeout_seconds: 120
  network_retries: 2
  structured_json: true

observation:
  schema_version: "2.0"
  require_evidence_frames: true
  context_policy: visual_only
  allow_candidate_global_ids: true

state:
  enabled: true
  snapshot_interval_windows: 10
  fail_on_state_error: false

scene_tracker:
  enabled: true
  camera_change_starts_new_scene: true
  preserve_entities_across_scenes: true

entity_registry:
  confident_match_threshold: 0.78
  ambiguous_match_threshold: 0.58
  max_missing_windows: 10
  temporary_entity_prefix: temp
  candidate_hint_weight: 0.05
  allow_delayed_merge: true
  delayed_merge_support_windows: 2

action_tracker:
  continue_max_gap_windows: 1
  end_missing_windows: 2
  repeat_action_min_gap_seconds: 5.0
  instant_actions:
    - press
    - switch
    - pick_up
    - put_down
    - hand_over
    - receive

transition_engine:
  high_confidence_threshold: 0.85
  medium_confidence_threshold: 0.60
  confirm_support_windows: 2
  max_pending_gap_windows: 1
  require_action_support_for_transition: true

context:
  max_entities: 15
  recent_window_count: 5
  max_active_actions: 8
  max_pending_attributes: 8
  max_serialized_characters: 6000

storage:
  output_root: outputs
  save_raw_responses: true
  save_sampled_frames: false
  save_entity_resolutions: true
  save_state_events: true
  save_state_deltas: true
  save_state_snapshots: true
```

至少新增：

```python
class StateConfig(BaseModel): ...
class SceneTrackerConfig(BaseModel): ...
class EntityRegistryConfig(BaseModel): ...
class ActionTrackerConfig(BaseModel): ...
class TransitionEngineConfig(BaseModel): ...
class ContextConfig(BaseModel): ...
```

## 5.1 严格配置校验

配置模型应使用：

```python
model_config = ConfigDict(extra="forbid")
```

禁止拼错配置字段后被静默忽略。

至少校验：

```text
0 <= ambiguous_match_threshold < confident_match_threshold <= 1
max_missing_windows >= 0
continue_max_gap_windows >= 0
end_missing_windows >= 1
repeat_action_min_gap_seconds >= 0
0 <= medium_confidence_threshold < high_confidence_threshold <= 1
confirm_support_windows >= 1
snapshot_interval_windows >= 1
max_entities >= 1
max_serialized_characters >= 100
warmup_windows >= 0
```

## 5.2 context_policy

支持：

```text
visual_only
weak_context
task_conditioned
```

默认必须为：

```text
visual_only
```

含义：

### visual_only

只向模型提供当前画面、窗口时间、候选实体上下文和通用场景类别。

不得把具体故障名称、具体回路名称或预期作业步骤作为视觉事实提示。

### weak_context

可以提供：

```text
电力设备标准化作业视频
变电站设备区域
```

但不能提供具体故障结论。

### task_conditioned

允许提供完整任务名称，但模型输出必须区分：

```text
visual_fact
task_conditioned_interpretation
```

第二阶段默认测试和 Golden Regression 只使用 `visual_only`。

## 5.3 配置生效要求

所有新增配置必须真实控制程序行为。

禁止出现配置已定义但业务代码未读取的情况。

---

# 六、重叠窗口的 Context / Commit 双区间

当前默认：

```text
window_seconds = 6
stride_seconds = 3
```

相邻窗口有 3 秒重叠。

仅靠提示词要求模型“不要重复”不能消除重复动作和状态变化，必须增加程序侧提交区间。

## 6.1 VideoWindow 扩展

扩展现有 `VideoWindow`：

```python
class VideoWindow(BaseModel):
    global_index: int
    run_index: int
    start_seconds: float
    commit_start_seconds: float
    end_seconds: float
    window_type: Literal["regular", "tail_completion"]
    processing_role: Literal["warmup", "commit"] = "commit"
```

约束：

```text
start_seconds <= commit_start_seconds < end_seconds
```

其中：

```text
Context Interval = [start_seconds, commit_start_seconds)
Commit Interval  = [commit_start_seconds, end_seconds)
```

## 6.2 commit_start_seconds 计算

对于完整视频中的第一个窗口，或没有任何可用前置窗口的 cold start：

```text
commit_start_seconds = start_seconds
```

只要存在按时间排序的前置窗口，包括 warmup 窗口，就使用：

```python
commit_start_seconds = max(
    current_window.start_seconds,
    previous_window.end_seconds,
)
```

因此从非零窗口启动且成功处理 warmup 时，第一个正式提交窗口不能再次提交 warmup 已覆盖的重叠区间；只有 cold start 时才提交该窗口的完整区间。

默认 6 秒窗口、3 秒步长示例：

```text
Window 0: [0, 6), commit [0, 6)
Window 1: [3, 9), context [3, 6), commit [6, 9)
Window 2: [6, 12), context [6, 9), commit [9, 12)
```

如果 `stride_seconds >= window_seconds`，则整个窗口都属于 Commit Interval。

## 6.3 证据时间映射

新增工具函数：

```python
def evidence_timestamps(
    evidence_frames: list[int],
    sampled_frames: list[SampledFrame],
) -> list[float]:
    ...
```

新增：

```python
def evidence_intersects_commit_interval(
    evidence_frames: list[int],
    sampled_frames: list[SampledFrame],
    window: VideoWindow,
) -> bool:
    ...
```

## 6.4 提交规则

Observation 可以使用整个窗口理解上下文，但：

* 证据全部位于 Context Interval 的新动作，不能创建新的 GlobalAction；
* 证据全部位于 Context Interval 的新属性值，不能创建新的状态转移；
* Context Interval 中的观察可以继续支持已有实体匹配；
* Context Interval 中的观察可以维持已有动作为 ongoing；
* 只要至少一个关键证据帧位于 Commit Interval，才允许创建新的动作或状态候选；
* 场景切换本身可以按窗口级信号提交，但必须记录对应窗口。

## 6.5 warmup_windows

当用户从非零窗口开始运行时，例如：

```bash
--start-window 239
```

若 `warmup_windows: 3`，程序应：

```text
先处理 236、237、238 窗口作为 warmup
再从 239 窗口开始正式提交
```

Warmup 窗口：

* 可以建立实体、场景和动作上下文；
* 不写入正式评测范围的 state_events；
* 必须在 `windows.jsonl` 标记 `processing_role=warmup`；
* 必须在 `run_meta.json` 记录真实 warmup 范围；
* 不得让用户误以为正式处理从 236 开始。

如果无法获取前置窗口，则明确记录：

```text
cold_start = true
```

---

# 七、Observation Schema 2.0

第二阶段继续保留局部 Observation，但需要解决第一阶段中的自由属性、动作信息丢失和上下文污染问题。

## 7.1 WindowObservation

```python
class WindowObservation(BaseModel):
    global_index: int
    start_seconds: float
    commit_start_seconds: float
    end_seconds: float
```

模型返回的时间字段仍然不能作为真实来源，必须由程序覆盖。

## 7.2 SceneObservation

```python
class SceneObservation(BaseModel):
    camera_change: bool = False
    view_type: ViewType = ViewType.UNKNOWN
    scene_visibility: VisibilityQuality = VisibilityQuality.UNKNOWN
    target_visibility: VisibilityQuality = VisibilityQuality.UNKNOWN
    continuity_hint: Literal[
        "continuous",
        "reframed",
        "camera_change",
        "unknown",
    ] = "unknown"
    description: str = ""
```

不要继续只使用一个全局 `visibility` 表示所有对象都清晰。

## 7.3 EntityObservation

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
    evidence_frames: list[int] = Field(default_factory=list)
```

`candidate_global_id` 只能从 ContextBuilder 提供的候选 ID 中选择。

模型不得自行创造：

```text
person_9999
device_8888
```

若模型返回未提供的候选 ID：

* 保留原始值到警告；
* 将规范字段设为 `None`；
* 不允许直接创建同名正式实体。

## 7.4 ActionObservation

```python
class ActionObservation(BaseModel):
    local_id: str
    actor_local_id: str
    action_type: str
    raw_action_type: str | None = None
    action_family: str | None = None
    normalization_status: Literal[
        "canonical",
        "alias_mapped",
        "out_of_vocabulary",
        "unknown",
    ] = "canonical"
    target_local_id: str | None = None
    tool_local_id: str | None = None
    phase_observation: ActionPhaseObservation
    description: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_frames: list[int] = Field(default_factory=list)
```

## 7.5 AttributeObservation

```python
class AttributeObservation(BaseModel):
    entity_local_id: str
    attribute_key: str
    value: str
    raw_attribute: str | None = None
    raw_value: str | None = None
    normalization_status: Literal[
        "canonical",
        "alias_mapped",
        "out_of_vocabulary",
        "invalid_for_entity_type",
    ] = "canonical"
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_frames: list[int] = Field(default_factory=list)
```

## 7.6 RelationObservation

新增：

```python
class RelationObservation(BaseModel):
    subject_local_id: str
    relation_type: str
    object_local_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_frames: list[int] = Field(default_factory=list)
```

第一版关系词表可以包含：

```text
holding
near
inside
in_front_of
operating
observing
pointing_to
wearing
attached_to
part_of
```

关系主要用于实体匹配和上下文构建，不在本阶段实现复杂关系图推理。

## 7.7 UncertaintyObservation

```python
class UncertaintyObservation(BaseModel):
    uncertainty_type: Literal[
        "identity",
        "action",
        "attribute",
        "visibility",
        "causality",
        "other",
    ] = "other"
    description: str
    related_local_ids: list[str] = Field(default_factory=list)
    evidence_frames: list[int] = Field(default_factory=list)
```

## 7.8 ObservationBatch

```python
class ObservationBatch(BaseModel):
    schema_version: Literal["2.0"]
    window: WindowObservation
    summary: str
    scene: SceneObservation
    entities: list[EntityObservation] = Field(default_factory=list)
    actions: list[ActionObservation] = Field(default_factory=list)
    attribute_observations: list[AttributeObservation] = Field(default_factory=list)
    relations: list[RelationObservation] = Field(default_factory=list)
    uncertainties: list[UncertaintyObservation] = Field(default_factory=list)
```

## 7.9 视觉事实和任务解释分离

当 `context_policy=task_conditioned` 时，不允许把任务解释混入 `summary` 或普通属性。

可选新增：

```python
class TaskConditionedInterpretation(BaseModel):
    description: str
    confidence: float
    supporting_local_ids: list[str]
```

但默认 `visual_only` 模式不输出该字段。

---

# 八、受控词表与非破坏性归一化

新增：

```text
inference/normalizer.py
```

定义：

```python
class ObservationNormalizer:
    def normalize(self, batch: ObservationBatch) -> NormalizationResult:
        ...
```

## 8.1 actions.yaml

调整为带元数据的结构：

```yaml
actions:
  observe:
    family: perception
    aliases:
      - watch
      - look_at

  inspect:
    family: perception
    aliases:
      - check
      - examine

  hand_over:
    family: object_transfer
    aliases:
      - handover
      - pass
      - give

  receive:
    family: object_transfer
    aliases:
      - take_from

  push:
    family: object_motion
    aliases: []

  pull:
    family: object_motion
    aliases: []

  hover:
    family: contact_proximity
    aliases:
      - suspend_over

  other:
    family: other
    aliases: []

  unknown:
    family: unknown
    aliases: []
```

至少包含：

```text
observe
inspect
approach
leave
hold
pick_up
put_down
place
hand_over
receive
carry
touch
hover
press
release
push
pull
point
rotate
switch
open
close
insert
remove
connect
disconnect
adjust
measure
record
operate
other
unknown
```

## 8.2 未知动作处理

若模型输出不在 canonical 或 aliases 中：

```python
action.raw_action_type = original_value
action.action_type = "other"
action.normalization_status = "out_of_vocabulary"
```

若模型明确输出 `unknown`：

```python
action.action_type = "unknown"
action.normalization_status = "unknown"
```

两者语义必须不同。

## 8.3 attributes.yaml

使用统一属性路径：

```yaml
attributes:
  door.state:
    entity_types:
      - device
      - component
    values:
      - unknown
      - closed
      - partially_open
      - open
    aliases:
      - door_state
      - door_status
      - 柜门状态

  cover.state:
    entity_types:
      - device
      - component
    values:
      - unknown
      - closed
      - partially_open
      - open
    aliases:
      - cover_state
      - cover_status
      - panel_cover_state

  indicator.lit:
    entity_types:
      - device
      - component
    values:
      - unknown
      - on
      - off
    aliases:
      - indicator_state
      - light_state

  indicator.color:
    entity_types:
      - device
      - component
    values:
      - unknown
      - red
      - green
      - yellow
      - white
      - blue
    aliases:
      - indicator_color
      - light_color

  switch.position:
    entity_types:
      - device
      - component
    values:
      - unknown
      - on
      - off
      - work
      - test
      - intermediate
    aliases:
      - switch_state
      - switch_position

  holding.state:
    entity_types:
      - tool
      - component
      - document
    values:
      - unknown
      - held
      - released
    aliases:
      - holding_state
```

## 8.4 属性校验

必须检查：

* 属性是否存在；
* 属性是否允许用于当前实体类型；
* 值是否属于允许值；
* alias 是否正确映射；
* 原始属性和值是否被保留。

非法属性不能进入 TransitionEngine 更新正式状态，但应保存在 Observation 和规范化警告中。

## 8.5 规范化结果

定义：

```python
class NormalizationWarning(BaseModel):
    warning_type: str
    local_id: str | None = None
    field_name: str
    raw_value: str
    normalized_value: str | None = None
    message: str
```

```python
class NormalizationResult(BaseModel):
    batch: ObservationBatch
    warnings: list[NormalizationWarning]
```

规范化警告写入每个窗口的处理结果和运行统计。

---

# 九、全局状态领域模型

新增：

```text
src/qwen_stream_video/domain/state.py
src/qwen_stream_video/domain/event.py
src/qwen_stream_video/domain/resolution.py
```

## 9.1 枚举

保留第一阶段 EntityType，并确保至少支持：

```text
person
device
component
tool
ppe
sign
environment
document
unknown
```

至少新增：

```python
class VisibilityState(str, Enum):
    VISIBLE = "visible"
    PARTIAL = "partial"
    OCCLUDED = "occluded"
    NOT_VISIBLE = "not_visible"
    UNKNOWN = "unknown"


class EntityLifecycleStatus(str, Enum):
    ACTIVE = "active"
    TEMPORARILY_MISSING = "temporarily_missing"
    INACTIVE = "inactive"
    MERGED = "merged"


class EntityResolutionStatus(str, Enum):
    MATCHED = "matched"
    CREATED = "created"
    AMBIGUOUS = "ambiguous"
    TEMPORARY = "temporary"
    REJECTED_HINT = "rejected_hint"


class ActionLifecycle(str, Enum):
    CANDIDATE = "candidate"
    STARTED = "started"
    ONGOING = "ongoing"
    POSSIBLE_ENDED = "possible_ended"
    ENDED = "ended"
    INSTANT = "instant"
    UNCERTAIN = "uncertain"
    INTERRUPTED = "interrupted"


class AttributeConfirmationStatus(str, Enum):
    OBSERVED = "observed"
    PENDING = "pending"
    CONFIRMED = "confirmed"
    CONFLICTED = "conflicted"
    REJECTED = "rejected"
```

## 9.2 EvidenceReference

```python
class EvidenceReference(BaseModel):
    run_id: str
    window_global_index: int
    local_id: str | None = None
    sample_indices: list[int] = Field(default_factory=list)
    timestamps_seconds: list[float] = Field(default_factory=list)
```

程序根据 `sample_indices` 映射时间，禁止模型直接提供伪精确时间。

## 9.3 TimeInterval

```python
class TimeInterval(BaseModel):
    lower: float
    upper: float
```

要求：

```text
lower <= upper
```

动作开始和结束使用时间区间，不使用抽帧精度无法支持的伪精确时间点。

## 9.4 SpatialObservation

```python
class SpatialObservation(BaseModel):
    window_global_index: int
    scene_id: str
    spatial_region: str
    confidence: float
```

## 9.5 AttributeState

```python
class AttributeState(BaseModel):
    attribute_key: str
    value: str
    confidence: float
    status: AttributeConfirmationStatus

    first_observed_window: int
    last_observed_window: int
    confirmed_window: int | None = None

    previous_value: str | None = None
    pending_value: str | None = None
    pending_confidence: float | None = None
    pending_support_windows: list[int] = Field(default_factory=list)

    supporting_observations: int = 0
    contradicting_observations: int = 0
    evidence: list[EvidenceReference] = Field(default_factory=list)
```

## 9.6 GlobalEntityState

```python
class GlobalEntityState(BaseModel):
    entity_id: str
    entity_type: EntityType
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)

    is_temporary: bool = False
    merged_into: str | None = None

    first_seen_window: int
    last_seen_window: int
    first_seen_time: float
    last_seen_time: float

    current_scene_id: str | None = None
    visibility: VisibilityState = VisibilityState.UNKNOWN
    lifecycle_status: EntityLifecycleStatus = EntityLifecycleStatus.ACTIVE

    appearance_signature: dict[str, str] = Field(default_factory=dict)
    spatial_history: list[SpatialObservation] = Field(default_factory=list)
    attributes: dict[str, AttributeState] = Field(default_factory=dict)

    confidence: float = 0.0
    evidence: list[EvidenceReference] = Field(default_factory=list)
```

## 9.7 GlobalActionState

```python
class GlobalActionState(BaseModel):
    action_id: str
    actor_id: str
    action_type: str
    action_family: str | None = None
    target_id: str | None = None
    tool_id: str | None = None

    lifecycle: ActionLifecycle

    start_window: int
    last_observed_window: int
    end_window: int | None = None

    start_time_interval: TimeInterval
    end_time_interval: TimeInterval | None = None

    observed_windows: list[int] = Field(default_factory=list)
    missing_window_count: int = 0
    confidence: float = 0.0
    evidence: list[EvidenceReference] = Field(default_factory=list)
```

## 9.8 SceneState

```python
class SceneState(BaseModel):
    scene_id: str
    view_type: ViewType
    start_window: int
    last_active_window: int
    continuity: str
    visible_entity_ids: list[str] = Field(default_factory=list)
```

## 9.9 GlobalState

```python
class GlobalState(BaseModel):
    schema_version: Literal["2.0"] = "2.0"
    run_id: str
    last_committed_window: int | None = None
    current_scene_id: str | None = None

    scenes: dict[str, SceneState] = Field(default_factory=dict)
    entities: dict[str, GlobalEntityState] = Field(default_factory=dict)
    actions: dict[str, GlobalActionState] = Field(default_factory=dict)

    active_action_ids: list[str] = Field(default_factory=list)
    pending_attribute_keys: list[str] = Field(default_factory=list)

    entity_counters: dict[str, int] = Field(default_factory=dict)
    action_counter: int = 0
    event_counter: int = 0
    scene_counter: int = 0
```

所有 ID 计数器必须属于 GlobalState，保证重放时结果可复现。

---

# 十、SceneTracker

新增：

```text
state/scene_tracker.py
```

定义：

```python
class SceneTracker:
    def update(
        self,
        state: GlobalState,
        observation: ObservationBatch,
    ) -> SceneUpdateResult:
        ...
```

## 10.1 场景更新结果

```python
class SceneUpdateResult(BaseModel):
    scene_id: str
    scene_changed: bool
    previous_scene_id: str | None = None
    continuity: str
    events: list[StateEvent] = Field(default_factory=list)
```

## 10.2 基础规则

第一版只使用已验证的 SceneObservation，不引入复杂视觉模型。

规则：

1. 第一个窗口创建 `scene_0001`；
2. `camera_change=true` 时创建新的 SceneState；
3. `continuity_hint=reframed` 时可以保留同一 scene_id，但记录 reframed；
4. 进入 closeup/detail 时，原场景实体设为 `not_visible` 或 `partial`，不能删除；
5. 返回 wide/medium 后允许重新匹配历史实体；
6. 镜头切换附近实体匹配降低空间连续性权重；
7. 镜头切换期间动作未观察到时，不立即结束动作；
8. 当前窗口新看到的局部组件允许先创建临时实体；
9. SceneTracker 不负责实体匹配，只负责提供场景上下文。

## 10.3 场景事件

至少支持：

```text
scene_started
scene_changed
scene_reframed
scene_visibility_changed
```

---

# 十一、EntityRegistry

新增：

```text
state/entity_registry.py
```

EntityRegistry 负责：

* 保存和查询 GlobalEntityState；
* 分配正式全局 ID；
* 更新实体最近出现信息；
* 更新外观签名和空间历史；
* 管理临时实体；
* 执行受控实体合并；
* 确保 ID 不复用。

## 11.1 全局 ID 规则

统一由程序生成：

```text
person_0001
device_0001
component_0001
tool_0001
ppe_0001
sign_0001
environment_0001
document_0001
unknown_0001
```

临时实体：

```text
temp_person_0001
temp_component_0001
```

ID 一旦分配，不得在同一次运行中复用。

## 11.2 核心接口

```python
class EntityRegistry:
    def create_entity(... ) -> GlobalEntityState:
        ...

    def get(self, entity_id: str) -> GlobalEntityState | None:
        ...

    def find_candidates(
        self,
        entity_type: EntityType,
        current_scene_id: str,
        current_window: int,
        max_missing_windows: int,
    ) -> list[GlobalEntityState]:
        ...

    def update_from_observation(... ) -> GlobalEntityState:
        ...

    def mark_not_observed(... ) -> None:
        ...

    def merge_temporary_entity(... ) -> EntityMergeResult:
        ...
```

## 11.3 实体生命周期

基础规则：

```text
本窗口观察到
→ active + visible/partial

连续若干窗口未观察到
→ temporarily_missing + not_visible

超过 max_missing_windows
→ inactive

镜头切换或特写期间未观察到
→ 暂不增加永久失活计数

临时实体被确认对应已有实体
→ merged
```

`inactive` 不等于删除，历史记录必须保留。

## 11.4 外观签名更新

外观属性采用保守更新：

* 高置信度新属性可以加入；
* 与已有稳定属性冲突时，不立即覆盖；
* 记录冲突次数；
* 低置信度属性不进入稳定外观签名；
* 不同视角缺失属性不视为冲突。

---

# 十二、EntityResolver

新增：

```text
state/entity_resolver.py
```

定义：

```python
class EntityResolver:
    def resolve(
        self,
        state: GlobalState,
        registry: EntityRegistry,
        scene_result: SceneUpdateResult,
        observation: ObservationBatch,
        sampled_frames: list[SampledFrame],
    ) -> EntityResolutionBatch:
        ...
```

## 12.1 解析结果

```python
class MatchScoreBreakdown(BaseModel):
    type_name_score: float
    appearance_score: float
    spatial_score: float
    relation_score: float
    recency_score: float
    candidate_hint_score: float
    total_score: float


class EntityResolution(BaseModel):
    window_global_index: int
    local_id: str
    global_entity_id: str
    status: EntityResolutionStatus
    selected_score: float | None = None
    second_best_score: float | None = None
    candidate_scores: dict[str, MatchScoreBreakdown] = Field(default_factory=dict)
    rejected_reasons: list[str] = Field(default_factory=list)
    evidence: list[EvidenceReference] = Field(default_factory=list)
```

```python
class EntityResolutionBatch(BaseModel):
    window_global_index: int
    mappings: list[EntityResolution]
    warnings: list[str] = Field(default_factory=list)
```

## 12.2 候选过滤

只选择：

* 实体类型相同；
* 最近 `max_missing_windows` 内出现；
* 未被 merged；
* 当前场景或可恢复历史场景中的实体；
* 生命周期不是永久无效状态。

## 12.3 硬约束排除

满足以下条件直接排除：

* `person` 匹配 `device`；
* 明确不同设备类别；
* 高置信度稳定外观属性冲突；
* 同一窗口两个同时可见局部实体映射到同一全局实体；
* 镜头连续且空间位置出现不可能跳变；
* candidate_global_id 指向不同实体类型；
* 已经在当前窗口被另一个更高分局部实体占用。

## 12.4 匹配评分

第一版采用可解释规则：

```text
总分 =
0.30 × 类型和名称相似度
+ 0.25 × 外观属性相似度
+ 0.15 × 空间连续性
+ 0.15 × 关系一致性
+ 0.10 × 最近出现程度
+ 0.05 × 模型候选 ID 提示
```

每一项必须返回 `[0, 1]`。

禁止只返回总分而不保存评分分解。

## 12.5 candidate_global_id 的地位

`candidate_global_id` 只能作为低权重提示。

必须满足：

* 候选 ID 存在于 ContextBuilder 提供列表；
* 候选实体类型一致；
* 不违反硬约束；
* 只贡献 `candidate_hint_weight`；
* 不能绕过阈值直接匹配。

## 12.6 阈值决策

```text
score >= confident_match_threshold
→ matched

ambiguous_match_threshold <= score < confident_match_threshold
→ ambiguous / temporary

score < ambiguous_match_threshold
→ created
```

若第一名和第二名分数差距过小，例如：

```text
best_score - second_best_score < 0.08
```

即使第一名超过阈值，也应降级为 ambiguous。

该差值应配置化。

## 12.7 一对一匹配

同一窗口必须保证：

```text
一个 local entity 只能映射一个 global entity
一个 visible global entity 只能被一个 local entity 占用
```

第一版可以采用：

1. 计算所有合法候选对；
2. 按总分降序；
3. 采用确定性贪心一对一匹配；
4. 分数相同时按 `local_id`、`global_entity_id` 排序打破平局。

禁止依赖 Python 无序字典遍历决定结果。

## 12.8 临时实体和延迟合并

ambiguous 时创建临时实体：

```text
temp_person_0001
```

若后续连续 `delayed_merge_support_windows` 个窗口都高分匹配同一正式实体，允许合并。

合并时必须：

* 生成 `entity_merged` 事件；
* 保留临时实体历史；
* 设置 `merged_into`；
* 迁移未冲突证据；
* 禁止删除旧记录；
* 禁止自动合并两个正式非临时实体。

---

# 十三、ActionTracker

新增：

```text
state/action_tracker.py
```

定义：

```python
class ActionTracker:
    def update(
        self,
        state: GlobalState,
        observation: ObservationBatch,
        resolutions: EntityResolutionBatch,
        sampled_frames: list[SampledFrame],
        scene_result: SceneUpdateResult,
    ) -> ActionUpdateResult:
        ...
```

## 13.1 局部引用转换

首先将：

```text
actor_local_id
target_local_id
tool_local_id
```

转换为正式全局 ID。

若 actor 无法解析：

* 不得创建 GlobalAction；
* 生成状态警告；
* 保留局部 Observation。

若 target 或 tool 暂时无法解析：

* 可以创建 `uncertain` 动作；
* 不得伪造实体 ID；
* 必须记录缺失引用。

## 13.2 动作键

基础动作键：

```python
action_key = (
    actor_id,
    action_type,
    target_id,
    tool_id,
)
```

但不能只凭动作键永久合并。

还必须考虑：

* 最近一次观察窗口；
* 时间间隔；
* 当前场景连续性；
* 前一个动作生命周期；
* 重叠窗口 Commit Interval；
* 是否为明显重复动作。

## 13.3 生命周期规则

```text
历史无匹配动作，且关键证据进入 Commit Interval
→ started

历史动作 started/ongoing，本窗口继续观察
→ ongoing

历史动作 ongoing，本窗口第一次未观察到
→ possible_ended

连续 end_missing_windows 未观察到
→ ended

动作属于 instant_actions，证据进入 Commit Interval
→ instant

镜头切换或严重遮挡时动作消失
→ uncertain，不立即 ended

同一动作在 Context Interval 重复出现
→ 只补充证据，不创建新动作
```

## 13.4 动作时间区间

使用证据帧相邻时间构造：

```python
start_time_interval = TimeInterval(
    lower=previous_sample_timestamp,
    upper=first_action_sample_timestamp,
)
```

若缺少前一帧，可以使用窗口开始时间作为 lower。

禁止保存无法由采样支持的三位小数精确动作开始时间。

## 13.5 重复动作判定

若同一动作键满足：

```text
当前窗口与 last_observed_window 间隔 <= continue_max_gap_windows
且时间间隔 < repeat_action_min_gap_seconds
且场景连续
```

优先视为同一动作继续。

若已 ended 且间隔超过阈值，则创建新 action_id。

## 13.6 全局动作 ID

格式：

```text
action_000001
action_000002
```

ID 一旦分配不得复用。

## 13.7 动作事件

至少生成：

```text
action_started
action_continued
action_possible_ended
action_ended
action_instant
action_uncertain
```

每个事件必须包含对应 EvidenceReference。

---

# 十四、TransitionEngine

新增：

```text
state/transition_engine.py
```

模型只输出当前属性观察，TransitionEngine 决定是否更新正式状态。

定义：

```python
class TransitionEngine:
    def update(
        self,
        state: GlobalState,
        observation: ObservationBatch,
        resolutions: EntityResolutionBatch,
        action_result: ActionUpdateResult,
        sampled_frames: list[SampledFrame],
        window: VideoWindow,
    ) -> TransitionUpdateResult:
        ...
```

## 14.1 输入过滤

以下 AttributeObservation 不能更新正式状态：

* 实体 local_id 无解析结果；
* `normalization_status=out_of_vocabulary`；
* `normalization_status=invalid_for_entity_type`；
* evidence_frames 为空且配置要求证据；
* 证据越界；
* 证据全部位于 Context Interval，且不是对已有 pending 状态的支持；
* confidence 低于 medium_confidence_threshold；
* 当前场景为 camera_change 且实体匹配为 ambiguous。

但这些观察仍应写入 Observation 和状态警告。

## 14.2 初始属性状态

若实体此前没有该属性：

```text
confidence >= high_confidence_threshold
→ confirmed 初始值

medium <= confidence < high
→ pending

confidence < medium
→ observed，不进入正式属性
```

初始值确认不生成 `before -> after` 转移，只生成：

```text
attribute_initialized
```

## 14.3 已知状态的同值观察

若观察值与当前 confirmed value 相同：

* 更新 last_observed_window；
* 增加 supporting_observations；
* 更新置信度；
* 清理与当前值冲突且已过期的 pending；
* 不生成重复状态转移事件。

## 14.4 已知状态的不同值观察

例如当前：

```text
door.state = closed
```

当前观察：

```text
door.state = open
```

默认不能立即转移。

确认条件满足以下之一：

### 条件 A：高置信度并有动作支持

```text
confidence >= high_confidence_threshold
且存在 open / operate / pull 等相关动作证据
且证据进入 Commit Interval
且实体匹配为 confident
```

### 条件 B：连续多窗口支持

```text
连续 confirm_support_windows 个窗口观察到相同新值
且场景连续
且实体解析不 ambiguous
```

### 条件 C：属性策略允许单窗口确认

只有在 `attributes.yaml` 中显式配置：

```yaml
confirmation_policy: single_high
```

才能单窗口确认。

## 14.5 pending 状态

第一次出现未满足确认条件的新值：

```text
pending_value = new_value
pending_support_windows = [current_window]
```

后续：

* 相同值：增加支持；
* 当前 confirmed 值：取消 pending 或记录冲突；
* 第三个不同值：记录 conflicted，禁止直接覆盖；
* 超过 `max_pending_gap_windows` 未继续支持：pending 过期。

## 14.6 动作支持映射

在 `attributes.yaml` 中允许定义：

```yaml
attributes:
  door.state:
    supporting_actions:
      open:
        - open
        - pull
        - operate
      closed:
        - close
        - push
        - operate
```

TransitionEngine 只使用已解析到同一实体的动作作为支持。

不得因为同一窗口存在任意 `open` 动作，就更新所有设备的 `door.state`。

## 14.7 新可见与状态变化

若属性上一窗口没有观察到，而当前首次观察到：

* 若 GlobalEntityState 中没有历史 confirmed value，按初始化处理；
* 若实体刚从 not_visible 恢复，当前值与历史相同，仅恢复可见性；
* 若当前值不同，仍按正常 pending/confirmed 规则处理；
* 禁止把“上一窗口未出现该组件”当作 `unknown -> current_value` 的正式转移。

## 14.8 状态转移事件

正式事件示例：

```json
{
  "event_id": "event_000021",
  "event_type": "attribute_transition",
  "window_global_index": 17,
  "entity_id": "device_0003",
  "attribute_key": "door.state",
  "before": "closed",
  "after": "open",
  "confidence": 0.88,
  "reason": "high_confidence_with_action_support",
  "evidence": [
    {
      "run_id": "...",
      "window_global_index": 16,
      "local_id": "E2",
      "sample_indices": [4, 5],
      "timestamps_seconds": [52.0, 53.0]
    },
    {
      "run_id": "...",
      "window_global_index": 17,
      "local_id": "E2",
      "sample_indices": [2, 3],
      "timestamps_seconds": [55.0, 56.0]
    }
  ]
}
```

---

# 十五、StateEvent 和 StateDelta

新增：

```text
domain/event.py
```

## 15.1 StateEvent

```python
class StateEvent(BaseModel):
    event_id: str
    event_type: str
    window_global_index: int
    timestamp_interval: TimeInterval | None = None
    entity_id: str | None = None
    action_id: str | None = None
    scene_id: str | None = None
    attribute_key: str | None = None
    before: Any | None = None
    after: Any | None = None
    confidence: float | None = None
    reason: str = ""
    evidence: list[EvidenceReference] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

## 15.2 事件类型

至少支持：

```text
scene_started
scene_changed
scene_reframed
entity_created
entity_matched
entity_ambiguous
entity_temporarily_missing
entity_reactivated
entity_merged
action_started
action_continued
action_possible_ended
action_ended
action_instant
action_uncertain
attribute_observed
attribute_pending
attribute_initialized
attribute_confirmed
attribute_transition
attribute_conflict
observation_gap
state_update_error
```

## 15.3 StateDelta

```python
class StateDelta(BaseModel):
    window_global_index: int
    scene_id: str
    entity_updates: list[str] = Field(default_factory=list)
    action_updates: list[str] = Field(default_factory=list)
    attribute_updates: list[str] = Field(default_factory=list)
    emitted_event_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
```

StateDelta 只描述本窗口程序侧发生的全局状态修改，不复制完整 GlobalState。

---

# 十六、StateReducer

新增：

```text
state/state_reducer.py
```

StateReducer 是第二阶段唯一允许提交 GlobalState 修改的协调器。

定义：

```python
class StateReducer:
    def apply_observation(
        self,
        state: GlobalState,
        observation: ObservationBatch,
        sampled_frames: list[SampledFrame],
        window: VideoWindow,
    ) -> StateReductionResult:
        ...
```

## 16.1 固定执行顺序

必须按以下顺序执行：

```text
1. SceneTracker.update
2. EntityResolver.resolve
3. EntityRegistry.commit_resolutions
4. ActionTracker.update
5. TransitionEngine.update
6. 更新未观察实体可见性
7. 更新未观察动作缺失计数
8. 生成 StateDelta
9. 提交 last_committed_window
```

禁止在不同模块中随意交叉修改同一状态。

## 16.2 原子性

单个窗口状态更新应具有原子性。

推荐方式：

```python
working_state = state.model_copy(deep=True)
result = reducer.apply_to_copy(working_state, ...)
state = working_state
```

如果 State Engine 中途失败：

* 原 GlobalState 不得部分修改；
* 写入 `state_update_error`；
* 根据 `fail_on_state_error` 决定继续或退出；
* 不影响该窗口原始 Observation 保存；
* 不能伪造 StateDelta。

## 16.3 失败窗口

如果模型 Observation 失败：

* 生成 `observation_gap` 事件；
* 不能推断该窗口实体消失；
* 活跃动作进入 uncertain 或保持；
* pending 属性不能因缺失窗口直接确认；
* 下一窗口继续处理。

## 16.4 确定性

所有遍历在影响 ID 分配和匹配结果时必须排序。

禁止依赖：

* set 的自然顺序；
* dict 的构建偶然顺序；
* 随机数；
* 当前系统时间；
* 文件系统目录顺序。

---

# 十七、ContextBuilder

新增：

```text
state/context_builder.py
```

替换当前：

```text
previous_window_summary
previous_entities
```

定义：

```python
class ContextBuilder:
    def build(
        self,
        state: GlobalState,
        current_window: VideoWindow,
    ) -> ObservationContext:
        ...
```

## 17.1 上下文内容

只向模型发送与当前窗口可能有关的信息：

```json
{
  "scene": {
    "scene_id": "scene_0002",
    "view_type": "closeup",
    "continuity": "continuous"
  },
  "candidate_entities": [
    {
      "entity_id": "person_0001",
      "entity_type": "person",
      "canonical_name": "worker",
      "appearance": {
        "clothing": "blue_uniform",
        "helmet": "white"
      },
      "last_seen_windows_ago": 1,
      "visibility": "visible"
    }
  ],
  "active_actions": [
    {
      "action_id": "action_000012",
      "actor_id": "person_0001",
      "action_type": "adjust",
      "target_id": "component_0004"
    }
  ],
  "pending_attributes": [
    {
      "entity_id": "device_0002",
      "attribute_key": "door.state",
      "candidate_value": "open"
    }
  ]
}
```

## 17.2 选择规则

只保留：

* 当前场景最近出现的实体；
* 最近 `recent_window_count` 个窗口出现的实体；
* 活跃或 uncertain 动作；
* pending 属性；
* 与活跃实体存在直接关系的设备和工具；
* 最近一次镜头切换信息。

不发送：

* 所有历史事件；
* 已结束很久的动作；
* 全量 GlobalState；
* 所有实体的完整历史；
* 大量自然语言窗口摘要；
* 已失活且与当前场景无关的实体。

## 17.3 候选 ID 限制

Prompt 必须明确：

```text
candidate_global_id 只能从 candidate_entities 中选择；
无法判断时返回 null；
不得自行创建正式 ID。
```

## 17.4 长度限制

序列化后不得超过：

```text
max_serialized_characters
```

超过时按以下顺序裁剪：

```text
最久未出现实体
→ 非活跃关系实体
→ 已 possible_ended 动作
→ 低置信度 pending 属性
```

禁止从 JSON 字符串中间直接截断导致无效结构。

---

# 十八、提示词修改

## 18.1 observation_system.txt

系统提示词至少明确：

1. 你是局部流式视频观察器；
2. 只能报告当前窗口可见事实；
3. 不使用未来信息；
4. 不输出完整全局状态；
5. 不决定正式实体身份；
6. `candidate_global_id` 只能从候选列表选择；
7. 看不到不等于消失；
8. 首次可见不等于发生状态变化；
9. 不根据任务名称补充不可见的回路、部件或故障结论；
10. 动作 phase 只是本窗口观察，不是正式生命周期；
11. 使用 canonical action 和 attribute_key；
12. 每个事实引用 evidence_frames；
13. 看不清使用 `unknown`，不要猜测；
14. 只输出符合 Schema 2.0 的 JSON。

## 18.2 observation_user.txt

动态提示词应包含：

```text
当前上下文策略
当前窗口 Context Interval
当前窗口 Commit Interval
抽样帧编号与时间
程序生成的候选上下文
动作词表摘要
属性词表摘要
Schema 2.0 输出说明
```

## 18.3 防止任务背景污染

`visual_only` 模式中不得发送具体视频文件名。

例如不得发送：

```text
VS1断路器操动机构储能回路故障排查标准化作业
```

只允许发送：

```text
这是电力设备标准化作业场景。
仅根据当前窗口画面描述可见事实。
```

## 18.4 JSON 结构化输出

若当前兼容接口支持，应在请求中启用：

```python
response_format={"type": "json_object"}
```

该配置由：

```yaml
model:
  structured_json: true
```

控制。

如果接口不支持，应输出明确警告并回退，不得静默忽略。

---

# 十九、Pipeline 集成

现有 Pipeline 扩展为：

```python
class StreamingVideoPipeline:
    def run(self, video_path: Path) -> RunSummary:
        metadata = read_video_metadata(video_path)
        windows = build_video_windows(...)
        selected_windows = select_windows_with_warmup(...)

        state = GlobalState(run_id=store.run_id)

        for window in selected_windows:
            try:
                sampled_frames = sample_window_frames(...)
                context = context_builder.build(state, window)
                prompt = prompt_builder.build(
                    window=window,
                    sampled_frames=sampled_frames,
                    context=context,
                )

                raw_result = qwen_client.observe(...)
                parsed_batch = parser.parse(raw_result)
                normalized_result = normalizer.normalize(parsed_batch)
                validated_batch = validator.validate(
                    normalized_result.batch,
                    sampled_frames,
                )

                store.write_observation(validated_batch)

                if config.state.enabled:
                    reduction = state_reducer.apply_observation(
                        state=state,
                        observation=validated_batch,
                        sampled_frames=sampled_frames,
                        window=window,
                    )
                    state = reduction.state
                    state_store.write_reduction(reduction)

            except ObservationProcessingError as exc:
                store.write_error(...)
                if config.state.enabled:
                    reduction = state_reducer.apply_observation_gap(...)
                    state = reduction.state
                    state_store.write_reduction(reduction)
                continue
```

## 19.1 模型观察与状态更新解耦

Pipeline 必须允许：

```text
Observation 已成功
State Engine 失败
```

这时：

* Observation 仍写入 `observations.jsonl`；
* 状态错误写入 `state_errors.jsonl`；
* 不得回滚已经成功保存的模型原始响应；
* 不得将 Observation 标记为模型失败。

## 19.2 Warmup 处理

Warmup 窗口仍调用 Observation 和 StateReducer，但：

* StateReducer 可以建立内部状态；
* 不计入正式 committed event 指标；
* 输出事件需标记 `warmup=true`；
* 默认不写入正式 `state_events.jsonl`，可以写入调试文件；
* 第一个正式 commit 窗口必须拥有 warmup 后的状态。

## 19.3 禁止并发改造

本阶段 Pipeline 继续按窗口顺序执行。

不要在本阶段引入线程池、异步队列或并行 StateReducer。

---

# 二十、Observation Replay

新增：

```text
state/replay.py
```

提供从已有 Observation JSONL 重建状态的能力，不调用千问 API。

## 20.1 CLI

新增：

```bash
qwen-stream-video \
  --replay-observations outputs/<run_id>/observations.jsonl \
  --config configs/base.yaml \
  --output-dir outputs/replay_run
```

## 20.2 Replay 输入

至少读取：

```text
observations.jsonl
windows.jsonl
run_meta.json
```

如果需要证据时间，还应读取 `windows.jsonl` 中的 sampled frame metadata。

## 20.3 V1 适配

当输入为 Schema `1.0` 时：

* 使用 ObservationV1Adapter；
* 将 `attribute` 映射为 `attribute_key`；
* 保留 raw_attribute；
* 将非法动作非破坏性映射为 `other`；
* 缺失 relations 使用空列表；
* 缺失 commit_start_seconds 根据窗口序列计算；
* 不伪造 candidate ID 或证据。

## 20.4 Replay 确定性

对相同输入执行两次 Replay：

```text
state_events.jsonl SHA256 必须一致
final_state.json SHA256 必须一致
```

运行时间、绝对输出路径等非确定字段不能混入状态语义文件。

---

# 二十一、状态存储结构

每次启用状态维护的运行输出：

```text
outputs/<run_id>/
├── run_meta.json
├── resolved_config.yaml
├── windows.jsonl
├── observations.jsonl
├── api_metrics.jsonl
├── errors.jsonl
├── normalization_warnings.jsonl
├── entity_resolutions.jsonl
├── state_events.jsonl
├── state_deltas.jsonl
├── state_snapshots.jsonl
├── state_errors.jsonl
├── final_state.json
├── raw_responses/
├── sampled_frames/
└── artifacts/
    ├── prompts/
    │   ├── observation_system.txt
    │   └── observation_user.txt
    ├── schemas/
    │   └── observation_v2.schema.json
    └── vocabularies/
        ├── actions.yaml
        ├── entity_types.yaml
        └── attributes.yaml
```

## 21.1 entity_resolutions.jsonl

每行保存一个窗口的 EntityResolutionBatch。

必须包含：

```text
local_id
global_entity_id
resolution_status
selected_score
second_best_score
score_breakdown
rejected_reasons
```

## 21.2 state_events.jsonl

只保存程序生成的正式 StateEvent。

不能直接复制模型描述作为正式事件。

## 21.3 state_deltas.jsonl

每个正式窗口一行，记录本窗口对 GlobalState 的修改摘要。

## 21.4 state_snapshots.jsonl

按：

```yaml
snapshot_interval_windows
```

周期保存完整 GlobalState。

必须额外保存最后一个窗口状态。

## 21.5 final_state.json

运行结束后原子写入最终完整状态。

推荐：

```text
先写 final_state.json.tmp
fsync
原子 rename 为 final_state.json
```

## 21.6 state_errors.jsonl

保存：

```text
窗口编号
状态阶段
异常类型
异常消息
是否影响 GlobalState
Observation 是否成功
原状态快照引用
```

## 21.7 artifacts 快照

必须保存本次实际使用的：

* Prompt 正文；
* JSON Schema；
* 动作词表；
* 属性词表；
* 实体类型词表。

不能只保存 SHA256 而不保存正文。

## 21.8 run_meta.json 扩展

至少新增：

```json
{
  "requested_start_window": 239,
  "requested_end_window": 279,
  "warmup_start_window": 236,
  "warmup_end_window": 238,
  "first_committed_window": 239,
  "last_committed_window": 279,
  "covered_start_seconds": 717.0,
  "covered_end_seconds": 843.0,
  "cold_start": false,
  "observation_schema_version": "2.0",
  "state_schema_version": "2.0",
  "state_enabled": true
}
```

---

# 二十二、命令行参数

保留第一阶段参数。

新增：

```text
--state
--no-state
--replay-observations PATH
--warmup-windows N
--snapshot-interval N
--context-policy visual_only|weak_context|task_conditioned
```

## 22.1 参数行为

### --state

启用 State Engine。

### --no-state

只生成局部 Observation。

### --replay-observations

不调用模型，从已有 Observation JSONL 重建状态。

与 `--video` 默认互斥。

### --warmup-windows

覆盖 YAML 中的 warmup_windows。

### --snapshot-interval

覆盖状态快照间隔。

## 22.2 配置优先级

继续遵守：

```text
命令行参数
> 环境变量
> YAML 配置
> 代码默认值
```

## 22.3 启动摘要

启动时打印：

```text
Observation Schema 版本
State Schema 版本
状态维护是否启用
Context Policy
Warmup 窗口数
实体匹配阈值
属性确认阈值
快照间隔
Replay 模式是否启用
```

API Key 仍然只能显示是否配置。

---

# 二十三、错误分类

在现有异常基础上新增：

```python
class StateEngineError(QwenStreamVideoError): ...
class SceneTrackingError(StateEngineError): ...
class EntityResolutionError(StateEngineError): ...
class EntityRegistryError(StateEngineError): ...
class ActionTrackingError(StateEngineError): ...
class TransitionError(StateEngineError): ...
class StateStorageError(StateEngineError): ...
class ObservationReplayError(StateEngineError): ...
class VocabularyNormalizationError(QwenStreamVideoError): ...
```

不得将所有状态错误都包装为无区分的 `RuntimeError`。

错误日志必须保留原异常链：

```python
raise EntityResolutionError(...) from exc
```

---

# 二十四、测试要求

使用 pytest。

所有 State Engine 测试必须使用构造的 Observation 或 Golden Fixture，不得调用真实千问 API。

## 24.1 test_observation_v2.py

至少包含：

```text
test_valid_observation_v2
test_unknown_schema_version_fails
test_window_commit_interval_validation
test_candidate_global_id_optional
test_relation_reference_validation
test_mutable_defaults_are_isolated
```

## 24.2 test_normalizer.py

至少包含：

```text
test_hand_over_is_not_replaced_by_unknown
test_receive_is_preserved
test_push_is_preserved
test_hover_is_preserved
test_oov_action_becomes_other_and_keeps_raw_value
test_unknown_action_remains_unknown
test_door_status_alias_maps_to_canonical_door_state
test_invalid_attribute_for_person_is_rejected
test_raw_attribute_is_preserved
```

## 24.3 test_commit_interval.py

至少包含：

```text
test_first_window_commits_full_interval
test_overlapping_window_commits_only_new_interval
test_non_overlapping_window_commits_full_interval
test_context_only_action_does_not_create_new_global_action
test_context_observation_can_continue_existing_action
test_tail_window_commit_interval_valid
```

## 24.4 test_scene_tracker.py

至少包含：

```text
test_first_window_creates_scene
test_camera_change_creates_new_scene
test_reframe_keeps_scene_id
test_camera_change_does_not_delete_entities
test_closeup_marks_previous_entities_not_visible
test_return_to_wide_allows_reactivation
```

## 24.5 test_entity_registry.py

至少包含：

```text
test_global_ids_are_monotonic
test_ids_are_not_reused
test_entity_not_seen_is_not_deleted
test_entity_becomes_temporarily_missing
test_entity_becomes_inactive_after_threshold
test_temporary_entity_merge_preserves_history
```

## 24.6 test_entity_resolver.py

至少包含：

```text
test_entity_type_mismatch_is_hard_rejected
test_two_visible_locals_cannot_map_same_global
test_candidate_hint_cannot_override_hard_constraint
test_candidate_hint_has_low_weight
test_confident_match_reuses_entity
test_low_score_creates_new_entity
test_close_scores_create_ambiguous_entity
test_resolution_is_deterministic
test_three_people_keep_ids_after_closeup_sequence
```

最后一个测试使用合成序列：

```text
wide: person A, B, C
closeup: only hand from B
wide: person A, B, C
```

返回全景后不能把：

```text
A → B
B → C
C → new person
```

整体平移。

## 24.7 test_action_tracker.py

至少包含：

```text
test_new_action_started_in_commit_interval
test_overlapping_action_is_continued_not_duplicated
test_action_possible_ended_after_one_missing_window
test_action_ended_after_configured_missing_windows
test_camera_change_makes_action_uncertain
test_instant_action_not_repeated_from_context_interval
test_repeated_same_action_after_gap_gets_new_id
test_unresolved_actor_does_not_create_action
```

## 24.8 test_transition_engine.py

至少包含：

```text
test_high_confidence_initial_value_is_initialized
test_initial_value_is_not_transition
test_same_value_does_not_emit_duplicate_transition
test_medium_confidence_requires_two_windows
test_high_confidence_transition_requires_action_support
test_context_only_attribute_does_not_create_transition
test_newly_visible_attribute_is_not_false_transition
test_conflicting_pending_value_is_cancelled
test_attribute_invalid_for_entity_type_does_not_update_state
test_cabinet_label_change_does_not_mutate_same_entity_without_match
```

## 24.9 test_context_builder.py

至少包含：

```text
test_context_contains_recent_entities
test_context_contains_active_actions
test_context_excludes_old_inactive_entities
test_context_candidate_ids_are_valid
test_context_respects_character_limit
test_context_truncation_keeps_valid_json
```

## 24.10 test_state_pipeline.py

构造至少 8 个连续窗口：

```text
1. 三人和两个柜体首次出现
2. 人员持续检查 5 号柜
3. 中间人员递交工具
4. 镜头切到控制面板特写
5. 指示灯首次可见
6. 返回全景
7. 人员打开 4 号柜门
8. 柜门开启被再次确认
```

验证：

* 三个人的正式 ID 稳定；
* 4 号柜和 5 号柜不合并；
* hand_over 不变成 unknown；
* 特写首次看到指示灯不生成伪状态转移；
* 重叠窗口没有重复创建同一个动作；
* 柜门变化在满足确认条件后生成一次转移；
* 最终状态可以 JSON 序列化。

## 24.11 test_observation_replay.py

至少包含：

```text
test_replay_does_not_call_model
test_v1_observation_can_be_adapted
test_replay_twice_produces_same_events
test_replay_twice_produces_same_final_state
test_replay_rejects_missing_window_metadata
```

## 24.12 Golden Regression

新增：

```text
tests/golden/stage2_sequence.jsonl
tests/golden/expected_stage2_events.jsonl
```

Golden Fixture 使用脱敏、手工构造的紧凑 Observation，不提交原始私有视频或 Base64 图片。

测试必须精确比较：

* 实体映射；
* 事件类型；
* 动作 ID；
* 状态转移；
* 最终状态关键字段。

禁止只比较文件存在。

---

# 二十五、状态质量分析脚本

新增：

```text
scripts/evaluate_state_run.py
```

支持：

```bash
python scripts/evaluate_state_run.py outputs/<run_id>
```

至少输出：

```text
窗口数量
成功 Observation 数量
状态更新成功数量
动作 OOV 数量和比例
属性 OOV 数量和比例
实体 created / matched / ambiguous 数量
临时实体数量
实体合并数量
全局实体总数
动作 started / ended / uncertain 数量
重复动作候选数量
属性 initialized / pending / transition / conflict 数量
无证据事件数量
camera_change 附近实体 ID 变化数量
平均上下文字符数
最终状态实体和活跃动作数量
```

## 25.1 基础质量指标

定义：

```text
Action OOV Rate
Attribute OOV Rate
Entity Ambiguous Rate
Entity ID Switch Candidate Count
Duplicate Action Candidate Count
Unsupported Transition Count
Evidence Coverage
```

第一版脚本可以通过确定性启发式检测候选问题，不要求提供人工标注精确率。

## 25.2 运行产物检查

脚本还应检查：

* state_events 中引用的实体是否存在；
* action actor/target/tool 是否存在；
* event_id 是否唯一；
* action_id 是否唯一；
* entity_id 是否唯一；
* last_committed_window 是否单调；
* snapshot 是否可解析；
* final_state 与最后 snapshot 是否一致；
* evidence sample index 是否能映射到时间。

---

# 二十六、README 更新

README 必须新增：

1. 第二阶段定位；
2. 模型 Observation 与 GlobalState 的职责边界；
3. State Engine 架构图；
4. Observation Schema 2.0；
5. Context / Commit Interval；
6. EntityRegistry 和 EntityResolver；
7. ActionTracker 生命周期；
8. TransitionEngine 确认规则；
9. 状态输出目录；
10. Observation Replay 使用方法；
11. 状态配置说明；
12. Golden Regression 测试方式；
13. 状态质量分析命令；
14. 当前仍未实现的实时队列和 RTSP；
15. 已知限制。

明确说明：

```text
GlobalEntity ID 是程序侧确定性分配的运行内 ID，
不等同于跨不同视频、不同运行的永久真实身份。
```

明确说明：

```text
本阶段仍然是本地视频模拟流式处理，
不保证模型调用速度满足真实时间要求。
```

---

# 二十七、安全和数据要求

继续遵守第一阶段安全要求。

不得提交：

* API Key；
* 原始私有巡检视频；
* Base64 图片；
* 大量真实模型响应；
* 包含敏感路径的完整运行目录；
* 未脱敏人员身份信息。

Golden Fixture 必须手工构造或脱敏。

`.gitignore` 至少包含：

```text
.env
outputs/
__pycache__/
.pytest_cache/
.ruff_cache/
*.pyc
```

状态输出中不得保存 API Key。

---

# 二十八、实施顺序

必须按照以下顺序逐步实现：

```text
S2-T01 配置严格化与第一阶段明确问题修复
→ S2-T02 Observation Schema 2.0
→ S2-T03 非破坏性词表归一化
→ S2-T04 Context / Commit Interval
→ S2-T05 GlobalState 领域模型
→ S2-T06 SceneTracker
→ S2-T07 EntityRegistry
→ S2-T08 EntityResolver
→ S2-T09 ActionTracker
→ S2-T10 TransitionEngine
→ S2-T11 StateReducer
→ S2-T12 ContextBuilder
→ S2-T13 Pipeline 集成
→ S2-T14 状态输出存储
→ S2-T15 Observation Replay
→ S2-T16 Golden Regression
→ S2-T17 状态质量分析脚本
→ S2-T18 README 和最终验收
```

每完成一个任务：

1. 保证代码可导入；
2. 运行对应单元测试；
3. 运行 `ruff check`；
4. 不允许积累到最后统一修复；
5. 在提交说明中列出实际变更文件。

---

# 二十九、验收标准

本阶段完成后必须满足以下要求。

## 29.1 工程验收

* 新增独立 `state/` 模块；
* Pipeline 不包含实体匹配和状态转移具体算法；
* 配置模型使用严格字段校验；
* 公共类和函数具有类型标注；
* 无循环导入；
* 状态模型可以 JSON 序列化和恢复；
* 第一阶段命令仍可运行。

## 29.2 Observation 验收

* 默认输出 Schema 2.0；
* 支持 V1 Observation Replay 适配；
* 动作 OOV 不再被破坏性覆盖；
* `hand_over`、`receive`、`push`、`hover` 可被保留；
* 属性采用 canonical attribute_key；
* 属性和值经过实体类型约束；
* candidate_global_id 只能引用 ContextBuilder 候选；
* Prompt 默认采用 visual_only；
* 任务名称不会污染视觉事实。

## 29.3 窗口验收

* 每个窗口具有 Context 和 Commit 区间；
* 重叠区域不会重复创建新动作；
* 重叠区域不会重复创建状态转移；
* 非零起始窗口支持 warmup；
* run_meta 记录真实处理范围；
* cold start 被明确标记。

## 29.4 实体验收

* 正式实体 ID 只由程序分配；
* 同一窗口不存在两个局部实体映射同一可见全局实体；
* candidate ID 不能绕过硬约束；
* 匹配结果保存评分分解；
* 模糊匹配不会强制覆盖已有实体；
* 镜头切换不删除历史实体；
* 三人经过特写后返回全景，ID 不整体漂移；
* 4 号柜与 5 号柜不会因镜头移动被直接当作同一实体属性变化。

## 29.5 动作验收

* 重叠窗口中的持续动作只有一个 GlobalAction ID；
* 动作具有 started / ongoing / possible_ended / ended 生命周期；
* 镜头切换期间动作不会立即 ended；
* instant 动作不会因重叠窗口重复创建；
* 已结束后再次发生的同类动作可获得新 ID；
* 动作证据可映射到实际采样时间。

## 29.6 状态验收

* 初次观察属性不会伪造成状态转移；
* 同值观察不会重复生成转移；
* 中等置信度新值需要多窗口支持；
* 高置信度关键转移需要对应动作支持或显式属性策略；
* 新可见组件不会直接产生伪状态变化；
* 冲突观察进入 pending/conflicted，而不是直接覆盖；
* 每个正式转移具有 before、after、reason 和 evidence；
* 失败窗口不会导致实体消失或动作错误结束。

## 29.7 存储验收

* 生成 entity_resolutions.jsonl；
* 生成 state_events.jsonl；
* 生成 state_deltas.jsonl；
* 生成 state_snapshots.jsonl；
* 生成 final_state.json；
* 状态写入失败有独立错误日志；
* Prompt、Schema 和词表正文被快照保存；
* 输出不会覆盖已有运行。

## 29.8 Replay 验收

* Replay 不调用模型 API；
* 可重放 Schema 1.0 和 2.0；
* 相同输入重复重放得到相同状态事件；
* 相同输入重复重放得到相同最终状态；
* 缺少必要窗口和证据元数据时明确失败。

## 29.9 测试验收

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

Golden Regression 必须通过精确事件和最终状态断言。

---

# 三十、最终交付内容

完成后输出：

1. 修改后的完整代码；
2. 新目录树；
3. 第二阶段架构说明；
4. Observation Schema 2.0 示例；
5. GlobalState 示例；
6. EntityResolution 示例；
7. Action 生命周期示例；
8. Attribute Transition 示例；
9. Context / Commit Interval 示例；
10. 新增配置说明；
11. 本地视频运行命令；
12. Observation Replay 命令；
13. 输出目录示例；
14. 单元测试和集成测试结果；
15. Golden Regression 结果；
16. 状态质量分析结果；
17. 本阶段修改涉及的文件列表；
18. 尚未实现的第三阶段功能；
19. 已知限制和后续优化建议。

不要只输出设计文档，必须实际修改仓库代码。

不要实现实时生产者—消费者队列、RTSP、摄像头、违规判断和多 Agent。本阶段通过验收后，再进入第三阶段。
