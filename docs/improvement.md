
# 一、先重新定义系统目标

当前实现的隐含目标是：

```text
每个窗口 → 生成一份完整的局部视频分析报告
```

应改成：

```text
每个窗口 → 识别相对上一状态发生了什么变化
```

模型不再负责输出完整世界状态，而只负责输出：

1. 当前新增了什么实体；
2. 哪些已有实体发生属性变化；
3. 哪些动作开始、持续或结束；
4. 发生了哪些确定的状态变化；
5. 哪些信息暂时无法确认。

完整状态由程序维护：

```text
当前窗口视觉观察
        +
程序中的历史状态
        ↓
增量更新
        ↓
新的全局状态
```

这是最核心的修改。否则无论怎么缩短提示词，模型仍会每三秒重新描述一次安全帽、工作服、配电柜、工具车。

---

# 二、把输出从“全量快照”改成“增量事件”

## 2.1 当前输出的问题

当前每个窗口都返回：

```json
{
  "entities": [
    "person_1完整描述",
    "person_2完整描述",
    "person_3完整描述",
    "device_1完整描述",
    "tool_1完整描述",
    "tool_2完整描述",
    "cart_1完整描述"
  ],
  "actions": [],
  "state_changes": [],
  "uncertainties": []
}
```

即使连续多个窗口完全没有变化，也会重复输出所有实体。

例如前几个窗口中，三个人员的安全帽、工装、手套、位置和设备状态被反复重新生成；API 输入和输出 Token 也随之保持在较高水平。

## 2.2 建议的新 Schema

改成下面这种增量结构：

```json
{
  "window": {
    "start_time": "00:08:03.000",
    "end_time": "00:08:09.000",
    "scene_id": "scene_1"
  },
  "summary": "三人继续在开启的配电柜前作业，无明确设备状态变化。",
  "scene_change": {
    "changed": false,
    "change_type": "none"
  },
  "new_entities": [],
  "entity_updates": [],
  "action_updates": [
    {
      "action_id": "action_1",
      "actor_id": "person_1",
      "action_type": "hold_and_insert",
      "object_id": "device_1",
      "tool_id": "tool_1",
      "status": "ongoing",
      "confidence": 0.72,
      "evidence_frames": [1, 3, 5]
    }
  ],
  "state_changes": [],
  "observations": [],
  "uncertainties": []
}
```

只有首次出现人物时才输出：

```json
{
  "new_entities": [
    {
      "temporary_id": "local_person_1",
      "entity_type": "person",
      "attributes": {
        "relative_position": "left",
        "ppe": ["helmet", "gloves"]
      }
    }
  ]
}
```

后续只有位置、姿态或持有物变化时才输出：

```json
{
  "entity_updates": [
    {
      "entity_id": "person_3",
      "changed_attributes": {
        "attention_target": {
          "before": "device_1",
          "after": "clipboard_1"
        }
      },
      "confidence": 0.78
    }
  ]
}
```

长期不变的字段不再输出。

---

# 三、实体 ID 不能继续完全交给模型管理

## 3.1 当前暴露的问题

结果中已经出现两类典型问题。

第一类是镜头近景后，原来的三名人员变成了：

```text
person_4
```

这是可以理解的，因为特写中只有手部，模型无法确定其对应 `person_1`、`person_2` 还是 `person_3`。

第二类更严重：早期的 `tool_1` 是黑色软管，后面的窗口里 `tool_1` 又被用于红色工具车，造成 ID 语义冲突。

因此必须遵循：

> 模型负责发现局部实体，程序负责分配和维护全局 ID。

## 3.2 增加 EntityRegistry

程序中加入：

```python
class EntityRegistry:
    entities: dict[str, EntityState]
    aliases: dict[str, str]
    next_ids: dict[str, int]
```

实体状态至少包括：

```python
@dataclass
class EntityState:
    entity_id: str
    entity_type: str
    canonical_name: str
    attributes: dict
    first_seen: float
    last_seen: float
    confidence: float
    scene_id: str
    missing_windows: int
```

模型只输出临时实体：

```json
{
  "temporary_id": "local_tool_1",
  "entity_type": "tool",
  "canonical_type": "flexible_tube",
  "appearance": "black_flexible_tube",
  "holder_local_id": "local_person_1"
}
```

程序再根据以下条件匹配：

```text
实体类型
规范化名称
所在位置
持有者关系
外观特征
上一窗口是否可见
当前场景是否连续
```

匹配成功：

```text
local_tool_1 → tool_1
```

匹配失败：

```text
创建 tool_3
```

模型不得自行覆盖已经存在的 ID。

---

# 四、近景镜头不要强行绑定人员

近景镜头只看到双手时，不应该直接生成一个普通的 `person_4`，也不应该强制认为是 `person_2`。

建议使用：

```json
{
  "temporary_id": "unknown_person_closeup_1",
  "entity_type": "person_fragment",
  "visible_parts": ["hands", "forearms"],
  "candidate_matches": [
    {
      "entity_id": "person_2",
      "confidence": 0.55
    },
    {
      "entity_id": "person_1",
      "confidence": 0.21
    }
  ]
}
```

只有出现足够证据时才合并：

```text
同样手套特征
镜头切换前后位置一致
所持工具一致
动作连续
```

否则保留为：

```text
unknown_person_closeup_1
```

这比错误地建立确定身份更可靠。

---

# 五、动作必须建立生命周期，而不是每个窗口重新描述

## 5.1 建立动作主键

建议使用：

```text
actor_id + action_type + object_id + tool_id
```

生成动作键：

```python
action_key = (
    actor_id,
    action_type,
    object_id,
    tool_id,
)
```

例如：

```text
person_1 + hold_and_insert + device_1 + tool_1
```

只要相邻窗口中的动作键相同，程序就判断为同一动作。

## 5.2 动作状态由程序辅助决定

模型可以输出本窗口观察：

```json
{
  "action_type": "cut",
  "actor_id": "unknown_person_closeup_1",
  "object_id": "wire_1",
  "tool_id": "scissors_1",
  "visible": true
}
```

程序根据上一状态确定：

```text
上一窗口不存在，本窗口出现 → started
上一窗口存在，本窗口仍出现 → ongoing
上一窗口存在，本窗口消失 → possible_ended
连续两个窗口消失 → ended
动作很短且窗口内完整发生 → instant
```

推荐不要仅凭一个窗口消失就立即判定 `ended`，因为可能只是：

* 手部被遮挡；
* 镜头暂时切换；
* 抽帧没有采到；
* 操作对象移出画面。

建议：

```yaml
action_end_missing_windows: 2
```

---

# 六、使用受控词表，禁止自由生成动作名称

当前同一个动作被描述成：

```text
伸手探入柜内操作
整理
插拔接线
操作接线
操作柜内设备
```

这些自然语言在阅读上没有问题，但不能直接用于状态统计。

需要同时保留两个字段：

```json
{
  "action_type": "manipulate_connection",
  "action_description": "右手在绿色端子排附近调整白色线缆"
}
```

## 6.1 建议的动作枚举

第一版可以只保留：

```text
observe
record
hold
approach
touch
insert
remove
connect
disconnect
adjust
press
rotate
switch
open
close
cut
strip
measure
inspect
place
pick_up
unknown
```

复合动作不要自由生成：

```text
持握并微调伸入
```

应拆成主要动作：

```json
{
  "action_type": "insert",
  "modifiers": ["holding", "slight_adjustment"]
}
```

## 6.2 建议的状态属性枚举

```text
door_state:
  unknown | open | closed | partially_open

connection_state:
  unknown | connected | partially_disconnected | disconnected

insulation_state:
  unknown | intact | cut | partially_removed | fully_removed

switch_state:
  unknown | on | off | work | test | intermediate

visibility_state:
  visible | partial | occluded | absent

holding_state:
  held | released | unknown
```

当前输出中出现了：

```text
insulation_status
insulation_sleeve_condition
```

两种名称。统一后只能使用：

```text
insulation_state
```

---

# 七、状态变化必须由“旧值—新值”触发

## 7.1 当前问题

模型有时描述：

```text
切口扩大
局部剥离更明显
套管仍未脱离
```

但并不总是写入 `state_changes`，而且状态维度名称也不统一。

## 7.2 建议的状态机

对于绝缘套管：

```text
unknown
  ↓
intact
  ↓
cut
  ↓
partially_removed
  ↓
fully_removed
```

模型只输出当前观察值：

```json
{
  "entity_id": "wire_1",
  "attribute": "insulation_state",
  "observed_value": "cut",
  "confidence": 0.91
}
```

程序对比全局状态：

```python
old_value = global_state["wire_1"]["insulation_state"]
new_value = observation["observed_value"]

if old_value != new_value:
    emit_state_change(old_value, new_value)
```

输出：

```json
{
  "entity_id": "wire_1",
  "attribute": "insulation_state",
  "before": "intact",
  "after": "cut",
  "first_observed_at": "00:08:30.000",
  "confirmed_at": "00:08:33.000"
}
```

对于不确定变化，可以采用两阶段确认：

```text
第一次观察：candidate
连续两个窗口相同：confirmed
```

例如：

```python
if confidence >= 0.85:
    confirm_immediately()
elif same_observation_count >= 2:
    confirm()
else:
    keep_candidate()
```

---

# 八、上一窗口状态需要大幅压缩

## 8.1 当前问题

当前 `next_previous_state` 几乎重新保存了完整实体和完整动作描述，并在下一次请求中再次传给模型。

这使得后续输入中出现大量重复文本：

```text
穿深蓝色工装
戴红色安全帽
戴白色手套
绿色端子排
蓝色继电器
工具车静止
```

## 8.2 新的 previous_state

只传模型完成判断真正需要的信息：

```json
{
  "scene_id": "scene_2",
  "entities": {
    "person_fragment_1": {
      "type": "person_fragment",
      "visible": true
    },
    "wire_1": {
      "type": "wire",
      "insulation_state": "cut",
      "connection_state": "connected"
    },
    "tool_3": {
      "type": "scissors",
      "holder": "person_fragment_1"
    }
  },
  "active_actions": {
    "action_4": {
      "type": "cut",
      "actor": "person_fragment_1",
      "object": "wire_1",
      "tool": "tool_3"
    }
  }
}
```

不要传：

* 服装的完整中文描述；
* 工具的颜色和长度；
* 长句形式的 `visible_state`；
* 过去所有 uncertainty；
* 无变化背景物体；
* 长篇 evidence 文本。

建议限制：

```text
previous_state JSON ≤ 500 字符
```

或者最多：

```text
10 个实体
5 个进行中动作
3 个待确认状态
```

---

# 九、提示词需要进一步压缩和强化

## 9.1 删除以下要求

窗口模型不再负责：

```text
完整场景重新描述
所有可见实体枚举
长期不变的 PPE 重复输出
长期不变的工具车描述
对整个窗口生成详细报告
对每个不确定项写长篇原因
```

## 9.2 增加硬性限制

建议在 System Prompt 中增加：

```text
1. 只输出相对于 previous_state 新增、变化或结束的信息。
2. previous_state 中未发生变化的实体不得重复输出。
3. summary 不超过 40 个汉字。
4. 每个 description 不超过 30 个汉字。
5. uncertainties 最多输出 3 条。
6. 不估计尺寸、长度、距离、角度或毫米级变化。
7. 除非存在标尺或清晰文字，不得输出“约 5 mm”“约 40 cm”等数值。
8. action_type、attribute 和 state value 必须从给定枚举中选择。
9. 模型不得创建全局实体 ID，只输出 temporary_id 或引用已提供的 entity_id。
10. 严格输出 JSON，不输出 Markdown。
```

## 9.3 增加无变化快速输出

如果当前窗口没有明显变化，允许直接返回：

```json
{
  "window": {
    "start_time": "00:08:09.000",
    "end_time": "00:08:15.000"
  },
  "summary": "作业持续，无明确变化。",
  "scene_change": {
    "changed": false,
    "change_type": "none"
  },
  "new_entities": [],
  "entity_updates": [],
  "action_updates": [],
  "state_changes": [],
  "observations": [],
  "uncertainties": []
}
```

不要为了“完成 Schema”而重新生成整幅画面的所有信息。

---

# 十、为采样帧添加明确编号

当前模型会直接写：

```text
00:08:27～00:08:29
```

但窗口只以 1 FPS 抽取 6 帧，模型实际上很难可靠判断精确连续时间。

建议在请求中给每帧编号：

```text
F0 = 00:08:24.480
F1 = 00:08:25.480
F2 = 00:08:26.480
F3 = 00:08:27.480
F4 = 00:08:28.480
F5 = 00:08:29.480
```

模型输出证据：

```json
{
  "evidence_frames": [3, 4, 5]
}
```

程序根据帧编号转换时间：

```python
evidence_timestamps = [
    sampled_timestamps[i]
    for i in evidence_frames
]
```

这样比让模型自由生成时间戳可靠。

---

# 十一、镜头切换必须单独处理

当前全景切换为柜内近景时，模型将场景和实体状态整体重建。建议增加：

```json
{
  "scene_change": {
    "changed": true,
    "change_type": "wide_to_closeup",
    "continuity": "uncertain"
  }
}
```

程序维护：

```text
scene_1：全景
scene_2：柜内特写
scene_3：返回全景
```

发生镜头切换时：

1. 不立即删除之前实体；
2. 将其标记为 `not_visible_in_current_scene`；
3. 新近景实体使用临时 ID；
4. 不允许新实体覆盖旧实体 ID；
5. 返回全景后重新激活已有实体；
6. 只有长期消失才标记为离场。

---

# 十二、修复 JSON 解析失败

当前窗口 9 出现 `JSONDecodeError`，说明仅依赖模型返回合法 JSON 不够。

建议增加完整容错链路。

## 12.1 保存原始响应

每个窗口都保存：

```text
raw_responses/window_0009.txt
```

即使解析失败，也能复盘。

## 12.2 分级解析

````python
def parse_model_output(raw: str) -> dict:
    # 1. 直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. 去除 ```json 代码围栏
    cleaned = strip_markdown_fence(raw)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 3. 使用 JSON repair
    repaired = repair_json(cleaned)
    return json.loads(repaired)
````

可以引入：

```text
json-repair
pydantic
```

## 12.3 解析失败后不要重新分析视频

第一次返回已经包含模型判断，只是 JSON 格式损坏。因此不要再次发送视频。

可以发起一个轻量文本修复请求：

```text
以下文本本应是 JSON，但格式损坏。
只修复 JSON 语法，不修改语义，不补充新信息。
```

## 12.4 重试限制

```yaml
json_repair:
  enabled: true
  max_retries: 1
  save_raw_response: true
```

仍失败则记录：

```json
{
  "status": "parse_error",
  "raw_response_path": "...",
  "state_updated": false
}
```

不要让错误窗口污染历史状态。

---

# 十三、增加 Schema 真实校验

当前 `schema_warnings` 大多为空，但实际仍然存在：

* ID 冲突；
* `tool_id` 指向错误类型；
* 动作对象与工具关系不合理；
* 状态属性名称漂移。

说明当前 Schema 校验只检查了“字段是否存在”，没有检查“语义一致性”。

建议增加以下校验。

## 13.1 引用完整性

```python
assert actor_id in known_entities
assert object_id in known_entities or object_id == "unknown"
assert tool_id in known_entities or tool_id is None
```

## 13.2 类型校验

```python
assert entities[actor_id].type in {
    "person",
    "person_fragment",
}
assert entities[tool_id].type == "tool"
```

这样可以发现“记录动作引用了工具车 ID”之类的问题。

## 13.3 枚举校验

```python
assert action_type in ACTION_TYPES
assert attribute in STATE_ATTRIBUTES
assert state_value in ATTRIBUTE_VALUE_MAP[attribute]
```

## 13.4 状态逻辑校验

例如：

```text
before == after
```

则不能产生状态变化。

```text
insulation_state: fully_removed → intact
```

若没有明显恢复证据，应标记为冲突，而不是直接覆盖。

---

# 十四、性能优化的正确顺序

当前性能问题不应只靠换模型解决。

## 第一优先级：减少输出

目标从：

```text
约 900～1500 completion tokens/window
```

降到：

```text
100～300 completion tokens/window
```

手段：

* 改增量 Schema；
* 不输出无变化实体；
* 缩短 summary；
* uncertainties 最多 3 条；
* 不重复 PPE 和背景；
* 限制 `max_tokens`。

## 第二优先级：减少输入

目标：

```text
previous_state 只保留稳定状态和进行中动作
```

删除：

* 长篇实体属性；
* 过去窗口的完整动作文本；
* 已解决的不确定项；
* 重复的自然语言状态。

## 第三优先级：更换低延迟模型

在语义和 Schema 稳定后，再对比：

```text
当前高能力模型
低延迟视觉模型
更低成本模型
```

否则直接换小模型，只会把当前的重复输出和 ID 漂移更快地产生出来。

## 第四优先级：窗口调度

真实流式运行时使用：

```text
只处理最新窗口
旧窗口不排队
```

如果模型正在处理窗口 10，而窗口 11、12、13 已经产生：

```text
丢弃 11、12
处理最新的 13
```

记录：

```json
{
  "skipped_windows": [11, 12],
  "reason": "inference_backlog"
}
```

这样延迟不会无限累积。

---

# 十五、建议调整输出文件结构

当前单个 JSONL 同时包含：

* 视频窗口；
* 模型分析；
* API 信息；
* 全局状态；
* 下一个状态。

建议拆成四类文件：

```text
outputs/run_xxx/
├── windows.jsonl
├── observations.jsonl
├── events.jsonl
├── state_snapshots.jsonl
├── api_metrics.jsonl
├── errors.jsonl
└── raw_responses/
```

## `windows.jsonl`

只记录窗口和采样信息：

```json
{
  "window_index": 10,
  "start": 510,
  "end": 516,
  "frame_timestamps": []
}
```

## `observations.jsonl`

记录模型原始增量输出。

## `events.jsonl`

记录程序合并后的确定事件：

```json
{
  "event_id": "event_12",
  "type": "state_change",
  "entity_id": "wire_1",
  "attribute": "insulation_state",
  "before": "intact",
  "after": "cut"
}
```

## `state_snapshots.jsonl`

每隔若干窗口保存一次完整状态，而不是每个窗口都重复保存：

```yaml
state_snapshot_interval: 10
```

## `api_metrics.jsonl`

保存：

```text
延迟
输入 Token
输出 Token
重试次数
缓存 Token
状态
```

---

# 十六、建议增加的评估指标

修改后不要只看描述是否正确，还要统计流式质量。

## 结构稳定性

```text
JSON 有效率
Schema 通过率
实体引用有效率
枚举合法率
```

## 跨窗口一致性

```text
实体 ID 切换次数
同一实体重复创建次数
动作碎片化次数
状态无依据翻转次数
```

## 增量效率

```text
每窗口平均新增实体数
每窗口平均重复字段数
无变化窗口占比
每窗口输出 Token
```

## 时序能力

```text
动作开始延迟
动作结束延迟
状态变化检测延迟
状态变化重复触发次数
```

## 性能

```text
平均 API 延迟
P95 API 延迟
Real-Time Factor
峰值积压窗口数
丢弃窗口数
```

建议第一阶段验收目标：

| 指标            | 建议目标 |
| ------------- | ---: |
| JSON 有效率      | ≥99% |
| 实体引用有效率       | 100% |
| 全局 ID 冲突      |    0 |
| 无变化窗口输出 Token | ≤150 |
| 有变化窗口输出 Token | ≤300 |
| 重复实体描述下降      | ≥70% |
| 状态属性枚举合法率     | 100% |

这些是工程目标，不代表模型能力保证。

---

# 十七、推荐的修改顺序

## P0：必须先改

1. 将输出改为增量 Schema；
2. 增加程序侧 `EntityRegistry`；
3. 增加程序侧 `GlobalState`；
4. 统一动作和状态枚举；
5. 不再让模型创建全局 ID；
6. 增加 Pydantic Schema 校验；
7. 增加 JSON repair 和失败重试；
8. 失败窗口禁止更新历史状态。

## P1：随后优化

1. 压缩 `previous_state`；
2. 限制 summary 和描述长度；
3. 禁止输出尺寸估计和伪精确数值；
4. 增加 `evidence_frames`；
5. 增加动作生命周期管理；
6. 增加镜头切换状态；
7. 拆分输出文件。

## P2：最后优化实时性

1. 比较不同千问视觉模型；
2. 控制最大输出 Token；
3. 减少输入图像分辨率；
4. 测试 4 帧和 6 帧差异；
5. 实现 latest-window-only 调度；
6. 实现自适应步长；
7. 增加无变化窗口跳过机制。

---

# 十八、建议下一版的最小目标

不要一次加入复杂 Agent、规则引擎或向量数据库。下一版只实现：

```text
6 秒滑动窗口
3 秒步长
6 帧输入
增量 JSON
程序管理实体 ID
程序维护当前状态
程序维护动作生命周期
JSON 自动修复
完整性能统计
```

用同一段 `08:00～09:03` 视频重新测试。

预期输出应从现在这种：

```text
每个窗口重新描述 5～8 个实体
每窗口约上千输出 Token
```

变成：

```text
无变化窗口：
“作业持续，无明确变化”
+ 0～2 个动作更新

有变化窗口：
新增工具、动作开始或状态变化
+ 证据帧编号
```

这样项目才真正从“连续生成局部报告”转变成：

> **基于滑动窗口的流式结构化状态抽取系统。**
