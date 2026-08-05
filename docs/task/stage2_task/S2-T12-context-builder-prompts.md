# S2-T12：ContextBuilder 与提示词

**状态：TODO**　**依赖：S2-T01、S2-T03、S2-T05、S2-T11**

## 目标

使用结构化、有限长度的全局状态上下文替换上一窗口自然语言摘要，并更新 Prompt 使模型只输出 Schema 2.0 局部视觉事实。

## 修改

- 新增 `src/qwen_stream_video/state/context_builder.py`。
- 实现 `build(state, current_window) -> ObservationContext`。
- 上下文只包含：
  - 当前场景；
  - 最近相关候选实体；
  - active/uncertain 动作；
  - pending 属性；
  - 与活跃实体直接关联的设备和工具；
  - 最近镜头切换信息。
- 排除长期失活实体、久远已结束动作、全量事件历史和完整 GlobalState。
- `candidate_global_id` 只能从 `candidate_entities` 选择；无法判断返回 null。
- 序列化长度不得超过 `max_serialized_characters`；按确定性优先级删除旧实体、非活跃关系实体、possible-ended 动作和低置信 pending。
- JSON 裁剪必须在对象层完成，禁止截断字符串。
- 更新 `prompts/observation_system.txt`：
  - 当前窗口事实；
  - 不维护正式身份；
  - 看不到不等于消失；
  - 首次可见不等于状态变化；
  - 不根据任务标题猜测；
  - 只输出 Schema 2.0 JSON。
- 更新 `prompts/observation_user.txt`，包含 Context/Commit 区间、采样帧编号与时间、候选上下文和词表摘要。
- `visual_only` 不发送具体文件名；`task_conditioned` 必须分离视觉事实与条件解释。
- 客户端在配置允许且接口支持时启用 `response_format={"type":"json_object"}`。
- 新增 `tests/unit/test_context_builder.py` 和 Prompt 构建测试。

## 不做

不改变 Pipeline 的执行顺序，不实现并发或第二次 LLM 状态推理。

## 验收

- Context 中所有候选 ID 均存在于 GlobalState。
- 上下文长度受限且始终为有效 JSON。
- 默认 Prompt 不含具体故障/回路名称。
- 模型不得自行创造正式全局 ID。
- 上一窗口摘要和自由候选实体列表不再作为主要状态上下文。
- `pytest tests/unit/test_context_builder.py -q`、Prompt 测试和 `ruff check` 通过。

## 完成记录

- 修改文件：
- 验证结果：
- 已知限制：
