# S2-T05：GlobalState 领域模型

**状态：TODO**　**依赖：S2-T02、S2-T04**

## 目标

建立可序列化、可复制、可重放的全局状态、状态事件、状态增量、证据和实体解析领域模型。

## 修改

- 新增：
  - `src/qwen_stream_video/domain/state.py`；
  - `src/qwen_stream_video/domain/event.py`；
  - `src/qwen_stream_video/domain/resolution.py`。
- 定义 `VisibilityState`、`EntityLifecycleStatus`、`EntityResolutionStatus`、`ActionLifecycle`、`AttributeConfirmationStatus`。
- 定义 `EvidenceReference`，包含 run、window、local ID、sample index 和程序映射的时间。
- 定义 `TimeInterval`，要求 `lower <= upper`，避免伪精确时间点。
- 定义 `SpatialObservation`、`AttributeState`、`GlobalEntityState`、`GlobalActionState`、`SceneState`、`GlobalState`。
- 所有实体、动作、事件和场景计数器保存在 `GlobalState` 中，保证 Replay 确定性。
- 定义 `StateEvent` 和至少覆盖场景、实体、动作、属性、缺口、错误的事件类型。
- 定义 `StateDelta`，只描述本窗口修改，不复制完整状态。
- 定义 `MatchScoreBreakdown`、`EntityResolution`、`EntityResolutionBatch` 等解析结果模型。
- 提供深复制、JSON 序列化和 JSON 恢复测试。

## 不做

不实现任何匹配、跟踪、转移或状态提交算法。

## 验收

- 所有模型严格校验、可深复制、可 JSON 序列化和恢复。
- 所有可变字段使用 `default_factory`。
- ID 计数器不会依赖系统时间或模块级全局变量。
- StateEvent 可追踪到证据；StateDelta 不包含完整 GlobalState。
- 相关单元测试和 `ruff check` 通过。

## 完成记录

- 修改文件：
- 验证结果：
- 已知限制：
