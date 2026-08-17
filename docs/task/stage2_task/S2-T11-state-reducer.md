# S2-T11：StateReducer

**状态：DONE**　**依赖：S2-T06 至 S2-T10**

## 目标

实现第二阶段唯一允许提交 GlobalState 修改的协调器，保证固定执行顺序、单窗口原子性、失败隔离和确定性。

## 修改

- 新增 `src/qwen_stream_video/state/state_reducer.py`。
- 实现 `apply_observation(state, observation, sampled_frames, window)`。
- 固定执行顺序：
  1. SceneTracker；
  2. EntityResolver；
  3. EntityRegistry 提交；
  4. ActionTracker；
  5. TransitionEngine；
  6. 未观察实体可见性更新；
  7. 未观察动作缺失计数；
  8. 生成 StateDelta；
  9. 更新 `last_committed_window`。
- 使用 `state.model_copy(deep=True)` 或等价机制实现原子更新。
- 状态阶段失败时：
  - 原状态不被部分修改；
  - 生成 `state_update_error`；
  - 根据 `fail_on_state_error` 继续或退出；
  - 原始 Observation 仍保留；
  - 不伪造 StateDelta。
- Observation 失败时生成 `observation_gap`，不得推断实体消失或错误结束动作。
- 影响匹配、ID 和事件顺序的集合全部显式排序。
- 定义 `StateReductionResult`，返回解析、动作、转移、事件、Delta、警告和新状态。
- 新增 reducer 原子性、失败窗口和确定性测试。

## 不做

不负责模型调用、文件持久化、Replay CLI 或并发调度。

## 验收

- 任一子模块异常不会留下半更新状态。
- 相同输入和配置重复执行得到相同实体、动作、事件和计数器。
- 失败窗口不会让实体消失、动作错误结束或 pending 被错误确认。
- Pipeline 之外的模块不能绕过 reducer 随意提交状态。
- reducer 相关测试和 `ruff check` 通过。

## 完成记录

- 修改文件：
- 验证结果：
- 已知限制：
