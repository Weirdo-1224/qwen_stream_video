# S2-T09：ActionTracker

**状态：TODO**　**依赖：S2-T04、S2-T05、S2-T08**

## 目标

将重叠窗口中的局部动作合并为具有稳定 ID、证据和生命周期的 GlobalActionState。

## 修改

- 新增 `src/qwen_stream_video/state/action_tracker.py`。
- 将 actor/target/tool 局部引用转换为解析后的全局 ID。
- actor 未解析时不得创建全局动作；target/tool 未解析时只能创建带缺失引用记录的 uncertain 动作。
- 基础动作键为 `(actor_id, action_type, target_id, tool_id)`，同时考虑时间间隔、场景连续性、已有生命周期和 Commit 区间。
- 生命周期规则：
  - Commit 区间首次出现：`started`；
  - 再次观察：`ongoing`；
  - 首次缺失：`possible_ended`；
  - 达到缺失阈值：`ended`；
  - instant 动作：`instant`；
  - 镜头切换/遮挡：`uncertain`；
  - Context-only 重复：只补证据。
- 使用证据采样时间构造 `TimeInterval`，不保存无法支持的伪精确时间点。
- 同动作键在允许间隔内延续；已结束且超过重复间隔时分配新 ID。
- GlobalAction ID 使用 `action_000001` 格式，计数器位于 GlobalState。
- 生成 started、continued、possible_ended、ended、instant、uncertain 事件。
- 新增 `tests/unit/test_action_tracker.py`。

## 不做

不判断动作是否符合作业规程，不做违规检测。

## 验收

- 重叠窗口持续动作只有一个 GlobalAction ID。
- Context-only instant 动作不会重复创建。
- 镜头切换不会让动作立即 ended。
- 已结束后超过阈值再次发生的同类动作获得新 ID。
- unresolved actor 不创建动作。
- `pytest tests/unit/test_action_tracker.py -q` 和 `ruff check` 通过。

## 完成记录

- 修改文件：
- 验证结果：
- 已知限制：
