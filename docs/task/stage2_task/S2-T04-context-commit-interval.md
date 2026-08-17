# S2-T04：Context / Commit 双区间

**状态：DONE**　**依赖：S2-T01、S2-T02**

## 目标

为重叠滑动窗口建立程序侧提交区间和 warmup 机制，从根源上避免重叠区域重复创建动作和状态变化。

## 修改

- 扩展 `VideoWindow`：
  - `commit_start_seconds`；
  - `processing_role: warmup|commit`。
- 约束 `start_seconds <= commit_start_seconds < end_seconds`。
- 实现 Commit 起点计算：
  - cold start 首窗提交完整区间；
  - 有前置窗口时，从 `max(current.start, previous.end)` 开始提交；
  - 非重叠窗口提交完整窗口。
- 新增证据时间工具：
  - `evidence_timestamps()`；
  - `evidence_intersects_commit_interval()`。
- Context-only Observation 可以支持实体匹配和延续已有动作，但不能创建新动作或新状态转移。
- 实现 `warmup_windows`：
  - 非零起始窗口先处理前置窗口；
  - warmup 只建立上下文，不写入正式评测范围事件；
  - 在 `windows.jsonl` 和 `run_meta.json` 明确标记。
- 无可用前置窗口时记录 `cold_start=true`。
- 新增 `tests/unit/test_commit_interval.py`，覆盖首窗、重叠、不重叠、尾窗、Context-only 新动作禁止和已有动作延续。

## 不做

不实现 ActionTracker 或 TransitionEngine 的完整业务规则，只提供它们需要的窗口与证据判断基础。

## 验收

- 默认 6 秒窗口、3 秒步长产生 `[0,6)`、`[6,9)`、`[9,12)` 的提交区间。
- 从非零窗口启动且有 warmup 时，第一个正式窗口不重复提交 warmup 覆盖区间。
- Context-only 新事实不能被标记为可提交。
- cold start 和真实处理范围可追踪。
- `pytest tests/unit/test_commit_interval.py -q` 和窗口回归测试通过。

## 完成记录

- 修改文件：
- 验证结果：
- 已知限制：
