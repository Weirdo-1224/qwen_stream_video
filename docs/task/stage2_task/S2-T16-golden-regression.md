# S2-T16：集成测试与 Golden Regression

**状态：TODO**　**依赖：S2-T01 至 S2-T15**

## 目标

建立覆盖身份、动作、状态、镜头切换和重叠窗口的稳定回归基线，防止后续优化破坏第二阶段核心语义。

## 修改

- 新增 `tests/integration/test_state_pipeline.py`，构造至少 8 个连续窗口：
  1. 三人和两个柜体首次出现；
  2. 持续检查 5 号柜；
  3. 人员递交工具；
  4. 控制面板特写；
  5. 指示灯首次可见；
  6. 返回全景；
  7. 打开 4 号柜门；
  8. 柜门开启再次确认。
- 新增 `tests/integration/test_camera_change_sequence.py`。
- 新增：
  - `tests/golden/stage2_sequence.jsonl`；
  - `tests/golden/expected_stage2_events.jsonl`；
  - 必要的期望最终状态 Fixture。
- Golden 数据必须手工构造或脱敏，不包含真实视频、Base64 图片或敏感人员信息。
- 精确断言：
  - 三人 ID 稳定；
  - 4 号柜和 5 号柜不合并；
  - hand_over 被保留；
  - 指示灯首次可见不生成伪转移；
  - 重叠窗口不重复动作；
  - 柜门转移只生成一次；
  - 动作和事件 ID 确定；
  - 最终状态关键字段准确。
- 对失败窗口、Context-only 证据、cold start、warmup 和 Replay 增加集成覆盖。
- Golden 测试不得只检查文件存在。

## 不做

不评估真实模型准确率，不调用真实 API，不提交原始私有视频。

## 验收

- Golden Fixture 的实体映射、动作 ID、事件序列、属性转移和最终状态精确匹配。
- 重复运行测试结果完全一致。
- 至少覆盖三人—特写—三人和双柜体不误合并场景。
- `pytest -q` 全部通过，`ruff check .` 通过。

## 完成记录

- 修改文件：
- 验证结果：
- Golden 结果：
- 已知限制：
