# S2-T15：Observation Replay

**状态：DONE**　**依赖：S2-T04、S2-T05、S2-T11、S2-T14**

## 目标

支持从已有 Observation 运行产物离线重建 GlobalState，不调用模型 API，并保证相同输入的状态语义结果确定一致。

## 修改

- 新增 `src/qwen_stream_video/state/replay.py`。
- CLI 新增：
  - `--replay-observations PATH`；
  - 与 `--video` 默认互斥；
  - Replay 不要求 API Key。
- Replay 至少读取 `observations.jsonl`、`windows.jsonl` 和 `run_meta.json`；需要证据时间时读取采样帧元数据。
- 按窗口顺序规范化、适配并调用 StateReducer，不调用 QwenClient。
- 新增 `ObservationV1Adapter`：
  - Schema 1.0 属性映射到 canonical key；
  - 保留 raw_attribute/raw_value；
  - 非法具体动作映射为 `other` 并保留 raw；
  - 缺失 relations 使用空列表；
  - 根据窗口序列计算 commit_start；
  - 不伪造 candidate ID 或证据。
- Schema 2.0 输入直接严格校验。
- 将运行时间、绝对路径等非确定字段排除出状态语义文件。
- 两次 Replay 的 `state_events.jsonl` 和 `final_state.json` SHA256 必须一致。
- 新增 `tests/integration/test_observation_replay.py`。

## 不做

不从视频重新抽帧或调用模型补全缺失 Observation，不修复无法恢复的证据。

## 验收

- Replay 模式没有任何模型请求。
- Schema 1.0 和 2.0 均可按明确规则重放。
- 缺少窗口或必要证据元数据时明确失败。
- 相同输入重复 Replay 事件和最终状态哈希一致。
- `pytest tests/integration/test_observation_replay.py -q` 和 `ruff check` 通过。

## 完成记录

- 修改文件：
- 验证结果：
- 已知限制：
