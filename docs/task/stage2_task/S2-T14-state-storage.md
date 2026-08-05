# S2-T14：状态输出存储

**状态：TODO**　**依赖：S2-T05、S2-T11、S2-T13**

## 目标

将程序侧状态结果以独立、可追踪、可恢复、不会覆盖历史运行的结构持久化，并保存实际使用的 Prompt、Schema 和词表正文。

## 修改

- 新增 `src/qwen_stream_video/storage/state_storage.py`，与第一阶段 Observation 存储职责分离。
- 状态启用时生成：
  - `normalization_warnings.jsonl`；
  - `entity_resolutions.jsonl`；
  - `state_events.jsonl`；
  - `state_deltas.jsonl`；
  - `state_snapshots.jsonl`；
  - `state_errors.jsonl`；
  - `final_state.json`。
- `entity_resolutions.jsonl` 保存 local/global 映射、状态、第一/第二分数、评分分解和拒绝理由。
- `state_events.jsonl` 只保存程序正式事件，不复制模型描述充当事件。
- 按 `snapshot_interval_windows` 保存快照，并强制保存最终窗口快照。
- `final_state.json` 使用临时文件、flush/fsync 和原子 rename。
- 状态错误记录阶段、窗口、异常链、是否影响状态和原状态引用。
- 保存 `artifacts/prompts/`、`artifacts/schemas/` 和 `artifacts/vocabularies/` 正文快照。
- 扩展 `run_meta.json`，记录请求范围、warmup、正式提交范围、覆盖时间、cold start 和两个 Schema 版本。
- 输出目录继续使用唯一 run_id，不覆盖历史结果。
- 新增 `tests/unit/test_state_storage.py`。

## 不做

不实现 Replay 读取或质量分析。

## 验收

- 所有状态文件逐行可解析，引用一致。
- final_state 原子写入，异常时不留下损坏正式文件。
- Prompt、Schema 和词表正文可直接审查，不仅保存哈希。
- API Key 和敏感内容不进入状态文件。
- 第一阶段输出继续保留，`observations.jsonl` 语义不变。
- `pytest tests/unit/test_state_storage.py -q` 和 `ruff check` 通过。

## 完成记录

- 修改文件：
- 验证结果：
- 已知限制：
