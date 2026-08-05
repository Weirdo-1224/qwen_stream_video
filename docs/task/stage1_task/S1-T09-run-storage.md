# S1-T09：运行结果存储

**状态：DONE（已按单 ObservationBatch 协议更新）**　**依赖：S1-T02、S1-T03、S1-T05、S1-T08**

## 目标

为每次运行创建不可覆盖、可追踪的输出目录，并将单个 `ObservationBatch` 写入 `observations.jsonl`。

## 修改

- 更新 `src/qwen_stream_video/storage/storage.py`：
  - `write_window_result` 接收 `ObservationBatch` 对象；
  - 将单个校验通过的 `ObservationBatch` 写入 `observations.jsonl`；
  - 保持原有窗口、指标、错误、原始响应、采样帧等存储逻辑。
- 重写 `tests/unit/test_storage.py`：覆盖唯一目录、目录冲突、JSONL 输出、原始响应保存、配置脱敏、采样帧保存及成功窗口无错误记录等场景；断言 `observations.jsonl` 中每行为单个 `ObservationBatch` 对象。

## 不做

不决定处理顺序，不负责调用或解析模型。

## 验收

- 连续运行不覆盖旧输出，错误可关联原始响应路径。

## 完成记录

- 修改文件：
  - 更新 `src/qwen_stream_video/storage/storage.py`。
  - 重写 `tests/unit/test_storage.py`。
- 验证结果：
  - `.venv/Scripts/python -m pytest tests/unit/test_storage.py -q`：7 个测试全部通过。
  - `.venv/Scripts/python -m pytest tests/ -q`：98 个测试全部通过。
  - `.venv/Scripts/python -m ruff check .`：无错误。
