# S1-T10：流水线、CLI 与收尾

**状态：DONE（已按单 ObservationBatch 协议更新）**　**依赖：S1-T01 至 S1-T09**

## 目标

组合前序模块为稳定的本地视频增量观察程序，解析后直接使用单个 `ObservationBatch`。

## 修改

- 更新 `src/qwen_stream_video/pipeline.py`：
  - 解析后直接使用 `batch`，不再出现 `batch.observations[0]`；
  - 上一窗口上下文仅传递 `summary` 和候选实体摘要（candidate_global_id / entity_type / description），不实现全局实体注册；
  - 保持单窗口失败记录后继续处理，`KeyboardInterrupt` 继续传播。
- 更新 `src/qwen_stream_video/cli.py`：
  - `DEFAULT_FAKE_RESPONSE` 更新为新的单窗口 `ObservationBatch` JSON。
- 更新 `tests/unit/test_pipeline.py`：覆盖 validate-only、dry-run、正常路径、窗口选择、单窗口失败隔离、KeyboardInterrupt 传播、状态传递开关；断言 observations.jsonl 写入单个 batch。
- 更新 `tests/unit/test_client.py`：假客户端响应使用新的单窗口协议 JSON。
- 更新 `README.md`（如需要）与 CLI 参数说明保持一致。

## 不做

不实现后续的实体注册、动作跟踪、状态转换。

## 验收

- 本地 MP4 可执行 dry-run、validate-only 和正常路径（通过 Fake 客户端）。
- 单窗口失败不终止整次运行，也不伪造成功 Observation。
- `pytest -q` 通过，`ruff check .` 无严重错误。

## 完成记录

- 修改文件：
  - 更新 `src/qwen_stream_video/pipeline.py`。
  - 更新 `src/qwen_stream_video/cli.py`。
  - 重写 `tests/unit/test_pipeline.py`。
  - 更新 `tests/unit/test_client.py`。
- 验证结果：
  - `.venv/Scripts/python -m pytest tests/ -q`：98 个测试全部通过。
  - `.venv/Scripts/python -m ruff check .`：无错误。
  - `.venv/Scripts/python run.py --video videos/test_15s.mp4 --validate-only`：正确报告 4 个窗口及预估帧数。
  - `.venv/Scripts/python run.py --video videos/test_15s.mp4 --dry-run --max-windows 2`：成功生成输出目录并写入 2 个窗口，不调用模型。
- 已知限制：
  - 正常路径（调用真实模型）需要有效的 `DASHSCOPE_API_KEY` 与可访问的 `base_url`；未在本地使用真实 API 验证，已通过单元测试使用 `FakeQwenClient` 覆盖正常数据流。
