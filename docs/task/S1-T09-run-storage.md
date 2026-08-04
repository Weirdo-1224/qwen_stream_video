# S1-T09：运行结果存储

**状态：DONE**　**依赖：S1-T02、S1-T03、S1-T05、S1-T08**

## 目标

为每次运行创建不可覆盖、可追踪的输出目录。

## 修改

- 生成唯一 `YYYYMMDD_HHMMSS_<experiment>_<short_hash>` 的 `run_id`。
- 写入元数据、最终配置、窗口、观察、API 指标和错误 JSONL。
- 依配置保存原始响应与采样帧；元数据记录哈希、模型来源、环境版本与最终统计。
- 仅完整校验通过的 Observation 才写入 `observations.jsonl`。
- 添加唯一目录、JSONL 和原始响应测试。

## 不做

不决定处理顺序，不负责调用或解析模型。

## 验收

- 连续运行不覆盖旧输出，错误可关联原始响应路径。

## 完成记录

- 修改文件：
  - 新增 `src/qwen_stream_video/storage/storage.py`：实现 `RunStorage` 类，支持生成唯一 `YYYYMMDD_HHMMSS_<experiment>_<short_hash>` 的运行目录，写入 `metadata.json`、脱敏 `config.json`、`windows.jsonl`、`observations.jsonl`、`metrics.jsonl`、`errors.jsonl`，并可选保存 `raw_responses/` 与 `sampled_frames/`。
  - 更新 `src/qwen_stream_video/storage/__init__.py`：导出 `RunStorage`。
  - 新增 `tests/unit/test_storage.py`：覆盖唯一目录、目录冲突、JSONL 输出、原始响应保存、配置脱敏、采样帧保存及成功窗口无错误记录等场景。
- 验证结果：
  - `.venv/Scripts/python -m pytest tests/unit/test_storage.py -q`：7 个测试全部通过。
  - `.venv/Scripts/python -m pytest tests/ -q`：82 个测试全部通过。
  - `.venv/Scripts/python -m ruff check .`：无错误。
