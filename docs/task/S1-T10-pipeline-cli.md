# S1-T10：流水线、CLI 与收尾

**状态：DONE**　**依赖：S1-T01 至 S1-T09**

## 目标

组合前序模块为稳定的本地视频增量观察程序，并完成阶段验证。

## 修改

- 实现流水线：元数据 → 窗口 → 抽帧 → 提示词 → 推理 → 解析/校验 → 存储。
- 单窗口失败记录后继续，`KeyboardInterrupt` 必须传播。
- 完成 CLI：视频/配置/输出、时间与窗口范围、实时、dry-run、validate-only、print-config 等阶段要求参数。
- `--validate-only` 不调用模型，只检查输入并报告窗口/预估帧；`--print-config` 隐藏密钥。
- 更新 README；运行 `pytest -q` 和 `ruff check .` 并修复阶段问题。

## 不做

不实现后续的实体注册、动作跟踪、状态转换。

## 验收

- 本地 MP4 可执行 dry-run、validate-only 和正常路径。
- 单窗口失败不终止整次运行，也不伪造成功 Observation。
- `pytest -q` 通过，`ruff check .` 无严重错误。

## 完成记录

- 修改文件：
  - 新增 `src/qwen_stream_video/pipeline.py`：实现 `StreamingVideoPipeline`，支持 validate-only、dry-run、正常路径、实时等待、单窗口失败隔离、上一窗口摘要传递与 `KeyboardInterrupt` 传播。
  - 重写 `src/qwen_stream_video/cli.py`：添加 `--video`、`--output-dir`、`--start-time`、`--end-time`、`--start-window`、`--end-window`、`--max-windows`、`--realtime`、`--dry-run`、`--validate-only`、`--no-state`、`--print-config`、`--verbose` 参数。
  - 更新 `src/qwen_stream_video/config.py`：新增 `VideoContextConfig` 并通过 `video_metadata` 别名读取 YAML 中的视频上下文。
  - 更新 `README.md`：与新工程结构、CLI 参数、输出格式保持一致。
  - 新增 `tests/unit/test_pipeline.py`：覆盖 validate-only、dry-run、正常路径、窗口选择、单窗口失败隔离、KeyboardInterrupt 传播与状态传递开关。
  - 新增 `tests/unit/test_cli.py`：覆盖 print-config、缺失视频、validate-only、dry-run、缺失 API Key 等场景。
  - 更新 `tests/unit/test_config.py`：增加 `video_metadata` 解析测试，并保留未知键忽略测试。
- 验证结果：
  - `.venv/Scripts/python -m pytest tests/ -q`：97 个测试全部通过。
  - `.venv/Scripts/python -m ruff check .`：无错误。
  - `.venv/Scripts/python run.py --video videos/test_15s.mp4 --validate-only`：正确报告 4 个窗口及预估帧数。
  - `.venv/Scripts/python run.py --video videos/test_15s.mp4 --dry-run --max-windows 2`：成功生成输出目录并写入 2 个窗口，不调用模型。
  - `.venv/Scripts/python run.py --video videos/test_15s.mp4 --print-config`：正常输出并隐藏 API Key。
- 已知限制：
  - 正常路径（调用真实模型）需要有效的 `DASHSCOPE_API_KEY` 与可访问的 `base_url`；未在本地使用真实 API 验证，已通过单元测试使用 `FakeQwenClient` 覆盖正常数据流。
  - 已移除 CLI 自动加载 `.env` 的逻辑，避免测试环境被污染；用户需手动导出环境变量或直接在 `config.yaml` 中提供密钥。
