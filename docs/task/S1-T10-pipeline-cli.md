# S1-T10：流水线、CLI 与收尾

**状态：TODO**　**依赖：S1-T01 至 S1-T09**

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
- 验证结果：
- 已知限制：
