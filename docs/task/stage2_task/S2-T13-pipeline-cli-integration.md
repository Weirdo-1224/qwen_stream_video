# S2-T13：Pipeline 与 CLI 集成

**状态：DONE**　**依赖：S2-T01 至 S2-T12**

## 目标

将 Observation 生成与 State Engine 顺序集成，保留 Observation-only 模式、失败隔离、warmup 和第一阶段命令兼容性。

## 修改

- 重构 `src/qwen_stream_video/pipeline.py`，明确两条链路：
  - Observation Generator：视频帧→局部 Observation；
  - State Reducer：局部 Observation→GlobalState。
- Pipeline 不包含实体匹配、动作生命周期或属性转移具体算法。
- 每个窗口执行：
  1. 构建 Context；
  2. 采样并调用模型；
  3. 解析、校验和规范化 Observation；
  4. 保存 Observation；
  5. 状态启用时调用 StateReducer；
  6. 保存状态结果。
- warmup 窗口参与上下文和状态建立，但不进入正式评测事件范围。
- 支持 Observation 失败、状态更新失败和 `KeyboardInterrupt` 的既有语义。
- 更新 `src/qwen_stream_video/cli.py`，新增：
  - `--state` / `--no-state`；
  - `--warmup-windows`；
  - `--snapshot-interval`；
  - `--context-policy`。
- 启动摘要显示 Observation/State Schema、状态开关、warmup、匹配阈值、确认阈值、快照间隔和模式。
- `--state` 与 `--no-state` 行为明确；第一阶段命令继续可用。
- 新增/更新 Pipeline 与 CLI 单元测试，全部使用 Fake 客户端。

## 不做

不实现 Replay（由 S2-T15 完成），不引入线程池、异步队列或并行 StateReducer。

## 验收

- Observation-only 模式只生成局部观察，不创建状态文件或按配置生成空状态。
- State 模式按窗口顺序更新 GlobalState。
- 单窗口 Observation 或 State 失败不伪造成功结果。
- `KeyboardInterrupt` 继续向上传播。
- 第一阶段 dry-run、validate-only、窗口选择和本地 MP4 命令兼容。
- Pipeline/CLI 测试和 `ruff check` 通过。

## 完成记录

- 修改文件：
- 验证结果：
- 已知限制：
