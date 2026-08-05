# S2-T01：配置严格化与已知问题修复

**状态：TODO**　**依赖：S1-T01 至 S1-T10**

## 目标

扩展第二阶段配置，修复第一阶段运行产物暴露出的配置失效、上下文污染和实验范围不可追踪问题，为后续 State Engine 提供严格配置基础。

## 修改

- 在 `src/qwen_stream_video/config.py` 新增 `StateConfig`、`SceneTrackerConfig`、`EntityRegistryConfig`、`ActionTrackerConfig`、`TransitionEngineConfig`、`ContextConfig`。
- 扩展现有配置：
  - `video.warmup_windows`；
  - `model.structured_json`；
  - `observation.schema_version="2.0"`、`context_policy`、`allow_candidate_global_ids`；
  - 状态开关、快照间隔、匹配阈值、动作缺失阈值、属性确认阈值和上下文长度限制；
  - 状态产物保存开关。
- 所有配置模型使用 `ConfigDict(extra="forbid")`，拼错字段必须在启动前失败。
- 校验匹配阈值、属性阈值、缺失窗口数、快照间隔、warmup 数量和上下文长度等边界关系。
- 默认 `context_policy=visual_only`；修正配置中的具体任务名称污染，视觉模式不得把具体故障或回路名称传给模型。
- 记录请求窗口、warmup 范围、正式提交范围、覆盖时间和 cold start。
- 补充 `configs/experiments/observation_only.yaml` 与 `state_tracking.yaml`。
- 新增或更新配置单元测试，验证每个新增配置确实被业务代码读取所需的接口可用。

## 不做

不实现 Observation 2.0 模型、状态算法、状态存储或 Replay。

## 验收

- 非法配置和未知字段在启动前明确失败。
- 默认配置为 `visual_only`，不会向模型发送具体视频文件名或故障名称。
- CLI、环境变量、YAML 和默认值优先级保持 `CLI > 环境变量 > YAML > 默认值`。
- 配置摘要不泄露 API Key，并显示 State、Schema、Context Policy、warmup 和快照配置。
- 第一阶段配置在明确兼容策略下仍可加载，或给出可操作的迁移错误。
- 对应配置测试通过，`ruff check` 通过。

## 完成记录

- 修改文件：
- 验证结果：
- 已知限制：
