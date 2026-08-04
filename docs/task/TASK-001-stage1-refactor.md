# TASK-001：第一阶段工程重构

## 文档约定

- 命名格式：`TASK-<三位序号>-<阶段>-<主题>.md`。
- 示例：`TASK-002-stage2-entity-registry.md`。
- 状态只能使用：`TODO`、`DOING`、`DONE`、`BLOCKED`。
- 后续任务应引用前置任务编号，并明确不在本任务范围内的内容。

## 基本信息

| 项目 | 内容 |
| --- | --- |
| 状态 | TODO |
| 前置任务 | 无 |
| 目标 | 将单文件视频分析原型重构为可安装、可测试的 Python 工程，并将模型输出改为当前窗口的增量观察。 |
| 参考 | `docs/stage/stage1.md` |

## 范围

### 要实现

1. 建立 `pyproject.toml` 和 `src/qwen_stream_video/` 布局；根目录 `run.py` 仅保留兼容入口。
2. 实现 Pydantic 配置模型、YAML/环境变量/命令行覆盖及启动校验。
3. 拆分视频元数据、滑动窗口、抽帧和图像 Data URL 编码模块。
4. 定义增量 `ObservationBatch` 数据模型及枚举，并实现 JSON、Schema 和语义校验。
5. 拆分 Qwen 调用、提示词构建、响应解析和异常分类；仅重试网络、429 与 5xx 错误。
6. 实现运行存储：配置快照、元数据、窗口、观察、API 指标、错误和原始响应均可追踪。
7. 实现清晰的 `StreamingVideoPipeline` 和命令行参数。
8. 增加 pytest 单元测试、README、`.env.example` 和 `.gitignore` 必要内容。

### 不实现

- EntityRegistry、ActionTracker、TransitionEngine；
- RTSP/摄像头输入、向量数据库、多 Agent、违规判断、自适应抽帧、Web 前端；
- 正式稳定的全局实体 ID（`candidate_global_id` 仅为候选）。

## 实施清单

- [ ] 工程骨架：`pyproject.toml`、`src` 包、CLI、兼容 `run.py`；`python run.py --help` 与 `qwen-stream-video --help` 可用。
- [ ] 配置系统：严格类型和边界校验；优先级为 CLI > 环境变量 > YAML > 默认值；日志不得泄露 API Key。
- [x] 视频模块：读取元数据，生成不越界的因果滑动窗口；保留 `global_index`，选择后重新计算 `run_index`。
- [x] 实时模式：以首个已选窗口起点为时间原点，`calculate_realtime_target(1000, 480, 486) == 1006`。
- [ ] 抽帧与编码：采样时间满足 `start <= timestamp < end`；帧数受 `min_frames`/`max_frames` 限制；不伪造重复图像。
- [ ] 观察协议：实现 `ObservationBatch` 及其子模型；所有可变字段使用 `default_factory`。
- [ ] 推理与校验：原始文本去代码块、提取 JSON、Pydantic 校验、语义校验；禁止 `eval()`；无效结果不得写入观察文件。
- [ ] 存储与流水线：每窗口独立失败并继续；保存 `outputs/<run_id>/` 中的运行信息、原始响应、指标与错误。
- [ ] CLI：支持 `--video`、`--config`、`--output-dir`、时间/窗口范围、`--realtime`、`--dry-run`、`--validate-only`、`--print-config` 等阶段文档要求的参数。
- [ ] 测试与文档：Mock/Fake Qwen 客户端；补齐配置、窗口、Schema、语义校验单元测试及 README。

## 关键约束

- 公共类和函数必须有类型标注；Python >= 3.10。
- 业务逻辑不得继续堆积在 `run.py`，也不得用空实现、硬编码结果或吞异常通过测试。
- 单窗口失败应写入 `errors.jsonl` 并继续；`KeyboardInterrupt` 必须向上传播。
- 每次运行使用唯一 `run_id`，不得覆盖历史输出；不得提交密钥、私有视频或大体积运行产物。

## 验收标准

- [ ] 可编辑安装成功：`pip install -e .`。
- [ ] `pytest -q` 全部通过，且测试不调用真实 API。
- [ ] `ruff check .` 无严重错误。
- [ ] 能处理本地 MP4，窗口与抽帧不使用右边界或未来帧。
- [ ] 模型结果为当前窗口增量 Observation，且经过 Schema 和语义校验。
- [ ] 非零起始时间的实时等待、全局窗口编号、失败窗口隔离均符合要求。
- [ ] 原始响应、运行元数据、最终配置、指标和错误可追踪，且无效 Observation 不写入 `observations.jsonl`。

## 完成记录

完成时填写：修改文件列表、验证命令与结果、已知限制，以及下一阶段输入。
