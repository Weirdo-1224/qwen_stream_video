# TASK-002：第二阶段全局状态引擎

## 文档约定

- 命名格式：`TASK-<三位序号>-<阶段>-<主题>.md`。
- 小任务命名格式：`S<阶段>-T<两位序号>-<英文主题>.md`。
- 状态只能使用：`TODO`、`DOING`、`DONE`、`BLOCKED`。
- 后续任务必须引用前置任务编号，并明确不在本任务范围内的内容。

## 基本信息

| 项目 | 内容 |
| --- | --- |
| 状态 | DONE |
| 前置任务 | TASK-001 |
| 目标 | 将窗口级局部 Observation 转换为身份稳定、动作连续、状态可确认、结果可重放的确定性 GlobalState。 |
| 参考 | `docs/stage/stage2.md` |

## 范围

### 要实现

1. 升级 Observation Schema 2.0，分离局部视觉事实与程序侧全局状态。
2. 修复动作和属性的破坏性归一化，保留原始模型值和规范化状态。
3. 为重叠窗口增加 Context Interval、Commit Interval 和 warmup 机制。
4. 建立 GlobalState、StateEvent、StateDelta、EntityResolution 等领域模型。
5. 实现 SceneTracker、EntityRegistry、EntityResolver、ActionTracker、TransitionEngine 和 StateReducer。
6. 使用 ContextBuilder 替换上一窗口摘要，限制候选 ID 和上下文长度。
7. 将 State Engine 接入现有顺序 Pipeline，并保留 Observation-only 模式。
8. 保存实体解析、状态事件、状态增量、周期快照、最终状态及状态错误。
9. 支持 Schema 1.0/2.0 Observation Replay，且不调用模型 API。
10. 建立单元测试、集成测试、Golden Regression 和状态质量分析脚本。
11. 保持第一阶段本地视频、dry-run、validate-only 和原有 CLI 使用方式可用。

### 不实现

- RTSP 或摄像头输入；
- 多窗口并行推理、异步队列和 latest-window-only 调度；
- 目标检测器、分割器或神经 ReID；
- 作业规程推理、违规判断、报警或多 Agent；
- 向量数据库、复杂图数据库或 Web 前端；
- 模型训练、微调或生产级实时性能优化。

## 实施清单

- [x] S2-T01：配置严格化与第一阶段明确问题修复。
- [x] S2-T02：Observation Schema 2.0。
- [x] S2-T03：动作和属性非破坏性归一化。
- [x] S2-T04：Context / Commit 双区间与 warmup。
- [x] S2-T05：GlobalState、StateEvent、StateDelta 和解析领域模型。
- [x] S2-T06：SceneTracker。
- [x] S2-T07：EntityRegistry。
- [x] S2-T08：EntityResolver。
- [x] S2-T09：ActionTracker。
- [x] S2-T10：TransitionEngine。
- [x] S2-T11：StateReducer。
- [x] S2-T12：ContextBuilder 与 Observation Prompt。
- [x] S2-T13：Pipeline 和 CLI 集成。
- [x] S2-T14：状态输出存储。
- [x] S2-T15：Observation Replay。
- [x] S2-T16：集成测试与 Golden Regression。
- [x] S2-T17：状态质量分析脚本。
- [x] S2-T18：README、全量回归和最终验收。

## 关键约束

- 千问只输出当前窗口局部 Observation；正式实体 ID、动作生命周期和属性状态只能由确定性代码生成。
- `candidate_global_id` 只作为低权重提示，不能绕过硬约束或匹配阈值。
- 当前窗口未观察到实体不等于实体消失；首次可见属性不等于发生状态转移。
- 状态更新必须可追踪到 run、window、local ID、证据帧和时间。
- StateReducer 是唯一允许提交 GlobalState 修改的协调器；单窗口更新必须具备原子性。
- 相同 Observation、配置和窗口序列重复 Replay，语义输出必须一致。
- 第一阶段输出文件继续保留，`observations.jsonl` 仍表示模型局部观察。
- 不得在本阶段引入并发 StateReducer 或真实流输入。

## 验收标准

- [x] `pytest -q` 全部通过，测试不调用真实 API。
- [x] `ruff check .` 无错误。
- [x] 第一阶段命令仍可运行。
- [x] 默认 Observation Schema 为 2.0，并可适配 Replay Schema 1.0。
- [x] 重叠区间不会重复创建动作或属性转移。
- [x] 三人经过特写再返回全景时，全局 ID 不整体漂移。
- [x] 4 号柜和 5 号柜不会被错误合并为同一实体的属性变化。
- [x] 持续动作只有一个 GlobalAction ID，并具有完整生命周期。
- [x] 初始属性、新可见属性和正式状态转移被明确区分。
- [x] 状态事件、增量、快照和 final_state 可追踪、可解析、可恢复。
- [x] 相同输入重复 Replay 的 `state_events.jsonl` 与 `final_state.json` 哈希一致。
- [x] Golden Regression 精确比较实体映射、事件、动作 ID、状态转移和最终关键状态。

## 完成记录

- 修改文件：配置、Observation/domain、normalizer、state、pipeline/CLI、storage、Replay、质量脚本、Golden 测试和 README。
- 验证命令与结果：`pytest -q` 119 passed；`ruff check .` 通过。
- Golden Regression 结果：脱敏 Golden Fixture 与集成测试通过。
- 状态质量分析结果：Replay 产物结构检查通过，无 structural errors。
- 已知限制：仍为本地 MP4 顺序处理，不支持 RTSP、并行推理、ReID、违规判断或多 Agent。
