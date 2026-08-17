# S2-T18：README 与最终验收

**状态：DONE**　**依赖：S2-T01 至 S2-T17**

## 目标

补齐第二阶段用户文档，执行完整回归、Golden Regression、Replay 确定性和状态质量检查，形成可交付的 Stage 2 实现。

## 修改

- 更新根目录 `README.md`，新增：
  - 第二阶段定位；
  - Observation 与 GlobalState 职责边界；
  - State Engine 架构；
  - Schema 2.0；
  - Context / Commit Interval 和 warmup；
  - EntityRegistry/Resolver；
  - ActionTracker 生命周期；
  - TransitionEngine 确认规则；
  - 状态输出目录；
  - Replay 使用方法；
  - 配置说明；
  - Golden 测试和质量分析命令；
  - 已知限制。
- 明确：
  - GlobalEntity ID 仅为单次运行内确定性 ID，不是跨视频永久真实身份；
  - 当前仍是本地视频模拟流式处理，不保证真实时间性能；
  - 尚未支持 RTSP、并行推理和违规判断。
- 更新 `.env.example`、`.gitignore` 和示例配置，禁止密钥、私有视频、Base64 和运行产物提交。
- 执行完整验收：
  - 可编辑安装；
  - 第一阶段命令兼容；
  - Observation-only；
  - State 模式；
  - Replay 1.0/2.0；
  - Golden Regression；
  - 质量分析；
  - 全量 pytest 和 ruff。
- 生成第二阶段交付说明，列出目录树、Schema/State/Resolution/Action/Transition 示例、运行命令、测试结果和已知限制。
- 更新本文件及 S2-T01 至 S2-T17 的完成记录和总览清单。

## 不做

不在最终验收阶段临时引入第三阶段架构改造或未经任务拆分的大规模重构。

## 验收

- `pip install -e .` 成功。
- `pytest -q` 全部通过。
- `ruff check .` 无错误。
- 第一阶段命令、Observation-only、State 和 Replay 均可执行。
- Golden Regression 精确通过。
- 两次 Replay 的事件和最终状态哈希一致。
- 状态质量脚本对 Golden 产物无结构错误。
- 文档中的命令、配置键和输出文件与代码一致。
- 无密钥、私有视频、Base64 或大体积真实响应进入仓库。

## 完成记录

- 修改文件：
- 验证命令与结果：
- Golden Regression：
- Replay 哈希：
- 质量分析：
- 已知限制：
