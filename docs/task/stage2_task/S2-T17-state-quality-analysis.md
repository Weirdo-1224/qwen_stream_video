# S2-T17：状态质量分析脚本

**状态：DONE**　**依赖：S2-T14、S2-T16**

## 目标

提供对一次状态运行产物的确定性质量统计和一致性检查，快速暴露 OOV、模糊匹配、重复动作、无支持转移和引用错误。

## 修改

- 新增 `scripts/evaluate_state_run.py`。
- 支持：
  - `python scripts/evaluate_state_run.py outputs/<run_id>`。
- 输出运行统计：
  - 窗口和成功 Observation 数量；
  - 状态更新成功数量；
  - 动作/属性 OOV 数量和比例；
  - 实体 created/matched/ambiguous/temporary/merged；
  - 全局实体总数；
  - 动作 started/ended/uncertain；
  - 属性 initialized/pending/transition/conflict；
  - 无证据事件；
  - camera change 附近 ID 变化候选；
  - 平均上下文字符数；
  - 最终实体和活跃动作数量。
- 定义启发式指标：
  - Action OOV Rate；
  - Attribute OOV Rate；
  - Entity Ambiguous Rate；
  - Entity ID Switch Candidate Count；
  - Duplicate Action Candidate Count；
  - Unsupported Transition Count；
  - Evidence Coverage。
- 检查引用和结构一致性：
  - 事件实体/动作引用存在；
  - ID 唯一；
  - `last_committed_window` 单调；
  - snapshot 可解析；
  - final_state 与最后 snapshot 一致；
  - evidence sample index 可映射时间。
- 错误时返回非零退出码；报告可输出文本，建议同时支持 JSON。
- 新增脚本单元测试或集成测试。

## 不做

不声称这些启发式指标等同于人工标注准确率，不做模型训练或在线监控。

## 验收

- 对合法 Golden 运行报告无结构错误。
- 人工注入重复 ID、悬空引用、无证据转移或 final/snapshot 不一致时能检测并非零退出。
- 指标计算确定、可复现。
- 脚本测试和 `ruff check` 通过。

## 完成记录

- 修改文件：
- 验证结果：
- 示例报告：
- 已知限制：
