# S2-T10：TransitionEngine

**状态：TODO**　**依赖：S2-T03、S2-T04、S2-T08、S2-T09**

## 目标

根据规范化属性观察、实体解析、动作支持、Commit 区间和多窗口证据，确定性维护属性状态并区分初始化、pending、冲突和正式转移。

## 修改

- 新增 `src/qwen_stream_video/state/transition_engine.py`。
- 过滤不能进入正式状态的属性观察：
  - 实体未解析；
  - OOV 或实体类型非法；
  - 必需证据缺失/越界；
  - Context-only 新值；
  - confidence 低于中阈值；
  - camera change 且实体解析 ambiguous。
- 初始属性：
  - 高置信度直接 confirmed，并生成 `attribute_initialized`；
  - 中置信度进入 pending；
  - 低置信度只记录 observed；
  - 初始化不得伪造成 before→after 转移。
- 已知同值：更新支持和证据，不重复发事件。
- 已知不同值：满足以下之一才确认：
  - 高置信度 + 同实体支持动作 + Commit 证据；
  - 连续多窗口支持；
  - 属性词表明确允许 `single_high`。
- 实现 pending 支持、取消、冲突和过期规则。
- 仅使用解析到同一实体的 supporting action。
- 实体从不可见恢复时，相同值只恢复可见性；首次可见组件按初始化处理。
- 正式转移事件必须包含 before、after、reason、confidence 和 EvidenceReference。
- 新增 `tests/unit/test_transition_engine.py`。

## 不做

不使用任务规程或 LLM 判断状态变化，不做风险评分。

## 验收

- 初次观察不生成 attribute_transition。
- 同值观察不重复生成转移。
- 中置信度新值需要配置数量的连续窗口支持。
- 关键高置信度转移在配置要求下必须有同实体动作支持。
- 新可见指示灯不会生成“由不存在变为亮起”的伪转移。
- 冲突值进入 pending/conflicted，不直接覆盖。
- `pytest tests/unit/test_transition_engine.py -q` 和 `ruff check` 通过。

## 完成记录

- 修改文件：
- 验证结果：
- 已知限制：
