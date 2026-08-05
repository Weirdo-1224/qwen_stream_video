# S2-T03：非破坏性词表归一化

**状态：TODO**　**依赖：S2-T02**

## 目标

实现动作和属性的非破坏性规范化，修复 `hand_over`、`receive`、`push`、`hover` 等有效模型输出被覆盖为 `unknown` 的问题。

## 修改

- 新增 `src/qwen_stream_video/inference/normalizer.py`，将 Schema 校验与词表规范化职责分离。
- 扩展 `vocabularies/actions.yaml`，至少覆盖：
  - `hand_over`、`receive`、`push`、`pull`、`hover`；
  - `pick_up`、`put_down`、`press`、`release`、`open`、`close`、`operate`、`measure`、`inspect`、`point`、`adjust`。
- 定义 canonical、alias、family 和 instant/continuous 等必要元数据。
- OOV 具体动作规范为 `action_type=other`，保留 `raw_action_type`，状态为 `out_of_vocabulary`。
- 视觉无法判断的动作保留 `action_type=unknown`，不得与 OOV 混淆。
- 扩展 `vocabularies/attributes.yaml`：
  - 使用如 `door.state`、`panel.cover.state`、`indicator.energy.lit`、`indicator.energy.color` 等 canonical key；
  - 定义值域、实体类型约束、别名、确认策略和支持动作映射。
- 将 `door_status`、`door_state` 等别名映射到同一 canonical key，同时保留原始字段。
- 将规范化警告返回调用方，后续可写入 `normalization_warnings.jsonl`。
- 新增 `tests/unit/test_normalizer.py`，覆盖动作保留、OOV、unknown、属性别名、实体类型约束和 raw 值保留。

## 不做

不更新 GlobalState，不判断实体身份，不确认属性转移。

## 验收

- `hand_over`、`receive`、`push`、`hover` 不再变成 `unknown`。
- OOV 与视觉 unknown 的语义严格区分。
- 非法实体类型属性被拒绝或标记为 `invalid_for_entity_type`，不能伪装成 canonical。
- 所有规范化结果保留原始动作、属性和值。
- `pytest tests/unit/test_normalizer.py -q` 和 `ruff check` 通过。

## 完成记录

- 修改文件：
- 验证结果：
- 已知限制：
