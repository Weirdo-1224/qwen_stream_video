# S2-T07：EntityRegistry

**状态：DONE**　**依赖：S2-T05、S2-T06**

## 目标

实现全局实体的确定性 ID 分配、生命周期、外观签名、空间历史、临时实体和受控合并。

## 修改

- 新增 `src/qwen_stream_video/state/entity_registry.py`。
- 实现 `create_entity()`、`get()`、`find_candidates()`、`update_from_observation()`、`mark_not_observed()`、`merge_temporary_entity()`。
- 正式 ID 格式为 `person_0001`、`device_0001` 等；临时 ID 使用 `temp_<type>_0001`。
- ID 由 GlobalState 计数器分配，一旦使用不得复用。
- 实体生命周期：
  - 观察到：`active`；
  - 短期未见：`temporarily_missing`；
  - 超阈值：`inactive`；
  - 临时实体合并：`merged`。
- 镜头切换、特写或严重遮挡期间不累加永久失活计数。
- 外观签名保守更新：高置信度加入，稳定属性冲突不立即覆盖，缺失属性不视为冲突。
- 临时实体合并必须保留历史、设置 `merged_into` 并生成结果供事件层使用。
- 新增 `tests/unit/test_entity_registry.py`。

## 不做

不计算局部实体与候选实体的匹配分数，不直接读取 candidate_global_id 决定身份。

## 验收

- ID 单调、唯一、不复用。
- 未观察到不等于删除；inactive 历史仍保留。
- 镜头切换不导致实体过早失活。
- 临时实体合并后证据和历史可追踪。
- `pytest tests/unit/test_entity_registry.py -q` 和 `ruff check` 通过。

## 完成记录

- 修改文件：
- 验证结果：
- 已知限制：
