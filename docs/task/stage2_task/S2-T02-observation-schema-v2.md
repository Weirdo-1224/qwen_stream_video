# S2-T02：Observation Schema 2.0

**状态：TODO**　**依赖：S2-T01**

## 目标

定义只描述当前窗口视觉事实的 Observation 2.0 协议，补充场景可见性、关系、规范化元数据和 Commit 区间，同时禁止模型直接维护全局状态。

## 修改

- 重写或扩展 `src/qwen_stream_video/domain/observation.py`。
- `WindowObservation` 增加 `commit_start_seconds`，程序必须覆盖模型返回的窗口时间字段。
- `SceneObservation` 使用独立的 `scene_visibility`、`target_visibility` 和 `continuity_hint`。
- `EntityObservation` 保留局部 ID、实体类型、外观、空间区域、证据和可选候选 ID；候选 ID 不代表正式全局身份。
- `ActionObservation` 增加：
  - `raw_action_type`；
  - `action_family`；
  - `normalization_status`。
- `AttributeObservation` 使用 canonical `attribute_key`，并保留 `raw_attribute`、`raw_value` 和规范化状态。
- 新增 `RelationObservation`，校验 subject/object 局部实体引用。
- `ObservationBatch` 默认 `schema_version="2.0"`，所有模型使用严格字段和 `default_factory`。
- 将 `visual_fact` 与 `task_conditioned_interpretation` 分离；默认模式只允许视觉事实。
- 更新 `domain/__init__.py` 公共导出。
- 新增 `tests/unit/test_observation_v2.py`，覆盖有效结构、未知版本、Commit 区间、关系引用、可变默认值和候选 ID 可选性。

## 不做

不做动作词表归一化、实体解析、动作跟踪或状态转移。

## 验收

- Schema 2.0 可 JSON 序列化并生成 JSON Schema。
- `start_seconds <= commit_start_seconds < end_seconds`。
- 非法枚举、越界 confidence、未知字段和无效引用无法通过校验。
- Observation 不包含正式 `global_entity_id`、正式动作生命周期或完整 GlobalState。
- `pytest tests/unit/test_observation_v2.py -q` 和 `ruff check` 通过。

## 完成记录

- 修改文件：
- 验证结果：
- 已知限制：
