# S1-T05：增量观察 Schema

**状态：DONE（已按 stage1.md 第 8 节统一迁移）**　**依赖：S1-T01**

## 目标

定义只描述当前窗口的 Pydantic 观察协议。本次迁移将内部协议从“observations 外层列表”改为“一个 ObservationBatch 对应一个 VideoWindow”。

## 修改

- 新增 `src/qwen_stream_video/domain/enums.py`：定义 `EntityType`、`ViewType`、`VisibilityQuality`、`ActionPhaseObservation`。
- 重写 `src/qwen_stream_video/domain/observation.py`：
  - 定义 `WindowObservation`、`SceneObservation`、`EntityObservation`、`ActionObservation`、`AttributeObservation`、`UncertaintyObservation`、`ObservationBatch`。
  - `ObservationBatch` 直接包含 `schema_version`、`window`、`summary`、`scene`、`entities`、`actions`、`attribute_observations`、`uncertainties`；
  - 所有 list/dict 字段使用 `Field(default_factory=...)`；
  - `confidence` 限制为 `[0, 1]`；
  - `ObservationBatch` 使用 `extra="forbid"`，禁止旧 `observations` 列表兼容结构。
- 删除旧 `Viewpoint`、`Visibility`、`ActionPhase`、`Entity`、`Action`、`Attribute`、`Uncertainty` 协议。
- 更新 `src/qwen_stream_video/domain/__init__.py`：导出新的公共接口。
- 重写 `tests/unit/test_observation.py`：覆盖有效单窗口 batch、非法枚举、confidence 越界、可变默认值隔离、拒绝旧 `observations` 列表。

## 不做

不做跨窗口实体注册或全局状态维护；`candidate_global_id` 仅为候选。

## 验收

- 无效结构无法通过 Pydantic 校验。
- 旧 `observations` 外层列表无法通过校验。

## 完成记录

- 修改文件：
  - 新增 `src/qwen_stream_video/domain/enums.py`。
  - 重写 `src/qwen_stream_video/domain/observation.py`。
  - 更新 `src/qwen_stream_video/domain/__init__.py`。
  - 重写 `tests/unit/test_observation.py`。
- 验证结果：
  - `.venv/Scripts/python -m pytest tests/unit/test_observation.py -q`：7 个测试全部通过。
  - `.venv/Scripts/python -m pytest tests/ -q`：98 个测试全部通过。
  - `.venv/Scripts/python -m ruff check src/qwen_stream_video/domain/ tests/unit/test_observation.py`：无错误。
