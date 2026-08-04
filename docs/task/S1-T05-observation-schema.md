# S1-T05：增量观察 Schema

**状态：DONE**　**依赖：S1-T01**

## 目标

定义只描述当前窗口的 Pydantic 观察协议。

## 修改

- 定义实体类型、视角、可见性、动作阶段枚举。
- 定义 `WindowObservation`、`SceneObservation`、实体/动作/属性/不确定性模型与 `ObservationBatch`。
- 置信度限制为 `[0, 1]`；所有 list/dict 使用 `default_factory`。
- 动作类型保留字符串，以便后续词表校验。
- 添加有效样例、置信度、枚举和可变默认值测试。

## 不做

不做跨窗口实体注册或全局状态维护；`candidate_global_id` 仅为候选。

## 验收

- 无效结构无法通过 Pydantic 校验。

## 完成记录

- 修改文件：
  - 新增 `src/qwen_stream_video/domain/observation.py`：定义 `EntityType`、`Viewpoint`、`Visibility`、`ActionPhase` 枚举与 `Attribute`、`Uncertainty`、`Entity`、`Action`、`SceneObservation`、`WindowObservation`、`ObservationBatch` Pydantic 模型。
  - 更新 `src/qwen_stream_video/domain/__init__.py`：导出观察协议公共接口。
  - 新增 `tests/unit/test_observation.py`：覆盖有效样例、置信度边界、窗口/动作时间顺序、非法枚举、边界框维度、可变默认值与批量观察。
- 验证结果：
  - `.venv/Scripts/python -m pytest tests/unit/test_observation.py -q`：10 个测试全部通过。
  - `.venv/Scripts/python -m pytest tests/ -q`：45 个测试全部通过。
  - `.venv/Scripts/python -m ruff check src/qwen_stream_video/domain/ tests/unit/test_observation.py`：无错误。
