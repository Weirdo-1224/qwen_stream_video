# S1-T06：观察语义校验

**状态：DONE（已按单 ObservationBatch 协议更新）**　**依赖：S1-T04、S1-T05**

## 目标

校验 Schema 通过后的 ID、引用、证据帧和动作词表，并直接处理单个 `ObservationBatch`。

## 修改

- 更新 `src/qwen_stream_video/inference/validator.py`：
  - `validate()` 直接处理单个 `ObservationBatch`；
  - 校验实体和动作 `local_id` 唯一，并校验属性 `entity_local_id` 无重复；
  - 校验 `actor_local_id`、`target_local_id`、`tool_local_id`、`entity_local_id` 引用存在；
  - 校验所有 `evidence_frames` 位于 `sampled_frames` 有效索引范围内，并去重排序；
  - 从 `vocabularies/actions.yaml` 校验 `action_type`；非法值映射为 `unknown` 并返回警告；
  - 使用当前 `VideoWindow` 覆盖 `batch.window` 的 `global_index`、`start_seconds`、`end_seconds`。
- 重写 `tests/unit/test_validator.py`：覆盖有效 batch、窗口字段覆盖、重复实体/动作 ID、缺失 actor/target/tool/属性实体引用、越界/负证据帧、证据帧去重排序、未知动作映射、空 target 允许等场景。

## 不做

不重试模型，不写入文件；不实现全局实体注册或跨窗口跟踪。

## 验收

- 无效语义结果不能作为有效 Observation 保存。

## 完成记录

- 修改文件：
  - 更新 `src/qwen_stream_video/inference/validator.py`。
  - 重写 `tests/unit/test_validator.py`。
- 验证结果：
  - `.venv/Scripts/python -m pytest tests/unit/test_validator.py -q`：12 个测试全部通过。
  - `.venv/Scripts/python -m pytest tests/ -q`：98 个测试全部通过。
  - `.venv/Scripts/python -m ruff check src/qwen_stream_video/inference/ tests/unit/test_validator.py`：无错误。
