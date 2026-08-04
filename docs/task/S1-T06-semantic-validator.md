# S1-T06：观察语义校验

**状态：DONE**　**依赖：S1-T04、S1-T05**

## 目标

校验 Schema 通过后的 ID、引用、证据帧和动作词表。

## 修改

- 实现 `ObservationSemanticValidator.validate()`。
- 校验实体/动作 `local_id` 唯一，动作和属性的实体引用存在。
- 校验证据帧索引范围，去重并排序。
- 读取 `vocabularies/actions.yaml`；非法动作映射为 `unknown`，保留描述并生成警告。
- 用实际 `VideoWindow` 覆盖模型返回的窗口时间和编号。
- 添加重复 ID、缺失引用、非法证据帧和有效引用测试。

## 不做

不重试模型，不写入文件。

## 验收

- 无效语义结果不能作为有效 Observation 保存。

## 完成记录

- 修改文件：
  - 新增 `vocabularies/actions.yaml`：动作词表，包含观察、检查、接近、离开、持有、拿起、放置、触摸、按压、旋转、切换、打开、关闭、插入、移除、连接、断开、调整、测量、记录、指向及 unknown 等动作。
  - 新增 `src/qwen_stream_video/inference/validator.py`：实现 `ObservationSemanticValidator` 类，加载动作词表，校验 ID 唯一性、动作实体引用完整性、证据帧索引范围，并对非法动作映射为 `unknown` 同时生成警告。
  - 更新 `src/qwen_stream_video/inference/__init__.py`：导出 `ObservationSemanticValidator`。
  - 新增 `tests/unit/test_validator.py`：覆盖有效观察、窗口字段覆盖、重复实体/动作 ID、缺失 actor/target 引用、越界证据帧、负证据帧、证据帧去重排序、未知动作映射及空 target 允许等场景。
- 验证结果：
  - `.venv/Scripts/python -m pytest tests/unit/test_validator.py -q`：11 个测试全部通过。
  - `.venv/Scripts/python -m pytest tests/ -q`：56 个测试全部通过。
  - `.venv/Scripts/python -m ruff check src/qwen_stream_video/inference/ tests/unit/test_validator.py`：无错误。
