# S1-T06：观察语义校验

**状态：TODO**　**依赖：S1-T04、S1-T05**

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
- 验证结果：
