# S2-T08：EntityResolver

**状态：TODO**　**依赖：S2-T06、S2-T07**

## 目标

以可解释、确定性、一对一的规则将当前窗口局部实体解析到全局实体，并对模糊情况创建临时实体而不是强制错误合并。

## 修改

- 新增 `src/qwen_stream_video/state/entity_resolver.py`。
- 实现 `resolve(state, registry, scene_result, observation, sampled_frames)`。
- 候选只包含实体类型一致、近期出现、未 merged、场景可恢复且生命周期有效的实体。
- 实现硬约束：
  - 实体类型或明确设备类别冲突；
  - 高置信度稳定外观冲突；
  - 同窗两个可见局部实体占用同一全局实体；
  - 连续镜头中的不可能空间跳变；
  - candidate hint 类型错误或候选已被占用。
- 实现评分分解：
  - 类型/名称 0.30；
  - 外观 0.25；
  - 空间 0.15；
  - 关系 0.15；
  - 最近出现 0.10；
  - candidate hint 0.05。
- 阈值决策：confident、ambiguous/temporary、created；第一第二名差距过小时降级 ambiguous。
- 同窗采用确定性贪心一对一匹配；平分按 local ID 和 global ID 排序。
- candidate_global_id 只作为低权重提示，必须来自 ContextBuilder 候选且不能绕过硬约束。
- ambiguous 创建临时实体；满足连续支持窗口后允许延迟合并。
- 每个解析结果保存分数分解、第二名分数、拒绝理由和证据。
- 新增 `tests/unit/test_entity_resolver.py`，包含三人—特写—三人序列。

## 不做

不调用大模型、ReID 网络、检测器或向量数据库。

## 验收

- 相同输入重复解析结果一致。
- 同一窗口不出现两个局部实体映射同一个可见全局实体。
- candidate hint 不能覆盖类型和外观硬冲突。
- close score 不被强制匹配；会创建 temporary/ambiguous。
- 三人经过特写再返回全景时 ID 不整体平移。
- 4 号柜与 5 号柜不会仅因镜头移动被合并。
- `pytest tests/unit/test_entity_resolver.py -q` 和 `ruff check` 通过。

## 完成记录

- 修改文件：
- 验证结果：
- 已知限制：
