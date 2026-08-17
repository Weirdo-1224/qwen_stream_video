# S2-T06：SceneTracker

**状态：DONE**　**依赖：S2-T05**

## 目标

使用已验证的 SceneObservation 确定性维护场景连续性，为实体解析和动作跟踪提供镜头切换、重构图和可见性上下文。

## 修改

- 新增 `src/qwen_stream_video/state/scene_tracker.py`。
- 实现 `SceneTracker.update(state, observation) -> SceneUpdateResult`。
- 第一个窗口创建 `scene_0001`。
- `camera_change=true` 创建新 SceneState；`continuity_hint=reframed` 保持 scene ID 并生成重构图事件。
- closeup/detail 时将前景外实体更新为 `not_visible` 或 `partial`，不得删除实体。
- 返回 wide/medium 时允许历史实体重新激活。
- 镜头切换附近降低空间连续性权重所需的场景标记。
- 镜头切换时未观察到的动作不得立即结束，仅提供 uncertain 上下文。
- 至少生成 `scene_started`、`scene_changed`、`scene_reframed`、`scene_visibility_changed`。
- 新增 `tests/unit/test_scene_tracker.py`。

## 不做

不做实体匹配、实体创建、动作创建或属性更新。

## 验收

- camera change 创建新 scene；reframe 不创建新 scene。
- 镜头切换和特写不会删除历史实体。
- closeup 后返回全景可恢复历史场景关联。
- 场景事件拥有窗口编号、场景 ID 和证据/理由。
- `pytest tests/unit/test_scene_tracker.py -q` 和 `ruff check` 通过。

## 完成记录

- 修改文件：
- 验证结果：
- 已知限制：
