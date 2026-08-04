# S1-T03：视频元数据与窗口

**状态：DONE**　**依赖：S1-T01、S1-T02**

## 目标

实现本地视频校验、因果滑动窗口和正确的实时模式时间原点。

## 修改

- 定义并读取 `VideoMetadata`；文件、OpenCV、FPS、帧数或时长无效时抛出明确异常。
- 定义 `VideoWindow`、`build_video_windows()` 和选择逻辑，覆盖普通、短视频和尾部补齐。
- 窗口必须有序、非空、不越过视频结束；保留 `global_index`，选择后重算 `run_index`。
- 实现 `calculate_realtime_target()`；以首个选中窗口起点为原点。
- 添加窗口测试，包括 `calculate_realtime_target(1000, 480, 486) == 1006`。

## 不做

不抽帧，不支持 RTSP 或摄像头。

## 验收

- 窗口场景测试通过，非零 `start_time` 不产生额外等待。

## 完成记录

- 修改文件：
  - 新增 `src/qwen_stream_video/video/metadata.py`：`VideoMetadata` 模型与 `read_video_metadata()`，使用 OpenCV 读取并校验 FPS、帧数、时长与分辨率。
  - 新增 `src/qwen_stream_video/video/window.py`：`VideoWindow` 模型、`build_video_windows()`、`select_windows()`、`calculate_realtime_target()`。
  - 更新 `src/qwen_stream_video/video/__init__.py`：导出视频模块公共接口。
  - 新增 `tests/unit/test_windows.py`：覆盖元数据读取、常规窗口、短视频、尾部补齐、越界检查、选择后保留全局编号、实时目标等测试。
  - 顺手修复 `scripts/make_test_video.py`、`scripts/analyze_results.py`、`scripts/merge_windows.py` 中的 ruff 报错，使 `ruff check .` 全绿。
- 验证结果：
  - `.venv/Scripts/python -m pytest tests/ -q`：25 个测试全部通过。
  - `.venv/Scripts/python -m ruff check .`：无错误。
  - `calculate_realtime_target(1000.0, 480.0, 486.0)` 返回 `1006.0`。
