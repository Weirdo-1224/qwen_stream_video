# S1-T04：抽帧与图像编码

**状态：DONE**　**依赖：S1-T02、S1-T03**

## 目标

从一个窗口抽取真实有效帧，生成 JPEG Data URL。

## 修改

- 定义 `SampledFrame` 和 `sample_window_frames()`。
- 保证 `start <= timestamp < end`；帧数由窗口时长和 `sample_fps` 计算并受 `min_frames`/`max_frames` 限制。
- 不伪造重复图像；真实帧不足最低要求时抛出明确异常。
- 实现 `encode_frame_to_data_url()`：等比缩小、不放大、JPEG/Base64/Data URL。
- 添加采样边界、帧数与编码测试。

## 不做

不保存帧，也不构建提示词。

## 验收

- 不采样右边界或未来帧；小图不被放大。

## 完成记录

- 修改文件：
  - 新增 `src/qwen_stream_video/video/sampling.py`：`SampledFrame` 模型、`sample_window_frames()` 与 `encode_frame_to_data_url()`。
  - 更新 `src/qwen_stream_video/video/__init__.py`：导出新增的公共接口。
  - 新增 `tests/unit/test_sampling.py`：覆盖采样数量、边界、max_frames 截断、真实帧不足、帧唯一性、序列化隐藏图像、小图不放大、大图等比缩小、Data URL 格式及参数校验。
- 验证结果：
  - `.venv/Scripts/python -m pytest tests/unit/test_sampling.py -q`：10 个测试全部通过。
