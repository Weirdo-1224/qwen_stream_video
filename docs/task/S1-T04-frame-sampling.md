# S1-T04：抽帧与图像编码

**状态：TODO**　**依赖：S1-T02、S1-T03**

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
- 验证结果：
