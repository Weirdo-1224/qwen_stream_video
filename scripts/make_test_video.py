"""生成一个用于测试的 15 秒 720p 视频。"""
import cv2
import numpy as np
from pathlib import Path

output_path = Path("videos/test_15s.mp4")
width, height = 1280, 720
fps = 24
duration = 15
total_frames = fps * duration

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

for i in range(total_frames):
    t = i / fps
    # 背景色随时间缓慢变化
    r = int(128 + 127 * (t / duration))
    g = int(128 + 127 * abs(np.sin(t * 0.7)))
    b = int(128 + 127 * abs(np.cos(t * 0.5)))
    frame = np.full((height, width, 3), (b, g, r), dtype=np.uint8)

    # 每隔 3 秒画一个移动方块，模拟视觉事件
    if int(t) % 3 == 0:
        x = int((t % 3) / 3 * width)
        y = height // 2 - 50
        cv2.rectangle(frame, (x, y), (x + 100, y + 100), (0, 0, 255), -1)

    # 时间戳文字
    cv2.putText(
        frame,
        f"Time: {t:.2f}s",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    writer.write(frame)

writer.release()
print(f"已生成测试视频：{output_path}，时长 {duration}s，分辨率 {width}x{height}，{fps}fps")
