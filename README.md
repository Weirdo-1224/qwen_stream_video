# Qwen 本地视频滑动窗口流式分析

一个最小、可直接运行的原型：将本地视频按时间切成因果滑动窗口，每个窗口均匀抽帧后调用千问视觉 API，输出逐窗口结构化 JSONL。

## 核心保证

分析窗口 `[start, end)` 时，程序只采样时间严格小于 `end` 的帧，因此模型无法看到未来窗口。默认快速离线运行；添加 `--realtime` 后，会等到视频逻辑时间到达窗口终点再发起请求。

## 项目结构

```text
qwen_stream_video/
├── run.py
├── config.yaml
├── requirements.txt
├── .env.example
├── prompts/
│   ├── system_prompt.txt
│   └── user_prompt.txt
├── videos/
└── outputs/
```

## 1. 安装

建议 Python 3.10+。

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## 2. 配置 API

复制环境变量文件：

```bash
# Linux/macOS
cp .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

编辑 `.env`：

```dotenv
DASHSCOPE_API_KEY=sk-xxxx
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3.7-plus
```

阿里云百炼目前推荐使用业务空间专属兼容地址。可直接在控制台复制对应地域的 `base_url`，覆盖 `.env` 中的 `DASHSCOPE_BASE_URL`。

## 3. 先做无 API 测试

把视频放入 `videos/`，先检查滑动窗口和抽帧是否正常：

```bash
python run.py --video videos/demo.mp4 --dry-run --max-windows 3
```

## 4. 调用千问 API

```bash
python run.py --video videos/demo.mp4 --max-windows 5
```

确认结果正常后，去掉 `--max-windows` 处理完整视频：

```bash
python run.py --video videos/demo.mp4
```

按真实视频时间等待：

```bash
python run.py --video videos/demo.mp4 --realtime
```

关闭上一窗口状态传递，做无状态对照实验：

```bash
python run.py --video videos/demo.mp4 --no-state
```

## 5. 默认参数

```yaml
window_seconds: 6.0
stride_seconds: 3.0
sample_fps: 1.0
min_frames: 4
max_frames: 12
```

窗口顺序示例：

```text
0-6 秒
3-9 秒
6-12 秒
9-15 秒
```

每次请求只传当前窗口的 Base64 JPEG 图像列表和上一窗口的压缩状态，不传完整历史视频。

## 6. 输出

每次运行会生成：

```text
outputs/<视频名_时间>/
├── run_meta.json
└── windows.jsonl
```

`windows.jsonl` 每行对应一个滑动窗口：

```json
{
  "window_index": 1,
  "window_start_seconds": 3.0,
  "window_end_seconds": 9.0,
  "sampled_timestamps_seconds": [3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
  "status": "ok",
  "analysis": {
    "window_summary": "工作人员在开关柜前操作门锁。",
    "entities": [],
    "actions": [],
    "state_changes": [],
    "observed_results": [],
    "uncertainties": []
  },
  "api": {
    "latency_seconds": 2.134,
    "usage": {}
  }
}
```

## 7. 修改分析任务

- 修改窗口参数：`config.yaml`
- 修改流式分析规则：`prompts/system_prompt.txt`
- 修改每窗口动态信息：`prompts/user_prompt.txt`
- 修改视频名称、类别和背景：`config.yaml` 的 `video_metadata`

## 8. 说明

- 千问 OpenAI 兼容接口支持以 `type: video` 传入按顺序排列的图像列表，并通过 `fps` 描述帧间时间关系。
- 本项目将本地帧编码为 `data:image/jpeg;base64,...` 后发送。
- 请求使用 `response_format={"type":"json_object"}`，并在本地再次进行 JSON 解析与基础字段检查。
- API 失败或 JSON 无法解析时，该窗口会记录 `status: error`，不会中断整个视频。
- 建议先使用 `--max-windows 3` 或 `5` 检查提示词与费用，再处理完整视频。
