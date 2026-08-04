# Qwen 本地视频滑动窗口流式分析

将本地视频按时间切成因果滑动窗口，每个窗口均匀抽帧后调用千问视觉 API，输出逐窗口结构化 JSONL 观察结果。

## 核心保证

- 分析窗口 `[start, end)` 时，程序只采样时间严格小于 `end` 的帧，模型无法看到未来窗口。
- 默认快速离线运行；添加 `--realtime` 后，会等到视频逻辑时间到达窗口终点再发起请求。
- 单窗口失败记录到 `errors.jsonl` 后继续处理，不伪造成功观察结果。
- 每次运行生成唯一输出目录，不覆盖历史结果。

## 项目结构

```text
qwen_stream_video/
├── run.py                    # 兼容入口，调用 qwen_stream_video.cli.main
├── config.yaml               # 默认配置
├── pyproject.toml
├── requirements.txt
├── .env.example              # 环境变量示例
├── prompts/                  # 保留的提示词模板文件
├── videos/                   # 输入视频目录
├── outputs/                  # 输出目录
├── src/qwen_stream_video/
│   ├── cli.py                # 命令行入口
│   ├── pipeline.py           # StreamingVideoPipeline
│   ├── config.py             # Pydantic 配置模型
│   ├── video/                # 元数据、窗口、抽帧、编码
│   ├── inference/            # Qwen 客户端、提示词、解析、语义校验
│   ├── domain/               # Observation Schema
│   └── storage/              # RunStorage 运行产物存储
└── tests/
```

## 安装

Python 3.10+。

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -e .
pip install -e ".[dev]"  # 开发依赖
```

## 配置 API

设置环境变量（推荐）：

```bash
# Linux/macOS
export DASHSCOPE_API_KEY=sk-xxxx
export DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export QWEN_MODEL=qwen3-vl-plus

# Windows PowerShell
$env:DASHSCOPE_API_KEY="sk-xxxx"
$env:DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:QWEN_MODEL="qwen3-vl-plus"
```

或在 `config.yaml` 中直接提供 `model.api_key`（请勿提交到版本控制）。

## 使用

### 1. 查看配置

```bash
python run.py --video videos/demo.mp4 --print-config
```

### 2. 检查窗口与预估帧（不调用模型）

```bash
python run.py --video videos/demo.mp4 --validate-only
```

### 3. 无模型抽帧测试

```bash
python run.py --video videos/demo.mp4 --dry-run --max-windows 3
```

### 4. 处理完整视频

```bash
python run.py --video videos/demo.mp4
```

### 5. 按真实视频时间等待

```bash
python run.py --video videos/demo.mp4 --realtime
```

### 6. 保存采样帧

```bash
python run.py --video videos/demo.mp4 --dry-run --save-frames --max-windows 3
```

### 7. 关闭上一窗口状态传递

```bash
python run.py --video videos/demo.mp4 --no-state
```

### 8. 限制窗口范围

```bash
python run.py --video videos/demo.mp4 --max-windows 5 --start-time 10 --end-time 60
```

## 命令行参数

```text
--config PATH            YAML 配置文件路径
--video PATH             本地视频路径（除 --print-config 外均为必需）
--output-dir DIR         输出根目录（覆盖 config.yaml）
--start-time SECONDS     忽略结束时间早于该值的窗口
--end-time SECONDS       忽略开始时间晚于或等于该值的窗口
--start-window INDEX     忽略全局序号小于该值的窗口
--end-window INDEX       忽略全局序号大于该值的窗口
--max-windows N          最多处理 N 个窗口
--realtime               按视频逻辑时间等待
--save-frames            保存本次运行的采样帧到 sampled_frames/
--dry-run                抽帧并构建提示词，但不调用模型
--validate-only          检查视频并报告窗口，不调用模型
--no-state               不将上一窗口摘要传入下一窗口提示词
--print-config           打印解析后的配置并退出
--verbose, -v            启用调试日志
```

## 默认参数

```yaml
video:
  window_seconds: 6.0
  stride_seconds: 3.0

sampling:
  sample_fps: 1.0
  min_frames: 4
  max_frames: 12
  max_image_side: 768
  jpeg_quality: 80
```

窗口顺序示例：

```text
0-6 秒
3-9 秒
6-12 秒
9-15 秒
```

每次请求只传当前窗口的 Base64 JPEG 图像列表和可选的上一窗口摘要，不传完整历史视频。

## 输出

每次运行会生成：

```text
outputs/<YYYYMMDD_HHMMSS_experiment_hash>/
├── run_meta.json
├── resolved_config.yaml
├── windows.jsonl
├── observations.jsonl
├── api_metrics.jsonl
├── errors.jsonl
├── raw_responses/
│   └── window_0000_0000.txt
└── sampled_frames/         # when --save-frames or storage.save_sampled_frames=true
    └── window_0000_0000/
        └── frame_000_0.000.jpg
```

- `run_meta.json`：运行元数据、视频 SHA256、最终模型来源、提示词哈希、最终统计。
- `resolved_config.yaml`：解析后的配置，API Key 已脱敏。
- `windows.jsonl`：所有选中的窗口。
- `observations.jsonl`：仅包含通过 Schema 和语义校验的观察结果。
- `api_metrics.jsonl`：每个窗口的 API 请求指标。
- `errors.jsonl`：失败的窗口及其错误信息，关联到原始响应路径。

`observations.jsonl` 每行对应一个窗口：

```json
{
  "schema_version": "1.0",
  "window_run_index": 0,
  "window_global_index": 0,
  "window_start_seconds": 0.0,
  "window_end_seconds": 6.0,
  "scene": {
    "description": "工作人员在开关柜前操作门锁。",
    "viewpoint": "front"
  },
  "entities": [],
  "actions": [],
  "uncertainties": [],
  "summary": "工作人员在开关柜前操作门锁。"
}
```

## 修改分析任务

- 修改窗口参数：`config.yaml` 的 `video` 和 `sampling` 部分。
- 修改流式分析规则：`src/qwen_stream_video/inference/prompts.py` 中的 `DEFAULT_SYSTEM_PROMPT`。
- 修改每窗口动态信息：`src/qwen_stream_video/inference/prompts.py` 中的 `DEFAULT_USER_PROMPT_TEMPLATE`。
- 修改视频名称、类别和背景：`config.yaml` 的 `video_metadata` 部分。

## 开发与测试

```bash
.venv/Scripts/python -m pytest tests/ -q
.venv/Scripts/python -m ruff check .
```

## 说明

- 千问 OpenAI 兼容接口支持以 `image_url` 传入按时间顺序排列的图像列表。
- 本项目将本地帧编码为 `data:image/jpeg;base64,...` 后发送。
- 模型输出在去 Markdown 代码块、提取 JSON、Pydantic Schema 校验和语义校验后才会写入 `observations.jsonl`。
- API 失败、解析失败或校验失败的窗口会记录到 `errors.jsonl`，不会中断整个视频。
- 建议先使用 `--validate-only` 或 `--dry-run --max-windows 3` 检查窗口、抽帧与提示词，再调用完整流程。
