# S1-T02：配置系统

**状态：DONE**　**依赖：S1-T01**

## 目标

用 Pydantic 构建严格、可追踪的应用配置。

## 修改

- 定义 `ExperimentConfig`、`VideoConfig`、`SamplingConfig`、`ModelConfig`、`ObservationConfig`、`RuntimeConfig`、`StorageConfig`、`AppConfig`。
- 支持 YAML、`DASHSCOPE_API_KEY`、`DASHSCOPE_BASE_URL`、`QWEN_MODEL` 与 CLI 覆盖，优先级为 CLI > 环境变量 > YAML > 默认值。
- 校验窗口、步长、采样率、帧数范围、图片边长、JPEG 质量和重试次数。
- 提供隐藏密钥的配置摘要及配置单元测试。

## 不做

不读取视频、不创建输出目录、不请求 API。

## 验收

- 非法配置在启动前明确失败。
- 测试覆盖环境和 CLI 优先级，日志不泄露密钥。

## 完成记录

- 修改文件：
  - `src/qwen_stream_video/config.py`：新增配置模型与加载逻辑。
  - `src/qwen_stream_video/cli.py`：增加 `--config`、`--print-config` 与配置错误退出。
  - `config.yaml`：按新配置结构重写。
  - `tests/unit/test_config.py`：新增配置单元测试。
  - `run.py`：修复 import 排序以通过 `ruff`。
- 验证结果：
  - `.venv/Scripts/python -m pip install -e ".[dev]"` 成功。
  - `.venv/Scripts/python -m pytest tests/unit/test_config.py -q`：14 个测试全部通过。
  - `.venv/Scripts/python -m ruff check src/qwen_stream_video/config.py src/qwen_stream_video/cli.py tests/unit/test_config.py`：无错误。
  - `python run.py --help` 与 `--print-config` 可运行；无效配置通过 CLI 返回非零退出码并给出明确错误。
