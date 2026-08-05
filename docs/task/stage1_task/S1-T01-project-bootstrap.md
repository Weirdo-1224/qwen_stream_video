# S1-T01：工程骨架

**状态：DONE**　**依赖：无**

## 目标

建立 Python 3.10+ 的可安装 `src` 工程，并保留旧启动方式。

## 修改

- 新增 `pyproject.toml`：运行/开发依赖和 `qwen-stream-video` 入口。
- 创建 `src/qwen_stream_video/` 包、`cli.py`、`pipeline.py`、`exceptions.py` 与 `domain`、`video`、`inference`、`storage` 子包。
- 根目录 `run.py` 仅保留 `qwen_stream_video.cli.main` 兼容入口。
- 更新 `.env.example`、`.gitignore`，禁止密钥和运行产物进入仓库。

## 不做

不实现业务逻辑、API 或视频处理。

## 验收

- `pip install -e .` 成功。
- `python run.py --help` 与 `qwen-stream-video --help` 可运行。

## 完成记录

- 修改文件：`pyproject.toml`、`run.py`、`.env.example`、`.gitignore`，以及新增的 `src/qwen_stream_video/` 包骨架。
- 验证结果：`.venv\\Scripts\\python.exe -m pip install -e .` 成功；`.venv\\Scripts\\python.exe run.py --help` 与 `.venv\\Scripts\\qwen-stream-video.exe --help` 均通过。
