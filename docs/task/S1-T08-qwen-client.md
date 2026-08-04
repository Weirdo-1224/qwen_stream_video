# S1-T08：Qwen 客户端

**状态：DONE**　**依赖：S1-T02**

## 目标

隔离 Qwen API 请求，输出原始响应与请求指标。

## 修改

- 定义 `RawInferenceResult`：原文、模型、延迟、请求 ID、token、尝试次数。
- 实现 `QwenClient`，按配置发起视觉请求。
- 只重试超时、连接中断、HTTP 429 和 5xx，次数服从 `network_retries`。
- 映射网络、限流、服务端错误为项目异常；提供 Fake/Mock 接口。

## 不做

不解析 JSON、不做校验、不存文件；JSON/Schema 错误不得重发请求。

## 验收

- 重试范围和次数可测试；单元测试不调用真实 API。

## 完成记录

- 修改文件：
  - 新增 `src/qwen_stream_video/inference/client.py`：定义 `RawInferenceResult` 数据类、`QwenClient` 与 `FakeQwenClient`。
    - `QwenClient.infer()` 使用 OpenAI 兼容接口发起视觉请求，构造包含系统提示词、用户提示词与 `image_url` 图像列表的消息。
    - 仅对超时、连接中断、HTTP 429 与 HTTP 5xx 进行重试，最大尝试次数为 `network_retries + 1`。
    - 将 `APITimeoutError`/`APIConnectionError` 映射为 `InferenceNetworkError`，HTTP 429 映射为 `InferenceRateLimitError`，HTTP 5xx 映射为 `InferenceServerError`；其他 4xx 客户端错误不重试直接抛出。
    - `FakeQwenClient` 返回固定响应并记录每次调用参数，供离线测试使用。
  - 更新 `src/qwen_stream_video/inference/__init__.py`：导出 `QwenClient`、`FakeQwenClient`、`RawInferenceResult`。
  - 新增 `tests/unit/test_client.py`：覆盖 `FakeQwenClient`、成功请求、超时重试后成功、连接错误/限流/服务端错误耗尽重试、非重试 4xx 不重试、零重试立即失败等场景。
- 验证结果：
  - `.venv/Scripts/python -m pytest tests/unit/test_client.py -q`：8 个测试全部通过。
  - `.venv/Scripts/python -m pytest tests/ -q`：75 个测试全部通过。
  - `.venv/Scripts/python -m ruff check src/qwen_stream_video/inference/client.py src/qwen_stream_video/inference/__init__.py tests/unit/test_client.py`：无错误。
