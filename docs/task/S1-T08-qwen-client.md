# S1-T08：Qwen 客户端

**状态：TODO**　**依赖：S1-T02**

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
- 验证结果：
