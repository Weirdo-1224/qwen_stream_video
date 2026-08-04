# S1-T07：提示词与响应解析

**状态：DONE（已按单 ObservationBatch 协议更新）**　**依赖：S1-T05、S1-T06**

## 目标

构建增量观察提示词，并安全解析模型原始响应为单个 `ObservationBatch`。

## 修改

- 更新 `src/qwen_stream_video/inference/prompts.py`：
  - 系统与用户提示词明确要求模型只输出当前窗口的一份 `ObservationBatch` JSON；
  - 提示词字段与新 `domain/observation.py` 协议对齐；
  - 用户提示词支持传入上一窗口候选实体摘要。
- 更新 `prompts/system_prompt.txt` 与 `prompts/user_prompt.txt`：与新的单窗口 Schema 对齐。
- 更新 `src/qwen_stream_video/inference/parser.py`：
  - 处理流程：原始文本 → 去 Markdown 代码块 → 提取最外层 JSON 对象 → `json.loads` → `ObservationBatch.model_validate` → 语义校验；
  - 不再处理或接受 `observations` 外层列表；
  - 解析错误映射为项目异常，禁止 `eval()`。
- 重写 `tests/unit/test_prompts_parser.py`：覆盖用户提示词构建、正常 JSON、Markdown 代码块、包裹文本、非法 JSON、非法 Schema、语义错误、顶层列表拒绝及 `eval` 禁止等场景。

## 不做

不调用模型；解析失败不触发重发；不实现全局实体注册。

## 验收

- 失败带明确异常，调用方可保存原文并继续下一窗口。

## 完成记录

- 修改文件：
  - 更新 `src/qwen_stream_video/inference/prompts.py`。
  - 更新 `src/qwen_stream_video/inference/parser.py`。
  - 更新 `prompts/system_prompt.txt` 与 `prompts/user_prompt.txt`。
  - 重写 `tests/unit/test_prompts_parser.py`。
- 验证结果：
  - `.venv/Scripts/python -m pytest tests/unit/test_prompts_parser.py -q`：11 个测试全部通过。
  - `.venv/Scripts/python -m pytest tests/ -q`：98 个测试全部通过。
  - `.venv/Scripts/python -m ruff check .`：无错误。
