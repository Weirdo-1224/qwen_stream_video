# S1-T07：提示词与响应解析

**状态：DONE**　**依赖：S1-T05、S1-T06**

## 目标

构建增量观察提示词，并安全解析模型原始响应。

## 修改

- 编写系统提示词：仅当前窗口、禁止未来推断、仅 JSON、证据帧为样本索引。
- 编写动态用户提示词：窗口、帧时间、视频背景和可选上一窗口摘要。
- 实现提示词构建器和解析流程：去 Markdown 代码块 → 提取 JSON → `json.loads` → Schema → 语义校验。
- 定义解析/Schema/语义异常；禁止 `eval()`。
- 添加正常、代码块、非法 JSON 和非法 Schema 测试。

## 不做

不调用模型；解析失败不触发重发。

## 验收

- 失败带明确异常，调用方可保存原文并继续下一窗口。

## 完成记录

- 修改文件：
  - 新增 `src/qwen_stream_video/inference/prompts.py`：定义 `PromptBuilder` 与默认系统/用户提示词模板，支持窗口、帧时间、视频背景与上一窗口摘要的动态拼接。
  - 新增 `src/qwen_stream_video/inference/parser.py`：定义 `ResponseParser`，完成去 Markdown 代码围栏、JSON 提取、`json.loads`、Pydantic Schema 校验与语义校验流程。
  - 更新 `src/qwen_stream_video/inference/__init__.py`：导出 `PromptBuilder` 与 `ResponseParser`。
  - 更新 `prompts/system_prompt.txt` 与 `prompts/user_prompt.txt`：与新 Observation Schema 对齐。
  - 新增 `tests/unit/test_prompts_parser.py`：覆盖用户提示词构建、正常 JSON、Markdown 代码块、包裹文本、非法 JSON、非法 Schema 及语义错误等场景。
- 验证结果：
  - `.venv/Scripts/python -m pytest tests/unit/test_prompts_parser.py -q`：11 个测试全部通过。
  - `.venv/Scripts/python -m pytest tests/ -q`：67 个测试全部通过。
  - `.venv/Scripts/python -m ruff check .`：无错误。
