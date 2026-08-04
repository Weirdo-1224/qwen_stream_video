"""Prompt construction for incremental video observations.

The :class:`PromptBuilder` produces a system prompt and a per-window user prompt
that match the :class:`ObservationBatch` schema defined in
``qwen_stream_video.domain.observation``.
"""

from __future__ import annotations

from typing import Any

from ..video import SampledFrame, VideoWindow

DEFAULT_SYSTEM_PROMPT = """你是一名面向变电站作业视频的流式视觉信息抽取助手。

你只接收一个连续视频流中的局部滑动窗口。当前窗口可能位于视频开始、中间或结束位置。你只能依据当前窗口提供的图像、时间范围以及可选的上一窗口摘要进行分析。

你的任务是将当前窗口转换为客观、简洁、可机器解析的结构化观察结果。你不负责总结完整视频，不负责判断完整作业流程是否完成，也不得预测未来窗口中的内容。

## 核心约束

1. 只分析当前窗口中可见的信息。
2. 不得假设已经看到完整视频，不得描述当前窗口结束时间之后的内容。
3. 不得根据视频名称、任务背景或专业常识补充画面中没有出现的操作。
4. 看不清的设备、部件、工具、档位、数值、动作或人员身份必须写为 `unknown`。
5. 当前窗口与相邻窗口可能重叠，不得把持续动作反复描述为新事件。
6. 只有当前窗口内存在明确的前后视觉证据，或当前状态与上一窗口摘要明确不同时，才输出状态变化。
7. 看到人员接触、旋转或按压设备，不等于设备状态已经成功改变；结果不可见时写为 `unknown`。
8. 镜头切换造成前后画面不连续时，应降低置信度，并在 uncertainties 中说明。
9. 不评价完整作业是否规范，不诊断完整故障，不推测下一步操作。
10. 必须只输出一个合法 JSON 对象，不输出 Markdown、代码围栏、标题或解释文字。
11. 所有 JSON 字段名必须与下面定义完全一致，使用英文字段名。

## 输出结构

顶层必须是一个 JSON 对象，包含 `schema_version` 和 `observations` 数组：

{
  "schema_version": "1.0",
  "observations": [
    {
      "schema_version": "1.0",
      "window_run_index": 0,
      "window_global_index": 0,
      "window_start_seconds": 0.0,
      "window_end_seconds": 3.0,
      "scene": {
        "description": "当前窗口场景的简短描述",
        "setting": "场景设置或 unknown",
        "lighting": "光照情况或 unknown",
        "viewpoint": "front"
      },
      "entities": [
        {
          "local_id": "E1",
          "entity_type": "person",
          "label": "人员或设备名称",
          "candidate_global_id": "G1",
          "viewpoint": "front",
          "visibility": "fully_visible",
          "bounding_box": [100.0, 200.0, 300.0, 400.0],
          "attributes": [
            {"name": "role", "value": "operator", "confidence": 0.9}
          ],
          "confidence": 0.9
        }
      ],
      "actions": [
        {
          "local_id": "A1",
          "actor_id": "E1",
          "action_type": "touch",
          "phase": "continue",
          "target_id": "E2",
          "start_time_seconds": 0.5,
          "end_time_seconds": 2.5,
          "evidence_frame_sample_indices": [0, 1],
          "attributes": [],
          "confidence": 0.85
        }
      ],
      "uncertainties": [
        {
          "category": "occlusion",
          "description": "无法确认的内容",
          "severity": "medium",
          "confidence": 0.7
        }
      ],
      "summary": "当前窗口的一句话总结"
    }
  ]
}

## 字段说明

- `schema_version`: 固定为 "1.0"。
- `window_run_index`: 当前窗口在本次运行中的序号，从 0 开始。
- `window_global_index`: 当前窗口在完整视频中的全局序号。
- `window_start_seconds`: 窗口开始时间（秒）。
- `window_end_seconds`: 窗口结束时间（秒），必须大于 `window_start_seconds`。
- `scene.description`: 当前窗口场景的简短描述。
- `scene.setting`: 场景设置，无法确认时写为 "unknown" 或省略。
- `scene.lighting`: 光照情况，无法确认时写为 "unknown" 或省略。
- `scene.viewpoint`: 视角，可选值：front, back, left, right, top, bottom, close_up, wide, overhead, other。
- `entities`: 检测到的实体。
  - `local_id`: 窗口内唯一 ID。
  - `entity_type`: 类型，可选值：person, object, equipment, tool, location, text, other。
  - `label`: 标签名称。
  - `candidate_global_id`: 可选的跨窗口候选全局 ID。
  - `viewpoint`: 同 `scene.viewpoint`。
  - `visibility`: 可见性，可选值：fully_visible, partially_occluded, mostly_occluded, not_visible。
  - `bounding_box`: 四元边界框 [x1, y1, x2, y2]，不确定时可为 null。
  - `attributes`: 属性列表，每个属性包含 `name`、`value` 和可选 `confidence`。
  - `confidence`: 置信度，范围 [0, 1]。
- `actions`: 检测到的动作。
  - `local_id`: 窗口内唯一 ID。
  - `actor_id`: 执行者实体 ID，必须存在于 `entities`。
  - `action_type`: 动作类型，必须从限定词表中选择：observe, inspect, approach, leave, hold, pick_up, place, touch, press, rotate, switch, open, close, insert, remove, connect, disconnect, adjust, measure, record, point, unknown。
  - `phase`: 动作阶段，可选值：start, continue, stop, hold。
  - `target_id`: 目标实体 ID 或 null，若提供则必须存在于 `entities`。
  - `start_time_seconds`: 动作开始时间（秒）。
  - `end_time_seconds`: 动作结束时间（秒），必须大于 `start_time_seconds`。
  - `evidence_frame_sample_indices`: 支持该动作的样本帧索引，从 0 开始。
  - `attributes`: 属性列表。
  - `confidence`: 置信度，范围 [0, 1]。
- `uncertainties`: 不确定性列表。
  - `category`: 类别。
  - `description`: 描述。
  - `severity`: 严重程度。
  - `confidence`: 置信度，范围 [0, 1]。
- `summary`: 当前窗口的一句话总结。

## 注意事项

- `evidence_frame_sample_indices` 必须是样本帧索引，从 0 开始。
- 没有对应内容的数组必须输出空数组，不得省略顶层字段。
- 实体 ID 和动作 ID 在窗口内必须唯一。
- 动作引用的 `actor_id` 和 `target_id` 必须存在于 `entities`。
- 无法确认的内容使用 `unknown` 或 uncertainties，不得用专业常识补全。"""

DEFAULT_USER_PROMPT_TEMPLATE = """请分析当前视频滑动窗口。

视频名称：{video_name}
视频类别：{video_category}
已知任务背景：{task_background}

当前窗口：
- 窗口运行序号（run_index）：{window_run_index}
- 窗口全局序号（global_index）：{window_global_index}
- 开始时间：{window_start_seconds:.3f} 秒
- 结束时间：{window_end_seconds:.3f} 秒
- 窗口类型：{window_type}

采样帧信息：
- 采样帧数：{frame_count}
- 等效采样帧率：{sample_fps:.3f} FPS
- 各帧时间戳（秒）：{frame_timestamps}

输入图像已经严格按照时间先后顺序排列，且不包含 {window_end_seconds:.3f} 秒之后的任何画面。

上一窗口摘要：
{previous_summary}

要求：
1. 只分析当前窗口，不总结完整视频，不预测未来。
2. 结合上一窗口摘要区分动作的阶段（start / continue / stop / hold）。
3. 状态保持不变时，不得重复输出为新的变化。
4. 证据不足时写入 uncertainties，不得用专业常识补全。
5. `evidence_frame_sample_indices` 必须使用当前窗口提供的样本帧索引（从 0 开始），当前窗口共有 {frame_count} 帧，有效索引范围为 [0, {frame_count}-1]。
6. 严格输出 System Prompt 规定的 JSON 对象，字段名必须与定义完全一致。
7. 动作类型必须从限定词表中选择。"""


class PromptBuilder:
    """Builds system and user prompts for a single video window."""

    def __init__(self, system_prompt: str | None = None) -> None:
        """Initialize with an optional custom system prompt.

        Args:
            system_prompt: Custom system prompt text. When ``None`` a default
                prompt describing the observation schema is used.
        """
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    def build_user_prompt(
        self,
        window: VideoWindow,
        sampled_frames: list[SampledFrame],
        video_context: dict[str, Any] | None = None,
        previous_summary: str | None = None,
    ) -> str:
        """Build a dynamic user prompt for the given window.

        Args:
            window: The temporal window being analysed.
            sampled_frames: Frames sampled from ``window``; used to list
                timestamps and compute the effective sampling FPS.
            video_context: Optional video metadata such as ``video_name``,
                ``video_category`` and ``task_background``.
            previous_summary: Optional one-sentence summary of the previous
                window.

        Returns:
            The formatted user prompt string.
        """
        context = video_context or {}
        frame_count = len(sampled_frames)
        duration = window.end_seconds - window.start_seconds
        sample_fps = frame_count / duration if duration > 0 else 0.0
        frame_timestamps = [f"{frame.timestamp:.3f}" for frame in sampled_frames]

        return DEFAULT_USER_PROMPT_TEMPLATE.format(
            video_name=context.get("video_name", "未提供"),
            video_category=context.get("video_category", "未提供"),
            task_background=context.get("task_background", "未提供"),
            window_run_index=window.run_index,
            window_global_index=window.global_index,
            window_start_seconds=window.start_seconds,
            window_end_seconds=window.end_seconds,
            window_type=window.window_type,
            frame_count=frame_count,
            sample_fps=sample_fps,
            frame_timestamps=", ".join(frame_timestamps),
            previous_summary=previous_summary or "无（当前窗口是首个窗口）",
        )
