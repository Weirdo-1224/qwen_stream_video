"""Prompt construction for incremental video observations.

The :class:`PromptBuilder` produces a system prompt and a per-window user prompt
that match the :class:`ObservationBatch` schema defined in
``qwen_stream_video.domain.observation``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..video import SampledFrame, VideoWindow

DEFAULT_SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parents[3] / "prompts" / "system_prompt.txt"
)
DEFAULT_USER_PROMPT_PATH = (
    Path(__file__).resolve().parents[3] / "prompts" / "user_prompt.txt"
)


def _load_template(path: Path, fallback: str) -> str:
    """Load a prompt template from disk, falling back to the embedded default."""
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8")
    except OSError:
        pass
    return fallback

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
12. `local_id` 只在当前窗口有效；`candidate_global_id` 只是候选全局标识，不代表正式稳定 ID。

## 输出结构

顶层必须是一个 JSON 对象，代表当前窗口的一份 ObservationBatch：

{
  "schema_version": "1.0",
  "window": {
    "global_index": 0,
    "start_seconds": 0.0,
    "end_seconds": 3.0
  },
  "summary": "当前窗口的一句话总结",
  "scene": {
    "camera_change": false,
    "view_type": "wide",
    "visibility": "clear",
    "description": "当前窗口场景的简短描述"
  },
  "entities": [
    {
      "local_id": "E1",
      "entity_type": "person",
      "name": "operator",
      "description": "操作人员",
      "appearance": {"role": "operator"},
      "spatial_region": "center",
      "candidate_global_id": "person_1",
      "confidence": 0.9,
      "evidence_frames": [0, 1]
    }
  ],
  "actions": [
    {
      "local_id": "A1",
      "actor_local_id": "E1",
      "action_type": "touch",
      "target_local_id": "E2",
      "tool_local_id": null,
      "phase_observation": "ongoing",
      "description": "操作人员在接触设备",
      "confidence": 0.85,
      "evidence_frames": [1, 2]
    }
  ],
  "attribute_observations": [
    {
      "entity_local_id": "E2",
      "attribute": "state",
      "value": "closed",
      "confidence": 0.8,
      "evidence_frames": [2]
    }
  ],
  "uncertainties": [
    {
      "description": "无法确认设备是否已完全闭合",
      "related_local_ids": ["E2"],
      "evidence_frames": []
    }
  ]
}

## 字段说明

- `schema_version`: 固定为 "1.0"。
- `window`: 当前窗口坐标。
  - `global_index`: 当前窗口在完整视频中的全局序号。
  - `start_seconds`: 窗口开始时间（秒）。
  - `end_seconds`: 窗口结束时间（秒），必须大于 `start_seconds`。
- `summary`: 当前窗口的一句话总结。
- `scene`: 场景描述。
  - `camera_change`: 当前窗口是否发生镜头切换。
  - `view_type`: 视角，可选值：wide, medium, closeup, detail, unknown。
  - `visibility`: 可见性质量，可选值：clear, partial, poor, unknown。
  - `description`: 当前窗口场景的简短描述。
- `entities`: 检测到的实体。
  - `local_id`: 窗口内唯一 ID。
  - `entity_type`: 类型，可选值：person, device, component, tool, ppe, sign, environment, unknown。
  - `name`: 名称或标签，无法确认时写为 "unknown"。
  - `description`: 简短描述。
  - `appearance`: 外观属性字典（可为空）。
  - `spatial_region`: 空间区域描述，无法确认时写为 "unknown"。
  - `candidate_global_id`: 可选的跨窗口候选全局 ID。
  - `confidence`: 置信度，范围 [0, 1]。
  - `evidence_frames`: 支持该实体的样本帧索引，从 0 开始。
- `actions`: 检测到的动作。
  - `local_id`: 窗口内唯一 ID。
  - `actor_local_id`: 执行者实体 ID，必须存在于 `entities`。
  - `action_type`: 动作类型，必须从限定词表中选择：observe, inspect, approach, leave, hold, pick_up, place, touch, press, rotate, switch, open, close, insert, remove, connect, disconnect, adjust, measure, record, point, unknown。
  - `target_local_id`: 目标实体 ID 或 null，若提供则必须存在于 `entities`。
  - `tool_local_id`: 工具实体 ID 或 null，若提供则必须存在于 `entities`。
  - `phase_observation`: 动作阶段，可选值：starting, ongoing, possibly_completed, instant, unknown。
  - `description`: 动作描述。
  - `confidence`: 置信度，范围 [0, 1]。
  - `evidence_frames`: 支持该动作的样本帧索引，从 0 开始。
- `attribute_observations`: 属性观察。
  - `entity_local_id`: 属性所属实体 ID，必须存在于 `entities`。
  - `attribute`: 属性名称。
  - `value`: 属性值。
  - `confidence`: 置信度，范围 [0, 1]。
  - `evidence_frames`: 支持该属性的样本帧索引，从 0 开始。
- `uncertainties`: 不确定性列表。
  - `description`: 描述。
  - `related_local_ids`: 相关实体 ID 列表。
  - `evidence_frames`: 相关样本帧索引，从 0 开始。

## 注意事项

- `evidence_frames` 必须是样本帧索引，从 0 开始。
- 没有对应内容的数组必须输出空数组，不得省略顶层字段。
- 实体 ID 和动作 ID 在窗口内必须唯一。
- 动作引用的 `actor_local_id`、`target_local_id`、`tool_local_id` 必须存在于 `entities`。
- 属性引用的 `entity_local_id` 必须存在于 `entities`。
- 无法确认的内容使用 `unknown` 或 uncertainties，不得用专业常识补全。"""

DEFAULT_USER_PROMPT_TEMPLATE = """请分析当前视频滑动窗口。

视频名称：{video_name}
视频类别：{video_category}
已知任务背景：{task_background}

当前窗口：
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

上一窗口候选实体：
{previous_entities}

要求：
1. 只分析当前窗口，不总结完整视频，不预测未来。
2. 结合上一窗口摘要区分动作的阶段（starting / ongoing / possibly_completed / instant / unknown）。
3. 状态保持不变时，不得重复输出为新的变化。
4. 证据不足时写入 uncertainties，不得用专业常识补全。
5. `evidence_frames` 必须使用当前窗口提供的样本帧索引（从 0 开始），当前窗口共有 {frame_count} 帧，有效索引范围为 [0, {frame_count}-1]。
6. 严格输出 System Prompt 规定的 JSON 对象，字段名必须与定义完全一致。
7. 动作类型必须从限定词表中选择。
8. 本窗口只输出一份 ObservationBatch，顶层不得包含 `observations` 数组。"""


class PromptBuilder:
    """Builds system and user prompts for a single video window."""

    def __init__(
        self,
        system_prompt: str | None = None,
        user_prompt_template: str | None = None,
    ) -> None:
        """Initialize with optional custom prompt templates.

        Args:
            system_prompt: Custom system prompt text. When ``None`` the prompt is
                loaded from ``prompts/system_prompt.txt`` if it exists, otherwise
                the embedded default is used.
            user_prompt_template: Custom user prompt template. When ``None`` the
                template is loaded from ``prompts/user_prompt.txt`` if it exists,
                otherwise the embedded default is used.
        """
        self.system_prompt = system_prompt or _load_template(
            DEFAULT_SYSTEM_PROMPT_PATH, DEFAULT_SYSTEM_PROMPT
        )
        self.user_prompt_template = user_prompt_template or _load_template(
            DEFAULT_USER_PROMPT_PATH, DEFAULT_USER_PROMPT_TEMPLATE
        )

    def build_user_prompt(
        self,
        window: VideoWindow,
        sampled_frames: list[SampledFrame],
        video_context: dict[str, Any] | None = None,
        previous_summary: str | None = None,
        previous_entities: list[dict[str, Any]] | None = None,
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
            previous_entities: Optional list of candidate entity summaries from
                the previous window, each with ``candidate_global_id``,
                ``entity_type`` and ``description``.

        Returns:
            The formatted user prompt string.
        """
        context = video_context or {}
        frame_count = len(sampled_frames)
        duration = window.end_seconds - window.start_seconds
        sample_fps = frame_count / duration if duration > 0 else 0.0
        frame_timestamps = [f"{frame.timestamp_seconds:.3f}" for frame in sampled_frames]

        previous_entities = previous_entities or []
        if previous_entities:
            previous_entities_text = "\n".join(
                f"- candidate_global_id: {e.get('candidate_global_id', 'unknown')}, "
                f"type: {e.get('entity_type', 'unknown')}, "
                f"description: {e.get('description', 'unknown')}"
                for e in previous_entities
            )
        else:
            previous_entities_text = "无（当前窗口是首个窗口或上一窗口无候选实体）"

        return self.user_prompt_template.format(
            video_name=context.get("video_name", "未提供"),
            video_category=context.get("video_category", "未提供"),
            task_background=context.get("task_background", "未提供"),
            window_global_index=window.global_index,
            window_start_seconds=window.start_seconds,
            window_end_seconds=window.end_seconds,
            window_type=window.window_type,
            frame_count=frame_count,
            sample_fps=sample_fps,
            frame_timestamps=", ".join(frame_timestamps),
            previous_summary=previous_summary or "无（当前窗口是首个窗口）",
            previous_entities=previous_entities_text,
        )
