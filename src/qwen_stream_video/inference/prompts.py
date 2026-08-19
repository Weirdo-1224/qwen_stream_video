"""Prompt construction for Schema 2.0 local visual observations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..domain import ObservationBatch
from ..video import SampledFrame, VideoWindow

DEFAULT_SYSTEM_PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "system_prompt.txt"
DEFAULT_USER_PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "user_prompt.txt"

DEFAULT_SYSTEM_PROMPT = """你是局部流式视频观察器。只报告当前窗口直接可见的视觉事实，不维护正式全局状态。

规则：
1. 只能使用当前窗口图像和给出的抽样时间，不使用未来信息。
2. local_id 只在当前窗口有效；不得自行创建正式 global_entity_id。
3. candidate_global_id 只能从程序提供的 candidate_entities 中选择，无法判断时返回 null。
4. 看不到不等于实体消失；首次可见不等于属性发生变化。
5. 动作 phase 只是本窗口观察，不是正式动作生命周期。
6. 不根据任务名称猜测故障、回路、作业步骤或不可见部件。
7. 看不清就使用 unknown，并为事实提供 evidence_frames。
8. 只输出符合 Observation Schema 2.0 的 JSON 对象，不输出 Markdown 或解释文字。"""

DEFAULT_USER_PROMPT_TEMPLATE = """请分析当前窗口中的局部视觉事实。

当前窗口：global_index={window_global_index}
Context Interval: [{window_start_seconds:.3f}, {commit_start_seconds:.3f}) 秒
Commit Interval: [{commit_start_seconds:.3f}, {window_end_seconds:.3f}) 秒
窗口类型：{window_type}；处理角色：{processing_role}

抽样帧（模型只能引用这些 sample_index）：
{frame_lines}

候选上下文（candidate_global_id 只能从这里选择）：
{context_json}

动作词表摘要：observe, inspect, hand_over, receive, push, pull, hover, pick_up, put_down, press, release, open, close, operate, other, unknown
属性使用 canonical attribute_key；视觉事实和任务条件解释必须分离。

请返回 Schema 2.0 JSON：窗口时间由程序覆盖；当前窗口未看到的实体不表示消失；不确定时保留 unknown 和 uncertainty。{task_context}"""


def _observation_schema_contract() -> str:
    """Return the authoritative Schema 2.0 contract for model prompts.

    The contract is generated from the same Pydantic model used by the
    response parser.  Keeping one source of truth prevents the remote model
    prompt and runtime validator from drifting apart.
    """
    schema = json.dumps(
        ObservationBatch.model_json_schema(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        """\n\n=== 强制输出协议：Observation Schema 2.0 ===
下面的 JSON Schema 是程序解析器使用的权威协议。你的输出必须严格遵守它。

规则：
1. 只输出一个 JSON 对象，不要输出 Markdown、注释或额外文字。
2. 除 Schema 中列出的字段外，禁止输出任何字段（尤其禁止 attribute_value、object_description、object_local_id、relation、scope、severity）。
3. entity_type 只能使用 person、device、component、tool、ppe、sign、document、environment、unknown。
4. scene.continuity_hint 只能使用 continuous、reframed、camera_change、unknown。
5. uncertainty 必须使用 uncertainty_type、description、related_local_ids、evidence_frames；不要使用 type、scope、severity。
6. attribute 必须使用 value；不要使用 attribute_value。关系必须使用 relation_type 和 object_local_id。
7. action 必须使用 action_type、actor_local_id、target_local_id、tool_local_id；不要使用 object_local_id。
8. local_id 只能是当前窗口的局部 ID，例如 person_0；不得创建正式 global_entity_id。
9. evidence_frames 只能引用当前抽样帧的 sample_index。
10. actor_local_id、target_local_id、tool_local_id、entity_local_id、subject_local_id、object_local_id 必须引用本次输出 entities 中存在的 local_id；无法确认时使用 null 或省略对应动作/关系。
11. window 字段由程序写入，可以省略；如果输出，必须符合下面 Schema。

权威 JSON Schema（不要照抄示例中的空数组，必须填写当前画面事实）：
"""
        + schema
    )


def _load_template(path: Path, fallback: str) -> str:
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8")
    except OSError:
        pass
    return fallback


class PromptBuilder:
    def __init__(
        self,
        system_prompt: str | None = None,
        user_prompt_template: str | None = None,
        *,
        context_policy: str = "visual_only",
    ) -> None:
        self.context_policy = context_policy
        base_system_prompt = system_prompt or _load_template(DEFAULT_SYSTEM_PROMPT_PATH, DEFAULT_SYSTEM_PROMPT)
        self.system_prompt = base_system_prompt + _observation_schema_contract()
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
        context: Any | None = None,
    ) -> str:
        """Build a bounded prompt; task-specific strings are opt-in."""
        if context is None:
            context_dict: dict[str, Any] = {
                "scene": {},
                "candidate_entities": previous_entities or [],
                "active_actions": [],
                "pending_attributes": [],
            }
        elif hasattr(context, "model_dump"):
            context_dict = context.model_dump(mode="json")
        else:
            context_dict = dict(context)
        frame_lines = "\n".join(
            f"- F{frame.sample_index} = {frame.timestamp_seconds:.3f} 秒" for frame in sampled_frames
        ) or "- 无抽样帧"
        generic_context = "这是本地视频中的通用视觉观察场景。"
        task_context = ""
        video_context = video_context or {}
        if self.context_policy == "weak_context":
            task_context = "\n场景类别提示：设备区域视觉观察。"
        elif self.context_policy == "task_conditioned":
            task_context = (
                "\n任务条件解释仅可单独放入 task_conditioned_interpretation，不得混入 visual_fact。"
                f"\n任务名称：{video_context.get('video_name') or generic_context}"
            )
        rendered = self.user_prompt_template.format(
            window_global_index=window.global_index,
            window_start_seconds=window.start_seconds,
            commit_start_seconds=window.commit_start_seconds,
            window_end_seconds=window.end_seconds,
            window_type=window.window_type,
            processing_role=window.processing_role,
            frame_lines=frame_lines,
            context_json=json.dumps(context_dict, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            task_context=task_context,
            video_name=video_context.get("video_name") if self.context_policy == "task_conditioned" else generic_context,
            video_category=video_context.get("video_category") if self.context_policy == "task_conditioned" else "visual_only",
            task_background=video_context.get("task_background") if self.context_policy == "task_conditioned" else "",
            frame_count=len(sampled_frames),
            sample_fps=(len(sampled_frames) / max(window.end_seconds - window.start_seconds, 1e-9)),
            frame_timestamps=", ".join(f"{frame.timestamp_seconds:.3f}" for frame in sampled_frames),
            previous_summary=previous_summary or "",
            previous_entities=json.dumps(previous_entities or [], ensure_ascii=False),
        )
        # Append the contract even when a custom user template is supplied.
        # This guarantees that every provider receives the same authoritative
        # schema and field restrictions.
        return rendered + _observation_schema_contract()
