from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_prompt.txt"
DEFAULT_USER_PROMPT_PATH = PROJECT_ROOT / "prompts" / "user_prompt.txt"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
REQUIRED_TOP_LEVEL_KEYS = (
    "window",
    "window_summary",
    "scene",
    "entities",
    "actions",
    "state_changes",
    "observed_results",
    "uncertainties",
)


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    total_frames: int
    duration_seconds: float
    width: int
    height: int


@dataclass(frozen=True)
class SampledFrame:
    timestamp_seconds: float
    frame_index: int
    image_bgr: Any


def parse_time_string(value: str) -> float:
    """解析时间字符串为秒。支持纯秒数、MM:SS、HH:MM:SS。"""
    value = value.strip()
    if ":" not in value:
        return float(value)

    parts = value.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
    raise ValueError(f"无法解析时间格式：{value}，请使用秒数、MM:SS 或 HH:MM:SS。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用本地视频滑动窗口模拟千问视觉流式分析。"
    )
    parser.add_argument("--video", required=True, help="本地 MP4/AVI/MOV 视频路径")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML 配置路径",
    )
    parser.add_argument("--output-dir", default=None, help="自定义输出目录")
    parser.add_argument(
        "--start-time",
        type=parse_time_string,
        default=0.0,
        help="分析起始时间，支持秒数、MM:SS、HH:MM:SS（默认 0）",
    )
    parser.add_argument(
        "--end-time",
        type=parse_time_string,
        default=None,
        help="分析结束时间，支持秒数、MM:SS、HH:MM:SS（默认视频结束）",
    )
    parser.add_argument(
        "--start-window",
        type=int,
        default=0,
        help="从第几个窗口开始处理（0-based，默认 0）",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="最多处理多少个窗口，适合先小规模测试 API 费用",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只验证视频切窗和抽帧，不调用 API",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="按视频时间等待后再处理窗口；默认快速离线模拟",
    )
    parser.add_argument(
        "--no-state",
        action="store_true",
        help="不向下一窗口传递上一窗口压缩状态",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    for section in ("video", "api", "runtime", "video_metadata"):
        config.setdefault(section, {})
    return config


def seconds_to_timestamp(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def inspect_video(cap: cv2.VideoCapture) -> VideoInfo:
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or total_frames <= 0:
        raise RuntimeError("无法读取视频 FPS 或总帧数，视频可能损坏或编码不受支持。")
    return VideoInfo(
        fps=fps,
        total_frames=total_frames,
        duration_seconds=total_frames / fps,
        width=width,
        height=height,
    )


def build_windows(
    start_seconds: float,
    end_seconds: float,
    window_seconds: float,
    stride_seconds: float,
) -> list[tuple[float, float]]:
    if window_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("window_seconds 和 stride_seconds 必须大于 0。")
    if start_seconds >= end_seconds:
        raise ValueError("end_seconds 必须大于 start_seconds。")

    segment_duration = end_seconds - start_seconds
    if segment_duration <= window_seconds:
        return [(start_seconds, end_seconds)]

    windows: list[tuple[float, float]] = []
    current_end = start_seconds + window_seconds
    while current_end <= end_seconds + 1e-6:
        windows.append(
            (max(start_seconds, current_end - window_seconds), min(current_end, end_seconds))
        )
        current_end += stride_seconds

    # 捕获末尾不足一个 stride 的内容，并且仍然只读取当前 end 之前的帧。
    if windows and end_seconds - windows[-1][1] > 0.05:
        windows.append(
            (max(start_seconds, end_seconds - window_seconds), end_seconds)
        )
    return windows


def sample_window(
    cap: cv2.VideoCapture,
    info: VideoInfo,
    start_seconds: float,
    end_seconds: float,
    sample_fps: float,
    min_frames: int,
    max_frames: int,
) -> list[SampledFrame]:
    duration = end_seconds - start_seconds
    if duration <= 0:
        raise ValueError("窗口持续时间必须大于 0。")

    requested = int(math.ceil(duration * sample_fps))
    frame_count = min(max_frames, max(min_frames, requested))
    sampled: list[SampledFrame] = []
    seen_indices: set[int] = set()

    # 取每个等长时间单元的中点，保证所有采样时间严格小于 end_seconds。
    for index in range(frame_count):
        timestamp = start_seconds + (index + 0.5) * duration / frame_count
        timestamp = min(timestamp, end_seconds - 1e-6)
        frame_index = min(
            info.total_frames - 1,
            max(0, int(timestamp * info.fps)),
        )
        if frame_index in seen_indices:
            continue
        seen_indices.add(frame_index)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
        if not success or frame is None:
            continue
        sampled.append(
            SampledFrame(
                timestamp_seconds=frame_index / info.fps,
                frame_index=frame_index,
                image_bgr=frame,
            )
        )

    if len(sampled) < min_frames:
        raise RuntimeError(
            f"窗口只成功读取 {len(sampled)} 帧，少于 min_frames={min_frames}。"
        )
    return sampled


def encode_frame_as_data_url(
    frame: Any,
    max_side: int,
    jpeg_quality: int,
) -> str:
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest > max_side:
        scale = max_side / longest
        frame = cv2.resize(
            frame,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
    )
    if not success:
        raise RuntimeError("JPEG 编码失败。")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def response_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif hasattr(item, "text") and isinstance(item.text, str):
                parts.append(item.text)
        return "".join(parts)
    return str(content)


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("模型输出不是 JSON 对象。")
    return parsed


def normalize_analysis(
    analysis: dict[str, Any],
    start_seconds: float,
    end_seconds: float,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    defaults: dict[str, Any] = {
        "window": {},
        "window_summary": "",
        "scene": {
            "location": "unknown",
            "camera_change": False,
            "visibility": "unknown",
        },
        "entities": [],
        "actions": [],
        "state_changes": [],
        "observed_results": [],
        "uncertainties": [],
    }

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in analysis:
            warnings.append(f"模型缺少顶层字段：{key}")
            analysis[key] = defaults[key]

    for key in ("entities", "actions", "state_changes", "observed_results", "uncertainties"):
        if not isinstance(analysis.get(key), list):
            warnings.append(f"字段 {key} 不是数组，已重置为空数组")
            analysis[key] = []

    # 时间以程序的因果窗口为准，避免模型自行改写时间。
    analysis["window"] = {
        "start_time": seconds_to_timestamp(start_seconds),
        "end_time": seconds_to_timestamp(end_seconds),
    }
    return analysis, warnings


def compress_state(analysis: dict[str, Any]) -> dict[str, Any]:
    entities: dict[str, Any] = {}
    for entity in analysis.get("entities", []):
        if not isinstance(entity, dict):
            continue
        entity_id = entity.get("entity_id")
        if not isinstance(entity_id, str) or not entity_id:
            continue
        entities[entity_id] = {
            "type": entity.get("type", "unknown"),
            "name": entity.get("name", "unknown"),
            "attributes": entity.get("attributes", {}),
            "confidence": entity.get("confidence", "unknown"),
        }

    active_actions = []
    for action in analysis.get("actions", []):
        if isinstance(action, dict) and action.get("status") in {"started", "ongoing"}:
            active_actions.append(action)

    confirmed_changes = []
    for change in analysis.get("state_changes", []):
        if isinstance(change, dict) and change.get("confidence") in {"high", "medium"}:
            confirmed_changes.append(change)

    return {
        "entities": entities,
        "active_actions": active_actions[-10:],
        "last_confirmed_changes": confirmed_changes[-8:],
    }


class QwenWindowAnalyzer:
    def __init__(
        self,
        model: str,
        max_tokens: int,
        timeout_seconds: float,
        retries: int,
        system_prompt: str,
        user_prompt_template: str,
    ) -> None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "未找到 DASHSCOPE_API_KEY。请复制 .env.example 为 .env 并填写 API Key。"
            )
        base_url = os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL)
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
        )
        self.model = os.getenv("QWEN_MODEL", model)
        self.max_tokens = max_tokens
        self.retries = max(0, retries)
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

    def analyze(
        self,
        frame_urls: list[str],
        start_seconds: float,
        end_seconds: float,
        sampled_fps: float,
        metadata: dict[str, Any],
        previous_state: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        user_prompt = self.user_prompt_template.format(
            video_name=metadata.get("video_name", "unknown"),
            video_category=metadata.get("video_category", "unknown"),
            task_background=metadata.get("task_background", "unknown"),
            window_start=seconds_to_timestamp(start_seconds),
            window_end=seconds_to_timestamp(end_seconds),
            frame_count=len(frame_urls),
            sample_fps=sampled_fps,
            previous_state_json=json.dumps(
                previous_state,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if previous_state is not None
            else "null",
        )

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                started = time.perf_counter()
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": self.system_prompt}],
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "video",
                                    "video": frame_urls,
                                    "fps": sampled_fps,
                                },
                                {"type": "text", "text": user_prompt},
                            ],
                        },
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                    max_tokens=self.max_tokens,
                )
                latency = time.perf_counter() - started
                text = response_content_to_text(completion.choices[0].message.content)
                analysis = parse_json_object(text)
                usage = completion.usage.model_dump() if completion.usage else None
                meta = {
                    "model": completion.model or self.model,
                    "request_id": completion.id,
                    "latency_seconds": round(latency, 3),
                    "usage": usage,
                    "raw_response": text,
                }
                return analysis, meta
            except Exception as exc:  # API、网络和 JSON 解析统一重试
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(min(2 ** attempt, 8))
        assert last_error is not None
        raise last_error


def save_sampled_frames(
    output_dir: Path,
    window_index: int,
    sampled_frames: list[SampledFrame],
) -> None:
    frame_dir = output_dir / "sampled_frames" / f"window_{window_index:04d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for item in sampled_frames:
        filename = f"{item.timestamp_seconds:010.3f}s_f{item.frame_index:08d}.jpg"
        cv2.imwrite(str(frame_dir / filename), item.image_bgr)


def main() -> int:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    video_path = Path(args.video).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    if not video_path.exists():
        print(f"视频不存在：{video_path}", file=sys.stderr)
        return 2
    if not config_path.exists():
        print(f"配置不存在：{config_path}", file=sys.stderr)
        return 2

    config = load_config(config_path)
    video_cfg = config["video"]
    api_cfg = config["api"]
    runtime_cfg = config["runtime"]
    metadata = config["video_metadata"]

    window_seconds = float(video_cfg.get("window_seconds", 6.0))
    stride_seconds = float(video_cfg.get("stride_seconds", 3.0))
    sample_fps = float(video_cfg.get("sample_fps", 1.0))
    min_frames = int(video_cfg.get("min_frames", 4))
    max_frames = int(video_cfg.get("max_frames", 12))
    if min_frames < 1 or max_frames < min_frames:
        raise ValueError("需要满足 1 <= min_frames <= max_frames。")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"OpenCV 无法打开视频：{video_path}", file=sys.stderr)
        return 2

    try:
        info = inspect_video(cap)

        segment_start = max(0.0, float(args.start_time))
        segment_end = (
            float(args.end_time) if args.end_time is not None else info.duration_seconds
        )
        segment_end = min(segment_end, info.duration_seconds)
        if segment_start >= segment_end:
            print(
                f"无效的时间范围：{segment_start}s -> {segment_end}s",
                file=sys.stderr,
            )
            return 2

        time_suffix = ""
        if segment_start > 0.0 or segment_end < info.duration_seconds:
            time_suffix = f"_T{int(segment_start)}-{int(segment_end)}"

        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else PROJECT_ROOT
            / "outputs"
            / f"{video_path.stem}{time_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        windows = build_windows(segment_start, segment_end, window_seconds, stride_seconds)
        start_window = max(0, int(args.start_window))
        if start_window > 0:
            windows = windows[start_window:]
        configured_max = runtime_cfg.get("max_windows")
        max_windows = args.max_windows if args.max_windows is not None else configured_max
        if max_windows is not None:
            windows = windows[: int(max_windows)]

        realtime = bool(args.realtime or runtime_cfg.get("realtime", False))
        carry_state = bool(runtime_cfg.get("carry_previous_state", True)) and not args.no_state
        save_frames = bool(runtime_cfg.get("save_sampled_frames", False))

        analyzer: QwenWindowAnalyzer | None = None
        if not args.dry_run:
            analyzer = QwenWindowAnalyzer(
                model=str(api_cfg.get("model", "qwen3.7-plus")),
                max_tokens=int(api_cfg.get("max_tokens", 1800)),
                timeout_seconds=float(api_cfg.get("timeout_seconds", 120)),
                retries=int(api_cfg.get("retries", 2)),
                system_prompt=read_text(DEFAULT_SYSTEM_PROMPT_PATH),
                user_prompt_template=read_text(DEFAULT_USER_PROMPT_PATH),
            )

        run_meta = {
            "video_path": str(video_path),
            "video_info": {
                "fps": info.fps,
                "total_frames": info.total_frames,
                "duration_seconds": round(info.duration_seconds, 3),
                "width": info.width,
                "height": info.height,
            },
            "analysis_range": {
                "start_seconds": round(segment_start, 3),
                "end_seconds": round(segment_end, 3),
                "start_time": seconds_to_timestamp(segment_start),
                "end_time": seconds_to_timestamp(segment_end),
            },
            "window_count": len(windows),
            "dry_run": args.dry_run,
            "realtime": realtime,
            "carry_previous_state": carry_state,
            "config": config,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        (output_dir / "run_meta.json").write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        jsonl_path = output_dir / "windows.jsonl"
        pretty_path = output_dir / "windows_pretty.jsonl"
        previous_state: dict[str, Any] | None = None
        wall_start = time.monotonic()

        print(f"视频：{video_path.name}")
        print(
            f"分析区间：{seconds_to_timestamp(segment_start)} -> "
            f"{seconds_to_timestamp(segment_end)}"
        )
        print(f"原始时长：{info.duration_seconds:.3f}s")
        print(f"窗口数：{len(windows)}")
        print(f"输出：{output_dir}")

        with (
            jsonl_path.open("w", encoding="utf-8") as output_file,
            pretty_path.open("w", encoding="utf-8") as pretty_file,
        ):
            for window_index, (start_seconds, end_seconds) in enumerate(windows):
                if realtime:
                    # 在真实时间到达窗口终点后才允许处理，模拟不能提前看到未来。
                    target_wall_time = wall_start + end_seconds
                    remaining = target_wall_time - time.monotonic()
                    if remaining > 0:
                        time.sleep(remaining)

                print(
                    f"[{window_index + 1}/{len(windows)}] "
                    f"{seconds_to_timestamp(start_seconds)} -> "
                    f"{seconds_to_timestamp(end_seconds)}"
                )
                record: dict[str, Any] = {
                    "window_index": window_index,
                    "window_start_seconds": round(start_seconds, 3),
                    "window_end_seconds": round(end_seconds, 3),
                    "window_start": seconds_to_timestamp(start_seconds),
                    "window_end": seconds_to_timestamp(end_seconds),
                    "causal_constraint": f"sampled_frame_time < {end_seconds:.6f}s",
                }

                try:
                    sampled = sample_window(
                        cap=cap,
                        info=info,
                        start_seconds=start_seconds,
                        end_seconds=end_seconds,
                        sample_fps=sample_fps,
                        min_frames=min_frames,
                        max_frames=max_frames,
                    )
                    timestamps = [item.timestamp_seconds for item in sampled]
                    effective_fps = len(sampled) / (end_seconds - start_seconds)
                    record["sampled_frame_indices"] = [item.frame_index for item in sampled]
                    record["sampled_timestamps_seconds"] = [round(value, 3) for value in timestamps]
                    record["effective_sample_fps"] = round(effective_fps, 3)

                    if save_frames:
                        save_sampled_frames(output_dir, window_index, sampled)

                    if args.dry_run:
                        record["status"] = "dry_run_ok"
                    else:
                        assert analyzer is not None
                        frame_urls = [
                            encode_frame_as_data_url(
                                item.image_bgr,
                                max_side=int(video_cfg.get("max_image_side", 768)),
                                jpeg_quality=int(video_cfg.get("jpeg_quality", 80)),
                            )
                            for item in sampled
                        ]
                        analysis, api_meta = analyzer.analyze(
                            frame_urls=frame_urls,
                            start_seconds=start_seconds,
                            end_seconds=end_seconds,
                            sampled_fps=effective_fps,
                            metadata=metadata,
                            previous_state=previous_state if carry_state else None,
                        )
                        analysis, warnings = normalize_analysis(
                            analysis,
                            start_seconds,
                            end_seconds,
                        )
                        record["status"] = "ok"
                        record["analysis"] = analysis
                        record["schema_warnings"] = warnings
                        record["api"] = {
                            key: value
                            for key, value in api_meta.items()
                            if key != "raw_response"
                        }
                        if warnings:
                            record["raw_response"] = api_meta["raw_response"]

                        if carry_state:
                            previous_state = compress_state(analysis)
                            record["next_previous_state"] = previous_state

                        summary = analysis.get("window_summary", "")
                        print(f"  {summary}")
                        print(f"  API latency: {api_meta['latency_seconds']}s")

                except Exception as exc:
                    record["status"] = "error"
                    record["error"] = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                    print(f"  ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)

                compact = json.dumps(record, ensure_ascii=False)
                output_file.write(compact + "\n")
                output_file.flush()
                pretty_file.write(
                    json.dumps(record, ensure_ascii=False, indent=2) + "\n\n"
                )
                pretty_file.flush()

        print(f"完成。逐窗口结果：{jsonl_path}")
        return 0
    finally:
        cap.release()


if __name__ == "__main__":
    raise SystemExit(main())
