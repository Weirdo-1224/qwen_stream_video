"""合并多个分段的 windows.jsonl 结果为一个完整序列。"""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def find_run_dirs(stem_prefix: str) -> list[Path]:
    """按时间顺序找到所有匹配的输出目录。"""
    dirs = [
        p
        for p in OUTPUTS_DIR.iterdir()
        if p.is_dir() and p.name.startswith(stem_prefix)
    ]
    dirs.sort(key=lambda p: p.name)
    return dirs


def merge_run_dirs(dirs: list[Path], merged_name: str) -> Path:
    if not dirs:
        raise RuntimeError("未找到可合并的输出目录。")

    merged_dir = OUTPUTS_DIR / merged_name
    merged_dir.mkdir(parents=True, exist_ok=True)

    compact_records: list[dict] = []
    pretty_records: list[dict] = []
    metas: list[dict] = []

    index_offset = 0
    for run_dir in dirs:
        jsonl_path = run_dir / "windows.jsonl"
        if not jsonl_path.exists():
            continue

        meta_path = run_dir / "run_meta.json"
        if meta_path.exists():
            metas.append(json.loads(meta_path.read_text(encoding="utf-8")))

        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                # 修正 window_index 为合并后的全局索引
                record["window_index"] = index_offset
                compact_records.append(record)
                pretty_records.append(record)
                index_offset += 1

    # 写入标准 JSONL
    with (merged_dir / "windows.jsonl").open("w", encoding="utf-8") as f:
        for record in compact_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 写入格式化 JSONL
    with (merged_dir / "windows_pretty.jsonl").open("w", encoding="utf-8") as f:
        for record in pretty_records:
            f.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n\n")

    # 写入合并后的 meta
    merged_meta = {
        "merged_from": [str(d) for d in dirs],
        "merged_at": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "total_windows": len(compact_records),
        "source_metas": metas,
    }
    with (merged_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(merged_meta, f, ensure_ascii=False, indent=2)

    # 可选：复制第一个 source 的 sampled_frames 等目录
    for run_dir in dirs:
        frames_dir = run_dir / "sampled_frames"
        if frames_dir.exists():
            dest = merged_dir / "sampled_frames"
            if not dest.exists():
                shutil.copytree(frames_dir, dest)

    return merged_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="合并分段 windows.jsonl 结果")
    parser.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        help="要合并的输出目录名（在 outputs/ 下），按顺序传入",
    )
    parser.add_argument("--output", default=None, help="合并后的目录名（默认自动生成）")
    args = parser.parse_args()

    dirs = [OUTPUTS_DIR / d for d in args.dirs]
    dirs = [d for d in dirs if d.exists()]
    print(f"将合并 {len(dirs)} 个目录：")
    for d in dirs:
        print(f"  - {d.name}")

    default_name = f"{dirs[0].stem}_merged_{datetime.now(tz=timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output_name = args.output or default_name
    merged_dir = merge_run_dirs(dirs, output_name)
    print(f"合并完成：{merged_dir}")
    print(f"总窗口数：{len(list((merged_dir / 'windows.jsonl').open(encoding='utf-8')))}")
