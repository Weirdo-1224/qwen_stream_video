"""解析并汇总合并后的 windows.jsonl 结果。"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="解析并汇总合并后的 windows.jsonl 结果")
    parser.add_argument(
        "--merged-dir",
        required=True,
        help="合并后的输出目录名（在 outputs/ 下）",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    merged_dir = project_root / "outputs" / args.merged_dir
    jsonl_path = merged_dir / "windows.jsonl"
    report_path = merged_dir / "analysis_report.txt"

    records = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    lines = []
    lines.append(f"总窗口数：{len(records)}\n")

    # 1. 状态统计
    status_counts = Counter(r["status"] for r in records)
    lines.append("=== 状态统计 ===")
    for status, count in status_counts.items():
        lines.append(f"  {status}: {count}")
    lines.append("")

    # 2. 每个窗口的摘要
    lines.append("=== 逐窗口摘要 ===")
    for r in records:
        idx = r["window_index"]
        start = r["window_start"]
        end = r["window_end"]
        status = r["status"]
        if status == "error":
            err = r.get("error", {})
            lines.append(f"窗口 {idx:2d} [{start} -> {end}] ERROR: {err.get('type')} - {err.get('message')}")
            continue
        summary = r["analysis"].get("window_summary", "")
        lines.append(f"窗口 {idx:2d} [{start} -> {end}] {summary}")
    lines.append("")

    # 3. 实体统计
    entity_counter = Counter()
    entity_details = defaultdict(list)
    for r in records:
        if r["status"] != "ok":
            continue
        for entity in r["analysis"].get("entities", []):
            eid = entity.get("entity_id", "unknown")
            name = entity.get("name", "unknown")
            key = f"{eid} ({name})"
            entity_counter[key] += 1
            entity_details[key].append((r["window_index"], entity.get("attributes", {}).get("visible_state", "")))

    lines.append("=== 实体出现频次 ===")
    for key, count in entity_counter.most_common():
        lines.append(f"  {key}: {count} 次")
    lines.append("")

    # 4. 动作统计
    action_counter = Counter()
    action_timeline = defaultdict(list)
    for r in records:
        if r["status"] != "ok":
            continue
        for action in r["analysis"].get("actions", []):
            desc = f"{action.get('actor_id')} {action.get('action')} {action.get('object_id')} [{action.get('status')}]"
            action_counter[desc] += 1
            action_timeline[desc].append(r["window_index"])

    lines.append("=== 动作统计 ===")
    for desc, count in action_counter.most_common():
        windows = action_timeline[desc]
        lines.append(f"  {desc}: {count} 次 (窗口 {windows})")
    lines.append("")

    # 5. 状态变化
    lines.append("=== 状态变化 ===")
    has_changes = False
    for r in records:
        if r["status"] != "ok":
            continue
        for change in r["analysis"].get("state_changes", []):
            has_changes = True
            lines.append(f"  窗口 {r['window_index']} [{r['window_start']} -> {r['window_end']}]: "
                  f"{change.get('entity_id')} {change.get('attribute')} "
                  f"{change.get('before')} -> {change.get('after')}")
    if not has_changes:
        lines.append("  无明确状态变化")
    lines.append("")

    # 6. 观察结果
    lines.append("=== 观察到的结果 ===")
    has_results = False
    for r in records:
        if r["status"] != "ok":
            continue
        for result in r["analysis"].get("observed_results", []):
            has_results = True
            lines.append(f"  窗口 {r['window_index']} [{r['window_start']} -> {r['window_end']}]: "
                  f"{result.get('object_id')} {result.get('result_type')} - {result.get('result')}")
    if not has_results:
        lines.append("  无明确观察结果")
    lines.append("")

    # 7. uncertainties 汇总
    lines.append("=== 主要不确定性 ===")
    for r in records:
        if r["status"] != "ok":
            continue
        for uncertainty in r["analysis"].get("uncertainties", []):
            target = uncertainty.get("target", "")
            reason = uncertainty.get("reason", "")
            if "柜内" in target or "操作" in target or "设备" in target:
                lines.append(f"  窗口 {r['window_index']}: {target} | {reason}")
                break
    lines.append("")

    # 8. 关键时间线
    lines.append("=== 关键时间线 ===")
    for r in records:
        if r["status"] != "ok":
            continue
        summary = r["analysis"].get("window_summary", "")
        # 提取关键变化
        if any(kw in summary for kw in ["绝缘", "电缆", "剥开", "接入", "工具", "操作"]):
            lines.append(f"  {r['window_start']} -> {r['window_end']}: {summary}")
    lines.append("")

    # 写入文件
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"解析报告已生成：{report_path}")


if __name__ == "__main__":
    main()
