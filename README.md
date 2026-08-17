# Qwen Stream Video：Stage 2 State Engine

项目按本地视频的因果滑动窗口运行。模型只输出当前窗口的局部视觉事实；程序侧再以确定性代码维护场景、实体、动作和属性状态。

## 架构边界

```text
视频帧
  -> Observation Generator（Qwen / Fake client）
  -> Schema 2.0 + 词表规范化
  -> SceneTracker
  -> EntityResolver / EntityRegistry
  -> ActionTracker
  -> TransitionEngine
  -> StateReducer（原子提交）
  -> events / deltas / snapshots / final_state
```

Observation 不包含正式 `global_entity_id`、动作生命周期或完整 GlobalState。Global Entity ID 只在一次运行内确定，不代表跨视频的永久真实身份。

## 安装与运行

```bash
pip install -e .
pip install -e ".[dev]"

python run.py --video videos/demo.mp4 --config configs/base.yaml
qwen-stream-video --video videos/demo.mp4 --config configs/base.yaml
```

离线检查和测试：

```bash
python run.py --video videos/demo.mp4 --validate-only
python run.py --video videos/demo.mp4 --dry-run --max-windows 3
python -m pytest -q
ruff check .
```

`--state` 启用状态维护，`--no-state` 只生成局部 Observation；`--warmup-windows`、`--snapshot-interval` 和 `--context-policy` 覆盖 YAML 配置。默认 `context_policy=visual_only`，不会把视频文件名、故障名称或具体回路名称发送给模型。

## Observation Schema 2.0

窗口包含 `[start, end)` 和程序计算的 `commit_start_seconds`：

```json
{
  "schema_version": "2.0",
  "window": {
    "global_index": 3,
    "start_seconds": 9.0,
    "commit_start_seconds": 12.0,
    "end_seconds": 15.0
  },
  "scene": {
    "camera_change": false,
    "view_type": "medium",
    "scene_visibility": "clear",
    "target_visibility": "clear",
    "continuity_hint": "continuous"
  },
  "entities": [],
  "actions": [],
  "attribute_observations": [],
  "relations": [],
  "uncertainties": []
}
```

`candidate_global_id` 只是 ContextBuilder 给出的低权重提示。动作 OOV 会变为 `action_type=other` 并保留 `raw_action_type`；视觉无法判断仍使用 `unknown`。属性使用 canonical key，例如 `door.state` 和 `indicator.energy.lit`，同时保留 raw 字段。

## Context / Commit 与 State Engine

默认 6 秒窗口、3 秒步长的提交范围是 `[0,6)`、`[6,9)`、`[9,12)`。重叠部分可以用于理解和延续已有事实，但不能创建新的动作或属性转移。非零窗口运行可通过 warmup 建立上下文；缺少前置窗口时 `run_meta.json` 会标记 `cold_start`。

EntityRegistry 分配 `person_0001`、`device_0001` 等运行内 ID，模糊匹配使用 `temp_*`，不会因特写删除历史实体。ActionTracker 维护 `started → ongoing → possible_ended → ended`，instant 动作和镜头切换分别处理。TransitionEngine 将初始值、pending、冲突和正式 `before → after` 转移区分开；初次可见属性不会产生伪转移。

## 状态输出

启用状态后，运行目录包含：

```text
observations.jsonl                 # 模型局部观察，语义保持不变
normalization_warnings.jsonl
entity_resolutions.jsonl
state_events.jsonl
state_deltas.jsonl
state_snapshots.jsonl
state_errors.jsonl
final_state.json
artifacts/prompts/                 # 实际使用的 Prompt 正文
artifacts/schemas/                 # Observation JSON Schema
artifacts/vocabularies/            # 实际词表正文
```

`final_state.json` 使用临时文件、flush/fsync 和原子 rename。状态事件、实体解析和属性转移都带窗口、证据样本及程序生成的 ID。

## Observation Replay

Replay 不调用 Qwen API：

```bash
qwen-stream-video \
  --replay-observations outputs/<run_id>/observations.jsonl \
  --config configs/base.yaml \
  --output-dir outputs/replay_run
```

Replay 支持合法 Schema 1.0 和 2.0。Schema 1.0 通过 `ObservationV1Adapter` 迁移字段，不伪造 candidate ID 或证据。相同输入、配置和窗口序列会得到相同的 `state_events.jsonl` 与 `final_state.json`。

## 质量分析与回归

```bash
python scripts/evaluate_state_run.py outputs/<run_id>
python scripts/evaluate_state_run.py outputs/<run_id> --json
```

脚本检查引用、ID 唯一性、快照与最终状态一致性、证据覆盖率、OOV、模糊实体、重复动作候选和无支持转移。Golden Fixture 位于 `tests/golden/`，不包含真实视频、Base64 或敏感信息。

## 配置

默认配置在 `configs/base.yaml`。配置模型严格禁止未知字段，优先级为：

```text
CLI > 环境变量 > YAML > 代码默认值
```

新增 State、SceneTracker、EntityRegistry、ActionTracker、TransitionEngine 和 Context 配置均会在启动前校验阈值关系。API Key 只能从 `.env` 或环境变量读取，不写入输出正文。

## 当前限制

当前仍是本地 MP4 的顺序模拟流式处理，不保证真实时间性能；尚未支持 RTSP、摄像头输入、并行推理、latest-window-only 调度、检测/ReID、违规判断、报警或多 Agent。
