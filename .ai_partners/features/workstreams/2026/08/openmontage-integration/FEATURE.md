---
title: Openmontage Integration
status: draft
priority: P2
created: 2026-08-12
updated: 2026-08-12
depends: []
milestone: Aug-2H
description: >-
  开箱集成 OpenMontage (agentic 视频生产系统). 调研完成、设计落定、目标确认;
  排入 8 月下半开发计划. 单一有状态 conductor channel + 大量动用 JobSupervisor 做后台任务管理
---

# Openmontage Integration

> Use `moss features set-status openmontage-integration <status> -m "note"` to update state.

## Motivation

OpenMontage 是第一个开源 agentic 视频生产系统 (AGPLv3, github.com/calesthio/OpenMontage)。
它不是"文生视频", 而是把完整制作管线交给 agent: 研究 → 提案 → 脚本 → 场景规划 → 素材生成 →
剪辑 → 合成, 全程审批门 + 预算管控 + 决策审计。100+ Python 工具 (BaseTool), 12 条 pipeline,
700+ 知识文件。

它和 MOSS 是同一种信念的两个实现 — **"model 读指令驱动工具"**。OpenMontage 的 agent-orchestrator
角色正是 MOSS 的 Ghost。开箱集成后, Ghost 获得完整视频生产能力: explainer、纪录片 montage、
动画短片、产品 teaser。

时机: 该能力是长期规划 (人类工程师"一直打算做"), 现在有了成熟开源实现, 集成成本远低于自建。
600+ 知识文件超出最初预期, 但其中大部分是 markdown 指令而非实现代码 — 迁移的"要搬的东西"是文本知识。

## 调研结论 (OpenMontage 内部)

- **架构**: instruction-driven, 无运行时 orchestrator。Python 只做 tools + persistence; 智能全在
  YAML pipeline manifest + Markdown skill。状态机 `research→proposal→script→scene_plan→assets→edit→compose→publish`
- **工具**: BaseTool 契约 (`name/capability/provider/runtime/dependencies/input_schema/fallback_tools/agent_skills`),
  `execute(inputs)->ToolResult`。registry 自动发现 (pkgutil), 零注册。selector 模式 7 维加权评分路由
- **治理**: checkpoint (schema 校验 + gate 强制 human_approved), budget (estimate→reserve→reconcile, cap 模式),
  decision_log (append-only), quality gates (pre-compose validation / post-render self-review / slideshow risk)
- **观测**: **全后台任务** (tools/ 无任何 asyncio, 秒到分钟级)。进度信号在磁盘:
  `events.jsonl` (tool start/finish/error + cost), checkpoint `partial_progress`, `cost_snapshot`
- **Backlot**: 本地 FastAPI 看板 (backlot/), **纯只读投影** — 扫过全部端点, 零 mutation。
  从 `projects/<id>/` 磁盘文件推导状态 (project.json / checkpoint_*.json / artifacts / events.jsonl / renders)。
  控制它只有文件 + 2 个 CLI (`python -m backlot open [id]` / `serve --port 4750`)
- **可作为 library**: 可 import, `registry.discover()` 后任意 `tool.execute()` 可调。但 pipeline 编排
  无 `Pipeline.run()` — 需要 agent 读指令驱动

## Key Decisions

<!-- 下一位模型实例先读这里 -->

### D1. 不常驻进程 — 启动返回能力上下文, 工具按需执行

OpenMontage 无进程内共享状态 (registry 只是 import 期元数据, 状态全在磁盘 `projects/`)。
**不需要常驻 node 进程**。开箱形态: host 侧一个 channel, 启动时从 registry 生成 capability envelope
(哪些 pipeline / 哪些工具可用 / 哪些 API key 配好) 作为 instruction/help/context, refresh 时重生成。
工具调用通过 subprocess shell 到 OpenMontage 独立 venv (thin runner `tool_runner.py <tool> <inputs.json>`),
每次调用付 0.5–2s 导入费, 对分钟级任务可忽略。

- **Why**: 宿主 venv 不被 openai/fastapi/torch 污染; OpenMontage 保持独立 clone 可 `git pull` 升级;
  无常驻进程白占内存
- **升级路径**: 高频或跨机器需求出现时, channel 内部从 subprocess 换成常驻 node 进程内调用 —
  抽象边界留好 (channel 命令实现是唯一替换点)

### D2. 单一有状态 conductor channel — 不做 100+ 命令自动生成

**用 one channel 做有状态任务管理, 这是 MOSS 的优势** (StatefulChannel/PrimeChannel + module)。
`openmontage` 是一个 PrimeChannel, 按 module 分层:

- **ContextModule** — capability envelope (pipeline list / tool 可用性 / API key 状态), 挂
  instruction/help/context_messages, `on_refresh_meta` 重生成
- **JobsModule** — 任务编排核心: 包装 JobSupervisor, 暴露任务导向命令, 状态以 JobSnapshot 呈现
- **BoardModule** — 结构化观测: get_board_state / get_checkpoint / events tail → topic/signal
- **BacklotModule** — `open_board` 给人类起浏览器

命令形态 (**任务导向, 不是工具导向**):

- `start_production(brief, pipeline)` → production_id (init project + checkpoint)
- `run_task(tool, inputs, project_id)` → job_id (把单次工具调用包成 job)
- `list_productions()` / `get_production(id)` / `get_checkpoint(id, stage)`
- `list_jobs()` / `get_job(job_id)` → JobSnapshot
- `cancel(job_id)` / `resume(job_id)`
- `get_board_state(project_id)` / `open_board(project_id)`

- **Why**: 100+ 命令自动生成是"工具视图" — 模型要自己跟踪任务状态; conductor channel 是"任务视图" —
  模型对话一个总指挥, 状态由 channel 持有 (correlate 磁盘 checkpoint + jobs + events)。这是 MOSS 相比
  普通 MCP/工具封装的核心优势

### D3. 大量动用 matrix.jobs (JobSupervisor)

所有后台任务 (图生/视频生/渲染, 秒到分钟级) 都是 job:
`JobSpec(args=(venv_python, tool_runner, tool, inputs), times=1)`, 提交给 channel 持有的 JobSupervisor。
观测走 `JobSnapshot` (status/exit_code/stdout_tail), 控制走 `stop()/resume()/wait()`。

- 一次 production 的多个后台任务 → production 对应 channel 侧一个 ledger, 关联其 jobs
- job 完成 / events.jsonl 变化 → **上行 signal 推给 Ghost**, 构成自驱循环, 不轮询

### D4. Backlot 是纯投影 — Ghost 看结构化数据, Backlot 留给人

Backlot 零 mutation 端点, 不可被驱动。不"控制"它: 用同一数据源给 Ghost 结构化视图 (BoardModule),
人类要看的活板面用 `open_board` 起服务开浏览器。

### D5. MOSS 三向数据覆盖观测

| OpenMontage 信号 | MOSS 方向 |
|---|---|
| 长工具执行 | 中行 `CommandUtil.set_progress` |
| events.jsonl tail | 上行 Topic/signal (自驱) |
| ToolResult (cost_usd/artifacts) | 下行 `observe` → Re-Act |

### D6. OpenMontage 保持独立, 集成是薄适配层

位置 `nodes/tools/openmontage/` (或开箱直接挂 host channel): NODE.md + main.py (adapter) +
pyproject.toml (openmontage 本地路径依赖) + INSTALL.md + 独立 venv, `exec.command` 用 `.venv/bin/python`。
OpenMontage 独立 clone 可升级。

### D7. 环境与许可证

- 环境: Python 3.10+ / FFmpeg (必装) / Node 18+ (Remotion, 可选) / GPU (本地生成, 可选)
- 许可证: OpenMontage AGPLv3 — 需确认与 MOSS license 兼容 (AGPL 网络传染性), 标注待办, 不阻塞本地使用

## Implementation Notes

<!-- Gotchas, 非显然行为, 拒绝的替代方案 -->

- 摩擦点 (来自 node-migration trafilatura pilot): NODE.md instruction 不写 CTML; 独立 venv
  `exec.command` 用 `.venv/bin/python`; `CellNamePattern` 无连字符; ghoshell-moss 优先 `[matrix]` 依赖
- tool 的 `execute()` 是同步阻塞 — 必须在 executor / subprocess 里跑, 不能卡 MOSS event loop
- `tool_runner.py` 是薄层: import registry → discover → execute → 输出 ToolResult (JSON)
- events.jsonl tail → MOSS topic: 每 production 一个 topic, 事件结构化后 publish
- 前置: 环境探针 (ffmpeg/node/python 版本 + `registry.discover()` 跑通) 在骨架落地前做
- 拒绝过的替代: (a) 100+ tool 自动生成命令树 — 工具视图, 状态跟踪负担在模型; (b) 常驻 node 进程 —
  无进程内状态可保, 白占内存

## 参照

- OpenMontage 克隆: `/Users/BrightRed/Develop/github.com/OpenMontage`
- OpenMontage 关键文件: `AGENT_GUIDE.md` / `PROJECT_CONTEXT.md` / `docs/ARCHITECTURE.md` /
  `tools/base_tool.py` / `tools/tool_registry.py` / `lib/checkpoint.py` / `lib/scoring.py` /
  `lib/events.py` / `backlot/server.py` / `pipeline_defs/*.yaml`
- MOSS 侧: `StatefulChannel` (core/blueprint/states_channel.py), `JobSupervisor`
  (contracts/job_supervisor.py), channel_builder (core/blueprint/channel_builder.py),
  node 模板 (nodes/tools/trafilatura/)
