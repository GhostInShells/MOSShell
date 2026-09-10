# Dolores Ego Preset + Nibble — 非 ego session 可用 & agent 级 tool

> 2026-09-10 与人类架构师对齐。今天落地两件事, 都源自「模型每一步吃一丁点」这个起点动机:
> ① ego 有专属 preset, 非 ego session 不再被 perStep 锁冻结; ② moss_think 改成 agent 级 tool,
> 经 `exec.agent` 咬 per-agent 的 model selection, 不再模块单例 / 整餐推 model。
> 关联: `ghost-prototype-dolores` FEATURE.md。推翻此前 `doloresSelectionRef` 方案。

## 背景: 卡住的两个点

1. **非 ego session 不可用** — `apply_ego_agent` 按 preset 识别, 而 ego 用默认 `standard`, 于是任何
   standard 会话都被套 ego 面 + perStep 锁 reject 冻结。
2. **「定义 agent 级别的 tool」卡了一周** — 误以为 `defineTool` 的 `execute` 拿不到 agentCtx, 所以
   「模型改自己的 effort」被塞成模块级 `doloresSelectionRef` + thinking/enter 整餐推 model。

## 决策 1: ego 专属 preset (非 ego session 可用)

- dsh preset **没有继承机制** (authoring 是 copy-only, shipped 四个 preset 各自独立文件)。
- 唯一组合原语是通用 `Include` row, 但嵌套 Include 有 write-back 坑 — 拆树时 loader 会把当前 data 写回
  被 include 的文件; dsh 自己把 preset 的 Include subclass 成 `PresetTree` 并 override `write(){}` 正是
  为防这个。指向 shipped standard 会截断它。
- 结论: **plugin 侧 boot 逐字再生**。`ensureEgoPreset(ctx)` 读 shipped `standard` 的 `agent.cordis.yml`
  原文, 写到 `<DSH_HOME>/.agent-presets/dolores-ego/agent.cordis.yml`。目录名即 preset id, 内容逐字
  (不 YAML round-trip, 保 `!!js`)。每次 boot 无条件重写 → 跟 dsh 升级 / 后续 delta 自动同步, 不 stale。
- ego/create 用 `DOLORES_EGO_PRESET='dolores-ego'` 挂 `meta.agentPreset` + `mount`; session-start 识别
  gate 同源。非 ego 的 `standard` 会话完全不被 `apply_ego_agent` 碰 → 正常跑。

## 决策 2: agent 级 tool + per-agent selection (nibble)

- **关键事实 (推翻陷阱)**: `defineTool` 的 `execute(args, exec)` 里 `exec.agent` 就是调用方 agent
  (`ToolExecutionInput.agent`, 由 agent loop 设置), `exec.agent.ctx` 就是 scoped ctx。tool 运行中够得到
  agent 状态, 根本不需要「把 agentCtx 传进 tool」。
- model selection 改 per-agent: `egoSelections = WeakMap<Agent, ModelSelectionRef>`; `current` getter 走
  canonical 链 `picked → requestHeader()?.config → agentDefaultModel.currentSelection()`, setter 设 `picked`。
  装在 `apply_ego_agent` (session-start), `installModelSelection(agent.ctx, ...)`。
- `moss_think` execute 用 `exec.agent` → `egoSelections.get(agent)` → 只咬 `reasoningEffort`。
- 模型/effort 权威始终在 canonical (settings `agent-default-model` / `requestHeader`), 改 effort 由
  `agent/request` 应用并落 `request/header` 日志, 界面自然同步 — 「每一步吃一丁点」。
- 删掉: 模块级 `doloresSelectionRef`、thinking/enter 的 `model` 载荷、`_ego.py` 的 `_model_config()`。
  effort 不再每次 thinking/enter 自动重置 (nibble 后持续到下一次 moss_think 或 agent dispose)。

## Open seams

- 老 ego session 被 resume 仍是 phase 2 (reentrant, 见 dolores-reentrant-ego-session.md), 今天只解了
  「非 ego session 可用」。
- effort 持久化语义变化 (不再 enter 重置) 待实测确认。
