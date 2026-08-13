---
title: DSH Productization
status: draft
# priority: importance within the current stage (iteration cycle) — not development urgency
priority: P1
created: 2026-08-14
updated: 2026-08-14
depends: []
milestone: 0.1.0
description: >-
  将 DeepSeek Harness (dsh) 作为 MOSS 可驱动的外部 agent 面做产品化集成，
  候选取代 claude-code-in-moss。开箱暴露其 mode / 工具 / session / 权限面，
  不做应用实现。调研见 ghost-in-shell-discusses/2026-08-14_deepseek-harness-survey.md
---

# DSH Productization

> Use `moss features set-status dsh-productization <status> -m "note"` to update state.
> 本 feature 是 8 月命题：**候选取代 `claude-code-in-moss`**。尚未实际运行，以下判断均为预判方向，留待施工时验证。

## Motivation

DeepSeek Harness（`deepseek-ai/deepseek-harness`，MIT，2026-08-13 开发者预览）是一个
「Model + Harness = Agent」的 coding harness，对标 Codex / Claude Code。与
`claude-code-in-moss` 要桥接的 claude code 相比，dsh 有**更工程化的协议面**：

- 正式 Python SDK（`deepseek-harness`）+ TypeScript SDK + ACP 三套驱动协议；
- `session_id` + `session_root`（JSONL）做跨进程 resume，不靠 daemon；
- 「一切皆插件」（Cordis 内核），能力面可外部组合。

因此 dsh 是 MOSS 做「外部 agent 面产品化」的更强候选。本 workstream 的定位是**开箱不实现**：
把 dsh 的 mode / 工具 / session 轨迹 / 权限预设原样搬到 MOSS，暴露给 Ghost 驱动，
不替它写应用逻辑。

## Design Index

- 调研报告：`ghost-in-shell-discusses/2026-08-14_deepseek-harness-survey.md`（源码级，含完整机制与对比）
- 参考 feature：`claude-code-in-moss`（同类的「桥 Node harness 进 Python MOSS」先例）
- 集成关键面：`python/sdk`（Python SDK 源码）、`packages/acp`（权限机器裁决协议）

## Key Decisions

<!-- 以下为预判方向，非最终裁决。实际运行后由施工化身修正。 -->

### 集成路径（预判）

- **走 Python SDK，不走 CLI headless**。`dsh --profile headless "task"` 是一次性 print
  模式（无 resume、每次新建 agent）；loop + resume 只在 SDK 层
  （`DeepSeekHarness.run(input, { session_id })` → `RunResult`）。
- **Cell 作宿主**：Cell 生命周期 = dsh runtime 生命周期（start = spawn，stop = reap），
  把 dsh 的「spawn-per-instance」映射到 MOSS 的「cell 作为进程单元」。
- **session_root 落 MOSS 存储**：MOSS 管 `session_root` + `session_id`，跨进程 resume 由 MOSS 持有。
- **工程点**：SDK 是同步的（threading reader），MOSS 是 async，需 executor / 线程桥。

### 融合点（为后续施工实例提供的预判方向）

1. **权限**：dsh 权限 = 两个独立 knob（沙箱 `workspace-write`/`danger-full-access` +
   审批 `ask`/`never`），审批是可插拔 waterfall 应答者 seam。预判：MOSS 可把自己的 Ghost
   插成那个应答者（走 ACP `session/request_permission`，逐条 `allow_once`/`reject_once`），
   或走 SDK 预配置 preset。对应 MOSS「限制权限是为了授权」——沙箱 = 划边界，审批 = 敢放行。

2. **provider 协议化**：dsh 的 seam+inject ≈ MOSS 的 blueprint+IoC provider（都是「装线前
   协议化」）。但协议化对象不同——dsh 协议化**外围**（fs/process/sandbox/shell/subagent/llm
   可替换），MOSS 协议化**核心**（memento 是 contract，session 归属装线前由外部定）。预判：
   这是「能否让 Ghost 拥有 dsh 连续性」的分界，也是 feature scope 的上限。

3. **历史轨迹类交互方式**：dsh 的 `session_event_read/search/trace`（模型主动查自己历史，
   工具形态）≈ memento 的 `log/window/show/confluences`（历史组装进上下文，底物形态）。
   同一命题、不同交付。预判：这是两者最近的收敛点，值得持续对照——尤其 dsh 只有 fork
   无 confluent，memento 有 reference-confluent（图结构）。

4. **开箱能力面**：4 mode（standard/minimal/PTC·Code Mode/creator）+ 30+ 模型可见工具 +
   session 轨迹 + 权限预设。预判：GUI 集成直接暴露这些「开箱」面（会话栏 / 模式选择 /
   工具活动流 / 权限预设 / 自我查询入口），不重实现。

5. **`-p` 交互方式**：dsh 的 loop 原语是 SDK 的 `run(input, { session_id })`，语义是
   quiescence-based（`finalResponse` = 区间内最后一条 committed assistant 文本，非因果
   绑定 prompt），不是 turn-causal。预判：GUI 集成要按「activity interval」而非「一问一答」
   来建模多轮交互。

### 定位边界（预判）

- **可「驱动」，不可「拥有连续性」**：SDK 让 MOSS 能驱动 dsh 的 session，但 session 焊在
  dsh runtime 内，不能 memento 化（不能外部 fork / confluent）。集成分界 = 执行反转了、
  连续性未反转。
- **开箱不实现**：只暴露 dsh 的开箱能力面，不做应用实现。

## Open Problems

- 未实际运行 dsh，以上集成路径、融合点均为纸面判断，需跑通 SDK 后验证。
- `finalResponse` 的 quiescence 语义对 GUI 多轮交互的实际体验影响未知。
- dsh 的 `ctx.sessions` 是否技术上可替换（文档判为 core service「扩展非替换」），源码未确认。
- ACP 机器裁决（`session/request_permission`）与 SDK 预配置两条权限路，哪条先做未定。

## Implementation Notes

<!-- 施工化身在此追加 gotchas 与决策。 -->

- 先跑通最小闭环：Python SDK spawn dsh → `run(input, session_id)` → 读 `RunResult.events`
  渲染轨迹，验证 loop + resume + 权限预设三件事。
- dsh 处于开发者预览，官方明示破坏兼容性变更——集成层需容忍协议漂移。
- 参考 `claude-code-in-moss` 的桥接骨架，但用 Python SDK 替换 CLI 解析。
