# Commit / Compact Ego Session

> dolores context 管理的第二阶段前置调研。2026-09-09 与人类架构师讨论收敛。
> 结论：**dsh session 机制原生支持 commit / compact 两个动作，无需在 dsh 侧发明新东西。**
> 但两者都不走官方 `session.fork`——统一走 Dolores Ego plugin 层，底层 `ctx.agents.create`。
> 关联：`ghost-prototype-dolores` FEATURE.md；此机制接 memento 的 commit/segment 之后。

## 意图

回合制 agent 可以花两分钟现场做一次 compact（模型读全历史 → 生成摘要 → replace）；实时交互 ghost 的
compact 必须**瞬间发生**，不能占用主循环的关键路径。

解法是把"慢的摘要生成"从关键路径上分期前置：**前序阶段性 commit 预先固化摘要，compact 时只做拼装。**
于是 dolores 需要两个 ego 动作：

- **commit** — 在 completed turn 边界 fork 一份，让后台 agent 生成该点之前的摘要，存为可拼装的 commit。
- **compact** — 上下文到限时 create 新 session，seed = 已就绪的 commit 摘要（作第一条 user message，xml-like
  容器）+ last-commit 之后的增量原文。动作本身零模型调用，毫秒~百毫秒级。

两个动作共享前提：**agentPreset 一致，工具构建继承（不产生毒 session）。**

## 两个动作 → dsh 机制映射

| 动作 | dsh 机制 | 原生支持 |
|---|---|---|
| commit = fork last completed turn | `session.fork`（省略 `atSeq` 即 last completed turn） | ✅ |
| compact = create 新 session + 摘要作第一条 | `ctx.agents.create`（seed 第一条 = 摘要 `user/message`） | ✅ |

摘要的 xml-like 容器 dsh **完全不管语义**：`deriveEventMessage` 对 `user/message` 是 verbatim 透传
（`surface.ts:96-98`，`return event.data`），不做 framing/parse。塞进 `<container>` 或任何 xml 形状，
dsh 一字不改喂给模型。

## 关键事实（源码锚点）

- **fork 只认 completed turn。** `atSeq` 校正到所在 turn 的 `turn/end`（`api-proxy.ts:2283-2289`）；
  open turn 则 `fork-unavailable`（`api-proxy.ts:2290-2297`）。→ commit 只能在静止点（turn 结束/idle）做。
- **fork 继承 agentPreset + 工具构建，不是毒 session。** fork handler 走
  `composeAgent(resolveSessionPreset(source))` 重建工具 + `meta.agentPreset`
  （`api-proxy.ts:2321-2336`）。注释明说：seeded history 是在这些工具下产出的，换组合会
  strand 已有的 tool call。
- **compact 的 create 是既有先例。** ego/create 已把 `messages` 数组逐条
  `session.append('user/message', ..., {surfaceOp:'append'})` 注入（`plugin.ts:381-390`）。摘要只是把
  "多条"换成"一条 xml-like 容器"，机制相同。create 的 `seed` 接受任意 seq 0 连续事件数组。
- **agentPreset 从 log 解析，不是只看 header。** `resolveSessionPreset({header, events})`
  （`api-proxy.ts:480-485`）：preset 存在 log 事件流里，fork 后 `resolveSessionPreset(source)` 能还原。
- **compact 的 seed 是重组，不受 fork 的"纯前缀"限制。** 只要满足通用约束：seq 0 连续、turn 平衡、
  无 dangling tool call。

## 分层判定：为什么官方 fork 不可用

官方 `session.fork` 是 `ctx.agents.create` 之上的 **UI 特化封装**，它锁死三个自由度，与需求逐条冲突：

| 官方 fork 锁死的 | 需求 | 冲突 |
|---|---|---|
| `seed = events.slice(0, cut)` 纯前缀 | compact 要"摘要作第一条 + 增量"，是重组 seed | ❌ 无摘要注入面 |
| `setup = composeAgent(...)` 固定 | 额外挂摘要工具 | ❌ setup 不是调用方写的 |
| `workspace.attachSession(childId)` 强制进 workspace | 后台 agent 不进 workspace | ❌ 无参数关闭 |

走 plugin 不是另起炉灶——官方 fork 与 plugin 要写的，底层是**同一个 `ctx.agents.create`**
（agent-loop 的 `createAgent`）。走 plugin 只是绕过官方 fork 强加的三条约束，拿回 seed 内容、
setup、workspace 三个自由度。

**分层结论：**

```
DshSession（通用抽象）       ← 保持官方动词，含 fork() 但仅作"继承式快照"语义
    ↓ 不承载 commit/compact
Dolores Ego plugin 接口      ← 新增 commit / compact 两个 plugin_call 路由
    ↓ 内部
ctx.agents.create({          ← 三者统一走这里
  seed: <摘要作第一条 | fork 前缀>,
  meta: { agentPreset, cwd },
  setup: <挂摘要工具 / 继承工具>,
  // 不调 attachSession → 自然不进 workspace
})
```

## Open Seams

1. **commit 的静止点**：fork 只认 completed turn。实时 ghost 中 turn 边界难界定，commit 要挑静止点或
   等 idle，否则 fork-unavailable。摘要生成由 fork 出的后台 agent（一次性、不进 workspace、挂摘要工具）
   消化，不阻塞主 ego。
2. **摘要重编码成 seed 事件**：commit 摘要是 MOSS 侧产物，compact 时须重编码成带 `surfaceOp:'append'`
   的 `user/message` 事件插入 seed 首条，再拼增量原文。整个 seed 须 seq 0 连续、turn 平衡。
3. **增量原文切片 + seq 重排**：last-commit 之后的原文仍在源 log，须按 commit 切点精确切片并重编号
   为 0 连续，切在 open turn 里 seed 校验直接拒。
4. **fork 后台 agent 的 model/provider 一致性**：历史里的 assistant 消息标注原 provider/model。后台
   agent 若换 provider 重放会冲突——同类型 agent 约束须钉死（child 与 parent 同 provider/model）。
