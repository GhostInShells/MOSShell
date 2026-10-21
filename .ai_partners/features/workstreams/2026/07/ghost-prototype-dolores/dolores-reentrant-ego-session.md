# Bypass Ego Session — 单轮旁路机制

> 2026-09-11 与人类架构师对齐 + dsh 源码调研，收敛为本设计。**取代**此前「n 控制器 fold + mark seq」
> 的旧设计（完整演变见 `git log -- <本文件>`）。文件名沿用，因为这仍是 perStep 锁的第二阶段。
>
> 目的：① 界面上已变旁路的 ego session 仍能响应 UI 对话（**单轮**）；② dsh session ref 机制生成的
> ego fork 可运行。
>
> 关联：`ghost-prototype-dolores` FEATURE.md；全貌 → [dolores-memento-plan.md](dolores-memento-plan.md)；
> 缺陷 → [dolores-todo.md](dolores-todo.md)。

## 准全双工：moment push 与 turn 边界（perStep 第三阶段，2026-10-19）

> 前两阶段针对**旁路**（非主 session 的单轮重入）。本阶段针对**主 session 自身的重入**：
> 一个 thinking 帧内，dsh 的 turn 结束后 body 动作可能仍在跑，期间到达的 signal（人的输入、
> 工具的异步 signal 回调、cell_event …）要能立即折进下一轮，而不是等 turn 边界把 moment
> 从 MOSS 抽干后无人消费。收敛出的核心命题：**主权在 body action，不在 dsh turn**。

### 根因（不可逆交接 + 被动投递）

- `BaseMomentsObserver.observe()`（`core/blueprint/moment.py:763-777`）是一次**不可逆交接**：
  同时抽干 `_injected_percepts`（`has_pending()` 变假）和轮换 `_echoes`（`need_observe` 折进
  `moment.previous` 后从 observer 消失）。mindflow 下一帧判据 `need_observe() or has_pending()`
  （`core/mindflow/_mindflow.py:1163`）据此决定是否再签 thinking。
- `moment/push` 只把 moment 塞进 `pendingMoments`（插件私有队列），**仅在活着的 pre-step 末尾
  排空**（`dsh_plugin/moss-dolores-ghost-plugin.ts` 的 `agent/pre-step` 分支）；turn 已结束时它
  要么滞留，要么被 `thinking/exit` 的 `flushPendingMoments` 投进 inbox（`agent.inject` 不唤醒、
  契约上可被丢弃）。于是：MOSS 的判据被抽干了，dsh 又没有活 turn 来消费 → 两头落空。
- 两个投递通道必须分清：`inject`（tool-result 的 moment → next-step inbox，turn 活着时续下一步）
  vs `pendingMoments`（`/moment/push` → 仅 pre-step 排空）。

### 收敛后的机制

| 点 | 结论 |
|---|---|
| turn/end 分类 | `aborted/interrupted/blocked/error` = 反向传导（abort）；`completed/max-tokens` 及自己的 `hook` cancel = 记号 |
| run 退出判据 | `turn_ended ∧ action_done ∧ 无待处理 signal`，或 `thinking abort`；经 settle 竞速串行化 |
| steer 与消费分离（D1=c） | steer = **空 steer**（唤醒开 turn，控制面）；moment 由 pre-step 排空 `pendingMoments`（数据面）。**steer 绝不消费 moment** |
| push moment | drain(observe) → push → `pendingMoments`，fire-and-forget，**无返回值探针** |
| thinking/exit | 无条件 close gate，但**不 flush** `pendingMoments`（100% 消费，除非硬重启） |
| safe flag | 安全 → observe+push；不安全 → 只通知不 drain（不安全 = turn/end 到达 → settle 结论之间） |

### settle 竞速（turn/end normal 后）

```
consume_turn_events → turn/end
  abort 类 → return（反向传导，body 交 mindflow 急停）
  normal → settle():
    2a. pushed_this_run 且 dsh 有未消费 pending moment → 空 steer → 新 turn → continue
    2b. race(wait_actions_done, signal_event)
        signal 赢 → drain + push + steer → continue
        action_done 赢 → exit
```

退出**只有** 2b 的 action_done 一条路；probe 空 / 本轮没 push 过，都不退出，都要落进 2b。
2a 必须在 2b 之前：否则 body 已 done 时竞速秒判 action_done，把已 push 但未消费的 pending
moment 漏掉。`drain+push` 里 push 必须先 await 完再 steer，否则新 turn 的 pre-step 时
`pendingMoments` 还没落进去。

### 工具语义

- `moss_observe`：保留阻塞到 observed（正交，不卷进 push/cancel；不阻塞会让模型反复 observe）。
- `moss_react` / `moss_wait_next`：期间 push 过 moment 就不 cancel，turn 自然走到 turn/end 探针。
- `final answer`：非 tool，即 turn/end normal → settle 探针。

### 待实测 / 未做（下一期）

numb 防抖、主动观测注入（tool use / agent message 这类必然有 next step 的事件）本期不做。
回归目标与实机判据见 [dolores-todo.md](dolores-todo.md)。

## 实现落地 (2026-09-12)

> 已在 `dsh_plugin/moss-dolores-ghost-plugin.ts` 实现, 待 GUI 实机验证。

- **A. pre-step 旁路分支** — `stepAgent.id !== doloresEgoSessionId` → 前置清理（折叠上一轮，见 B）+ `bypassTurns.set(id, turn)` + 降级（思考模式改 `low` + `sandbox/mode` 改 `read-only`）+ `next()` + `{kind:'enter', messages:[bypassInstruction(), ...decision.messages]}`。
- **B. pre-step 前置折叠（清空表面）** — 下一轮 pre-step 时按 `bypassTurns` 折叠上一轮，用 replace op 把该 turn 的 surface 节点折叠成一条空 `user/message`（空 assistant/message 才是真零残迹，但 session invariant 要求 assistant/message 命名 open step，pre-step 时 openStep 仍为 null 不可用）。**弃用 `session/event` 后置清理**：它经 session carrier 的 scope 分发，plugin ctx 不在其 `emit` 上行链，收不到事件（实测不触发）。
- **C. tools 全拒** — `agentCtx.tools.guard(exec => exec.agent.id !== doloresEgoSessionId ? reason : undefined)`，动态判定。
- **preset 元数据** — `ensureEgoPreset` 额外写 `preset.yml`（name/description），picker 不再显示裸 id 或复用 standard 描述。
- **删除** `notifySessionFrozen` + `session/frozen`（原「冻结」被旁路取代，D29 零件事）。
- **不做界面创建否决** — 曾考虑 `session/created` 同步 throw 否决非当前 ego，但会误拒 fork 出的旁路，且非必要（界面误建的 dolores-ego session 也会走旁路降级，无害），故不实现。

## 意图

- **旁路（bypass）** = ego-class session，但不是主 session。
- 旁路行为：**插入 instruction → 跑一轮 → turn/end 时把这一轮从 surface 清掉**。
- 与主路逻辑**完全不同**：不走 `thinking/enter|exit`，不消费 `pendingMoments`，工具调用全部立刻拒绝
  并返回「你现在是旁路」提示。
- **谁驱动不重要**（UI 打字 / MOSS 单次请求 / fork 后 followup — 都走同一机制）。

## 三个正交指标（不可混）

| 指标 | 载体 | 权威来源 |
|---|---|---|
| **ego-class 身份** | `agent.session.header.agentPreset === 'dolores-ego'` | durable，写进 `SessionHeader`（`dsh-session/lib/types/types.d.ts:71-77`），resume 后仍可读 |
| **主 session 身份** | `mainEgoSessionId`（模块级） | 进程内，`ego/create` 设 |
| **thinking 状态** | `thinkingGate` + `doloresThinkingToken` | **只属于主 session**，旁路永不触碰 |

旁路判定 = `agentPreset === 'dolores-ego' && agent.id !== mainEgoSessionId`。
**纯身份判定，任何时刻不看 gate / token。**

## 机制

### A. pre-step 分支（驱动无关）

`agent/pre-step` 是 waterfall，payload 带 `{agent, messages, turn, step, signal}`。分支在
`apply_ego_agent` 里按 agent ctx 注册：

```ts
agentCtx.on('agent/pre-step', async ({ agent, turn, signal }, next) => {
  if (agent.id !== mainEgoSessionId) {            // 旁路
    bypassTurns.set(agent.id, turn)               // dict[SessionId → turn]
    const decision = await next()
    if (decision.kind === 'reject') return decision
    return { kind: 'enter', messages: [...bypassInstruction(agent), ...decision.messages] }
  }
  await thinkingGate.wait(undefined, signal)      // 主路：不变
  ...pendingMoments 前缀...
})
```

关键事实：`step()` 会把 `decision.messages` 逐条
`session.append('user/message', msg, {surfaceOp:'append'})`。**所以插入的旁路 instruction 必然成为这个
turn 内的 surface 节点 —— 这正是 cleanup 要遮蔽的区间。** 主路注入的 epoch/context 走同一条路。

### B. turn/end 清理

- **不挂 pre-step 自清** —— 旁路可能不再有下一轮，清理会丢（人类架构师判定）。
- **不挂 `agent/turn-stopping`** —— 它在正常 stop 路径才触发，abort/error 路径被跳过；`turn/end` 在
  loop 的 `finally` 里 append，**abort/error 也发**。
- 挂 session 事件（agent-scoped，只收本 agent 的 session 事件）：

```ts
agentCtx.on('session/event', (session, event) => {
  if (event.type !== 'turn/end') return
  const turn = bypassTurns.get(session.id)
  if (turn === undefined || event.data.turn !== turn) return
  bypassTurns.delete(session.id)
  collapseTurn(session, event.data.turn)          // try/catch，fire-and-forget 里不能抛
})
```

`collapseTurn`：从 `session.events` 取该 turn 的 `turn/start`~`turn/end` seq 窗口，取窗口内**当前
surface 上存在**的节点，然后：

```ts
session.append(<节点>, {
  surfaceOp: { op: 'replace', start: firstSurfacedSeq, end: lastSurfacedSeq },
  sourceEventSeqs: [...windowSurfaceSeqs],
})
```

session 侧校验（`dsh-session/lib/index.js`）：`replacementRange`(339-349) 要求 start/end 是**当前**
surface 节点且顺序正确；`assertProvenance`(320-337) 要求 sourceEventSeqs 去重、更早、**覆盖全部被遮蔽
节点**；`assertToolResultRewrite`(369-393) **只约束 `tool/result` 的替换** —— 所以
`user/message`/`assistant/message` 的 replace 可以跨越 tool call/result 对（compaction 的 tool-pairing
校验是它自己加的，session 不强制）。规范样例见 `dsh-compaction-basic/lib/index.js:605-615`。

### C. 工具全拒

在旁路 agent ctx 上挂 guard：

```ts
agentCtx.tools.guard(() => '你现在是旁路会话，工具不可用；请只做单轮回应。')
```

- guard 经 `agent.ctx` 注册只对该 agent 生效；`guardReason` 走该 agent 的 scope chain。
- 它在 dispatch **之前**跑，返回字符串即 `Error: <reason>` / `isError:true`，**tool body 根本不执行**。
  → 模块级 `pendingYield`/`pendingCalls`（`pendingCalls` 还是 callId-keyed）根本到不了，不存在跨 session
  污染。

### D. fork 为什么「天然可运行」

MOSS 侧 `DshSessionRef`（turn 区间坐标）→ `seed_from_log` 取 verbatim balanced 前缀
（`src/ghoshell_moss/deepseek_harness/trajectory.py:55-75`）→ 走
`ctx.agents.create({sessionId, meta:{agentPreset:'dolores-ego', parentSession, seedLength}, seed, setup})`。
机制是**身份驱动**的：fork 只要带 `dolores-ego` preset，`agent/session-start` 就装配它、pre-step 就落入
旁路分支 —— **不需要为 fork 单独写一条路**。

## 决策（2026-09-11 人类架构师）

1. **replace 语义** —— 「保留到它还是 ego session 的状态，后面的全部不追加到 surface。UI 产生的后果我们
   不管。」（UI 走 append-origin transcript，replace 不抹掉人类已见的轮次。）
   → 待确认的精确形态：替换节点用「空 content 的 `assistant/message`（`deriveEventMessage` 返回
   `null`，模型派生历史零残渣）」还是「表征 ego 状态的最小 marker 节点」。
2. **log-only mark** —— 不是「没有」，是「能写不能读」（见下节）。结论：mark 的家应在 **MOSS 侧
   （memento 的 ref）**，不落 dsh log。
3. **session 重建** —— ghost 给 ego 一个**重建接口**；ego 检测到要重建时**准备好一个 ref**；**下一轮
   thinking 进来时先重建 ego**（不提前建好等 enter）。重建逻辑 = **fork 目标 session，从 ref 开始，截取
   ref 之后的 surface**，重建 ego session。建立时 **instruction 相同**，但 **memory 要集成 memento，
   生产 branch view**。未来切 branch 同理。

### log-only mark 的确切结论（重要）

- `Session.append` 运行时**可以**追加任意类型（plugin 现在就在 append `session/frozen`）。
- 但持久化读取门 `dsh-session-persistence/lib/index.js:1119`：
  `if (KNOWN_SESSION_EVENT_TYPES.has(event.type) || event.ignorable === true) continue;`
  —— 未知类型必须带 `ignorable: true`，**否则整条 log 拒绝加载**。
- `Session.append` 构 envelope 时只写 `{type, seq, time, data, ...surfaceMetadata}`
  （`dsh-session/lib/index.js:1444-1475`）—— **没有 `ignorable` 写入口**；`ignorable` 只在
  seed/normalized 路径被识别，且不可由调用方设置。
- 两条备选（若一定要在 dsh log 里做 mark）：① 复用**已知 log-only 类型**当载体（如 `session/title` /
  `plan/mode` / `permission/preset`，完整列表 = `KNOWN_SESSION_EVENT_TYPES`）；② mark 留在 MOSS 侧。
- **副作用**：现有 `notifySessionFrozen` 的 `session/frozen` 不在已知类型表里 → 潜伏的 resume 破坏 bug，
  已登记 D29。

## dsh 源码锚点（关键路径，减少重复调研）

> 路径根：`~/.dsh/profiles/node_modules/@deepseek-ai/`（随 dsh 升级会变，行号是 2026-09-11 快照）。

| 需要 | 锚点 |
|---|---|
| pre-step payload `{agent,messages,turn,step,signal}` | `dsh-agent/lib/types/runtime-types.d.ts:235-241` |
| pre-step 默认 decision `[...claimed, context]` | `dsh-agent-loop/lib/index.js:490-497` |
| pre-step 返回的 messages → `user/message` append | `dsh-agent-loop/lib/index.js:552` |
| `turn/start` / `turn/end`（`finally`，含 abort/error） | `dsh-agent-loop/lib/index.js:525-526, 588-595` |
| `agent/turn-stopping` 只在正常 stop 路径 | `dsh-agent-loop/lib/index.js:571-576`；声明 `runtime-types.d.ts:301-305` |
| `agent/status` idle⇄running | `dsh-agent/lib/types/runtime-types.d.ts:169-172` |
| `session/event`（post-commit, agent-scoped） | `dsh-session/lib/types/index.d.ts:56-66` |
| `Session.surface.nodes` / `.events` / `.append` / `.deriveMessages` | `dsh-session/lib/types/index.d.ts:106-267` |
| `SurfaceOp = 'append' \| {op:'replace',start,end}` | `dsh-session/lib/types/types.d.ts:393-397` |
| replace 校验：range / provenance / tool-result rewrite | `dsh-session/lib/index.js:339-349, 320-337, 369-393` |
| `deriveEventMessage`：空 content `assistant/message` → `null` | `dsh-session/lib/index.js:278-286` |
| canonical replace 样例（compaction） | `dsh-compaction-basic/lib/index.js:605-615`；选择/边界 `:516-545` |
| `tools.guard` 只对该 agent 生效 | `dsh-tools/lib/types/index.d.ts:605-616` |
| `guardReason` 走 agent scope chain | `dsh-tools/lib/index.js:2812-2820` |
| guard 拒绝在 dispatch 前 → `Error: <reason>` / isError | `dsh-tools/lib/index.js:3116-3127` |
| `ctx.agents.create` options（seed/meta/setup/agentPreset） | `dsh-agent/lib/types/index.d.ts:65-110` |
| `SessionStore.fork(source, boundary?, childSessionId?)` | `dsh-session/lib/types/index.d.ts:400-413` |
| `SessionHeader.agentPreset`（durable 身份） | `dsh-session/lib/types/types.d.ts:71-77` |
| log-only 读取门（未知类型须 ignorable） | `dsh-session-persistence/lib/index.js:1119`；已知类型表 `dsh-session/lib/index.js:1054+` |
| `append` 不写 ignorable | `dsh-session/lib/index.js:1444-1475` |
| MOSS 侧 ref 物化 | `src/ghoshell_moss/deepseek_harness/trajectory.py:55-75`；`types/refs.py` |

## Open Seams

1. **replace 节点精确形态** —— **已定**（2026-09-12）：空 content 的 `user/message`。空 `assistant/message`
   才是真零残迹，但 session invariant 要求 assistant/message 命名 open step，pre-step 折叠时 openStep 仍为 null 不可用。
2. **拒绝 tool 会起新 step** —— denied tool 仍返回 `tool/result`，loop 会再跑一步，模型可能反复试；guard
   无法 `concludeTurn`。instruction 要硬性禁止调工具。
3. **主身份恢复** —— `mainEgoSessionId` 是进程内模块态；重启后 resume 一个 ego session 时它是 `null` →
   会误判成旁路。主身份的恢复规则（MOSS 重新声明 / 首个 resume 认领）待定。
4. **清理 append 的健壮性** —— **已实现**：`collapseTurn` 调用处 try/catch + `ctx.logger.warn`。
5. **session/event 收不到（已弃用）** —— 实测 `agentCtx.on('session/event')` 与 `ctx.on('session/event')`
   都不触发：session/event 经 session carrier 的 scope 分发，plugin ctx 不在其 `emit` 上行链。改用 pre-step
   前置清理（B）。
