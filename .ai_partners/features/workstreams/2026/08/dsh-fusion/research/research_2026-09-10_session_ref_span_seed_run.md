# Session Ref 的 span / seed / run 方案

> 2026-09-10 与人类架构师收敛。承接昨天落地的 `DshSessionRef`(a6e3f873 单 turn 坐标),
> 今天把它从「点」扩成「段」, 并钉死 read / seed / run 三个面的实现口径。
> 关联: dsh-fusion FEATURE.md; 路径 B (dolores ego 推理中枢) 与路径 A (gui agent) 共享的基建。

## 结论摘要 (当前干净形态)

```
ref (span) → 冷读源 log → tail-truncate 成 seed → ctx.agents.create({seed, setup: preset+restrict(read-only)})
            → followup → whenIdle → 读结果 → dispose
```

- **read / export**: 冷读 log (不 materialize agent), fold 成纯文本。
- **run**: 冷 seed + 独立 one-shot, **不经过 subagent 框架** (subagent 会把结果回流进父 agent 的
  turn, 驱动另一个 agent 运行 — 这是红线)。
- **无 fork 中间 session**。ref 直接物化成 seed, 不再 fork 一个 pinned P 当父。
- 全部不依赖具体 agent; seed 是纯事件, 可与 dolores ego 解耦。

## ref = span (turn 主 / seq 辅)

`DshSessionRef` 从单 turn 坐标改为 **turn 区间**:

- `session_id` + `start_turn` / `end_turn` (区间两端, 含端) + `start_seq` / `end_seq` (可选加速器)。
- **turn 是主坐标, seq 是派生。** turn 是「哪个 turn」的稳定语义锚, ref 长期持有; seq 是某条 log 的
  内部位置, **没记录时从 turn 反查** (`turn/start` / `turn/end` 事件各带 `data.turn`)。
  覆盖两阶段时序: turn 中间产出 ref (只有 turn, seq 待补) 与 turn/end 后 commit (补 end_seq)。
- `end_turn` / `end_seq` 决定 seed 切点; `start_turn` / `start_seq` 只决定 read 窗口左端。

落点: `src/ghoshell_moss/deepseek_harness/types/refs.py`。

## seed 重建 (tail 截断, 已验证)

`seed_from_log(events, ref)` — 纯函数, 输入源 log (seq 连续从 0), 输出 verbatim 前缀。

dsh seed 契约 (`core/agent/src/index.ts:102-109`): 必须 **contiguous from seq 0**、lossless-JSON、
balanced (无 open turn / dangling tool call)。校验 (`core/session/src/index.ts:525-527`):
`snapshot.seq !== index` → 直接抛 "seed must be contiguous from 0"。

为什么尾部截断满足它:

- 范式先例: 内置 fork provider 就是 `events.slice(0, lastTurnEnd.seq + 1)`
  (`subagent-fork-in-process/src/index.ts:48-54`)。
- `seq = log.length` 契约贯穿全系统 (`core/session/src/index.ts:512-513`), 持久化也保持 seq 连续
  含 raw chunk (`core/session/src/types.ts:233`), `session.export` 的 JSONL 与后端持久 artifact
  逐字节相同。
- 切在 `turn/end` 天然 balanced。

一条硬约束: **seed 是逐字节全量的 log 前缀, 不能挑事件、不能重排** (丢了任何 log-only 事件 seq 就断链)。
推论: 尾部截断只产「fork 式 seed」(原样继承), 产不出「compact 式 seed」(摘要作首条 = 重组, 另一条路)。

切点规则 (镜像 apiproxy fork, `api-proxy.ts:2303-2304`): 落在 `end_turn` 的 `turn/end` 上, 再向后吞
trailing standalone 事件 (session/title / injection) 到下一个 `turn/start`。open turn → `SeedUnavailable`
(与官方 `fork-unavailable` 同源)。

落点: `src/ghoshell_moss/deepseek_harness/trajectory.py::seed_from_log`。

## read / export (冷读 log, 不是 surface)

- `session.history` 明确 "uses attached state or persistence inspection **without acquiring an Agent**"
  (`api/sessions.ts:283-284`) — 冷读不 materialize、不建 session。
- **read 走 log, 不走 surface。** dsh surface(`deriveMessages`) 尊重 `SurfaceOp.replace`, 是 dsh 自己的
  派生物; MOSS 不复刻它。只有「模型当前可见窗口」才需要 surface — 那个从活 session 的 plugin 侧拿。
- **export** = `session.export` 的 raw JSONL (原样 log, 无渲染) + MOSS 自折叠。dsh 没有 `>`/`~`/`@`
  那种文本导出; `ToolEventView` 明说永不持久化 (`types/events.py` "render intent, 永不持久化"), 所以
  `@` 只能从原始 `tool/call` 的 name+arguments 自渲染。

fold 符号 (已落地 `trajectory.py::render_transcript`):

- `>` 用户输入 (`user/message`, 仅 `source.kind == "user"`, 非 injection)。
- `~` 模型输出 (`assistant/message` 的 text 块; 空 content 的 usage 帧跳过)。
- `@` 工具调用 (`tool/call` 的 name+arguments; `tool/result` 不渲染 — read 只要动作不要结果)。
- 其余 (turn/chunk/request/todo/injection) 跳过; `limit_turns` 保留最近 N turn。

## run (冷 seed + 独立 one-shot, 不走 subagent)

**为什么不用 subagent 框架**: subagent 的定义性行为是「父 agent 在自己的 turn 里调 tool 发起 child,
child 结果作为 tool result 回流进父的 turn, 父据此继续跑」— 即 **child 结果驱动另一个 agent 运行**,
命中红线。另外 subagent 还带 subagent list / `subagent/end` 通知, 都是父绑定。

**冷 seed 路径**: `ctx.agents.create({seed})` **不需要父 agent** — RPC fork 就在 apiproxy 的 ctx 上直接调
它 (`api-proxy.ts:2323`)。所以:

```
ctx.agents.create({
  sessionId: <ref.id 或派生 id>,
  seed,                          # verbatim 前缀
  meta: { cwd, seedLength, agentPreset },
  setup: async (agentCtx) => {
    await composePreset(resolveSessionPreset(source))
    agentCtx.tools.restrict({ deny: [写工具] })   # read-only
  },
})
handle.followup(prompt); await handle.whenIdle(); 读最后一条非空 assistant message; handle.dispose()
```

- **read-only = `setup` 里 `agentCtx.tools.restrict({deny:[...]})`** (`core/tools/src/index.ts:1071`,
  scoped 到该 agent、返回 disposer) — subagent 框架内部用的正是这个原语 (`child-agent.ts:174`), 我们
  直接在自家 setup 用。
- 只有被创建的 agent 跑一轮; 结果是一个值, 不回流任何父。

**销毁 (dispose)**: 必须。dsh 活 session 是无界 Map、无 LRU、无 idle 回收 (`core/session/src/index.ts:792`),
不销毁 = 千级活 agent 泄漏。且 **dispose 不删 log**: `retireCore` = `flush` + drop live tracking
(`session-persistence/src/coordinator.ts:1154-1161`), append-only 的 jsonl/sqlite 留存。时点 = whenIdle +
读结果之后 (先 dispose 会把未闭合 turn 的 reason 标成 `disposed/aborted`)。

## 关键源码锚点

| 事实 | 锚点 |
|---|---|
| seed 契约 (连续/balanced/无 dangling) | `core/agent/src/index.ts:102-109` |
| seed 校验 seq===index | `core/session/src/index.ts:525-527` |
| seq = log.length 契约 | `core/session/src/index.ts:512-513` |
| 持久化 seq 连续含 chunk | `core/session/src/types.ts:233` |
| fork provider 尾部截断范式 | `subagent-fork-in-process/src/index.ts:48-54` |
| store fork 不挂 workspace; RPC fork 才 attach | `core/session/src/index.ts:1081-1100` vs `api-proxy.ts:2344-2349` |
| RPC fork cut 吞 trailing standalone | `api-proxy.ts:2303-2304` |
| RPC fork 在 apiproxy ctx 上调 agents.create (无需父 agent) | `api-proxy.ts:2323` |
| history 冷读不 acquire Agent | `api/sessions.ts:283-284` |
| tools.restrict (read-only) | `core/tools/src/index.ts:1071` |
| 活 session 无 LRU | `core/session/src/index.ts:792` |
| dispose 不删 log | `session-persistence/src/coordinator.ts:1154-1161` |

## 真实验证 (2026-09-10, http://127.0.0.1:3080)

对活 dsh 的 2-turn session (`session-454825ea...`) 只读验证 `seed_from_log` / `render_transcript`:

- `session.history` 单页拿全 1156 事件, **seq == index 连续** (0..1155) — 尾部截断的 `seq = log.length`
  前提在真实持久化上成立。
- **`data.turn` 是 1 起** (turn=1, turn=2), 不是 0 起。ref 的 `start_turn`/`end_turn` 应直接取
  `data.turn` 的值 (代码按 `data.turn == ref.end_turn` 匹配, 对 0/1 起无假设)。
- `seed_from_log(end_turn=2)` → 1156 事件前缀, 末尾 `session/end-seed` (seq 1155)。该 session 被
  resume 过, 全量 stored log 成了 constructor seed, 所以 end-seed 落在最尾; 我们的"吞 trailing
  standalone 到下一个 turn/start"镜像 RPC fork, 把 end-seed 一并吞进 — 语义正确 (seed 确实终止于此)。
- `render_transcript` 真实输出: `>` 用户 / `@ read_image(...)` / `~ 模型回答`, 空 content 帧与
  tool/result 正确跳过。真实事件里还有 `permission/preset`、`sandbox/mode`、`approval/policy`、
  `agent/inbox/spliced` 等 standalone 类型 — seed 里 verbatim 保留, fold 里正确忽略。

## Settled vs Open

**Settled**: ref span (turn 主 seq 辅); seed 尾部截断重建 + 契约; read 走 log 不走 surface; run 走冷 seed
独立 one-shot 不走 subagent; read-only 用 `tools.restrict`; dispose 后 log 留存。

**Open (留给下一轮, 不阻断本次落地)**:

1. **read fold 的渲染细节**: 缩进规则 (当前每行 2 空格)、`@` 是否要 result、reasoning/image 块处理、
   参数截断长度 — 方向已定, 具体格式待定。
2. **run 的 RPC 契约**: Python 侧如何把 seed/ref 传给 plugin 路由 (传 ref 让 plugin 进程内重建, 还是传
   seed JSON 让 plugin 薄执行), 以及 plugin 路由的 request/response 形状。**属 dolores 装线**。
3. **provider/model 一致性**: seed 历史里 assistant 消息标注原 provider/model, run 时换模型会与重放
   历史冲突 (dolores-commit-compact 的 Open Seam 4)。冷 seed 路径是否锁「child 与源同 provider/model」。
4. **run 的 durable log 累积**: 每次 one-shot run 都落一条 durable session log; per-life 下是该留, 但
   千级 run 要 O3 的归档/索引/GC 治理。
5. **谱系**: run session 是否设 `meta.parentSession = 源` (可追溯) 还是无父顶层 — 可见性已接受, 纯属
   取舍。

## 留给 dolores ego 装线

- **plugin** (dsh web 侧): 接收 seed (或 ref), 做 `ctx.agents.create` + preset 组合 + `tools.restrict`
  (read-only) + followup/await/read/dispose, 经 `ctx.webServer.register` 暴露一个 RPC 路由给 Python 侧调。
- **可能复制独立 plugin, 不与 dolores ego 现有 plugin 混** (人类架构师 2026-09-10 判断)。
- Python 侧 (`deepseek_harness`) 只交付 ref/seed/fold 纯函数 + 测试; run 的 RPC 契约形状待 dolores 侧定。
