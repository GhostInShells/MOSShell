# dsh agent loop 驱动模型与 MOSS 治理边界

> 源码锚定、可回放、非推测。源码路径均相对 `research/source/deepseek-harness/`。
> 本次调研回答一个核心问题：**dsh 的 session 循环是上游驱动还是自动，以及 MOSS 如何在其上治理生命周期**。
> 结论收敛于一个可落地的治理闭环：`wake`（感知）+ `perStep 门控`（注入/阻塞）+ `interrupt`（丢弃）。

---

## 一、session 循环：上游驱动（wake 驱动），非自动

### 探索路径

- `session.create` contract：`packages/host/apiproxy/src/api/sessions.ts:246-262`
- `AgentOptions`：`packages/core/agent/src/runtime-types.ts:24-31`
- `wakeDriver()`：`packages/core/agent-loop/src/agent.ts:172-193`
- `turn()` 结束条件：`agent.ts:324`

### 结论

dsh 的 session 循环**不是完全自动的**，而是 **wake 驱动**：

1. `session.create` 注释明确：**"Creates a real session and its idle agent"**（sessions.ts:247）——创建出来的是 **idle** agent，不自己跑。
2. `AgentOptions` 只有 `provider` / `model` / `maxTokens` 三个字段（runtime-types.ts:24-31），**没有 auto/autonomous flag**。
3. `wakeDriver()`（agent.ts:172-193）是**唯一**启动 driver 的地方：`setPhase({kind:'running'})` → `kick()`。它只在 `send(wakeup=true)` 时被调用。
4. `turn()` 结束时 `if (!this.inbox.hasPending) return false`（agent.ts:324）——队列清空就回 idle 停住，等下次 wake。

### "自动"的两层含义（避免误解）

- **一次 prompt 后的自动推进**：session.prompt 发一条消息后，agent 会自己跑完整条工具链（工具结果回填 next-step 继续），直到无 tool-call 或 max-tokens。这一层 dsh 是"自动"的。
- **session 创建后 / 无人驱动的自动**：不存在。创建的是 idle agent，必须显式 prompt 唤醒。

---

## 二、创建 agent 的参数面

### 探索路径

- `CreateAgentOptions`：`packages/core/agent/src/index.ts:80-133`
- `AgentOptions`：`packages/core/agent/src/runtime-types.ts:24-31`
- `AgentHandle` / `AgentFactory`：`packages/core/agent/src/index.ts:172-214`

### 结论

`CreateAgentOptions`（index.ts:80-133）六个字段：

| 字段 | 含义 |
|---|---|
| `sessionId` | 必填，agent/session 共享身份 |
| `meta?` | cwd / parentSession / seedLength / origin('subagent') / delegationDepth / agentPreset |
| `seed?` | 初始 replay/fork 历史（fork 时父 session 的 log 前缀） |
| `agentOptions?` | 即 `AgentOptions`（provider/model/maxTokens） |
| `signal?` | 创建期取消信号 |
| `setup?` | 创建期组合 scoped world 的回调 |

关键注释（index.ts:128-130）：**"Setup composes, it never drives"** —— setup 只做组合，不驱动。**"Drive the agent only after creation resolves"** —— 创建后必须由调用者显式驱动。这从设计上坐实了"创建 ≠ 运行"。

---

## 三、idle 边界：next 推进与每 step 注入点

### 探索路径

- `turn()` / `step()` / `preStep()` 私有调度方法：`agent.ts:246-401`
- `agent/pre-step` waterfall：`agent.ts:234-242`

### 结论

**next（step/turn）推进是内部运行时驱动的**：`kick()` → `while (await this.turn()) {}`，turn 内部 `while(true)` 跑 step。`turn()/step()/preStep()/wakeDriver()` 是 `ReactLoopAgent` 的 private 方法，循环硬编码，**不可细粒度重写**。

**但每个 step 前有原生注入点** `agent/pre-step`（waterfall，agent.ts:234-242）：

```js
const decision = await this.dispatch.waterfall(
  'agent/pre-step', { messages: claimed, ...position, signal },
  () => ({ kind: 'enter', messages: context === undefined ? claimed : [...claimed, context] }),
)
```

上游 listener 可以：
- **注入**：往 `messages` 数组塞额外消息（MOSS 上下文注入的精确位置）
- **reject**：返回 `{kind:'reject'}` → turn 以 `blocked` 结束 → driver 回 idle（agent.ts:267-270）

这是 async waterfall，listener 可以 `await` 任意外部条件——**每个 step 前停下治理的机制**。循环内部自动推进，但推进到 pre-step 这个闸口会等上游。

### "停下来"的两种形态

| 形态 | 机制 | 结果 |
|---|---|---|
| 自然停 | 队列清空（nextStep+nextTurn 空）且 turnEnds → break | driver 回 idle，`agent/status` 事件 |
| 主动停 | pre-step reject | turn `blocked` → idle |

没有第三种"强制暂停"；中途想停只能 `cancel()`。

---

## 四、pre-step 阻塞的时序（event 广播前后）

### 探索路径

- `turn()` 循环逐行：`agent.ts:246-330`

### 结论

```
255  turn/start 广播                    ← session.append
266  preStep() → waterfall('agent/pre-step') 阻塞点 ← 上游注入动作在这里
279  step/start 广播
283  user/message 广播（claimed + 注入的 context）
287  step() 模型调用执行
293  step/end 广播
```

**pre-step 阻塞在 step 的所有 event（step/start、user/message、step/end）之前，在 turn/start 之后。**

对 MOSS 的含义：MOSS 先收到 turn/start（知道新一轮已开），在 pre-step 闸口阻塞注入，注入的上下文随后作为 user/message 进历史（agent.ts:282-284），模型调用才真正开始。

### 上游拿不到 step 的持久化 id

pre-step 的 waterfall 参数只有 `{ messages: claimed, turn, step, signal }`（agent.ts:225, 234-242）：
- `turn` / `step` —— 纯整数序号
- `messages` —— 已 claim 的 UserMessage 对象（有 `message.id`）
- `signal`

**拿不到**：这个 step 在 session log 里的事件 seq（step/start 未 append）。pre-step 是一个"step 尚未诞生"的时点。上游能锚定的只有整数 `(turn, step)` 和 inbox 消息自己的 `message.id`。要拿 step 的持久化标识，得等 step/start 广播（但那时 pre-step 已过）。

---

## 五、工具调用与 step/turn 边界

### 探索路径

- `step()` 内部 `while(true)`：`agent.ts:339-400`
- 工具结果回流：`packages/core/agent-loop/src/tool-calls.ts:146-160, 262-289`

### 结论

**一次 step = 一次模型调用 + 它触发的工具组。** `step()` 内部的 `while(true)`（agent.ts:339）只在 error 重试时 `continue`（agent.ts:370），正常路径必 return：

```js
const toolCalls = message.content.filter(b => b.type === 'tool-call')
if (toolCalls.length === 0) return { kind: 'completed' }   // 无工具 → step 完
const { concluded } = await executeToolCalls(...)           // 有工具 → 执行组
return concluded ? { kind: 'completed' } : null             // 否则 null
```

**工具链不是在一个 step 里跑完的，而是跨多个 step**：每次模型调用带工具 → step() 返回 null → turn 外层开新 step（target='next-step'，agent.ts:300）→ 直到某次无工具调用 → completed → turn 结束。

**工具结果两条回流路**：
1. 进历史：`appendToolResult` → `session.append('tool/result', ...)`（tool-calls.ts:281-288）→ 下个 step 的 `deriveMessages()` 读到
2. 进队列：`result.additionalContexts` → `acceptContext` → next-step inbox（tool-calls.ts:156 + agent.ts:397）→ 下个 preStep claim 取走

### 边界语义

| 边界 | 触发 | 覆盖 |
|---|---|---|
| step | 一次模型调用 | 该模型调用 + 它触发的工具组 |
| turn | claim 到 next-turn 消息（来自 followup/send） | 从 prompt 到最终回答的整条交互链 |

**工具调用是 step 级的内容，不触发 turn；turn 跟 prompt 绑定。** 工具 `concludesTurn` 是工具级的显式 turn 终止开关。

---

## 六、session event 的流式数据面

### 探索路径

- `assistant/chunk`：`packages/core/session/src/types.ts:266`
- `assistant/message`：`types.ts:273`
- `tool/call`：`types.ts:279`
- `StreamChunk`：`packages/llm/llm/src/types.ts:291-303`
- `ReasoningBlock`：`llm/types.ts:59-63`

### 结论

session event 提供** token 级保真**的流式数据面。`assistant/chunk` 事件携带 `StreamChunk`（七变体 union，llm/types.ts:291-303）：

| 变体 | 内容 |
|---|---|
| `block-start` | 块类型标记（index + blockType） |
| `text-delta` | 正常 content 文本增量 |
| `reasoning-delta` | **thinking 流**（llm/types.ts:294） |
| `tool-call-delta` | **tool use 流**（id + name + argumentsDelta） |
| `block-end` | 组装完成的完整 ContentBlock |
| `usage` | token 计数（含 reasoningTokens） |
| `finish` | 结束原因 |

interleaved thinking 靠 `index` 关联交错增量（llm/types.ts:285 "Block indexes correlate interleaved deltas"）。完整事件时间线：

```
step/start
  assistant/chunk ×N   (thinking-delta / text-delta / tool-call-delta / block-end)
  tool/call            (callId + name + raw arguments JSON)
  tool/result          (content + isError + meta)
  assistant/message    (组装完成的完整消息，含 usage)
step/end
```

`assistant/message` 的 content blocks 完整保留 thinking（`ReasoningBlock`）、tool-call（`ToolCallBlock`）、text（`TextBlock`）。这就是 MOSS 做 logos 流式输出的完整数据源。

---

## 七、perStep 阻塞的 TS 机制：promise 反绑

### 探索路径

- approval race 范式：`packages/interaction/user-approval/src/index.ts:304-344`
- `ApprovalService.request`：`user-approval/index.ts:257-276`

### 结论

**不是 AbortSignal，是可 resolvable Promise（deferred）。** signal 是单向取消（abort 后不可逆），不能当"解锁"。

TS 表达"阻塞直到条件满足"：`await new Promise(r => resolvers.push(r))`，外部条件满足时 `resolve()`。dsh 的 approval 就是先例——`decide()` 里 promise（结果）+ signal（取消）并排竞速（user-approval/index.ts:330-343）：

```js
return await new Promise((resolve) => {
  const onAbort = () => resolve('cancelled')
  signal.addEventListener('abort', onAbort, { once: true })
  void answer.then((outcome) => resolve(outcome))
})
```

### py/ts 并发模型对照

| 语义 | Python | TS（dsh 侧落地） |
|---|---|---|
| 等 articulator 输入 | `await event.wait()` | `await gate`（resolvable promise） |
| interrupt 丢弃 | `task.cancel()` | `signal.abort()` + gate 绑 abort 竞速 |
| 多等待者 | `asyncio.Event` 天然支持 | waiters 数组（`resolvers.push(r)`） |

js 体系回调不互相持有句柄，必须用 promise 反绑（resolve 闭包连接等待方与触发方），这是和 python `asyncio.Event`（共享句柄，set 广播）的本质区别。

---

## 八、prompt 外 wake：host/session-status 走 mux

### 探索路径

- `agent/status` → `host/session-status` 转译：`packages/host/apiproxy/src/api-proxy.ts:3560-3562`
- `HostFrame` 类型：`packages/host/apiproxy/src/api/events.ts:138`

### 结论

**wake 的 mux 可见信号是 `host/session-status {sessionId, running}`。**

```js
// api-proxy.ts:3560-3562
ctx.on('agent/status', ({ agent, status }) => {
  queue.push(frame({ type: 'host/session-status', sessionId: agent.id, running: status === 'running' }))
})
```

`agent/status` 本身是 live Cordis 事件（不走 mux），但 apiproxy 进程内订阅它、转译成 `host/session-status`，通过同一个 mux queue 推给外部。所以 ghost 走 `dsh mux → ghost` 这条路就能收到 wake，**不需要 plugin → ghost**。

- `running:true` = idle→running（agent 被唤醒），`running:false` = running→idle（干完了）
- timing：`running:true` 在 `turn/start` 之前（agent/status 在 wakeDriver 时 emit，早于 kick→turn）
- 频率：只在 idle↔running 翻转时推一次，比每次 turn 都推的 turn/start 干净，天然不可丢

### prompt 外 wake 的技术机制

wake 的唯一通道是 `agent.followup()` / `agent.steer()`（`wakeup=true`，agent.ts:172-193 只被这两者触发）。`inject`（wakeup=false）**不 wake**——它填 FIFO queue，但 idle 的 agent 不会因此醒。

`inbox.hasPending` 检查（agent.ts:324）的精确语义：只决定**已经在 running 的 driver** 是否继续开新 turn；**idle 时队列非空不会自动醒**，必须 followup/steer 显式 wake。

goal-round-driver 是"prompt 外 wake"的现成实现：goal armed 时，driver 在 agent idle 后调 `agent.followup()`（goal-round-driver/index.ts:192）。

---

## 九、interrupt 机制与上下文

### 探索路径

- `Agent.cancel`：`agent.ts:134-140`
- `AgentCancelCause`：`packages/core/session/src/types.ts:143-147`
- turn abort 处理：`agent.ts:302-305`

### 结论

interrupt = `cancel(cause, options)`，底层是 abort signal：

```js
cancel(cause, options = {}) {
  if (!options.keepInbox) {
    this.inbox.clear()                       // 默认清 pending 队列
    if (this.phase.kind !== 'idle') this.phase.wakeRequested = false
  }
  if (this.phase.kind !== 'idle') this.phase.abort.abort(cause)   // abort 当前 signal
}
```

abort 后，当前 step 的 `signal.throwIfAborted()` 抛出 → turn 的 catch（agent.ts:302-305）把 `turnEnds = {kind:'aborted', reason: cause}` → `turn/end` 广播。api 层对应 `session.cancel`（sessions.ts:371，keepInbox 语义）。`AgentCancelCause` 四种：`user` / `parent` / `hook` / `disposed`。

**是否进上下文：不进。** interrupt 不产生模型可见的消息：
1. `turn/end {reason: aborted}` 是边界标记，log-only，不进 derived history（无 surfaceOp）
2. 中断时未完成的 `assistant/chunk` 流在 `throwIfAborted` 处断掉，`assistant/message` 不 append → 半截输出不进 derived history
3. 中断前已完成的 step 的 assistant/message、tool/result **还在** derived history（interrupt 不回溯删除）

### 丢弃"锁住的 step"需 gate 响应 abort

perStep 的 gate 是 resolvable promise，**默认不响应 abort**。interrupt 要丢弃卡在 pre-step 的 step，gate 必须绑 abort：cancel 时 abort 门信号，`await gate` 提前结束，preStep 下一个 `throwIfAborted` 抛出 → turn aborted。否则 gate 挂到 resolve 之后才抛。approval 的 race 模式（user-approval/index.ts:330-343）就是标准答案：gate 是"articulator 输入到达"和"interrupt 取消"双竞速。

---

## 十、收敛：MOSS 治理闭环（人类方案）

> 本节是人类工程师在讨论中落定的设计方向，基于上方源码锚定事实。

### 拓扑约束

只在三条路里完成，不走 plugin → ghost：

```
ghost → plugin           (web api / rpc)
plugin → dsh runtime     (create agent / perStep / rpc 开关)
dsh mux → ghost          (session event + host frame)
```

### 五点设计

1. **main agent session 运行都认为是在 shell 内** —— 不再告诉它"现在不是"。
2. **任何 text chunk 都认为是 logos** —— 通过 prompt 提示。
3. **tool call 里的 moss observe** —— 触发 trajectory drain 动作。
4. **tool call 里的 moss ctml append** —— 被解析成 logos，通过 plugin 回调给 plugin 阻塞住的 tool。
5. **agent session 在 ghost runtime 里任何时候都监听事件，但只有 articulator 循环里才监听 logos**。

### 治理闭环

```
wake:      host/session-status{running:true} 走 mux → ghost 拉 mindflow/articulator
perStep:   main agent 静态门控（agent/pre-step 阻塞，等 articulator 输入）
articulator: 等输入（tool use 的发现），非输出；ghost 从 tool/call 发现 → rpc 解除 tool 阻塞
interrupt: attention 循环开始先发 interrupt，丢弃锁住的 step（gate 需绑 abort）
```

关键 trick：articulator 等的是**输入**（tool use 发现），不是 tool 输出。ghost runtime 从 mux 收到 `tool/call` 作为 articulator 输入，回调 plugin 解除 agent 侧 tool 阻塞。main agent 的 perStep 是静态门控（总是卡，等 articulator）。

### 正确性依赖（两个）

1. **rpc 解锁的 retry 是必需**，不是兜底优化：articulator on 的 rpc 单次失败 → gate 永久 pending → agent 冻结。retry 需幂等（resolve 多次无害）。
2. **gate 要补超时/abort 兜底**：retry 只覆盖"rpc 传输失败"，覆盖不了"ghost/mindflow 根本不回应"（崩溃、articulator 起不来）。gate 需 `Promise.race([gate, timeout])` 或绑 abort——超时放行还是报错，取决于 dsh 侧"宁可跑空转还是宁可停住"。

---

> 记录：本次调研起于"session 循环是上游驱动还是自动"这个判断，沿 turn/step 循环、pre-step 注入点、session event 数据面一路下沉，最终落在 MOSS 治理 dsh agent loop 的完整闭环。关键转折有两处：一是发现 `session.create` 创建的是 idle agent + `AgentOptions` 无 auto flag，坐实了"wake 驱动"；二是发现 `agent/status` 经 apiproxy 转译成 `host/session-status` 走 mux，让 wake 信号落到三条路拓扑内、无需 plugin→ghost。落地时 js 侧的 promise 反绑 + signal 竞速（approval race 范式）与 python 的 asyncio.Event 协调是同一语义的两套表达，这一块待实现时由模型接。
