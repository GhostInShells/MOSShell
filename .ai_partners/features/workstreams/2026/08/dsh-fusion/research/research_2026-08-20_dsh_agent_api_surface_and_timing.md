# dsh 两个调用面的 API 时序对照

> 源码锚定、可回放、非推测。源码路径均相对 `research/source/deepseek-harness/`。
> 本次调研回答一个具体问题: **dsh 里 session 级别 agent 的接口, 哪些是 http rpc 外侧可调、哪些是 plugin in-process 内侧可调, 各自时序语义是什么**。
> 触发点: MOSS 侧 `MShellTrajectory` 的 append-only 上下文治理模型, 关键在 `inject` / `steer` 的时序与表面。

---

## 一、两个调用面

dsh 的 session 级别 agent 有两套调用面, 同一语义的两种编码:

| | 外侧 http rpc(ghost 可达) | 内侧 plugin in-process(dsh 进程内) |
|---|---|---|
| 承载 | apiproxy `RpcMethodMap`, `POST /api/session.*` | `Agent` / `Session` / `ctx.agents` 对象直接方法 + 事件扩展点 |
| 谁调 | ghost(经 plugin web api 转发) | plugin 直接拿对象 |
| 锚点 | `packages/host/apiproxy/src/api/sessions.ts` `SessionsApi` / `subagents.ts` `SubagentsApi` | `packages/core/agent/src/runtime-types.ts:64-144` `Agent` 接口 |
| 特征 | 强类型 envelope, 跨进程, 有 rpcId 回执 | 同进程直调, 无序列化 |

权威注册表是 `packages/host/apiproxy/src/api/rpc-map.ts` 的 `RpcMethodMap`(键即 wire path, 如 `POST /api/session.prompt`)。

---

## 二、时序词汇表

| 取值 | 语义 |
|---|---|
| 同步立即 | 调用返回即生效(void), 如 `cancel` 的 abort、`send/followup/steer/inject` 的入队 |
| FIFO 入队 | 进 inbox, driver 在后续边界 claim |
| 下个 step 边界 | 在下一个 pre-step 被 claim 进模型 |
| 新 turn | 唤醒 idle agent, 开独立 turn |
| 阻塞(await) | async, 调用方 await 到完成 |
| fire-and-return | rpc 返回 `accepted`, 实际效果在其后异步发生 |

---

## 三、统一 API 时序表

| 类别 | ctx api (in-process) | web rpc 等价物 | 执行时序 | wake | 消费边界 / 丢失 |
|---|---|---|---|---|---|
| 驱动 | `followup(msg)` | `session.prompt{mode:queue}` | FIFO 入队 + 立即 wake | ✓ | 独占一个 turn 的唯一普通消息; cancel/dispose 丢弃 |
| 驱动 | `steer(msg)` | `session.prompt{mode:steer}` / `session.updateQueue{action:steer}` | idle: 立即开 turn; running: 下个 step 边界 | ✓(idle) | 下个 pre-step claim; 被 reject 的 step 停靠到下次 wake; cancel/dispose 丢弃 |
| 驱动 | `inject(msg)` | **无** | FIFO 入队, **不 wake** | ✗ | 下个 pre-step claim; 可能 miss 已 claim 的当前 step; cancel/dispose 丢弃 |
| 驱动 | `send(msg, target, wakeup)` | `session.prompt`(仅 wakeup=true 子集) | 同步返回; wakeup 定是否 wake | 可选 | target 选 next-turn / next-step 边界 |
| 驱动 | `cancel(cause, {keepInbox})` | `session.cancel`(keepInbox=true) | 立即 abort(signal) | — | 中断当前 turn; keepInbox 保留队列否则清; 首 cause 生效 |
| 驱动 | `whenIdle()` | 无(`host/session-status{running:false}` 可观测) | 异步阻塞 | — | resolve 于整体静默后 |
| 驱动 | `runMaintenance(task)` | 无 | idle 期 claim + 异步跑 | ✗(保持 idle) | 非 turn 维护任务 |
| 生命周期 | `ctx.agents.create(opts)→Handle` | `session.create` | 异步阻塞(setup+发布+loop start) | — | 建 idle agent; setup 只组合不驱动 |
| 生命周期 | `ctx.agents.resume(opts)→Handle` | 无直接 | 异步阻塞 | — | 恢复持久 session |
| 生命周期 | `AgentHandle.dispose()` | 无 | 异步 | — | 停 loop 卸载 |
| 读写 | `session.append(type,data,opts)` | 无(经 mux event 出) | 同步 commit | — | 追加 typed event 到 durable log; 热路径不阻塞 I/O |
| 读写 | `session.events` / `session.seq` | `session.history` / `subagent.history` | 同步读快照 / rpc 异步 | ✗ | 读 log, 不激活 agent |
| 读写 | `ctx.agents.get/list/roots` | 无 | 同步 | — | 查 live agent(仅 live, 不含 cold) |
| 读写 | `agent.options`(readonly) | `session.models` / `session.selectModel` | 读 / rpc 变更 | ✗ | models 带 `routable` 就绪闸 |
| 读写 | `session.append('session/title')` | `session.rename` | 同步 / rpc | ✗ | 钉标题 |
| 读写 | `ctx.agents.create(seed=前缀)` | `session.fork` | 异步阻塞 | — | 派生 child |
| 读写 | — | `session.list` / `session.search` | rpc 异步 | ✗ | 跨 session 目录/搜索(persisted 域) |
| 读写 | — | `session.attachment` | rpc 异步 | ✗ | 读持久化图片 |
| 扩展点 | `agent/pre-step`(waterfall) | 无 | 阻塞(每 step await) | — | reject/enter; per-step 门控点 |
| 扩展点 | `agent/request`(waterfall) | 无 | 阻塞 | — | 替换 frozen call config |
| 扩展点 | `agent/request-error`(waterfall) | 无 | 阻塞 | — | retry |
| 扩展点 | `agent/turn-stopping`(serial) | 无 | 阻塞(awaited) | — | turn 收尾前, 可 steer 反对 |
| 扩展点 | `agent/status` 等 lifecycle emit | `host/session-status`(mux 转译) | emit(不阻塞) | — | idle⇄running 翻转 |

---

## 四、关键发现

### 1. `inject` 无 web rpc 等价物

http 面只有 `prompt`(队列/steer)/ `cancel` / `updateQueue`, 没有"注入上下文但不唤醒"的动词。
trajectory 模型的核心动词 `llm_agent.inject(frame)` **只能走 in-process**(plugin 直接 `agent.inject()`),
ghost 从 http 侧喂不进来。这意味着 epoch/frame 的 append-only 上下文注入必须由 plugin 在 dsh 进程内完成。

### 2. `inject` 的三条时序事实(append-only loop 的命门)

锚点 `runtime-types.ts:135-143`:

- **不 wake**: idle agent 收到 inject 后仍停在 idle, 等 followup/steer 来唤醒。
- **下个 pre-step claim**: running 时在下个 step 边界进模型。
- **可能 miss**: "may miss a request whose pre-step already claimed its batch" —— pre-step 已 claim 的当前 step 吃不到这次 inject, 要等下个 step。

`steer` 与 `inject` 的分野(`runtime-types.ts:126-133`): steer 对 idle 会开 turn(wake),
对 running 在下个 step 边界消费; 被 reject 的 step 会停靠到下次 wake。`followup` 则是"独占一个 turn + 必然 wake"。

### 3. `run_step` / `step_inputs` 是 MOSS 自造帧, 不是 dsh 动词

dsh 的 turn/step 循环是 `ReactLoopAgent` 私有 hardcoded、wake 驱动的(`turn()/step()/preStep()` 为 private,
见 08-19 调研第三节), **没有 `run_step` 可调**。trajectory 伪代码里的 `llm_agent` 是 MOSS 侧抽象,
落到 dsh 上只有两条路: (a) 自定义 driver 自己跑 turn/step; (b) 挂 `agent/pre-step` 扩展点, 借 dsh 内部循环。

### 4. fire-and-return: ghost 的 rpc 不阻塞在 turn 上

`session.prompt` 返回 `{accepted:true}` 于 host admission 之后(`sessions.ts:347-353`), `session.cancel`
返回 `{accepted:true}` 于 admit cancel signal 之后(`sessions.ts:371`), 两者都不等 turn 完成。ghost 驱动是异步解耦的。

---

## 五、对 MOSS trajectory 模型的落点

trajectory 的 `inject(epoch_start_point)` / `inject(pop_frame())` 映射到 dsh 侧 `agent.inject()`,
落在 in-process 路径(plugin → dsh runtime)。这带来两个约束:

1. **注入者必须是 plugin 本身**, ghost 无法经 http 完成 append-only 注入。
2. **inject 的 miss 语义要纳入正确性依赖**: 若某 step 的 pre-step 已 claim, 该帧 delta 落到下个 step ——
   对"每帧 delta 都进历史、靠前缀缓存命中"的模型, 丢一帧不致命(下一帧会含全量 facade delta), 但 epoch_start_point
   的全量注入不能 miss, 需在 turn/epoch 边界(而非 step 中途)注入。

---

## 六、MOSS 侧封装表面 (人类方案形态)

> 本节是讨论中落定的方案形态, 非源码锚定。分层/live 细节待实装时补齐。

dsh session 之上有**三个封装表面**:

1. **dsh agent** — 最底层封装: 让单个 dsh session 变得 ghost 可开、可 loop。面 = 上表驱动动词
   (`followup/steer/inject/cancel/whenIdle/runMaintenance`) + `agent/pre-step` 门控,
   即 "agent call protocol / agent surface"。
2. **main (暂定名 ego)** — 全局唯一单例, 编排多个 dsh agent: 主 session (in-shell, perStep 门控)
   + 旁路 commit session (异步 compact, 完事 archive) + ...。
3. **clone** — 与 main 同自我认知的另一 dsh agent, 旁路观测主路状态, 路径锁
   (dsh 无路径权限原语 → 创建时锁 + prompt 软约束)。

**subagent 不管** — 那是 fork 的产物, 出局。

### compact / 预热模型 (append-only 约束)

- 实时双工**不能同步 compact**; 必须触发式异步 commit, 到阈值重绘上下文
  (staging moments + 历史 commits 摘要)。
- commit body 精确控制在 1~2k; 估算 50k 稳态预算, ~300k 触发重压缩。
- 切换前预热防首次请求过慢 → 创建 agent 时完成上下文注入, 走
  `agent/session-start` → `agent.inject(全量重绘)` (idle 注入 park 不 miss)。

### 待实装时定 (deferred)

- 旁路 commit session 是 live agent (自带 loop 做 summarize) 还是 passive log。
- 重绘上下文回注主 session: `inject` 新 epoch vs 换新 create 的 agent。
- clone 旁路观测走 `session.history` (rpc 只读) 还是 MOSS 侧 shell state。

---

> 记录: 本次调研起于 `MShellTrajectory` 落地后"inject/steer 时序与表面"的追问, 把 08-19 调研里分散的 session.* / subagent.* / Agent 对象 / Session 对象 / 扩展点五张表收敛成一张带时序语义的统一表。核心转折是发现 `inject` 无 web rpc 等价物——append-only 上下文注入天然是 in-process 操作, 这决定了 trajectory 注入点必须放在 plugin 侧, 而非 ghost 的 http 侧。
