# dsh session 两层架构与 message 协议面（纠正 append-only 结论）

> 源码锚定、可回放、非推测。源码路径均相对 `research/source/deepseek-harness/`。
> 本次调研**纠正**此前（08-20 API surface 调研）「dsh session 是 append-only 上下文」的框架性结论，
> 并补全 message 协议面、pre-step decision 结构、agent 运行时的锁/注入语义。
> 触发点：为 perStep 门控（锁）+ 上下文注入建模，追问 session 的数据/上下文两层、message 协议、decision 结构。

---

## 一、session 两层架构（核心纠正）

dsh session 不是「append-only 上下文」，是**两层**：

| 层 | 性质 | 可否变 | 锚点 |
|---|---|---|---|
| **log** | 事件真相源，append-only，永不删 | 不可变 | `core/session/src/index.ts:426` `private log` |
| **surface** | 模型可见投影，`nodes` 是进模型的事件 seq 列表 | 可 `replace` | `core/session/src/surface.ts` `SurfaceManager` |

- 只有 `user/message` / `assistant/message` / `tool/result` 三种事件能进 surface（`surface.ts:15-19`）。
- `deriveMessages()` 读 surface 投影成 `Message[]`（`index.ts:726-747`）。
- **surface replace 只影响模型侧**，刻意不影响人类 transcript（`isAppendSurfaceEvent` 注释：a landed replacement would erase conversation the user already saw）。
- **compaction 是 replace 的唯一 producer**：compaction-basic（历史→summary）与 compaction-tool-result-pruner（tool/result 剪枝）都以 `{surfaceOp:{op:'replace', start, end}}` append。
- 因此「compact 只能追加」是错的：compact 用 replace 把旧 surface 范围 shadow 掉，log 只增，surface 变小。

## 二、`session.append` 返回值

返回 `SessionEvent<T>`（非 void），含本次 append 分配的 `seq` / `time` / `data` 快照（`index.ts:604-655`）。`seq = log.length`（append 当时）。调用方读 `.seq`（例 `agent.ts:349` `append('assistant/chunk', …).seq`）。surface-eligible 事件**必须**传 `surfaceOp`（编译强制，`index.ts:607`）。

## 三、fork 是 log 复制，不是 surface 复制

`_forkSeed` 返回 `events.slice(0, boundary + 1)`（`index.ts:1137`），是**完整事件 log 前缀**（含 chunk / turn 边界 / inbox splice / request-header），不是 surface nodes 投影。约束：boundary 必须连续存在（`INVALID_BOUNDARY`）、不能落在 open turn 内（`OPEN_TURN`，`index.ts:1128-1135`）。fork 出的是「父 log 的已完结 turn 前缀」。

## 四、ContextForm 是封闭 union，不可插件扩展

`ContextForm`（`llm/llm/src/message.ts:48-60`）是**封闭 union**（6 值：`instructions` / `catalog` / `snapshot` / `notice` / `relay` / `recall`），**不是** interface，插件运行时加不了新值。注释明说「grows one value at a time as producers gain the structured fields their form needs」。与 `MessageSource.kind`（merge-extensible，插件能加 kind）是两个正交轴：kind 回答「谁产生」，form 回答「什么类型的东西」。`form` 可选、只挂在 `plugin` source 上，省略 = 默认 opaque。

## 五、无 lock/pause/suspend 原语

`Phase`（`agent-loop/src/agent.ts:38-46`）只有 `idle | maintenance | running`；`AgentStatus`（`agent/src/runtime-types.ts:50`）只有 `idle | running`。**没有 `lock()/pause()/suspend()`**。最接近「锁住不运行」的三个机制，语义各不同：

1. `runMaintenance(task)`（`agent.ts:142-162`）——claim idle 相位跑非 turn 任务，waking input 停靠到任务结束。
2. `cancel({kind:'disposed'}, {keepInbox:true})`——abort 当前 turn、保留队列，disposed 不再唤醒（`agent.ts:172-181` latch 条件排除 disposed）。
3. `agent/pre-step` 返回 `{kind:'reject'}`——per-step 否决，turn 以 `blocked` 结束（`agent.ts:267-270`）。

## 六、inject / steer 同队列，唯一区别 wakeup

`agent.ts:122-132`：`followup`=`send(next-turn, wake)`，`steer`=`send(next-step, wake)`，`inject`=`send(next-step, 不 wake)`。inject 与 steer **target 完全相同**（next-step），唯一差别是 `wakeup`。inject 不唤醒，idle 时纯排队等 followup/steer 唤醒。

## 七、pre-step 入参 / 出参（decision 结构）

**入参**（waterfall payload，`agent.ts:234` + dispatch 注入 agent）：`{ agent, messages, turn, step, signal }`。`messages` = inbox claim 出来的 UserMessage（已从 inbox 移除）。**不含** assembly / tools / runtime context 投影 / 最终 request。

**出参**（`PreStepDecision`，`runtime-types.ts:53-55`）：

```ts
type PreStepDecision =
  | { kind: 'reject' }
  | { kind: 'enter'; messages: UserMessage[] }
```

`preStep` 内部补 `assembly` 成 `PreparedStep`（`agent.ts:50-52,242`）。**`enter.messages` 是自由改写点**——可替换/注入/过滤/清空。空 messages 两态：step 0 时空 → turn `completed` 收尾不花 model call（`agent.ts:274-277`）；step > 0 时空 → 照常跑一次 model call（基于历史无新输入，`agent.ts:271` 不拦此 case）。这是比 `inject()` 更原子、无并发窗口的注入通道。

## 八、next() 是洋葱模型委托，不重入 perStep

Cordis waterfall（`vendor/cordis/src/events.ts:234-243`）：`next()` = `cbs.shift() ?? inner`，从监听器队列取下一个调用，取空用 `inner`（default）。**不重新触发 waterfall、不死锁**，链必然在 `inner` 终止（inner 只返回 `{kind:'enter', messages:[...claimed, context]}`，不 dispatch）。不调 `next()` 直接 return = 短路整条链。

## 九、messages 一条一个 event

`enter.messages`（`UserMessage[]`）在 `turn()` 里被**展开**成 N 个独立 `user/message` event（`agent.ts:282-284`），每条一个 `seq`。pre-step 监听器拿不到这些 seq（append 在 preStep 返回之后）。

## 十、Message 协议：role 锁死，无 metadata 槽位

`Message`（`llm/llm/src/message.ts:129-138`）只有 `id / role / content / source` 四字段。`role: 'system' | 'user' | 'assistant'`，readonly + deep-frozen；`UserMessage` 钉死 `role:'user'`。**无 metadata 字段**——唯一带 provenance 的通道是 `source`（merge-extensible `MessageSourceMap`，`plugin` 源带 `plugin: string` + `ContextForm`）。`SessionEvent`（`session/src/types.ts:404-436`）亦无通用 `meta`，唯一例外是 `tool/result` 的 `meta?: JsonValue`（tool 私有）。

## 十一、inject 在 pre-step 里是下一 step 才拿到

`preStep` 里 `claim` 在 waterfall **之前**（`agent.ts:229`），已移空当前 next-step。监听器里再 `agent.inject(msg)`（`inbox.splice('next-step', …, [msg])`）落进**下一轮** claim 的队列，当前 step 已错过。**正确做法是 `enter + messages`**（原子改写当前 step 进什么），不是 pre-step 里 inject。

## 十二、session/event 同步广播

`append` 同步 push log + 同步 emit `session/event`（`index.ts:636-648`），`invokeContainedSessionObservers` 同步逐个调用 listener、per-listener 隔离（`index.ts:382-399`）。**广播（emit）同步，持久化（落盘）异步**（persistence 插件异步 buffer + flush）。invariant companion 通过 `internal/dispatch` 同步 stage 校验（`session/src/invariant.ts:233-241`）。

## 十三、assistant/message 有 open-step 约束（append 到 pre-step 非法）

`invariant.ts:118-121`：`assistant/message` 必须 `requireOpenStep`（step/start 之后、step/end 之前）。pre-step 阶段 openStep 为 null，append `assistant/message` 会违反 turn/step 结构不变量（invariant companion 启用时当场 fail，不启用则留结构非法 log）。**`user/message` 无此约束**（`invariant.ts:145-146` `break`），可在 turn/step 之间自由 append。pre-step 注入 model-facing 内容的正规通道是 `user/message`（enter 改写），不是 assistant/message。

---

> 记录：本轮起于「perStep 能否当锁」的追问，沿 session 数据/上下文两层、message 协议、pre-step decision 结构下沉，纠正了 08-20 调研的 append-only 框架。核心转折是确立「log + surface」两层——compact 的 replace 只 shadow surface、不动 log，这推翻了「append-only 上下文」对 compact 方案的约束。随后补全 message 协议（role 锁死、无 metadata、一条一个 event）与 pre-step decision（enter+messages 是原子注入通道），为下一轮「按步骤动手」提供事实依据。
