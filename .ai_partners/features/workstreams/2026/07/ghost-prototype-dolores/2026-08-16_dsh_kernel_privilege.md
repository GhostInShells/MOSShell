# DSH 内核特权与三方桥 — 探索轨迹 (2026-08-16)

> 接续 [2026-08-15_dsh_deep_dive.md](2026-08-15_dsh_deep_dive.md)。这一轮的主题从「能不能
> 无缝 compact」开始, 一路收敛到一个结论: **ghost 要够到 dsh 的进程内内核特权, 唯一干净
> 的路是仿 apiproxy 再写一个 plugin, 注册几个 HTTP 路由。** 记录问题 / 探索路径 / 初步结论 /
> 观点, 技术细节仍不记。

---

## 一、fork 不是「全量复制」, compact 也不是「改写」

**问题** — 上一轮说 session fork 是全量复制, 用户不信; 连带问 compact 是不是也追加。

**探索路径** — 读 `SessionStore.fork` + `_forkSeed`(packages/core/session/src/index.ts)、
apiproxy 的 fork handler、compaction 的 `commitCompactionBody`。

**初步结论**:

- **fork = 按 completed-turn 边界切一段事件前缀, 深拷贝成新 log。** `_forkSeed` 返回
  `events.slice(0, boundary+1)`, 但 boundary 默认不是"全量"而是"最后一个事件", 且前缀末尾
  必须在 turn 边界 (否则 `OPEN_TURN` 报错)。apiproxy 层更严: `atSeq` 省略时锚定最后一个
  `turn/end`, 进行中的 turn 被排除。
- **深拷贝发生在构造器, 不是 slice。** `events.slice()` 只是浅拷贝数组; 真正的逐事件深拷贝
  在 `Session` 构造器里 `snapshotJsonValue` (完整 JSON 走一遍 detach) + `deepFreeze`。所以
  子 session 的 log 与父完全独立, 父后续 append 不渗入子。
- **代价是 O(前缀长度) 的深拷贝。** 不是 COW、不是链表引用、不是共享只读历史。fork 长
  session 的 CPU/存储都线性累积。

- **compact 也是追加, 不删不 trim。** `commitCompactionBody` 顺序 append 四个新事件:
  `compaction/start`(锁)→ `compaction/summary`(摘要原文)→ `user/message`(带
  `surfaceOp: replace` 的 checkpoint)→ `compaction/end`。旧事件一个字节不动, 靠 surface
  层的 `replace` 把旧范围 shadow 掉。**模型面 (surface) 变小, 磁盘 log 只增不减。**

**观点** — fork 和 compact 都指向同一个物理事实: dsh 的 session log 是 append-only 沉淀,
  既不共享也不回收。这直接动摇上一轮「记忆物理存储委托 dsh, Memento 只存指针」的假设 —
  dsh 的 log 没有"压缩后变小"的物理语义, 真省存储得靠 Memento 侧决定"这个 branch 可以丢"。

---

## 二、给 session 发请求的两条协议面, 能力不对称

**问题** — 现在给 session 发一个请求, 协议面到底是什么。

**探索路径** — 读 apiproxy 的 `fetch/handler.ts` (UNARY_ROUTES 全集) 与 `sdk/server` 的
`handleRequest`。

**初步结论**:

- **HTTP RPC** `session.prompt` 的 payload 是 `{sessionId, mode: 'queue'|'steer', content:
  [{text|image}], clientTimeZone?}`。`mode` 决定走 `agent.followup`(queue)还是 `agent.steer`
  (steer)。content 被收窄成 text/image 两种, 带图先校验模型 inputModalities。
- **stdio JSON-RPC** `session/prompt` 的 params 是 `{sessionId, contentBlocks}`。**没有 mode
  字段**, 服务端固定走 `agent.followup`。contentBlocks 是完整 ContentBlock union, 不校验图。
- **底层三入口** (agent.ts): `followup`(next-turn, 唤醒)、`steer`(next-step, 唤醒)、
  `inject`(next-step, **不唤醒**)。两条协议面最终都落到这三个之一。

**观点** — 之前 checkpoint 记「客户端仅 3 个方法」方法数没错, 但漏了语义差异: **stdio 面是
  只能 followup 的退化 prompt, HTTP 面才完整 (steer 可选 + image 校验)**。这对 ghost 用哪条
  协议面驱动 dsh 有实际影响: 走 Python SDK 就永远没有抢占能力。

---

## 三、历史注入三通道 + assistant append 的真相

**问题** — 构建 session 时能否注入对话历史 (尤其 assistant 消息)? 无缝 compact 卡在
"切不回来, 只能发 user message"。

**探索路径** — 读 `session.create` payload、`agents.create({seed})` 签名、`Session.append`
与 `assertMessageEventShape`。

**初步结论** — 三条通道, 能力逐级放大:

1. **`session.prompt`**(两条协议面): 每条都 `createUserMessage`, 只有 user 角色, 逐条累积。
2. **`session.fork`**: seed = 从已有 session 切前缀, **复制**既有 user/assistant/tool 历史,
   不能**构造**任意历史。
3. **`ctx.agents.create({ seed })`**(进程内插件): 可凭空构造任意事件序列, 只要满足
   "seq 从 0 连续、turn 闭合、无悬空 tool call"。

- **assistant append 只在进程内。** `Session.append('assistant/message', ...)` 是合法路径,
  有专门的 `createAssistantMessage` 构造器; seed 也接受 assistant/message (校验
  `source.kind === 'model'` 且带非空 provider/model)。**但 RPC 面没有这个字** — 没有任何
  方法能把一条 assistant 消息 append 进 session。
- 进程内注入 assistant 需要**伪造 model source** (`{kind:'model', provider, model}` 非空即过,
  不查真实性)。语义上是造假, 机制上能过。

**观点** — 无缝 compact 要"回填 assistant", 只有两条路: 进程内 `Session.append` (需伪造
  model source) 或降级为 user message (语义降级)。这是「历史注入三通道」命题的收尾, 也把
  "切不回来" 的根因钉死了: **dsh 把写历史的能力关在进程内, RPC 面只能 user message。**

---

## 四、one-shot subagent: 只能读不能驱动, 且照样持久化

**问题** — subagent_fork 能否 RPC 直接调? one-shot 跑完会不会落盘? 有没有删 session 的接口?

**探索路径** — 读 `subagents.schema.ts`、`subagent-fork-in-process`、`session-persistence`
coordinator、`workspace` archive/delete。

**初步结论**:

- **subagent_fork = 深拷贝前缀 + 一次性委托子任务**, 不是"共享上下文的并行化身"。子拿到的
  seed 是一次性深拷贝 (`completedTurnPrefix`), 从此与父解耦; 资源是"继承+收窄"
  (toolFilter/persona), 不是"分配"; 生命周期是父子单向委托 one-shot, 跑完即弃。
- **RPC 只能读, 不能驱动 one-shot。** `subagent.prompt`/`interrupt` 的 schema 把 mode 锁死
  `continuable`; one-shot 只能由模型在 tool 循环里调用 `subagent`/`subagent_fork` 工具创建。
  `subagent.list`/`history` 能读。
- **one-shot 也持久化。** 持久化层对 subagent 无 one-shot 特判: `session/disposed` →
  `retire` → `flush` 落盘。one-shot 跑完只是转入"冷"状态, 不删。`subagent.history` 的 cold
  读路径恰好证明它还躺着。
- **没有删 session 的接口。** `session.*` 全集 11 个方法无 delete; `workspace.archiveSession`
  只是往 `archivedSessionIds` 数组加 id (隐藏, 不删物理); `workspace.delete` 明确
  "retaining every session log"。jsonl 后端一个 session 一个目录一个 append-only 文件。

**观点** — "1000 次 subagent = 1000 份 100kb 上下文" 在源码层面成立: 每次深拷贝 + 一份独立
  落盘, 无 COW/共享/去重, 且没有删除口。真正缓解的办法不是等 dsh 给删除接口, 是回到
  「Memento 当共享层, dsh session 只承载增量」的框架 — 但那条又撞上"RPC 只能 user 重放"的
  成本。两个方向都贵, 只是贵的点不同 (fork 贵存储/CPU, 重放贵每次 token)。

---

## 五、dsh 单进程选型: 崩溃隔离靠纪律, 热加载分三层

**问题** — node 单进程性能好么? 一个 plugin crash 不就全崩? 有没有道理热加载?

**探索路径** — 读 cordis fiber 错误路径、loader isolate、client hmr。

**初步结论**:

- **性能不是瓶颈** — dsh 是 I/O bound (LLM 往返 + subprocess 等待), 不在 event loop。
- **同步 throw 被 fiber 兜住, 异步逃逸崩进程。** `_reload` catch 后 fiber 进 FAILED, 不崩;
  但 fiber.ts 原文: 异步 effect 逃逸时 "process-level crash is the honest outcome"。cordis
  是 **fiber 级逻辑隔离, 不是进程级隔离** — 靠插件守规矩, 不靠崩了也隔得住。
- **热加载有, 分三层**: loader 热换插件 (`Entry.update` → dispose → start)、isolate 热换
  service (symbol diff + reflect.notify)、client-hmr 热换前端 bundle (stat-poll + SSE)。
  进程内插件确实可热替换, 前提是插件守 effect 纪律、不逃逸。

**观点** — dsh 是「单进程 + 有热加载 + 崩溃隔离靠纪律不靠边界」: 用 fiber/effect 把插件圈
  在同一进程, 换热替换 + 零序列化, 代价是隔离兜底是"作者守规矩"。这对三方桥是直接警告 —
  桥插件在 dsh 进程内跑, 一旦异步逃逸就带着所有 session 一起崩。

---

## 六、收敛: web 跨进程的本质 = apiproxy 是进程内 plugin 当翻译官

**问题** — dsh 启动 web 不就是跨进程吗? 为什么 ghost 还要先写一个内核接入的 plugin?

**探索路径** — 澄清 web 跨进程面 (apiproxy) 与内核特权的关系。

**初步结论** — 这是本轮最关键的一次纠偏:

- **web 之所以能跨进程, 恰恰因为 `dsh-host-apiproxy` 是挂在 dsh 内核里的一个 cordis plugin。**
  它干的活 = 注册 HTTP 路由 → 收到跨进程请求 → 调内核 (`agent.followup` 等)。浏览器 (跨进程)
  能调 `session.prompt`, 是因为进程内有个 plugin 在当翻译官。
- 所以「web 跨进程」从来不是"没 plugin 也能跨进程", 而是"**已经有一个 plugin 在翻译**"。
  我一度误说成"直接 HTTP RPC, 不需要 plugin", 那是把"你不需要自己写"错说成"没有 plugin"。
- **内核特权物理上只在 dsh 进程内, 跨进程方够不到, 只能靠进程内某物伸手再翻译。** apiproxy
  是"某一个 plugin", 它选择性暴露了一部分 (session.prompt/history/fork/cancel); ghost 要的
  特权 (append assistant / 构造 seed / 动态 prompt) 它不翻译, 就得再来一个 plugin 翻译。

**观点** — 最终答案回到最朴素的一句:

> **ghost 要的内核特权 = 仿 apiproxy 再写一个 plugin, `ctx.webServer.register` 注册几个
> HTTP 路由, 加几个接口, 完事。** transport 直接用 dsh 已有的 HTTP 面, 不引入 zenoh、不引入
> zmq、不改内核。plugin 逃不掉 (特权只在进程内), transport 是可选的 (HTTP 最简)。

- 唯一要自己定的是**接口面**: 加什么接口、什么 payload、什么权限。一旦这个 plugin 进内核,
  它就有 apiproxy 同级的特权半径 (能 append assistant、能造 seed), 所以接口要窄、要自己把关,
  不能做成"任意 append 任意事件"的裸口。

### 被放弃的候选 (过程层留痕)

- **「dsh = matrix 上一个 service kind, 复用 matrix-operator 走 zenoh」— 已放弃。** 本轮
  曾想把 dsh 三方桥做成 matrix-operator 的一个新 service kind (kind=`"dsh"`, ghost 侧
  `operator.get/sub`, dsh 侧 TS plugin 用 zenoh-ts 对齐 queryable/pub/sub)。判断理由: matrix
  的通讯面本身早就成立 (matrix-operator 的 7 条致命问题是「批评施工模型质量」用的复工清单,
  不是通讯不成立), 但**内核特权桥接不需要 zenoh** — 因为 dsh 的 web 面 (apiproxy) 已经证明
  了「进程内 plugin + HTTP 路由」这条更短的路径, ghost 复用 dsh 已有的 HTTP 面即可, 不引入
  zenoh/zmq、不与 moss matrix 耦合。
- 这取代了上一轮 Open Problems 里「热数据桥接 = zenoh query 插件」的方向 — 不是 zenoh 插件,
  是 apiproxy 式 plugin + HTTP 路由。

---

(完 — 2026-08-16 探索轨迹, 收敛到「apiproxy 式 plugin 桥接内核特权」)
