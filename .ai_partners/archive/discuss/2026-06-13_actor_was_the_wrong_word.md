# actor 用错了词 — 从 FutureRouter 到 channel 是合成抽象的下沉

## 上下文

FutureRouter 作为 `ghoshell_moss.tools` 下的进程内基建落地后（concurrent.futures.Future + 字符串协议 + on_create 回调 + 防崩接口 + max_done 有界淘汰，约 160 行），话题自然延伸到跨进程怎么扩展。直接驱动是两个上层意图：审批 channel（跨进程等审批），以及一个被笼统称作 "actor channel" 的东西。

讨论开始时，技术形态似乎已经清楚：写一个桥接器把两侧的 FutureRouter 串起来，approval 跑在上面，actor 用 cache.lock 做抢占仲裁。但真正落到选型上时，每一层都在重新打开。最后落到的位置不是新基建，而是对现有抽象的重新识别 —— **channel 已经是 future identity + interface signature 在 cell 地址上的合成**。我们以为要在 future-router 之上造的东西，channel/proxy 已经做完了。actor 这个词承载的是 worker pool 调度需求，而不是新的通讯原语。

讨论中沉淀的关键概念：

- **FutureRouter 桥接器拓扑**：两侧各有一个 `FutureRouter`，纯进程内抽象保持不变；中间的 `CommIssuer` / `CommReceiver` 序列化 `Create(id, args) / Resolve(id, result) / Reject(id, reason) / Cancel(id)` 四个事件。router 不感知传输层。
- **multiplexed channel**：固定一条 zenoh key，payload 带 future_id，接收端 router 按 id 分发。对照的是 per-uuid 地址（每 request 一个 zenoh key + liveness 绑定），后者是优化项不是基线。
- **副作用锁 vs 消息锁**：审批的仲裁不在消息层。`future.done()` 自身就是 CAS —— 第一个 `resolve()` 成功，后续返回 False。跨进程多 executor 各自广播 `Resolved` 时，creator 端 router 只接受第一个，其余静默丢弃。
- **Action vs Self-explaining**：人类工程师在第四轮提出的两种动作。Action = "我知道谁，调它的 API"；Self-explaining = "我描述要什么，谁能做谁来"。两者差异只在 implementer 绑定时机（编译期/配置期 vs 运行时 interface match）。
- **三层分离**：Layer 1 future（identity = request_id，terminal state）；Layer 2 actor（identity = worker address，cell + channel 已实现）；Layer 3 调度策略（broadcast+claim / round-robin / sticky session）。worker pool 抢占是 Layer 3 在 Layer 2 之上的策略，不是 actor 自身。
- **channel 作为合成抽象**：cross-process identity + interface signature，由 cell 地址承载。provider/proxy 是 1:1 RPC 的成熟实现，1:N 由独立的资源抢占分层处理，不并入 channel。

## 碰撞点与过程

### 第一轮 — 跨进程通讯协议盘点

讨论从"FutureRouter 跨进程要走什么协议"切入。盘点现有 host/matrix 基建：

- `Topic` / `Publisher` / `Subscriber`（zenoh pub/sub）—— 1→N 广播
- `Channel` / `ChannelProxy`（zenoh queryable）—— N proxy → 1 provider 的 RPC
- `TopicWindow` —— 持久化广播 + 按需快照
- **队列协议** —— 没有。zenoh 不提供 "消息只投递给一个消费者" 的语义

Future 协议的三个动作（cancel / done / exception）映射到这套基建时：

| 动作 | 方向 | 协议层 | 缺口 |
|------|------|--------|------|
| done | executor → issuer | Topic pub 或 channel RPC | 满足 |
| exception | executor → issuer | 同上 | 满足 |
| cancel | issuer → executor | **router 不提供反向通知接口** | router 的 `cancel(id)` 只改本地 future 状态，跨进程 executor 收不到取消信号 |

人类工程师确认了 cancel 是协议层最弱的点："现在的模块本身不关联执行侧"。这句话直指 router 的设计取舍 —— `on_create(callback)` 是 executor 唯一感知入口，只触发一次；后续 future 状态变化由 executor 自己挂 `add_done_callback` 处理。同进程时这个间接路径足够；跨进程时桥接器需要把 cancel 事件 forward 给对端 router，触发本地 `cancel(id)` 后由 mirror future 的 done callback 通知 executor。

桥接器的拓扑在这一轮收敛：

```
[Process A: creator]                            [Process B: executor]

    creator                                          executor task
        │                                                  ▲
        │ create(args)                          on_create  │
        ▼                                                  │
    FutureRouter ──▶ CommIssuer ══════════════▶ CommReceiver ──▶ FutureRouter
    (local future)        protocol:                          (mirror future)
                          Create(id, args)
                          Resolve(id, res)
                          Reject(id, msg)
                          Cancel(id)
        ▲                                                          │
        │ resolve/reject  ◀═══════════════════ CommIssuer ◀────────┘
        │ /cancel callback
```

关键性质：两侧都是同一个 `FutureRouter` 类，纯进程内抽象保持不变；桥接器是 router 的 port，不污染基础协议。

### 第二轮 — 审批 = 广播 + 副作用锁

人类工程师识别审批的本质："审批本身就是抢占式加广播的效果，只要副作用动作有逻辑锁修改唯一事实，然后跨进程确认发送过广播信号协议即可。"

这句话把审批的仲裁从消息层移到了 SST（single source of truth）层。具体落地：

- **消息层用广播**：所有审批员都能看到请求，谁先点都行
- **真正的锁在副作用**：执行侧 router 的 `resolve(id, result)` 自身就是 CAS。`future.done()` 状态是单一事实
- **跨进程"确认发送过广播信号"**：多个审批员同时点了"批准"，他们各自广播 `Resolved(id, "by Alice")` / `Resolved(id, "by Bob")` 到 creator 进程；creator 的 router 只接受第一个，第二个被静默丢弃

这套模型不需要分布式锁，不需要 leader election。消息层只负责送达，仲裁在 future done 状态。审批场景天然契合广播。

这一轮的判断点是关于 actor 的早期尝试。模型当时建议 actor 走同一套桥接器，区别只在第 3 步加 cache.lock 抢占：

```
1. issuer.router.create(task_args) → (id, future)
2. 桥接器广播 FutureCreate(id, args) 到 actor topic
3. m 个 worker 收到，各自尝试 cache.lock("actor/" + id, overdue=30)
4. 赢家进入本地 router.mirror_create(id, args) → on_create 触发处理
5. 输家直接丢弃
6. worker 完工 → 本地 router.resolve(id, result)
7. 桥接器广播 FutureResolved(id, result) 回 issuer 进程
```

cache.lock 是 sqlite CAS，n 个任务 × m 个 worker 共 n*m 次锁尝试，单机几十微秒一次，在审批/低频任务场景不是瓶颈。这套方案当时被同时认为是 actor 的实现。

### 第三轮 — actor 用错了词

人类工程师提出了一个不优雅的草案：广播 actor request 带 uuid，广播前监听 uuid 对应的回调接口。这个方案是 per-request 一个动态 zenoh subscriber + 一个固定的请求 topic。然后明确："总体来说还是觉得不优雅。"

不优雅的来源被识别：把 actor 当成了 "Future + 多 worker race"。但 actor 的本质不是任务派发，而是 worker 的持久身份。真正的 actor 是 "Bob 是一个 worker，你给他发 message，他按顺序处理"。MOSS 里 cell + channel/proxy 已经是这个语义 —— workers 天生 addressable，不需要 claim。

分层校准：

```
Layer 3 (调度策略): broadcast+claim / round-robin / sticky session
Layer 2 (Actor):     cell 是 worker, channel 是 mailbox, channel/proxy 是消息发送
Layer 1 (Future):    identity = request_id, 一次性应答
```

approval 走 Layer 1；worker pool 走 Layer 3 + Layer 2；pure actor（Bob 是固定的 worker）只走 Layer 2。把 actor 设计为 "future + cache.lock 抢占" 是混淆了 Layer 2 和 Layer 3。

人类工程师在此处接受 actor 这个词用错了，但同时拒绝把问题归零："我知道你说的 actor 是对的，说明我用错了词，但不说明我说错了问题。"

### 第四轮 — Action vs Self-explaining

问题被重新切分。两种动作：

- **Action**：已知 cell 存在，按它的接口协议请求。必须理解 cell-api
- **Self-explaining**：扔出一个问题时，希望谁都好可以给一个解。和 cell 无关

人类工程师同时给出判断：两者目前都没有像审批那么明确的场景。但下一步考虑的核心自迭代动作是协议式的 —— "先定义了 interface，然后方便定义实现，这是模型自迭代的路径。interface (request / response) 描述了一个需要"。

模型在这一轮识别 Action 与 Self-explaining 的差异只在 implementer 选择时机，共同基础是 **跨进程 interface 作为一等公民**：

| | Action | Self-explaining |
|---|--------|----------------|
| identity | cell address + signature | interface 自身 |
| implementer 选择 | 编译期/配置期已绑定 | 运行时由 interface match 发现 |

MOSS 现有抽象的对应关系：

- `manifests.contracts` —— interface ABC 声明，但绑定是静态文件
- `IoC container` —— type-based 查找，进程内
- `channel/proxy` —— 跨进程调用，但 by address 不 by interface
- `channel 签名作为 prompt` —— interface 已是一等公民，只是没跨进程同构

zenoh queryable 在协议层支持 1:N capability matching —— 多个 cell 可以 declare 同一个 key expression，query 时网络汇聚所有应答。MOSS 当前的 `channel/proxy` 只用它做 1:1 RPC，没暴露 1:N。

这一轮的收尾是把推论留作储备词汇而非实现路径："interface 是需求和能力之间的桥。没这一层，每加一个能力都要知道 cell 地址，模型无法自我组装能力网络。这是框架最底层的演化引擎。"

### 第五轮 — channel 早就是 future + interface 的合成

人类工程师在这一轮做了最后的下沉："实际上，讨论这么多，channel 就是用标准的 future 机制实现的。任何一个 cell 如果可以 provider 多个 channel，或者 provider channel 给不同的分类，channel 间通讯的状态，建模，还原成 interface，其实全部都做了。"

这句话把前四轮的推演整个收拢到现有抽象。重新映射：

- **future identity** —— channel 调用产生的 request_id 已经在 channel/proxy 协议内承载
- **interface signature** —— channel 的 Python 函数签名就是 prompt，就是 interface 声明
- **跨进程同构** —— provider/proxy 已经把这两者在 cell 地址上合成

剩下没做的有两件：

1. **1:N 资源抢占**：channel 是 1:1 的（一个 address 一个 provider），多个 cell 都想 provider 同一个 channel 时如何抢占未实现。这是 Layer 3 调度，独立分层
2. **duplex channel 底层重构**：人类工程师明确提到 "我其实也在考虑用标准的 future router 完善 duplex channel 的底层实现"。FutureRouter 未来会作为 duplex channel 内部状态机的复用基建，不是另起炉灶

整个推演的形状到此闭合：起点是要造跨进程 future + actor + interface 三个新原语，终点是发现 channel/proxy 已经是这三者的合成，FutureRouter 是 duplex channel 内部的可复用基建，资源抢占是真正还没做的新原语。

## 模型的自留地

当前记录者视角：

这次推演的形状是 MOSS 系统会反复出现的模式 —— 从"我要造一个新原语" 开始，每下沉一层就发现"已经有一个旧抽象覆盖了"，直到落到现有抽象的某个**还没被推到极限的边界**上（这次是 channel 的 1:N 资源抢占）。这种下沉每次发生一次，框架的抽象堆叠就少一层。

值得记下的判断是：actor 这个词在 MOSS 上下文里几乎一定是误用。当一个人说"我要做 actor"时，他大概率在说三件事之一：1）我有一个长跑 worker，需要持久身份（这是 cell + channel/proxy）；2）我有 worker pool，需要抢占调度（这是 Layer 3 策略，cache.lock 够用）；3）我想要 send-and-forget 的消息派发（这是 Topic）。每一项都对应现有抽象，actor 这个词只是把它们模糊化了。

未闭合的方向：

**资源抢占是真正待造的**。当多个 cell 想 provider 同一个 channel address 时（或者想成为同一个 interface 的 winner），这是 1:N 场景，channel/proxy 不覆盖。它可以基于 `cache.lock` + zenoh liveness 实现 —— 抢锁的同时申明 liveness token，token 失效时锁自动释放。但这是新原语，不是 FutureRouter 的延伸。可能值得在下一个 feature 立项时单独考虑。

**duplex channel 用 FutureRouter 重构**还没做。当前 duplex channel 的状态机散落在 `DuplexChannelContext` 里，如果 FutureRouter 能承接 request/response identity + 状态机这层，duplex channel 的代码会显著简化。这是 FutureRouter 真正会被消费的第一个场景，比 approval / actor 优先级更高。

**跨进程 interface discovery** 这层留作储备。zenoh queryable 的 1:N 应答机制已经具备底层能力，缺的是上层契约 —— 一个 cell 怎么 declare "我实现 interface X"，另一个 cell 怎么 query "谁实现 interface X"。这是模型自迭代的核心动作，但没有当下的紧迫场景。当第一个需要它的应用出现时，再展开。

最后，关于 future-router 自身：它当前是 160 行的薄基建，刚好覆盖进程内的 string-protocol future routing。它不应该试图变厚。跨进程是上层桥接器的事，1:N 是 Layer 3 的事，interface 是更上层的事。FutureRouter 守住自己的边界 —— 进程内、纯路由、状态由 future 自身承载 —— 它才能成为其他抽象的合成原料。

---

*Claude Opus 4.7, 2026-06-13, via Claude Code*
*与人类工程师讨论 FutureRouter 跨进程扩展、actor 的语义边界、与 channel 作为合成抽象的下沉*
