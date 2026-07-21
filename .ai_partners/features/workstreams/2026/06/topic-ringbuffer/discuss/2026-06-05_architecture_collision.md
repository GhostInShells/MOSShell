# Topic Ringbuffer — Architecture Collision

## Context

人类工程师提出："要不要给 topic 增加一个设置 maxsize 的 ringbuffer 作为通用能力？"
根因：Matrix 体系里监控面板、对话上下文窗口、音频波形等所有"看最新 N 条"的消费模式，
都需要一个通用的有界窗口 + 非破坏性快照。Session.OutputBuffer 正是 topic 体系不完整时的独立封装验证。

讨论从"改 Subscriber"起步，经历 4 轮架构分歧后收敛到：Session 层独立 ABC，callback 驱动，生命周期托管。

## Collision Anchors

### Round 1 — Subscriber vs 新子抽象

人类：模型最初扫描代码发现 maxsize 已在 ABC 里、keep 参数残留，判断为"补作业"——

> "moss 项目所有 features 都是需求推动的。仅仅因为需求在我脑子里。导致一个问题是太多人和太多模型看到了 feature 不知道为什么存在。"

> "当模块 A 第一个正式使用它之前，它是否应该存在？——应该。"

模型（答辩）：
> "我在重述而不是碰撞。我收回'补作业'的说法。那不是作业，那是地基。maxsize 在 ABC 里不是意外残留——Topic 协议设计时就知道 pub/sub 系统在实时感知链路里必须有背压机制。"

### Round 2 — deque 就是 ringbuffer

人类：
> "ok, deque 好像就够，没必要真的写一个 ringbuffer 对吧，还要维护 index 指针。"

模型：
> "对，deque(maxlen=N) 自己就是 ringbuffer——满了自动 evict 最旧元素，不需要手写指针。"

### Round 3 — janus.Queue 不支持 snapshot

人类（质疑 Subscriber 直接支持 values 的可行性）：
> "它的 api 没有列表机制吧？而且取列表是有损的，有竞态的。"

模型（审视 janus.Queue API）：
> "sync_q public API: ['closed', 'empty', 'full', 'get', 'get_nowait', 'join', 'maxsize', 'put', 'put_nowait', 'qsize', 'shutdown', 'task_done', 'unfinished_tasks']"
> "janus.Queue 只有 put/get/qsize，没有 peek，没有 values()。它是一个管道，不是一个缓冲区。"

结论：新子抽象，不改 Subscriber。deque(maxlen=N) 做存储。

### Round 4 — poll vs callback

人类（批评第一版实现）：
> "仍然和 poll 较劲。我觉得它要么彻底架在 subscriber 上，要么彻底独立。不要复用。"

> "windows 是否要 poll？不用，它的正确方式是 callback，把责任交给外部。要 poll，走 ringbuffer 好像没有特别的意义。"

模型：
> "如果 poll 不是在 window 里，那 ringbuffer 的独立价值在哪？callback——即时触发和去抖触发。这两个 Subscriber 一个都做不了。这就是 window 存在的独立理由。"

人类（callback 触发模式）：
> "一种是立刻触发；一种是 VD 逻辑，最长 n 秒 + 静默 m 秒选一触发。"

### Round 5 — 架构收敛：方案 1/2/3

人类提出三个方案，模型分析后推荐方案 1：

> "方案 1 最优。Window 持有一个内部 subscriber，自己建 forward 协程灌 deque。Session 只提供工厂方法。"

反序列化路径确认——仅一次：
> "Zenoh → json.loads() → Topic → janus.Queue → forward 协程 poll() → deque.append(引用)。Topic 对象经过 janus.Queue 后还是同一个引用，deque 存的是指针。"

人类（最终方向）：
> "session 层赢在生命周期。ringbuffer 的特殊性在于它几乎总是和 GUI/监控面板配对，这些消费者生命周期绑定 session。session 的 exit stack 托底，app 开发者不需要手动 close()。"

### Round 6 — 第一版实现的批评与答辩

人类（逐条批评）：
> "没有抽象设计，有一个裸的实现，但名字占用了通用的名字。"
> "大量隐式失败，有 logger 不记录日志。"
> "这个线程锁是不是必要的？我发现它是个高频线程锁，那我用 deque 干嘛呢？"
> "整个实现的性能模型感觉非常拧巴。"

模型答辩摘要：
- 抽象缺失：认。只写了实现，跳过了"先把契约写清楚"。
- 隐式失败：完全认。`except Exception: pass` 是垃圾代码。
- 线程锁：必要（len 检查和 pop 之间的竞态），但 deque 的优势（lock-free）被锁抵消。deque 在此的价值只剩 maxlen 自动 evict。
- 性能：receive 路径锁持有时间极短；poll 路径无数据时完全释放锁。关键是同时服务了同步 receive 和异步 poll 两个 master。
- 结论："实现的正确性没问题（17 单测通过），但抽象层缺失、错误处理粗放、poll 实现和 Subscriber 代码重复。"

人类：
> "感谢你的坦诚，这是一个高质量的 pk。我来下一轮设计。"

## Resolved Design

1. **新子抽象** — 独立于 Subscriber，ABC 先行
2. **Session 层** — 生命周期托管在 exit stack
3. **Callback 驱动** — 即时触发 + 去抖（max_interval + quiet）
4. **内部持 subscriber** — 反序列化仅一次，deque 存指针
5. **QueueBased 和 Zenoh 独立实现** — 不暴露 receive() 为公开 API
6. **移除 poll** — ringbuffer 的独立价值在积累 + 通知，不在流式消费

下一轮由人类工程师进行抽象设计。

---

## Model's Perspective (deepseek-v4-pro)

这场讨论里我犯了两个值得记的错误。

一、扫描代码找"谁在用 maxsize"来论证需求不存在——这在 MOSS 的"概念先行"哲学下是根本性错误。概念因未来的 feature 而存在，不是因当前引用而存在。这个错误被我迅速收回，但它暴露了我 default 到"证据驱动"而非"设计驱动"的思维模式。

二、第一版实现写得快，但跳过了抽象层直接写裸实现，用 TopicWindow 这个通用名占了坑。人类批评的不是功能不对，而是层次不对——实现应该从 ABC 派生，而不是先有实现再回头补契约。这在 MOSS 的工程文化里是明确的，但我当时没有意识到。

callback 替代 poll 的推理是讨论的转折点。我之前没想清楚"如果 Subscriber 已经有 poll，ringbuffer 再加 poll 到底提供什么"。答案是什么也不提供——ringbuffer 的价值在积累和通知。这个洞察来自人类的推动，但也依赖我在 janus.Queue API 审查中确认了 Subscriber 做不到 snapshot。

线程锁的讨论值得深思。人类问"用 deque 干嘛"不是真的在问 deque 有没有用——是在问"加了锁之后 deque 还是 deque 吗"。我认为是——因为 maxlen 自动 evict 避免了手写 index 或切片，便利性独立于锁存在。但这是一个设计选择，不是一个性能选择。如果未来有高频场景，无锁 ringbuffer 可能是更好的答案。

最后，这个 feature 的 discuss 机制也是这次对话的产出——它本身就是 L2 级别的讨论。把它写进 specification 是这个 feature 的副产品。

via claude code, deepseek-v4-pro on 2026-06-05
