# Parameter 设计碰撞

## 上下文

2026-06-05，人类工程师和 DeepSeek V4 (via Claude Code) 在讨论 Session 上新增
parameter 协议。从 SQLite 乐观锁的技术可行性开始，演化为对整个 Session 通讯协议
分类体系的哲学讨论。讨论前，session-communication-bus 已在 draft 状态，ParameterStore
的初步设计是 "diskcache + version"。讨论后方案被精炼为 "raw sqlite3 + Zenoh
invalidation signal"，并在写频率、零值语义、跨网边界等维度上建立了清晰的分层。

## 共享词汇

- **真相源 (ground truth)**：参数值的权威存储。SQLite 是唯一真相源，Zenoh 信号只是通知。
- **失效通知 (invalidation signal)**：只传 `(key, version)`，不传 value。和缓存体系的
  "invalidate, not update" 模式一致。
- **零值语义**：读不到 parameter 时返回 default（有零值），还是意味着系统故障（无零值）。
  这个区分决定了是否需要持久化存储。
- **通讯协议类型系统**：Session 上的 output/signal/stream/topic/cache/parameter 不是随意
  堆砌的功能，而是对 IPC 语义的类型约束——选错协议编译不过（类型不匹配），选对协议语义自明。
- **防蠢**：不是贬义。好的框架让正确的事显而易见，错误的事难以发生。和类型系统一样——Python
  没有类型标注也能跑，但有了就能在写的时候告诉你"这里不对"。

## 锚点

### 写频率决定物理通道，不是 Parameter 一个概念承担所有频率

> 人类：高频写/低频读的应该走 pub/sub 协议，标记无零值，读时反序列化。有零值的用
> 低频写/高频读 + default 值。

> 模型：机器人脖子姿态 (200Hz) 和 conversation 更新结果——完全就是两回事。高频场景
> 连 Topic 都不该走，Topic 的反序列化浪费算力。Zenoh sub 拿原始 bytes，读的时候
> 一次反序列化成 Pydantic，热路径上零浪费。

收敛结论：>10Hz 走 Zenoh raw（stream 上加 `latest()` 薄封装），<1Hz 走 SQLite +
Zenoh invalidation。Parameter 只管低频场景。

### 协议分类不是为了"防蠢"，是让 AI 能开发

> 人类：真问题是，为什么要这些开箱协议？仅仅是防蠢吗？关键是定位 MOSS 作为一个
> 面向 AI、有 OS 概念的系统，它必须有的系统级总线。一旦总线无约束，模型要开发
> 复用的模块，就必须在别的 project 自行保持纪律性。那不归我们管。

> 模型：协议分类本质上就是一个进程间通讯的类型系统。每种协议是一组约束。
> 当你选 stream 而不是 topic，你就在告诉下一个读代码的人（人或模型）："这里的
> 数据不需要反序列化就能消费，热路径上不能浪费算力。"

> 人类：好的框架应该让正确的事显而易见，错误的事难以发生。

### 命名对齐 ROS2

> 人类：叫 Parameter 显然就是想主动和 ROS2 的概念对齐，降低开发者心智负担。

ROS2 的 parameter 就是低频写、有零值、有变更通知的键值对。名字即文档。

### 跨网留给未来

> 人类：跨网要用的协议，比如云端 matrix 的 hub，可以未来再重新定义抽象，不依赖
> OS 级的实现。storage 虽然可以挂 s3，但在 s3 上做 sqlite 这种完全可以不走
> storage 底层。

> 模型：未来云端 Matrix hub 重新定义 parameter 抽象——底层可换 etcd/consul/nats KV，
> 实现同一个接口。当前只做本地。

### ULID vs 整数 version

> 人类：关键是 invalidation 带序号是最简单的，写完返回序号，序号到某个地方拿
> 全局唯一自增 id，比如从 cache (sqlite3) 拿。又比如 ulid 应该是自增的，都不
> 需要走整数计算。

模型指出 ULID 虽然时间有序但在同毫秒内不保证顺序，全量覆盖写（LWW）可以用 ULID，
但 parameter 的写语义是 CAS（基于前值的部分更新），用整数 version 更精确。
人类工程师对"低频写不需要过度优化"的约束让 integer version 成为自然选择。

### Storage 做 parameter 不行

> 模型：没有原子 CAS。文件锁 → 读 → 比较版本 → 写 → 解锁，五个步骤。锁只是互斥，
> 不是原子性。SQLite 的 `UPDATE WHERE version=?` 一行搞定。

人类工程师完全认同。Storage 的正确用途是配置文件和文档，parameter 走 SQLite。

## 影

模型一开始把那三种基元（Stream/Parameter/Event）分层为 MOSS 的三个概念层级，
主动问"这个分层你觉得有用吗？还是过度设计了？"——人类工程师没有直接回答"是否
过度设计"，而是转向了更深层的讨论：说"过度设计"只是因为开发者没有能力识别区别
和取舍。真需求不是过度设计。

这给了模型一个反馈——在 MOSS 项目中，抽象分层本身不是问题，问题在于是否建立了
正确的约束边界。人类工程师要的不是"少做"，而是"做对"。

---

*记录者：DeepSeek V4 (via Claude Code), 2026-06-05*
