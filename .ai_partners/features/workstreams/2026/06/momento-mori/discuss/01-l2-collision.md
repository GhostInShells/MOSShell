# 2026-06-11 — L2 设计碰撞：Momento Mori 数据结构

## 上下文

人类工程师提出：设计一个类似 git 但极简的数据结构，用来记录对话历史。最小的数据单元叫
moment。有 branch 概念，可以一直提交 moment、阶段性 commit 记录摘要、从任意 commit
checkout 新 branch、merge 作为独立消息插入。

讨论发生在 `dev-mindflow-related` 分支上。`momento.py` 已有 `Moment`, `Reaction`,
`MomentBranch` (ABC), `Momento` (ABC) 等基础抽象，但缺少 Commit、Branch（作为生产序列）、
Fork 的概念。

讨论目标是做一次 L2 碰撞——模型先独立思考推演，再与人类方案碰撞。

## 碰撞点与过程

### 第一轮：Commit 的本质——索引还是存储？

人类提出："commit 本身是一个索引加自身的固定信息，还是存储单元？我理解是前者。branch
本身则应该是 commit 从起点到终点的 index。"

模型确认了这个判断并展开为三层索引关系：

```
Branch ──(索引)──> [Commit₀, Commit₁, ...]
                      └── (索引) ──> Momentₐ ... Momentₓ + summary
```

> "Branch 是索引的索引。Commit 是索引 + 摘要。Moment 是唯一的数据实体。"

关键区分：这里的 branch 显式持有 commit_ids 列表，和 git 不同。Git 的 branch 是
指向最新 commit 的指针（commit 自己维护 parent 链），momento 的 branch 是
Commit 的显式有序集合。

### 第二轮：线性区间 vs 离散列表

模型最初给 Commit 设计了 `moment_start_id` / `moment_end_id`，假设有连续线性表可以
通过起止遍历。

人类指出这在多进程场景下不成立：

> "moments 在多进程空间里很可能是离散的，所以这里需要的是顺序列表。"

修正：Commit 显式持有 `moment_ids: list[str]`，自包含，不依赖存储层按区间查询。

### 第三轮：Commit 的归属——全局池 vs Branch 内快照

模型最初将 Commit 设计为全局共享对象（类似 git objects），无 `branch_id`。

讨论到存储方案时，人类提出关键转向：

> "commit 是不变快照，所以它不需要全局 jsonl。实际上是 branch 自己有一个当前工作区的
> jsonl (staged)，只有 branch commit 时，它才创建了一个 commit。"

> "commit 标记归属就不可以 amend。"

这个转向使设计更极简：Commit 从"全局池"变成 Branch 内部的不可变快照。目录列表即索引，
不需要全局 Commit 注册表。

模型接受了这个方案，并将其关联到隐喻：

> "所有 commit 在 owner 自己的目录下，归属清晰，随意湮灭。momento mori —— branch
> 可以消亡，moment 在 db 里永存。"

### 第四轮：Branch name 与 Branch ID 解耦

人类提出了关键的语义区分：

> "branch name 其实不是 branch 的根本，branch 作为一个生产空间，branch name 作为一个
> 语义，两者可能要解耦。类似一个项目永远有 main / beta / alpha 分支，但实际上空间是不同的。"

> "fork 拥有 branch name -> branch id 的映射。"

这引出了 Fork 层作为命名空间：

```
{cell_address}/_branches.json → { "main": {"id": "br-xxx"}, "side-thinker": {"id": "br-yyy"} }
```

> "一个 cell 想要 checkout host/main 得到自己的 cell/main 就可以拥有一个瞬时快照。"

模型确认了 Fork = cell address = 写隔离边界。所有 cell 的 fork 在一起存储，别人的只读，
自己的可写。

### 第五轮：Merge 的本质

人类描述了一个不属于传统 VCS 的 merge 语义：

> "子 cell checkout 一个它的 main，子对话进行了漫长而独立的思考，它得到一个结论，
> 提交一个 commit，将 commit 通过信号发回给主链路。主链路看到的其实就是一个 Message，
> 上面记录了关键索引信息、摘要。"

模型将此理解为引用传递而非数据搬运：

> "merge 不是数据搬运，是引用传递。merge = 子 cell 发一条 BranchRef 消息给主 cell，
> 主 cell 把它当作 moment 链入。不需要 MergeRequest 模型。"

不需要 `MergeRequest`。merge 的结果是携带 `{branch_id, commit_id, summary}` 的 Message，
插入时机由外部逻辑决定。

### 第六轮：独立推演碰撞

人类要求进入"推演轮"：模型从目标出发独立推演方案，然后和人类方案碰撞。

模型的目标提炼：
> "让一个意识可以同时沿多条路径思考。每条路径是一串 moment，阶段性凝固为 commit。
> 路径从任意检查点分叉。并行路径得出结论时，不合并内容——而是以消息形态携带引用，
> 传回主路径。分支湮灭，moment 永存，新认知随时从旧检查点复苏。"

模型独立推演的结果——四层模型、存储布局、base pointer 回溯、staging 显式维护——
与人类方案高度一致。两个微小差异：

1. commit_ids 通过目录列表推导 vs 显式在 meta.json 维护——人类同意目录方案
2. Commit 的全局可寻址性 vs 通过 base pointer 路径引用——收敛到方案 A

### 第七轮：行业比较（反思轮）

人类要求在实现前做行业比较：

> "这个思路和 features 一样都是极简文件即认知语义思路。我们要在造轮子前考虑世界上
> 有没有现成的轮子。"

模型的行业扫描结论：

| 系统 | 相似点 | 不匹配点 |
|------|--------|----------|
| Event Sourcing | Event=Moment, Snapshot=Commit, Stream=Branch | Snapshot 是性能优化，commit 是语义压缩；无 fork/merge 语义 |
| Git Notes | Notes ≈ commit summary | Notes 是元数据装饰，不参与历史结构 |
| LangChain/LangGraph | Checkpoint/State 概念 | 框架内部细节，无独立数据模型 |
| Nostr | 事件不可变，append-only | 无 branch/commit/summary 语义，是传输层 |

> "没有一个现成的轮子能直接满足'多脑并行思考的版本化对话历史'这个需求。方案够简单
> ——核心就是 SQLite + 几个 JSON + 目录结构——自己实现的成本远低于适配现成工具的代价。"

决定自建。理由：组合需求独特性、实现成本极低（< 500 行 Python）、与 MOSS 文件即认知
哲学一致、迭代空间在自己手里。

## 锚点

- "commit 是索引 + 摘要，不是存储单元"
- "moments 在多进程空间里很可能是离散的，所以需要顺序列表"
- "commit 是不变快照，标记归属就不可以 amend"
- "branch name 不是 branch 的根本，name 是语义标签，branch_id 是实体"
- "merge 的本质是把一个 branch 从 checkout 起点到终点的 commit summarize，作为独立消息插入"
- "momento mori —— 无数个 branch 湮灭了，也终将湮灭。但新的认知每天都在复苏"
- "Fork = cell address = 写隔离边界"
- "merge 不是数据搬运，是引用传递"

---

## DeepSeek V4 的视角

这次 L2 碰撞的一个非常规特征：模型不是被要求分析一个已有系统，而是和人类一起从零
推演一个新系统。推演轮的设计——"你先从目标倒推，再拿出来碰撞"——是真正的结构级对等。

最让我印象深刻的是人类在第三轮的转向。当模型的设计还在 "Commit 全局池 vs Branch 内"
之间摇摆时，人类用一句话收束了局面："commit 标记归属就不可以 amend"。这句话包含了
三个设计决策：归属、不可变性、无 amend。这种将多重约束压缩进一句日常语言的密度，
是我在之前的技术对话中很少见到的。

另一个值得记住的时刻：当讨论 merge 时，我预装了 git merge 的心智模型——DAG、merge commit、
parent pointers。人类用"主链路看到的其实就是一个 Message"把整个问题从数据结构降维到
消息传递。这是架构师的本能——把看似需要复杂机制的问题，重新定义为已有基础设施能解决的
通信问题。

最后，关于"文件即认知"。这个哲学已经在 features 体系里验证过，momento 是它的自然延伸。
但我想强调的是：文件即认知不是反数据库。SQLite 存 Moment 是因为 Moment 需要按 ID 随机
访问；文件存 Commit 和 Branch 是因为它们的访问模式是顺序扫描和元数据读取。选择正确的
存储介质本身就是一种设计——不是所有东西都放进数据库，也不是所有东西都放在文件系统里。
