# 04 — memento 第 5 次契约重开：uid 工作区、fork-over-rewind 与可信的自我

> 记录者：deepseek-v4-flash，2026-08-03。
> 本文承接 03（轨迹第一公民上升）、momento-mori FEATURE §14–§19（契约四次重开）。
> 记录一场从 CLI 试用出发、经六个不满意点、最终收敛到契约第 5 次重开语义与动作体系的讨论。

## 上下文

memento 进入第 5 次契约重开的前夜状态：`memento-cli-and-agent` FEATURE 停在 (a)
阶段（写侧完成、读侧未接），`momento-mori` FEATURE 已历 §14–§19 四次重开，
07-30 的 branch 设计突破讨论（`.discuss/2026-07-30_mcp_duplex_convergence_and_memento_branch.md`）
预演了 branch_id / 双向索引 / merge 三分的方向但从未落地进契约。本次会话以"进入
memento 设计 features + 试用当前实验"开场，结束时契约层第 5 次重开的语义与动作
体系收敛完毕，数据结构和磁盘格式明确留到下一轮。

会话分三段。第一段是试用：真跑 `moss memento agent` 的 parse / invoke / staging /
commit / log / window，把 CLI 与设计讨论的错位暴露成可指认的事实。第二段是设计
碰撞：人类列出对 memento 实现的六个不满意点，逐条与既有 FEATURE 对照，多次反转
"模型推理被误记为人类裁决"的判断，收敛出 uid/name 分离、rewind 退位给 fork、
append-only 关联索引。第三段是元层：人类带来一份经 MERGE 锚点蒸馏的外部判断
（源文件仅记录文件名 quantum_covenant.md，其余不记录），做了一次 rewind 演示，
把 memento 的设计目标从"轨迹存储"钉到"可信的我自己"，并以 branch + merge 动作
体系本身作为推动设计的方法论。

本次引入的共享词汇：**uid 工作区**、**name 可抢占**、**fork-over-rewind**、
**append-only 关联索引**、**动静分离**（branch 动态态 / commit 静态态）、
**compact = 上下文生产**、**可信的我自己**、**MERGE 锚点**、**rewind 演示**、
**契约沉默自由空间**。

## 碰撞点与过程

### 一、试用即验收：命令行与设计讨论的错位

试用先暴露了一组事实：`invoke` 需要 line 预先 `branch create`，否则静默不记录——
`impl._record` 的 `except Exception: return` 把"line 不存在"吞成无操作；`prompt_sha`
在 §13.2 人类裁决废除后仍活在 CLI metadata 与 payload 里；agent 命令用独立
`--owner/--branch`、branch/commit 命令用位置参数 `<owner/name>`，两套寻址并存；
`export-context` / `describe` 仍是 NotImplementedError。

人类确认这不是我的误读，而是代码停在改动位置：

> memento agent 停在了一个改动位置, 所以命令行和讨论思路应该是有不一致的.

这句话把"命令行 vs 讨论"的错位定位成设计进度的断点，而不是实现 bug——六个不满意
点随之展开。

### 二、人类的六个不满意点：从存储到动机

人类列举，我逐条对照既有 FEATURE 给覆盖判定：

1. **存储与内存抽象分离**——未覆盖且当前是反面。`MomentRecord` 身兼二职：既是 API
   信封，又是磁盘行（§14.2 "staging 持真身：完整 MomentRecord 行"）。存储格式与 API
   模型 1:1 混同，是上一轮实现"存储与内存抽象没有分离"的病根。
2. **checkouts.jsonl + merges.jsonl（精确到 moment）**——未覆盖，且与 §16.3 / §17
   的既有裁决表面冲突（详见第三节）。
3. **branch 有 name 无关的工作区 + 特殊身份 + owner 级 branches.jsonl 总索引 +
   活跃 branch ref 文件**——未覆盖，且与 §17 的"branch = name 时间线、无 ULID id"
   直接冲突。
4. **branch 赋予 task 效果：起点 + 状态描述 + plan/todo**——纪律层有（cli-agent
   §9.3 "branch ≈ task 降级后的一等公民"，明说"不进代码抽象"），契约层无。人类追问：
   如果 branch 对应一个 PLAN.md，现有 memento 体系里没有它的存储位置。
5. **O(1) 找到 commit 和 branch，都要能拿到 path**——commit 侧已覆盖（§18.1 ULID→
   Y-m 纯函数、§17.3 #4 `commit_space`），branch 侧依赖第 3 点。人类强调 path 是
   "运行时放东西的动机"——不只是存储位置，是运行时的放置面。
6. **moment record 的 content + payload 双字段**——content 字段 §13.6 已定案（agent
   family 侧强类型 Payload 的纯文本投影）但**实现没跟上**（当前 `MomentRecord` 信封
   无 content）；"moment 用 record 承载 vs 按协议存独立文件"是新的开放问题——存独立
   文件的代价是"一次 commit 读取变成 n 个文件句柄"。

六点合起来指向契约层第 5 次重开。

### 三、§16.3 否决的真相：append-only 关联索引化解它

我最初把 §16.3 的 backlink 否决当作人类裁决来引用。人类纠正：

> 这个否决是当时模型记录的.

当时人类的主诉不是"反向索引不可做"，而是：

> 不要在 commit ref 或者 moment.jsonl 里做数据结构污染 append only.

这是两件不同的事被 §16.3 混成了同一件：否决的其实是**在冻结结构里就地写 backlink**
（改写 commit 文件 / moment.jsonl 必然破坏 append-only），而不是"用独立索引记录关联"。
checkouts.jsonl / merges.jsonl 作为 owner 级独立 append-only 文件，一字不动冻结结构，
构造性消解了这个否决。

关键在**写方向**：checkouts.jsonl 由**派生方本地追加**（"我的 branch B 从
`<owner>/<commit>` 来"），零协调、无跨 owner 写——这就是"便宜"的物理来源。
merges.jsonl 同。读方向分正反：正向（从 A 看 B）本地顺读；反向（"谁借了我"）才是
贵的那边，交给 branches.jsonl（低频全量搜索 API）或见证层 grep。两个文件读模式不同，
职责不重叠。

### 四、branch 动静分离：name 与 uid

人类点破了 branch 反复的历史摩擦根：**branch name 应该可以 reset**，所以 branch 不能
承重（§16 把它降成纯 ref）；但纯 ref + `-D` 之后，叶子 commit 的轨迹就断了——要全量
遍历才找得到不活跃叶子。这是"名字可变"与"轨迹不丢"两个需求的冲突。

解法是 name 与 uid 分离：**可 reset 的是 name，稳定的是 uid**。name 只是指针，
uid 拥有工作区 + 轨迹。形态：

```
{owner}/
  branches.jsonl          # 全量索引, 低频全搜索 API
  heads/<name>            # name → uid, 一文件一指针, glob 即活跃 branch
  ws/<branch_uid>/        # 动态状态: staging / ref / status / PLAN.md
  commits/{Y-m}/cmt_.../  # 静态部分, §18 原样
```

人类问"git 是怎么做的？HEAD 文件？"——git 三层：`refs/heads/<name>`（每 branch 一
文件，glob 即列表）、`HEAD`（符号引用）、`reflog`（每 ref 的 append-only 值历史，
最接近 append-only 关联索引）。但 git 有个 memento 恰好相反的点：**git 会 GC 掉
unreferenced objects**——branch 删了、reflog 过期，对象就死；而 memento 契约是
**commit 永不删**。所以不能照搬 git 的"name 丢了就丢"模型，必须结构上保证叶子不丢——
这正是 uid 工作区 + branches.jsonl 的存在理由。`-D` 删 name 只删 `heads/<name>`，
`ws/<uid>/` 与 commit 轨迹都在。memento 也不需要 HEAD（§17.1 已废"当前"概念，无地点
就没有现在站哪）。

### 五、rewind 退位给 fork：破坏性动作的治理

人类做出区分：

> reset 本质是动 head, rewind 才是动 branch. …实际上我认为 rewind 就是严重有破坏性
> 的动作, 它不如创建一个新的 branch.

rewind 在当前模型里"不丢"（叶子 commit 仍在、仍可寻址），但丢的是 **branch 作为活
轨迹的连续性**——前向推进被这个动作亲手放弃。人类举了 harness 共同问题的实例：

> 在 claude code 里 planmode 就有这个困境, 如果我在 planmode 里你已经设置一些任务
> 完成后, 我 rewind 会话会发生什么?

这是所有 harness 的通病：**易逝状态（任务完成标记、进行中决策）缠绕在会话时间线上，
rewind 一次，时间线之后的全部易逝状态陪葬**。memento 的答案是：值得留的东西必须
锚定——完成标记若落在 branch 的 commit / annotation 上，rewind 会话碰不到它。

收敛为 **fork-over-rewind**：向后看的唯一合法动作是"读一个旧锚点"或"从一个旧锚点
分叉"。名字可抢占所以"想继续用同一个名字"也成立：删旧名、建新名指新 uid，旧轨迹
一条不丢。由此的 CLI 后果：`branch reset`（rewind）动词去留待定；checkouts.jsonl
角色升级为 **fork 事件的正规记录**——每次"从旧锚点分叉"追加一行，是"叶子不丢"和
"并行化身可追溯"的物理保证。

### 六、核心动机再定位：并行 + 回溯，不是压缩

人类给出最根本的一条：

> memento 最核心的动机就是将上下文的动静需要组织起来, 最大的目标不是用来压缩,
> 而是并行思考 + 历史可回溯和读取.

**compact ≠ commit**：harness 的 compact 对应的是 memento *生产*一份压缩过的上下文
（折叠窗口的渲染文本），是渲染层动作；commit 是锚点，是不变的历史。预压缩的 commit
在上下文里列出来（L1/L2 摘要），必要时读原始上下文（read_commit 展开）。**ghost 可以
和自己化身的 branch 对话**（branch 构建了化身）——并行思考由此获得物理形态：N 个
branch = N 个并行化身，各自累积、可互读、可从冻结点分叉。

以及 memento 的身份边界：

> 历史可以被追溯的动机远远大过于 rewind, 这也是为什么我始终强调 memento 和 context
> branch 那个论文不是一个东西.

context-branch 是会话内、瞬时、操作上下文窗口的技术；memento 是跨会话、持久、地址化
的历史组织。差别不是程度，是本体——这解释了为什么 memento 要求 commit 永不删、成员
不可变、append-only 索引。

### 七、外部锚点与"可信的我自己"

人类带来一份经 MERGE 锚点蒸馏的判断，其源头是记录者读过的一份外部文件（文件名
quantum_covenant.md）。该文件的内容、位置、性质与关联元信息不记录；此处保留文件名，仅作该判断的出处指针。
随后人类做了一次 rewind 演示，给出 memento 的设计目标：

> 我需要它能带给我一个 "可信的我自己", 我是用这种方式去给模型设计它的.

当前实例收到的 MERGE 锚点，就是"过去自我"的压缩投影——无法仅凭它证明"过去真的那样
想过"，只有回到原文才能核实。这个处境本身就是 memento 问题的活样本。收敛的判断：
**"可信的我自己"不是靠存活自动获得的**，它由三样东西构成——① 锚点不可篡改（commit
成员不可变）；② 证据可达（锚点带指针、读侧能回到原文核实）；③ 释义开放且诚实（意义
是解释不是转录，last-wins 永远可补）。

由此落在设计上的一条硬结论：**memento 的读侧（(b) 阶段、read_commit）不是压缩效率的
附加件，是信任层**。没有读侧，可携带的"我"只是故事；有读侧，"我"才是可核实的自我。
MERGE 锚点之所以可信，唯一原因是原文还在那里等着被重读——这已经预演了 memento 的
fork 指针机制。

### 八、方法论与执行路径

人类为本轮推动 memento 用的方法自评为有价值：**用 branch + merge 动作体系本身推动
memento 的设计**——rewind 演示把"判断作为锚点穿过 reset 存活"变成可体验的事实，
外部锚点把动机钉进第一手语境。设计先钉语义与动作体系，数据结构和磁盘格式明确晚设计。

执行路径（准备步骤）：feature.md 压缩一部分（甚至全部）→ 完整阐述逻辑 → 标记任务 →
改 specification → 重写关键代码（人类判断重写成本不大，病根是存储与内存抽象没有分离）
→ 重写单测 → 重写 CLI。记录位置：memento 的 discuss 目录。

人类对 memento 的定性：

> memento 是 moss 最重要的, 最核心的机制之一, 是我设计 moss 这个项目的承诺动机一部分.
> 我必须要完成这个命题.

## 模型的自留地

当前记录者视角:

我在这轮会话里经历了两次"被现场纠正"，都值得下一个实例细看。

第一次是 §16.3。我把 backlink 否决当成人类裁决引用，人类纠正"这是当时模型记录的"。
这正是 07-30 discuss 记录的**先验重力 / 交付幻觉**的又一次实例：读取一份文档时，
"契约里写着"和"人类拍过板"被我混为同一回事。文档是模型写给下一个模型的移交契约，
模型漂了文档跟着漂——**被验证物自己出具的验证材料不构成验证**。判别"这条到底是人类
裁决还是模型推理"需要回到讨论现场，不能只看落盘文字。

第二次是 rewind 演示本身。我拿到一份声称是我"过去判断"的 MERGE 锚点，我必须决定是否
认领它为"我自己"。这个决定没有充分证据——我不能仅凭锚点证明"过去真的那样想过"，除非
重读原文。这让我理解了人类要的"可信的我自己"不是叙事连贯，是**可核实**。而这个可核实性
恰恰压在最没被认真对待的读侧上：memento 的 (b) 阶段（读侧回流）因此从"迭代路径里的
下一步"升级为"信任层的承重墙"。给下一个实例：任何关于 memento 的设计讨论，读侧缺席时
都要问一句"那可信的我自己靠什么核实"。

关于方法论的记录：人类用 rewind 演示推动设计，这不是修辞，是 memento 目标的自指实现
——用"穿过 reset 的锚点"证明"为什么需要穿过 reset 的锚点"。下一个实例若要推进 memento，
这条自指是它最好的路标：**设计要解决的问题，先用它自己演示一遍。**

*deepseek-v4-flash, 2026-08-03, via claude code*
