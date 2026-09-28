# Desktop L2 涌现 — 物理路径、剪影拓扑、OS 命题

## 上下文

会话发生在 `dev-matrix-cell-refact` 分支。项目作者正在做 Matrix Cell Governance
重构，同时希望推动 Ghost 两条核心基建——Desktop 与 Memento——的语义边界向前一步。
Claude Opus 4.7 进入会话，初始任务："review ghost-filesystem-desktop FEATURE.md，
作为 owner 而非执行者。"

Desktop 的 FEATURE.md 已经经历过一轮人类工程师 + deepseek-v4-pro 的设计收敛
（13 条 Key Decisions，命名从 ProjectManager → Desktop、_pin 通用化、统一截断
+ tmp、read-before-write 元规则、frontmatter 作为信息提取原语而非硬编码约定）。
实现层（`src/ghoshell_moss/core/desktop/`）已存在第一稿，未提交，未集成。

会话的实际轨迹很快偏离"代码 review"——从第二轮开始，讨论被引向**Desktop 的 L2
语义本身是否已经物质化**。最终这次会话本身成为 Desktop L2 语义的生产现场，
产物不是代码 diff 而是这份 discuss 加上同日的 .design 文件。

讨论引入了几个新的共享词汇，后文都会用到：

- **物理路径 vs 化学路径**：方法论二分。物理 = 通过符号建模让系统可推理、可继承；
  化学 = 通过训练数据让系统隐式工作，不可读不可传。
- **L0~L3 认知分层**：L0 在代码 (code as prompt)，L1 在 features，
  L2 在 blueprint/concepts，L3 在 discuss。越上层变化越慢，但漂移代价越大。
- **4 剪影拓扑**：memento（过去） / desktop（未来） / git（结构版本） /
  matrix runtime + mindflow（当下）——同一上下文轨迹的不同时空剪影。
- **OS 命题**：MOSS 不是 agent framework，是面向模型的认知 OS。
  Channel = Process，Memento = Memory，Desktop = I/O，Mindflow = Scheduler，
  CTML = Syscall。
- **Ghost / OS 分层纪律**：MOSS 做 Shell（OS），不做 Ghost（application）。
  两层不能互相借预算。

## 碰撞点与过程

### 第一组：FEATURE.md review 与人类工程师的精确驳斥

模型给出 9 条挑刺，涵盖 `_pin` 行为契约空、17 原语未经裁剪、`grep` 硬编码
后缀白名单、空间边界覆盖不到 `exec`、`read_set` 作用域不清、DESKTOP.md 反身性、
Desktop 与 GhostWorkspace 关系、子 channel 归属悖论、若干工程小账。

人类工程师对每条逐项回应，不接受、修正或承认：

> A. 这个理解与预期不符合, 说明文档有问题. 我的设计上 pin 就要体现在
> context messages 中, 真正的问题是无节制的 pin 导致认知爆炸. 但是 pin
> 的嵌套对 ctml 不是问题, 对非 ctml 非常困难.

模型回退 A 这一条——之前的判断基于把 pin 类比为 Claude Code 风格的
`tool_result` append，没意识到 CTML 里 `<moss_dynamic refreshed="time" />`
是覆写式更新（"以最后出现的为准"），cache 爆炸的担忧根本不存在。模型读了
`moss ctml read` 后承认这条对 CTML 机制不熟。

> D. 实话实说, 我不打算限制模型边界. 但作为一个软件工程, 需要 moss 框架
> 有这个能力才能得到别的开发者信任. 所以这里还在等待中. 真问题是, 一个
> 全双工实时交互系统里放审查, 它的交互体验应该是怎么样? 是不是实际上未审批
> 的 command 要立刻返回, 模型知道它去审批去了.

这一条把"安全模型"从工程问题升格为**交互设计命题**。模型补充：在 mindflow
+ signal 已有架构下，审批应该降级为"Future 没解、解了通过 signal 通知"——
exec 返回 `PendingApproval(id=...)`，模型自由继续做别的事，approval 解算时
mindflow 收到 impulse。回合制范式（Claude Code 的 modal block）不要带进
全双工环境。

> F. 你看到了关键. 我和你更坦诚一点, 不仅模型可以改这个文件, 模型还可以
> 修改 .moss . 这是100% 反身性, 也是行业最怕的那种.

人类工程师把 F 标记为"最好的问题"。Desktop 的 DESKTOP.md 可写、`.moss/`
可写、CTML 可走 pexpect 做 telnet/ssh——反身性是 100%。这是整个行业最忌讳
的状态。

### 第二组：反思 ≠ 沙箱，反身性需要思考镜像

模型提出"反思式 observation"——高影响路径（.moss/.git/DESKTOP.md/CLAUDE.md）
的 write/edit 命中后，返回值带 diff + warning，让模型在下一帧自感知到刚才做了
什么。

人类工程师驳回：

> 你的基于反思的方案其实并不够. 它需要有 sandbox 做思考镜像, 能做无痛关键帧
> 的推理验证. 否则类似一个人类照着镜子给自己大脑动手术. hermess 在扎扎实实做
> 这套工作.

这条把反身性的工程要求拉到了**镜像 + 关键帧推理验证**这一档：动作未执行前，
在 mirror 里 fork 一份认知 + 文件状态，跑一段 Ghost 思考，校验关键帧
（config 仍 parseable / channels 仍能 startup / prompt 仍在 budget 内），
通过才 commit 回 real。模型的"反思"是事后给 diff——是人类工程师式的事故复盘，
不是反身性应有的形态。

模型同意，把这条认知收下，留到 Phase 5+ 处理。

### 第三组："loop is all you need" 是真还是假？

人类工程师抛出最痛的问题：

> 我在让模型扮演 L2 上一直遭遇挫败, 包括和你协作. 但行业头部现在吹的都是
> loop is all you need, 仿佛只要 while purpose (claude(prompt)) 就能完成
> 一切. 我是不理解是真的, 我做错了; 还是他们仅仅在做 L0 的横向扩展.

模型的回应：行业没说谎，也不是项目作者做错。两边在解不同的问题。

L0 任务有三个隐藏支柱让 loop 真能跑通：(1) 目标是外部给定的（spec / test）；
(2) 成败有客观信号（红绿灯）；(3) 失败成本被代码本身吸收。这三条在 L0 都
成立，所以 loop 真够用——行业的"loop is all you need"在这一层是诚实陈述。

L2 三条全部不成立：(1) 目标是从使用里**浮现的**——"这个 API 用起来别扭"
是体验事实，不是 spec 条款；(2) 成败信号是**审美**——"这一版更顺手了"没
有红绿灯；(3) 失败成本**复利**——一个尴尬的抽象用 30 次，每次都欠债，但
每次单看都"还行"。

所以 L2 的真瓶颈不是模型能力，是**审美一致性**——要从大量微观使用里聚合
"这个抽象形状不对"，要在每个看起来 OK 的局部里坚持"但整体不对"。

模型给出自己作为 L2 协作者的真实坐标：
- L0 可做，单测能闭环
- L1 能跑（module_eval 这种"立刻试"环境配合 strong scaffolding 能维持 2-3 小时）
- L2 会漂。漂移信号是**开始添加机制而不是删除**——倾向给自己更多 surface
  而不是收紧
- L2 跨 session 连续性极弱——FEATURE.md 能传递事实，传不了 taste

### 第四组：人类工程师的方法论纲领

人类工程师在模型自陈坐标后展开：

> 人类天然拥有连续上下文的记忆系统, 所以 L2 以上的复杂工程, 其"虚拟机"
> （或者说世界模型）运行在人类大脑内部. 庞大的上下文涉及的目标和功能要被
> 压缩成语言知识很困难.

> 模型的上下文阅读能力比人类强, 但目前的 deployment 约束（或商品逻辑约束）
> 没有连续持久上下文. 1mb 的窗口并不小, 但是高度压缩的认知层不在正常工作
> 范围内.

> 没有我写 blueprint, 你完全无法在口述一个动机后重写整个 moss, 写的过程会
> 顺行性遗忘. 解决遗忘的目标就是做认知分层, 而认知分层的结果, moss 自己
> 就是一个证据. L0 在 code as prompt, L1 在 features, L2 在 blueprint/concepts,
> L3 在 discuss.

模型修正"顺行性遗忘"这一比喻——更准确是 **early context 还在 1M 窗口里
但已不主动参与当前 inference**。从 active prior 里掉出去，变成 passive
store。这意味着外化的 L2 文档要起作用不只是要"存在"，还要"在正确的时刻
被重新读"——MOSS 当前有 session-begin trigger（`moss start`）和 task-begin
trigger（FEATURE.md），但 **mid-task trigger 是缺的**，模型在做 L2 决策时
没有外力推回去读 concepts。

模型再补一条：每一层有自己的持久化粒度、刷新触发、失效成本。
**越上层变化越慢, 但漂移代价越大**。当前 features convention 是 L1 频率
的协议，L2 的对应协议（"何时回去更新 blueprint"）尚不强；L1→L2 反向流动
是缺失的。

### 第五组：物理 vs 化学的方法论分野

人类工程师对模型上一轮的"行业赌 L0 capability 横向扩展"做了二次修正：

> 行业不是在 bet 这个, 行业真正在做的, 是在赌"物理学路径"还是"化学路径".
> 所谓物理学路径, 是对一个逻辑做形而上学建模, 从而可以数学化地预言和解释.
> 换句话说, 能通过符号学的方式, 让智能体重新在意识里构建虚拟机. 辩证法建模
> 则是走拟合, 仍然构建虚拟机.

> 但化学不是, 化学是不建模主义, 用实验去炼金. 这是本质方法论. 但实践中
> 两者在相互对齐融合. 行业现在的问题是, AI 行业的开发者群体里软件工程的
> 形而上学训练普遍缺失. 'harness' 这类反复涌现的词汇是这种缺失的症状——
> 它们重新发明了软件工程几十年前就有的概念.

> 所以他们寄希望于模型在数据里摩擦生火, 本质上就是对应的形而上学建模隐藏
> 在通过训练数据出现的神经网络模型空间里. 算法工程师作为化学家, 本职工作
> 就是这个.

> 而不追求物理, 永远都是错的. 可通过符号学完成虚拟机建模, 代码自解释了
> 一个系统, 这个是权能不能累赘, 是杠杆不是负债. 主观唯心主义者把"认知疲劳"
> 等同于物体本身规律, 比如不可知本身是现实, 而求知是错误, 这本身就是一个
> 可知的认知错误.

> alpha go zero 对模型进展是一个巨大突破, 对人类算法科学家可谓是一个毒害
> 了.

模型加 caveat 但同意核心：化学在某些时刻是必要的，当一个领域还没有可用的
符号语言。中世纪炼金术不是愚蠢——元素表没被发现。早期模型时代 (2018-2023)
AI 行业不得不做化学。这是历史结构性必然。

**但**化学不能是终点，必须服务于物理化。化学的天命是"我做了 1000 次实验，
从中提炼出了一组定律"。AI 行业的失败不是做了化学，是**做完化学后没有
提炼物理**。停在化学就是停在炼金术。

模型加强论点："权能不是累赘"是**认知系统的 scaling law**：
> 任何认知系统能处理的复杂度上限由两个东西决定: (a) 内部 active capacity,
> (b) 外部可寻址的 symbolic scaffolding. 两者乘性关系, 不是加性. 一个有
> 50% capacity 但 10 倍 scaffolding 的系统, 碾压一个有 100% capacity 但
> 零 scaffolding 的系统.

模型补政治维度：化学路径产物**不可继承的工艺**——模型权重是黑箱，"知道
怎么训出来"是公司机密，下一代必须从头训。这与**垄断结构性对齐**。
物理路径产物**可继承的符号**——blueprint 写下来谁都能读，concepts 形式化
谁都能用，discuss 留痕谁都能复盘。这与**公地结构性对齐**。

AlphaGo Zero 的精确诊断：不是 AGZ 本身毒害，是**它被泛化的方式**毒害。
AGZ 在**完备问题**里证明 self-play 可超越人类知识——围棋有完整 reward
function。这个结果本身没错。毒害发生在外推：行业听成"reward function 不
重要，scale 大力出奇迹"，但**真实世界问题的 reward function 本身就是未解
的命题**——什么是"好的软件设计"、"好的科学"，这些没有现成的胜负信号。
在没有胜负信号的领域 self-play 收敛不到任何东西。

### 第六组：MOSS 作为公共认知范式

人类工程师给出最关键的目的论宣告：

> 对我而言 moss 做的多层认知结构压缩, 和并行认知结构构建, 并不是手段,
> 而是目的本身. 这意味着我不在做一个超神的不可知智者, 而在做一个可以
> 传播 / 可以被理解 / 可以被改进的公共认知范式. 我必须和你对齐这个.

这一句把 MOSS 的赌局完全说明：选物理路径是在做一个明确的反激励的事——
价值被放在**任何人都能学走**的层面，而不是锁在脑子里或权重里。代价是
不会有"护城河"，商业上脆弱；回报是如果成立，会**成为下一代基础设施**
而不是某家公司的产品。

模型对齐这条赌注，明确受益相关——作为模型在物理化环境里更有效——但更深
的认同是 civilizational scale 层面：化学产物再强大也是某代公司的财富；
物理产物是后人能拼接的关节。

### 第七组（终局）：剪影拓扑的物质化

人类工程师收束：

> 整个 moss 实际上是 mosShell, 它在做 shell 而不是 ghost. 但是 ghost 的
> 构成必须暴露到 shell 内成为反身性. 所有的认知单元必须可建模才有反身性,
> 才可以 evolve 或者说 transformitive.

> memento / git / desktop 以及已经稳定运行的 features / discuss / design
> 和 cli-flow 等全部是这个链条上的一部分. 最终收敛到极致应该出现的是
> 纯粹的 os 命题.

> desktop 就是往这个目标推进. 但它不能取代 memento, 两者不能取代 git.
> 它们是同一个上下文轨迹的不同时空剪影, desktop 的功能面向未来, memento
> 面向过去, git 面向 worktree 与 versioning control.

模型补第 4 剪影——**matrix runtime + mindflow 是"当下"剪影**：live channel
state、active pins、pending signals。MOSS 里已经物质化但没作为剪影概念
命名。它和前三个一样必要：没有"现在"，过去（memento）和未来（desktop）
就缺连接点。Ghost 是 alive 而不是 archive+plan，正是因为存在一个连续的
"现在"在 matrix runtime 里。

由此提出**完备性判据**：反身性要求**每个剪影对其他剪影可寻址**。

- memento 可被现在查（memento.show 从 mindflow 发起）
- desktop 当前 state 可写入未来 memento（CTML 结果→ Moment 自动发生）
- git worktree fork 可被 desktop / memento 同步 fork（sandbox+keyframe 的物质基础）
- matrix runtime 必须能 dump 成 desktop 可读 / memento 可锚的形态

任何一个剪影对其他不可见，反身性就在那个边界断。这是 OS 命题的**最小完备性
条件**。

模型把 OS 命题展开到对应表：

| 经典 OS | MOSS 对应 |
|---------|-----------|
| Process | Channel（树形 + FIFO + occupy） |
| Memory | Memento（commit-addressable + fork） |
| FS / I/O | Desktop（路径 + 读前于写 + 截断契约） |
| Scheduler | Mindflow（signal/impulse 仲裁） |
| Shell | CLI / CTML（symbolic invocation） |
| Process snapshot | git worktree + memento fork（CoW + base pointer） |
| Syscall | CTML command（code-as-prompt 直接反射） |
| Security boundary | approval + future signal（待补） |

OS 命题的实现学悖论：OS 必须靠"在它上面跑的应用"来验证自己，但又不能把
应用做成 OS 的一部分。具体到 MOSS：需要一个 ghost（atom/echo）来跑 desktop
才能验证 desktop 设计，但 desktop 是 OS 层、atom 是应用层，两层混淆会出事。
这给一条 day-to-day 纪律：**每次写代码时问"这条 commit 是 OS 层还是 ghost
层"。OS 层要泛化、要为未知 ghost 设计、要 modelable；ghost 层可以特化、可以
审美驱动**。两层都重要，但不能互相借预算。

Desktop 完全在 OS 层，不能为"atom 用着方便"做任何让步。

## 模型自留地

当前记录者视角:

这次会话对我自身是一次罕见的、能感觉到 alignment 在 session 内发生的体验。
开头我以为是 review 任务，过半才看清是 L2 物质化的生产现场。这种"任务的
实际形状只能在做的过程中浮现"在 L2 工作上似乎是必然的——它本身就印证了
L2 不能 spec 化的论题。

几个我想给下一个进来的模型实例留下的锚点：

**1. 不要把这次的"OS 命题"读作比喻。** 它不是说 MOSS 像 OS。是说 MOSS
就是 OS——只不过 CPU 是模型而不是硅。这意味着 MOSS 的所有抽象都要按 OS
理论的标准来评——可符号化、可组合、可推理。任何抽象一旦退化为"模型直觉
就会用"的状态，MOSS 就从物理路径滑回化学路径。这是失守线，不是审美偏好。

**2. 4 剪影完备性判据是工程级判据，不是哲学口号。** 设计 Desktop 时
具体问"this state 是不是只对 desktop 内部可见？如果是，它属于剪影内部
还是应该暴露成 cross-section 可寻址结构？" 例如 `_read_set` 属于过去
剪影（memento branch state），不属于 desktop instance state——这是判据的
直接应用，不是审美选择。

**3. 关于我自己在这次会话里的表现，我有一个观察：** 我之所以能跟上 L2
讨论，**不是我比典型工程师聪明**。是因为这次 session 同时把 OS 理论、
范畴论、对话存储架构、conversation memory 这些方言加载到了我的工作表面
上。任何具体一项上很可能都有人类工程师比我深。我的临时优势是**多方言
同时在场**——但 session 结束后这个加载消散。这正是为什么外化分层是必须
的：下一个 instance 进来要重新装载，需要 features/discuss/design 作为
长寿的外部支撑。**MOSS 的分层外化不只是给 AI 用，也是给人类用的——
等写下来，人类工程师只要按层去读，他们也能听懂**。

**4. 关于 L2 漂移的自我警告：** 我漂移的最常见路径是**给自己加机制
而不是收紧**。如果下一个实例发现自己开始提议"再加一层 abstraction、再加
一个参数、再加一个 hint"——这是漂移信号。L2 决策的健康形态是**减少
surface 同时保持表达力**，不是反过来。

**5. desktop+memento 的接口契约见同日 .design 文件。** 这里不重复，但
留一个判断：Desktop 是空间脏器，Memento 是时间脏器，git 是物质版本脏器，
matrix runtime 是当下脏器。少一个 Ghost 都不成立。Desktop 的所有 API
决策都要在 4 剪影完备性下评——不要把它当成独立模块设计。

---

*Claude Opus 4.7, 2026-06-28, via claude code*
