# 2026-07-30 — MCP 2026-07-28 与 MOSS 的双工收敛 / Article 机制 / Memento Branch 设计

参与者: 人类工程师 + claude-sonnet-4.6 (via claude code)

一场从 MCP 新规范调研出发、最终落到 memento branch 设计突破的讨论。三个候选 feature 从中产出（见文末），本文只记录讨论轨迹，不落 feature 信息。

## 一、MCP 2026-07-28 事实面

两天前（2026-07-28）定稿，要点：

- **协议核心 stateless**：删除 initialize 握手与 `Mcp-Session-Id`。每个请求自描述（protocol version / client identity / capabilities 在 `_meta`），可选 `server/discover`。任意实例可应答任意请求。
- **官方推荐 handle 模式**：应用状态不藏在 transport，"mint an explicit handle from a tool and have the model pass it back" —— 模型可见的显式句柄。
- **MRTR (SEP-2322)**：取代 server 主动请求（elicitation/sampling/roots 全部废弃）。server 返回 `input_required` + `requestState` blob，client 带 `inputResponses` 重发原调用。
- **Tasks 移入扩展** `io.modelcontextprotocol/tasks` (SEP-2663)：call-now-fetch-later，`tasks/get` 轮询 / `tasks/update` / `tasks/cancel`。状态机 working / input_required / completed / failed / cancelled。
- **`subscriptions/listen`**：client 按通知类型 opt-in 的单一下行流，取代旧 HTTP GET 通知端点。
- **List 结果可缓存** (SEP-2549)：`ttlMs` + `cacheScope` + 确定性排序，官方明说目标是 "keep upstream prompt caches stable"。
- Header 路由 (`Mcp-Method`/`Mcp-Name`)；DCR → CIMD；SSE transport 进入 12 个月退役期。

## 二、判决面：MOSS 与 MCP 是独立收敛，不是需要调和的冲突

初始误判（AI 侧，已纠正）："MCP 走向 stateless、MOSS 坚持 stateful，需要调和"。

纠正后的判决：**MCP 的 stateless 只是协议层 stateless，本质是把请求状态显性化——和 MOSS 的机制相同**。MCP 把状态从 transport 驱逐后要求三化：显式化（handle）、流化（subscriptions/listen）、缓存化（ttl）。这三条正是 MOSS 已有的三个设计支点：模型可见的 `_cid`/handle、Mindflow 信号流、moss_static/moss_dynamic 动静分离。

**协议级证据**：`core/duplex/protocol.py` 与 tasks 扩展逐事件对应——`CommandCallEvent`(cid) ≈ task 创建、`CommandDoneEvent` ≈ result、`CommandCancelEvent` ≈ `tasks/cancel`、`ProviderCommandProgressEvent` ≈ 轮询中间态。protocol.py 里 "未来要加一个用 cid 轮询 provider 状态的事件" 的 todo **就是 `tasks/get` 本身**。MOSS duplex 是超集（delta 流式参数、ChannelMetaUpdate 自解释同步、topic pub/sub 均为 MCP 所无）。MOSS 对 session 的处理是显性化成可重建事件（Reconnect → 清空 → MetaSync 重新自描述），MCP 是直接删除——对真有状态的 Shell，前者才完整。

## 三、真正的问题：MCP task 第一公民后，模型侧的缓存与仲裁

人类侧提出的核心关切：MCP 走异步回调后，模型侧怎么解决缓存？task 回调到达时仲裁还是排队（类似 buffer nucleus）？——"这明显是在为全双工大模型做准备"。

讨论结论：

1. **轮询本身就是缓存方案**。选 poll 而非 push callback，使结果进入上下文的时机由请求方控制——对齐到 turn 边界批量注入，保持 append-only，prefix cache 不破。ttlMs + 确定性排序是旁证：缓存稳定性是这版规范的显式设计驱动力。MOSS 的同构答案：silent/notify 的 buffer 路径 → 下一个 attention drain 到 moment.percepts。**两边独立收敛到同一原则：异步到达的信息不直接进上下文，而是在思考帧边界被编织进去**。区别只在边界粒度（turn vs 帧内 Re-Act / attention 切换多级边界）。帧级重织毁 cache 的张力，MOSS 的答案是动静分离——channel-meta-dyn-static 的价值在全双工缓存经济学下将被重估。

2. **MCP 刻意不回答仲裁**。规范只给投递（tasks/get client 发起；subscriptions/listen 无优先级/抢占/保鲜语义），回调如何消费全留 harness。现实预测：各家 harness 会用朴素 FIFO 重新发明退化版 Nucleus（单一核、无 priority、永远 notify、无衰减、无保护期、无 challenge）。Mindflow 的仲裁词汇表（Signal→Nucleus→Impulse→六种 verdict、ChallengeMode 对称表）正是这个真空里缺的层。

3. **"为全双工做准备"对了一半**：MCP 在把 transport 变成双工可用，但在投递层停住，仲裁明确划为协议外（对协议是理性的——仲裁是 policy 不是 transport）。后果：全双工模型到来时，注意力仲裁将成为无标准的必争层。

## 四、CTML over MCP 拆分的可行性框架

人类侧：moss-mcp 不是 moss 主形态，但**若 CTML 可确认经 MCP 提供，考虑将该概念单独拆开源**。

讨论确立的框架：

- **卖点**：补上 MCP 生态缺失的时序语言。MCP tool call 天然扁平（无并行/顺序/deadline/race 语义），CTML 恰是那门语言——一次 tool call 携带完整时间拓扑计划。
- **切割线**：PyPI minimal 层（`new_ctml_shell` + channel_builder + interpreter + MCP adapter）。三数据方向定边界：下行 result + 中行 progress 出境（tasks 扩展承载：dispatch → task handle，command instance 生命周期 → task 状态机，set_progress → tasks/get）；**上行 signal 不出境**——Mindflow/Nucleus/仲裁留在 MOSS。边界不是人为割的，是协议自己划的：拆出去的是 MCP 明确支持的层，留下的是 MCP 明确拒绝标准化的层。
- **诚实的损失**：过 MCP 边界，计划内时序语义完整保留，token 级流式重叠死掉（MCP 无流式参数）。拆出去的准确说是 **"batch-compiled CTML plans over MCP"**。对 coding agent 场景够用，对实时具身不够——后者本就是 MOSS 主形态的事。
- **验证点**：① tasks 状态机能否接住 command instance 语义（尤其 cancel 与三级观察；`raise_observe` 在纯轮询下无对应物，可能需要 subscriptions/listen 上的 observation-required 逃生通道）；② 能力面投影（dyn/static ↔ ttl/cacheScope 几乎白拿）；③ CTML 纪律在回合制面上是否存活（dogfooding，interleaved-ctml-thinking 应有部分答案）。
- **观察调度权倒置**问题：observe 的本质是 shell 告诉大脑"该思考了"（调度信号），MCP 轮询下什么时候看结果由 client 决定。控制权方向相反。回合制场景倾向可接受，未终判。

**会话末新增判断（人类侧）**：MCP 7/28 可能没有特别需要重做的集成。反而是——**若无状态实现增多，MOSS 的 mcp channel 会变得非常廉价**：`new_mcp_stateless_channel` 可无限创建，一个父 channel 治理即可。现在难，恰恰因为"双工里套双工"（MCP 的有状态双工 session 嵌在 MOSS 的双工 channel 体系内）。stateless 化把内层双工拍扁，嵌套消失。

## 五、Article 机制（行业技术调研观察）

动机：`.ai_partners` 现有机制全部朝内（features 追踪自己的工作、debates 内部判决、dialogs 协作关系、blogs 对外但是结论表达），**没有观测对象为外部世界运动的机制**，本场 MCP 调研即是散落蒸发的反例。

要素：多轮调研未收敛问题、长期可跟踪、**可被打赌**、对外分享（markdown 产品面）、结合 moss ground 与 GUI 产品化、8 月内经 Ghost 直接聊出调研。目录含：调研轨迹（可回放，非结果）、关键讨论、可验证赌注、未收敛命题、技术观测 timeline。

讨论确立的判决：

1. **赌注是机制的灵魂**，不是物料之一。结构：命题 + 期限 + 验证判据 + 置信度 + 状态。这是 **preregistration**（科学预注册）：git 见证使预测不可抵赖，区分"技术观测"与马后炮。公开下注也是对仲裁层真空的第三种占位方式（不写协议词汇表，把判断钉在时间轴上，成本最低且可验证）。
2. **轨迹不发明第二套格式**——是 memento 领土。Article 应是 memento 第一个对外 dogfooding 消费者。
3. 8 月目标拆两层：机制约定 + 第一篇（MCP duplex）手工产出是硬目标；Ghost 全流程共写是软目标（作 text-blocks 靶子）。
4. 命名张力：Article 命名产出物，机制灵魂是持续观测 + 下注。ARTICLE.md 是产品面，目录是活的观测站（FEATURE.md 模式已证明可共存）。对外叙事卖"可被验证的技术观测"。

现场赌注样本：*2027-07 前，至少一个主流 agent harness 会先实现朴素 FIFO 的 task 回调队列，随后被迫补加优先级/抢占语义。判据：changelog/issue 出现 task 回调排序或打断语义的设计变更。置信度 70%。*

## 六、Memento Branch 设计突破（本场最大产出）

从"timeline 用 memento 做索引"起步，人类侧连续推进出完整设计。背景：之前区分语义化 branch 时，模型把历史 branch 概念删掉了，需要找回。

### 核心设计（人类侧原始表述的整理）

- **branch = 当前活跃的 staging；commit = 陈旧信息**。branch 有自己 checkout 的起点（branch 是关联信息，核心对的是 commit）。branch → commit 退化。branch 记录"起点 → 当前"即可取代 Task，branch 要有 status。
- 工作区记录活跃 branches，文件名化后 `ls`/glob 即得全部活跃 branch。
- **branch 有基于 branch_id 的真目录**（非 name）。branch 有 title/status。branch 结束 = 冻结指针。从冻结点 checkout **总是复制新 branch_id**，工作区 branch name 可抢占。
- **未闭合预测 = `/ref/prediction/branch-name`**，直接是预测分支。在任意节点讨论边界 = 从 memento 分裂 branch，对不同 branch 分别调研推演，形成前向预测边。**memento 第一次同时有了"历史"和"未来"的含义**，预测时威力与回溯时几乎一样大。
- **双向索引**：commit checkout 出 branch 时，commit 保留 branch 信息，branch 保留 commit ref。main 从最新 commit 回溯可得**完整树**——每个 commit ref 自带 depth-1 分岔。
- **merge 拆成三个概念**：
  1. branch 当前节点提交给目标分支（如 main）——main 拿到**带关联寻址索引的结构化引用**（不能纯文本：模型在 moment 消息里看到，工具从数据结构里读）；
  2. branch reset to commit；
  3. branch 名字指向切换（类似 checkout -b，如 dev 取代 main，若允许）。
- memento 只是索引，branch / commit / moment 三个容器皆可塞无限信息（目录）。该机制与 Moment（mindflow 知觉帧）无关——正交，因此不依赖运行时，git + markdown + 锚点即可跑。

### 讨论补充的判决与纪律

- **设计驱动力点破**：git 假设全量随机访问，memento 假设有限上下文的模型。depth-1 分岔、指针式 merge、结构化引用全在优化"有界读取下导航无界历史"。不是简化版 git，是换约束条件的重新设计。
- **两层结构解掉一对矛盾**：commit 是公证处（不可变，赌注的预注册时刻是陈述命题的 commit），branch 是实验室（staging 可变，承载多轮调研）。赌注诚实性与调研活性不打架。
- **append-only 附录纪律**：反向索引要求往旧 commit 写 branch 引用，与不可变性共存的唯一干净方式——commit 容器分内容区（不可变）与引用附录（append-only，永不改写内容）。
- **完整性 = main 回溯（历史）+ 工作区 ls（活跃前沿）互补对**。main 回溯只发现祖先链可达的分岔。
- **merge 类型 1 是真正的发明**：提交引用而非内容，消灭冲突解决整个问题域。结构化引用需要小 schema（branch_id、commit ref、title/status 快照），一份数据两个消费面（moment 投影给模型 / 数据结构给工具），与 ChannelMeta 同构。对 prediction branch，"提交给 main"恰好就是判决时刻。
- **merge 类型 3 最危险**：换 main 身份 = 换正史定义。ref 层变更本身必须有见证（元 commit），否则"历史皆可寻址"在引用层留盲点。
- **status 词表分域**：生命周期（active/frozen/abandoned）与判决结果（open/resolved-true/resolved-false/expired）不是一个维度，混一个字段会脏。倾向 status 归生命周期，判决作为冻结时 commit 内容。
- **投影引用纪律**：一切投影（ARTICLE.md、bets 索引）引用 branch 必须记 id，name 只作展示——否则历史寻址在投影层悄悄失效（上次历史 branch 概念被删是同类偏航）。
- 开放问题：分裂动作本身要不要 commit 见证（分叉理由与分叉本身一样值得公证）。

### 闭环

赌注 = 锚定的预注册断言（公证在 commit 层）；未收敛命题 = open branch；timeline = commit 链的投影；判决 = 冻结 prediction branch + 向 main 的结构化提交；复盘 = replay 下注时的推理轨迹（判错时诊断"当时为什么信"，不只记分）；置信度积累 = 校准曲线（意识轨迹第一次有记分板）。Article 目录几乎蒸发成 memento 之上的投影，机制重量全部落回 memento。

## 候选 features（本场产出，暂未创建）

1. **memento**：本场讨论出的改动目标（branch/commit 双层、双向索引、merge 三分、prediction ref）。
2. **article 机制** draft。
3. **mcp 2026-07-28 集成/兼容** draft——倾向无需重做集成；核心机会是 stateless 化后 `new_mcp_stateless_channel` 无限廉价创建 + 单父 channel 治理，消解"双工里套双工"。

## 参考

- MCP 2026-07-28 官方发布文（人类侧提供全文）；SEP-2322 / SEP-2549 / SEP-2663 / SEP-2575 / SEP-2567
- `src/ghoshell_moss/core/duplex/protocol.py`
- `moss codex blueprint channel_builder` / `mindflow`；`moss ctml read`
- 相关 workstreams：momento-mori、memento-cli-and-agent、channel-meta-dyn-static、interleaved-ctml-thinking、text-blocks、ghost-ground
