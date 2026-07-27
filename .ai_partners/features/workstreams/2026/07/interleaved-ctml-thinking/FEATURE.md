---
created: 2026-07-23
depends: []
description: 约定 interleaved thinking 过程中模型输出 CTML 控制外部世界的交互范式： 边想边铺执行轨、observe 只看执行游标、wait/interrupt
  做思维剪枝。
milestone: null
priority: P1
status: completed
status_note: 5-verb tool surface landed and blind-tested via MCP
title: Interleaved CTML Thinking — 交错思考流中的 CTML 控制范式
updated: '2026-07-25'
---

# Interleaved CTML Thinking — 交错思考流中的 CTML 控制范式

> Use `moss features set-status interleaved-ctml-thinking <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

模型输出的 token 在技术上分三类：thinking / tool_use / output(text)。目标是把行业协议
推向：模型输出 **CommandToken 流**（`core.concepts.command:CommandToken`），输入/输出同时、
全异步、没有 ToolUse/ToolResult 这种阻塞点，最多有 stop reason。

痛点起点：`Thinking + Text` 太慢，思考过程无交互。借用 interleaved thinking，让 thinking 与
CTML(tool_use 形态) 交错输出，模型可以「想一段、铺一段执行、再想一段」。但早期把交互动词
暴力归并后，出现**思维奔逸**：需要观察结果的命令没被观察，模型跑两三轮才意识到。

本 workstream 正式约定这套控制范式，终局用在 **ghost thinking 模式**下用 CTML 做思考间交互。
MCP 只是验证载体（回合制），不是范式的归宿。

### 验证方法论（人类工程师约束）

人类工程师**不用直觉**，靠思维内推理-模拟-验证。但这套接口的用户是模型不是人，脑内模拟
「模型会怎么反应」缺 ground truth。收敛闭环 = 人类推理出候选建模 → 模型通过 MCP 当被试
跑真实行为 → 用行为数据裁决。本次讨论中模型已当被试产出真实数据点（见下）。

## Design Index

- Key design documents: `design/`
- Key discussion records: `discuss/`  ← 本次碰撞轨迹待补录（人类决定是否记）

## Key Decisions

### K1. 模型是「边想边铺轨、且跑在轨道前面」的写头，不是回合制调用者

append 模式下模型是写头：一边 reasoning 一边往一条执行轨道铺 CTML。**生成快、执行慢**——
笔尖到第 1000 token 时，执行游标才爬到第 300。「笔尖与读头的间距」是 duplex 的全部张力。
第一段还没执行完，第二段已在排队。这不是隐喻，是 `interpreter(kind='append')` 的实际行为。

**否决**：把「生成」与「执行」当同一时钟的回合制模型。反复回合制化是本次讨论中模型（被试）
的主要偏航模式，出现≥3 次，根因是 tool_use/tool_result 的先验。

### K2. 模型只有三个真实动作，围绕「我与游标的相对位置」

1. **继续铺**(append) — 默认状态，是介质本身，**不需要动词**。
2. **看读头在哪**(observe cursor) — 停笔瞥一眼游标：哪些 done / 哪条 ongoing / 哪些 pending。
3. **对读头前方轨道动手** — 两分叉：`wait_idle`(让读头跑完) 或 `interrupt`(掐掉未执行段、重铺)。

`interrupt` 的语义天然只对 **pending 段**有效——已执行的收不回（说出去的话），这是干净的
物理约束，不是设计选择。原语 `wait_idle`/`interrupt`/`wait` 均已存在于 `__main__`。

### K3. wait 契约必须烧进承诺时机，不能是尾随 flag（首次准确率的根因）

模型选错「要不要观察结果」几乎总发生在自回归的**那一个 token**：写下命令的瞬间已心理 commit，
`wait_done` 作为尾随 flag 是「动作已选定之后的第二决策」，被自回归连贯性偏置带跑 → 漏掉。
杠杆：让契约选择与动作选择塌缩到同一时刻。

**部分否决 / 修正**：一度倾向「契约主要由命令返回类型签名承载（`-> Observe` / `-> None` /
`@nonblocking`），动词只做例外覆盖」。这个方向不算错（`-> Observe` 机制确实已存在），但被 K4
证明**不是主要矛盾**——主要矛盾是观察盲区，不是动词歧义。

### K4. 「思维奔逸」的真根因是**观察盲区**，不是动词歧义 —— 本次最重要的翻转

MCP 实测（模型当被试）：模型在「想差不多了、该决定 wait 还是 interrupt」的关键帧，
`get_moss_dynamic_info`（现 observe 通道）**只返回 channel interface，不返回执行游标**。
铺了 5 条命令下去，模型看不见现实时间里执行到哪 → 只能猜 → 猜就是思维奔逸。

> 不是模型选错动词，是**模型做剪枝决策时眼前是黑的**。

数据在 `Interpretation` 里全都有（`success/pending/cancelled_tasks`、`task_done_at`、
`command_tokens`），**只是没投影给模型看**。修复 = observe 返回一张「轨道游标图」：按生成序
列出每条命令的状态(done+result / ongoing / pending)，而非现在的「有什么能力」interface 表。

### K5. observe 拆成两个语义：执行游标(拉/高频/轻) vs 能力重绘(推/低频/变更触发)

现在 observe 把两件事焊死：观察执行(游标) + 观察能力(dynamic interface 刷新)。二者由不同时钟
驱动，必须拆：

| 信息流 | 驱动时钟 | 获取方式 |
| --- | --- | --- |
| 执行游标 (progress/result) | **我的决策时钟** — 想差不多才看 | **拉**，绑在动词上 |
| 能力 meta (channel interface) | **世界时钟** — channel 上线/state 切换才变 | **推**，变更时注入 |
| 感知帧 (vision 等) | **世界时钟**，高 churn、只有最新帧有价值 | **推**，激进 drain |

**vision 之痛 = 误分类的症状**：它是世界时钟的推型数据，却被塞进拉型通道（observe 顺带刷 meta），
于是进入历史的时机绑在了模型的观察时机上——每瞥一眼游标就被迫吞一帧图进历史。两个时钟错位，
痛是必然，不是参数没调好。

**结论**：`observe`(=旧 wait_done) **只返回游标(progress+result)，meta 一律不走这个通道**。
不是折中，是分类学结论——游标是唯一时机与模型决策耦合的信息流。

### K6. 「原子 vs 极简」是假谱系 —— 建模不收敛的根源

人类一直在「observe 返回多少」这一根轴上滑（纯原子只回结果 ↔ 极简一把全刷），任何点都别扭，
因为问题是**二维**的：信息流由不同时钟驱动，一维上任何点都得同时伺候三种时钟，结构上无解。
二维分解后矛盾溶解：**动词侧做到纯原子（一个动词只管游标），上下文侧做到极简（模型不管理
meta/感知，世界自己重绘）**。两个理想各归其维，不再打架。

### K7. 右翼不做左翼的事 —— 范式边界（8 字蝴蝶）

模型交互是 8 字形：**左循环 = mindflow 治理的输入/感知，右循环 = shell 治理的 action+result**。
interleaved thinking 只治右循环。

- `wait input`(第 5 类) **外移 mindflow**，不进动词表。
- **外部世界变化的感知，放到下一轮去感知就够了**——不在右翼里模拟左翼的主动感知。
- `on the fly`(第 3 类) 溶解成介质底色，不设动词。

**主动放弃**：最初想在 interleaved thinking 里做「主动发起的模拟全双工体验」，让 MCP 也能
三循环。结论——**彻底放弃，不削足适履**。MOSS 架构本身有用、开创性不止一点，没必要为了让回合制
MCP 模拟全双工而扭曲范式。MCP 是有损投影，回合制是流式的退化，别把回合制假设烧进终局协议。

### K8. append 即 observe —— 一张投影 + 两根正交轴，K3/K4/K7 在此合流（第二轮体验前的收敛）

把 dynamic messages 从返回里剥掉（K5）后，`moss_exec(wait_done=False)` 的返回瞬间就是
**compiled 态**——`feed → wait_compiled` 即返回，不等 `wait_stopped`。这张 compiled 快照
和 observe 拿到的游标图**是同一张投影**：同一个 drain、同一套 payload 规则，只是被读取的
时钟点不同。由此三动作塌缩成「一张游标投影上的取样点」，区别只在两根正交轴：

| 轴 | append | observe | interrupt |
| --- | --- | --- | --- |
| 是否铺轨(写) | 铺 | 不铺 | 反向铺（收回 pending） |
| 读前阻塞多久 | 零额外等待（compiled 即读） | 可选 `budget=N` | 等到 idle |

**返回 payload 在两轴上恒定**——永远是那张游标图。于是没有「观察原语」这回事，只有
「一次游标读」+ 可选铺轨 + 可选等待。`observe` 是**退化的 append**（只读不写，故永不被拒）。

**唯一的对称性破缺，恰好落在 fail-closed 上。** 把 append 拆成读 + 写：

- **读**：无条件成功，永远返回游标图（失败就在图里）。
- **写**（铺新 logos）：**存在未处理的 `is_critical_failed()` 时被拒**——读照常返回，
  但新段不进 compiled。模型想继续铺轨，必须先当场理解并处理失败。

所以 `append = 无条件的 observe + fail-closed 的写`。这不是被动「把失败粘在 buffer 等模型自愿来看」，
是**互锁**：append 是模型前进的必经动作，前进即触发感知。纯拉侧靠「前进即观察」做出了 push 效果，
无需左翼。被拒的 append 天然就是 K2「actually, I'm just going to...」剪枝的触发器 —— 失败联锁
即 re-plan 触发器，不是额外机制。

**三条决策在此合流**：
- **K3**（承诺时机）：每次 append 本身就是一次 observe，模型不再需要在 commit token 那一刻做
  「要不要观察」的第二决策——下次铺轨无条件带回游标图。commit 时刻剩下的唯一强制决策是
  fail-closed 逼停，而它按定义漏不掉。「首次准确率」问题溶解。
- **K4**（观察盲区）：盲区只能存活在「既不 append、也不 observe」的那一刻。
- **K7**（右翼边界）：而那一刻按定义就是「不再铺轨、移交下一轮」——正是左翼该接手处。
  right-wing 能关的全关了，收尾静默段本来就归 mindflow。

**残余边界（诚实标记，非缺陷）**：append 联锁只覆盖「模型仍在铺轨」时段。若模型思考结束、
不再 append，此时一个 fire-and-forget 后台任务才 critical fail——无 append 可拦，这条落到
下一轮感知（mindflow）。分工线正好落在 K7 边界上。MCP 阶段验的是「fail-closed 后 critical
漏检率降多少」的相对量，不指望绝对归零（无左翼，push 不了）。

**投影 payload 规则（drain 分层）**，游标图按此投影：
- **存在性与 payload 丰度解耦（K9，重大发现）**：存在性由 `task.done()` 定，payload 丰度另说。
  空串 / None 都是合法 outcome，是「存在但无内容」，不是「不存在」。旧投影用「长度>0 /
  not is_empty」当存在判据，把两者焊死 → 一条空返回命令的**存在性从投影里蒸发**（B1 真根因）。
- **作用域收窄（人类定，纠正模型的过度外推）**：非空占位**只对 `observe=True` 的命令做**，不是
  每个 command caller 都逐条记录。模型曾主张「所有命令存在性都不该蒸发」→ 被否。关键：
  `on_done_task:219` 里 `is_critical` 自动并入 `observe=True`，所以「只保 observe=True」已天然
  覆盖 critical failure，不存在「失败漏在 observe 门外」的缝。非 observe 的空成功照旧折叠省 token。
- 空成功（`success` + 空 result + `observe=False`）：**折叠成计数**，不浪费 token，**连身份都不必给**。
- 未来优化（体验时增量，不现在定）：时间戳投影（可能吃太多 token）、返回值包裹格式。
- 运行中：`callername + progress`（progress 是活的中间态串）。
- 成功非空 result：**drain-once**，浮出即推进水位线（走全双工 buffer / `on_task_done`）。
- `is_critical_failed` / `observe=True`：**drain 后仍粘住**，不 TTL、不自动清，直到显式 acknowledge。
- `output` 消息道（已经 say 出去的）：只记「已发出」，**不重放内容**（否则重复入历史，vision 之痛的另一入口）。
- stop reason 是分类不是二值：全跑完 / 某条 observe / exception / interrupted / timeout。
- 布局：整张图按时间序，**底部 = 当下**（ongoing/pending 计数/stop reason 在底），历史 result 在上老→新。

**K9. 空 payload 的合法包裹约束 —— `is_empty()` 丢弃不是纯 bug，是在挡真实协议约束。**

`CommandTaskResult.as_messages()`(command.py:911/915) 与 `Interpretation.on_done_task()`
(interpreter.py:224) 三层串联，每层用 `is_empty` / `len>0` 过滤空消息。**这不能简单删掉**：
主流消息协议（尤其 Anthropic 不支持 role）下，一个**没有 xml tag 包裹的空字符串 content 是非法
协议体，可能直接搞崩请求**。旧代码丢弃空消息，是在防这个。

所以修正不是「别丢空的」，是**给空 outcome 一个合法的非空包裹**：空 result 投成带 cid 的最小合法
标签（如 `<command name="...:cid" at="..."/>` 或 `<result command="...:cid">(no output)</result>`），
既满足「存在性不蒸发」，又满足「协议体合法非空」。**约束点在 `as_messages` / `on_done_task` 这一层
定一条更好的规则**，而不是在 MCP 拼接层补丁。这条同时关掉 B1（存在性蒸发）和协议崩溃两个风险。

**数据源纠偏**：游标图建在 **`Interpreter.managing_tasks()` 的活 `CommandTask` 指针**上
（`.state` / `.progress` / `.task_result()` / `.done_at`），**不是** `Interpretation.status_messages()`
的聚合计数快照——后者拿不到 progress、拿不到逐条 result。拉/推分界下沉到 task 粒度：
**活的读 interpreter，盖棺的进 buffer**（`on_task_done` 喂，水位线挂 buffer 不挂 interpreter，
因 append 换 interpreter 会丢）。

**observe=True 语义纠偏（回合制污染第 4 次复发的修正）**：`CommandTaskResult.observe` docstring
的「停止、取消后续」是**默认 Re-Act 协议**（非原生协议下强造观察回合）。interleaved 里模型本就
交错，**observe=True 不强停**——退化成「下次观察时必须浮出」的 drain 优先级标记。一条 ctml 可以
全跑完、n 个 observe=True 一起浮出。别把默认停机语义套进目标协议。

**放弃 `InterleavedLogosThinking` 抽象**：搞抽象只服务 ego（证明「我预判 CTML 会过期」的前瞻）。
到 MCP 等场景为了和 CTML prompt 对齐，必然具体说 CTML。`Interpreter` ABC 已承载 feed/commit/
managing_tasks/observe，无需新抽象。三动作接口值不值得固化，留给体验数据裁决，不现在推理。

### 7 类交互预期的最终归并

| 原始分类 | 归并去向 |
| --- | --- |
| 1 quick reaction | `react`(phatic，即时 flush，永不 stop) — 候选新原语，少见故值得显式 |
| 2 fire-and-forget | 默认 append，`-> None` / `@nonblocking` 签名承载 |
| 3 on the fly | 介质底色，无动词 |
| 4 observe result | `observe` = 一次游标读（退化的 append，只读不写）；append 本身即 observe（K8）。**非强 stop** |
| 5 wait input | 外移 mindflow（左翼），不进动词表 |
| 6 interrupt | `interrupt` 原语（已存在），思维剪枝「actually...」 |
| 7 wait max N | `observe(budget=N)` / `wait_idle(timeout=N)`（已存在） |

真正需模型**主动选**的动词只剩 `react` 与 `interrupt`——都是「打破默认流」的强信号，天然显眼、
不易漏。其余沉入签名类型或介质。

## Implementation Notes

- **stop reason 类型化**：只有 `observe` 族合法产生 stop reason，`react`/`cast`/append 不停流。
  呼应「无 ToolUse/ToolResult 阻塞点，最多 stop reason」。
- **缓冲不是问题**：`Interpretation`(`core.concepts.interpreter`) 已是完整全双工缓冲，
  progress 已实时提交，把 result 也实时挂上即可。`call_soon=True` 的 cancel 已产出 Interpretation。
  一度误判为「需新建 pending-results 缓冲、要动内核」——**否决，缓冲早已存在**。
- **context messages 是推通道的实现**：interleaved thinking 无 context messages 协议，历史由
  外部拼接（可 pin / drain）。这不是协议缺失的权宜，就是推通道本身。游标消息=消费后 drain；
  meta 消息=pin-latest 同名覆盖（instruction 已有「moss_dynamic 以最后为准」语义，差在拼接层
  真的丢旧的）；感知帧=only-latest 激进 drain。
- **MCP 验证协议（待执行，本次未落码）**：改 `cli/moss_as_mcp.py:bootstrap()`（很薄）——
  `observe` 只返游标视图(从 Interpretation 投影)，meta 变更作独立分区消息。裁决指标：
  ① 该 observe 时首次即 observe 的比率；② 冗余 observe 次数；③ interrupt 时机误差
  （游标已过某点还试图反悔）。A/B 对照：游标纯净 vs 游标+meta 混合。
- **落地面**：现 MCP 四工具（`moss_instruction`/`get_moss_dynamic_info`/`execute_ctml`/
  `interrupt_execution`）是 `MossRuntime` 方法薄包装。`MossRuntime.moss_observe` 现读当前
  interpreter 的 `status_messages()` + 顺带 refresh_metas —— 正是 K5 要拆的焊点。
- **实现策略：纯增量，不改旧路径**。新造游标投影 + 新工具面，旧的 `moss_observe` /
  `Interpretation.status_messages()` / `moss_exec` 原封不动 —— 它们即 A/B 的**基线组**，
  且不惊动 ghost runtime 等其他消费者。
- **新投影必须统一承载两类错误**（第二轮体验 traceback 逼出）：
  - 运行期 `is_critical_failed`：task 已进 `managing_tasks()`，`state=failed` + `errmsg` 有内容
    → 从活指针投影。
  - 编译期 `INTERPRET_ERROR`：命令没进 `managing_tasks()`、轨道没铺成，异常在
    `interpreter.py:557 wait_compiled` 抛出（`self._parsing_exception`），当前冒泡成 MCP tool error，
    不进任何投影 → 新投影须 catch `_parsing_exception`，投成游标图里一行 stop reason。
  - 这两者是 K8 fail-closed 的孪生：一个「写被未处理失败拦」，一个「写自身编译即失败」。

### 第二轮体验基线（2026-07-24，CTML v1 英文版，当前 MCP 面，8 次调用）

**测量目的**：改造前的基线组，供 A/B 对照。以下是**旧路径实际行为**，不是目标行为。

- **B1 结果全黑 —— 更正：不是回归，是投影层拿长度当存在判据的设计缺陷（见 K9）**。四条读路径
  （append compiled 态 / `get_moss_dynamic_info` / `execute_ctml(wait_done=true)` / `<observe/>`+wait_done）
  都只回聚合计数。初判为「上一轮 wait_done 能带回、现在不能的回归」，**读码后推翻**：
  root 是 `as_messages`/`on_done_task` 三层 `is_empty`/`len>0` 过滤（command.py:911/915,
  interpreter.py:224）。上一轮偶尔能带回，只因那次 result 恰好非空、过了长度关；空 result 命令
  则连存在性一起蒸发。**不是 commit 回归，是存在性判据错误。修复见 K9。**
- **B2 失败静默**：`ls /nonexistent_dir`（`@observe` 的 exec）快照显示 `done: 1`，errmsg 无处可寻。
  静默失败最纯形态（项目已知失败模式）。
- **B3 计数无身份**：四次快照给 `cancelled: 4/2/2/1`，无法判断哪些被取消、为何。
  → 补 payload 规则：**cancelled 非零也需身份，不能只计数**。
- **B4 MCP 反转写头/读头时钟比**：append 回合延迟 ~15-20s，轨道执行 ~3s。流式里笔尖跑在游标前
  （K1 张力），MCP 里游标恒跑在笔尖前，compiled 快照几乎总是「全跑完」。**「MCP 是有损投影」的
  精确机制 = 反转时钟比**。K1 张力只能在流式载体复现，MCP 验不了。
- **B5 联锁不存在**（预期基线）：制造 INTERPRET_ERROR 后下一次 append 无阻拦通过。fail-closed 是纯增量。
- **B6 编译期错误冒泡成 tool error**：`<totally.fake:command/>` → INTERPRET_ERROR 直接抛出成 MCP
  tool error，不进任何投影（见上「两类错误」）。
- **B7 CTML v1 英文版盲测**：
  - 首次可用性好，8 次调用 1 次 parse 错：`text__` body 裸 `&&` → parse fatal。文档说 XML-like
    content *should* wrap CDATA，但裸 `&` 也炸。建议红线改为「body 含任何 `&` 或 `<` 必须 CDATA」，
    首错率可归零。报错质量高（行列号 + 提示）。
  - **文档兑现不了的支票**：instruction 承诺 `<result>` 会在后续消息出现、`@observe` 要求「stop and
    wait for the observation to arrive」——但当前 MCP 面 observation 永不 arrive（B1）。按文档等 =
    等到空，这本身是首次准确率隐患。改造修通 result 投影后此支票才兑现。

**改造优先级(数据钉死)**：① result/游标投影修通（`managing_tasks()` 活指针 + `on_task_done` buffer，
兼容两类错误）→ ② fail-closed 写拒 → ③ MCP 面对齐。**先 ① 再 ②**：否则 fail-closed 拦下失败后
模型仍看不到 errmsg，联锁退化成「不让过又不说为什么」。

## K10 落地：跨-interpreter 历史丢失 —— 已选「标准化」路径 (2026-07-25)

**决策**：走「标准化」——MOSShell 原生一个跨-interpreter 的 shell 观察器,
生命周期独立于单个 interpreter, 无消费不堆砌。这就是 K8「水位线挂 buffer 不挂
interpreter」缺的那个 buffer 的正式形态与归属。

### 落地架构（三层）

```
[MCP 层]  cli/moss_as_mcp.py:bootstrap
          ├── 旧 4 工具原封不动 (A/B 基线组)
          └── 新 4 工具 (K1-K9 收敛): ctml_append / ctml_peek / ctml_observe / ctml_interrupt
               |  server-scoped watcher, 挂在 async with moss_host.run() 之内
               v
[Host 层] host/interleaved_thinking.py:InterleavedThinkingToolset
          |  订阅 shell.Tracer 钩子, 缓冲 ShellEvent, 提供 4 原语:
          |  buffered / drain / status / wait_interpreter_done
          v
[Core 层] core/concepts/shell.py:Tracer Protocol
          |  5 方法: is_running / is_closed / on_task_pushed / on_task_done / on_interpreter_stopped
          |  fire-and-forget, 转交职责不做防御 (tracer 自持 is_closed 语义)
          v
          core/ctml/shell/ctml_shell.py:CTMLShell.add_tracer
          core/ctml/interpreter.py:CTMLInterpreter.on_close_callback
          (interpreter close 是 on_interpreter_stopped 的唯一权威 fire 点,
           覆盖 async-with exit / stop_interpretation / shell 退出 三条路径)
```

### 关键设计沉淀

- **Tracer Protocol 而非多回调注册**：加事件类型 = 加 Protocol 方法 + 加 tracer 实现,
  不改 shell add_xxx_callback 的散乱面。
- **exit 取值点选在 `Interpreter.close()` 末尾**：唯一 fire 点, shell 层不再各自 fire,
  避免了 async-with 退出路径漏发的隐蔽 bug (Step 2 修复过一次)。
- **task_done 不唤醒 waiter**：`wait_interpreter_done` 语义是「等 interpreter 到 idle」,
  不是「等任一 task done」。分离两根信号避免半唤醒竞态。
- **K9 空 outcome 兜底在 TaskDone.as_message 层**：空 result 投成
  `<result command="...">(no output)</result>`, 存在性不蒸发 & 协议体合法非空。
- **InterpreterStopped 只在有 exception 时入 buffer**：清洁停止表现在
  `wait_interpreter_done` 语义里, 不用生成噪音事件。
- **线程模型**: fire 是同步直调 (可能来自 channel 线程或 asyncio 线程), toolset 用
  `threading.Lock` + `ThreadSafeEvent` 处理跨-loop asyncio wake, 锁临界区极小、
  绝不嵌套、event.set() 一律在锁外。
- **MossRuntime ABC 未动**：MCP 层直接组合 `state.toolset.shell` + `state.watcher`,
  作为「先跑通、增函数不删除」的第一版, 待体验数据确认是否上升到 runtime facade。

### 新 MCP 工具语义速览

| 工具 | 语义 | K 决策映射 |
|---|---|---|
| `ctml_append(logos)` | 铺 CTML, `wait_compiled` 后立即返回 watcher.drain() + status | K1/K8 (append=observe) |
| `ctml_peek()` | 只读 `buffered() + status`, 不 drain, 不阻塞 | K5 (拉侧原子读) |
| `ctml_observe(budget)` | 等 interpreter idle 或超时, drain + status | K2 (观察游标) / K5 |
| `ctml_interrupt()` | `shell.clear()` 掐 pending, drain 已完成结果 + status | K2 (对读头前方动手) |

### 已落码 (commits)

- `c2c22dab` — Shell Tracer Protocol (core 层, 5 tests)
- `b5c8f37b` — InterleavedThinkingToolset + interpreter on_close_callback (host 层, 11 tests)
- (this) — MCP bootstrap 挂 server-scoped watcher + 4 新工具

**残留待验**：
- 第三轮 MCP 盲测（模型当被试）：验 K4 观察盲区是否修通, B1（存在性蒸发）是否闭合,
  A/B 对照 (旧 execute_ctml vs 新 ctml_append) 首次准确率差异。
- fail-closed 写拒（K8 append 遇 critical failure 拒 push）尚未做, 需体验数据先驱动优先级。

## K11 落地: 五动词工具面 + docstring 卫生 (2026-07-25)

**结构性洞见**: create_task + 双 Event lifecycle. interpreter 在 fire-and-forget task 内跑完整生命周期,
MCP 函数只 await 生命周期节点 (compiled / stopped Event), 不阻塞在 async with 内. **中断从此是同步动作,
不干涉执行**. 这一步之前尝试过让 MCP 函数直接 async with interpreter, 结果每种动词的 close 时序都会坏
一种语义 —— 用户手写伪代码钉住方向后才收敛.

**关键洞见**:

1. **budget 是等待时限, 不是运行时限** (人类明确纠正). 上限 30s, 只截断本次等待, 绝不中断命令.
   任务续跑, 结果在下次任意动词调用带回. 回合制 MCP 面禁止无限阻塞的硬约束 —— 模型下发时无需猜任务耗时,
   是真 on-the-fly.

2. **observe 常态退化的消解**. 上一版设计: async with 立即 close, `shell.interpreting()` 拿到 closed 的
   旧 interpreter, `is_running()=False`, observe 退化为只 drain 一次快照. **新架构 interpreter outlive
   MCP 函数**, observe 抓到的是活的 interpreter, `wait_stopped()` 真等到任务完成 —— K5 拉侧的观察能力
   自此完整.

3. **切口同步进 drain**. `_set_result`(command.py:1450-1479) 全同步, done-hook 直接调用不走 call_soon.
   `interpreter(kind='clear')` 内部 `await clear() + stop_interpretation()` 返回时, 被掐 task 的
   cancelled 事件已同步进 watcher buffer. K10 依赖的底层同步性事实至此在盲测中验证 —— replan 与
   interrupt 的切口在同一回合返回.

4. **docstring 只面向模型 (K11 教训)**. MCP tool 的 docstring 通过反射变成模型可见的 tool description.
   内部代号 (K1/K2/K8)、内部 API 名 (`kind='clear'`/`wait_stopped`/`shell.interpreting()`)、调试语境
   ("wait_until_idle 时序 gap")、对话锚点 ("actually, I'm just going to..."、"前进即观察") 都会污染
   tool schema. 开发讯息走 `#` 注释, docstring 只留 4 类信息: 动词做什么 / 什么场景用 / 参数含义 / 返回什么.

   **写完必须反射一次拉出 description 字段验证**, 不能只看源码字符 —— 单看源码字符时 K/kind 术语与语义
   讲解交织, 靠人眼分辨会漏.

   **当轮就改的纪律**: 用户第一次指出 docstring 泄露时, 承认+继续泄露 (下一版又塞新的内部术语进去)
   是不干活. 承认必须触发当轮修复 + 当轮反射验证, 不允许递延.

**七工具面 (K10 后的最终形态)**:

- 会话地基 (2): `moss_instruction` (1 次拉出静态面) / `get_moss_dynamic_info` (N 次拉刷动态增量)
- CTML 动词 (5): `ctml_append` / `ctml_exec` / `ctml_observe` / `ctml_replan` / `ctml_interrupt`
- 删除: `execute_ctml` (被 append/exec 取代) / `interrupt_execution` (被 interrupt/replan) /
  `ctml_peek` (归 debug 不进模型面)

**动词与内核映射**:

| 动词 | kind | task 内等待 | MCP 函数返回时机 | 用户可见 budget |
|---|---|---|---|---|
| append | 'append' | wait_compiled → set(compiled) → wait_stopped | compiled | — |
| exec | 'append' | 同上 | compiled + stopped (可 budget 截断) | ≤30s |
| observe | 不建 | — | `interp.wait_stopped()` if running | ≤30s |
| replan | 'clear' | 同 append | compiled | — |
| interrupt | 不建 | — | `shell.clear()` 同步返回 | — |

**盲测验证** (2026-07-25, 五动词全通):

- `ctml_append(sleep 5s)` → `running:True, ongoing:sleep:s1`, 立即返回, interpreter outlive ✓
- `ctml_observe(no budget)` → 等到 sleep 完成拿 `success:1` ✓ (**核心突破** — 之前只能拿快照)
- `ctml_exec(sleep 6s)` → 阻塞至 done 直接 `success:1, running:False` ✓
- `ctml_replan(掐 60s sleep + 铺 1s)` → `cancelled:1 + ongoing:short` 同步返回 ✓
- `ctml_interrupt(掐 60s sleep)` → `cancelled:1 + running:False` ✓

**残留 (下次起点)**:

- **fail-closed 写拒** (K8): append 遇 critical failure 应拒 push, 尚未做. 需体验数据先驱动优先级.
- **MCP restart 副作用**: server 重启会带走 accepted cell 但 node 子进程可能成孤儿. 当前依赖手动
  `matrix.nodes:run` 恢复; 值得设计 server 生命周期与 node 子进程的正式契约.
- **channel 上线推模式** (问题 2): `get_moss_dynamic_info` 目前只拉不推. ghost thinking 场景下长 thinking
  内感知不到 channel 变更, MCP 面下也只能靠模型手动刷新. 需要 channel 上线的推通道设计.
- **编译期 "did you mean"** (第三轮盲测发现): 通道路径拼错时无纠错. 我读过完整投影仍拼错 (多了一层 .probe),
  错误里加一句 "closest: matrix.mesh.interleaved_probe:slow" 即可归零同类首错.
- **进度投影** (第三轮盲测数据点): 运行中 task 的 `progress` 活串未在 status 里投影, 只有 `ongoing:name`.
  interrupt 决策的输入残缺. FEATURE.md payload 规则里已经列过 ("callername + progress"), 待实现.

---

**本文件状态**: K1–K11 已收敛并落码验证. 五动词工具面盲测通过. workstream **completed**.