---
title: Interleaved CTML Thinking — 交错思考流中的 CTML 控制范式
status: draft
priority: P1
created: 2026-07-23
updated: 2026-07-23
depends: []
milestone:
description: >-
  约定 interleaved thinking 过程中模型输出 CTML 控制外部世界的交互范式：
  边想边铺执行轨、observe 只看执行游标、wait/interrupt 做思维剪枝。
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

### 7 类交互预期的最终归并

| 原始分类 | 归并去向 |
| --- | --- |
| 1 quick reaction | `react`(phatic，即时 flush，永不 stop) — 候选新原语，少见故值得显式 |
| 2 fire-and-forget | 默认 append，`-> None` / `@nonblocking` 签名承载 |
| 3 on the fly | 介质底色，无动词 |
| 4 observe result | `observe` 动词 = 看游标(progress+result)，**真 stop 点** |
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
