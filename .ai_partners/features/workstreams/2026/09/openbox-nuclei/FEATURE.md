---
created: 2026-09-16
depends:
- moss-openbox-modes
- mindflow-interleaved-thinking
description: '治理 openbox 默认 nuclei 的机制面: 把 win/lose 两侧的策略显式化, 补 inject / knock 两个缺口,
  把 FIFO / PriorityQueue 作为独立的交付粒度纪律立项, 并把 signal meta 的 docstring 改成非机制的自然语言 (英文).'
milestone: null
priority: P1
status: completed
status_note: 落地清单 1-8 全部有终局 (1/2/4/7 已落, 3/5 dropped, 6 缓, 8 另起); 3 个已知问题随关闭带走 (injected_percepts
  无上限/quiet 无 drain, aside suppress 重拼重复注入, hint 未落 buffered 路径)
title: Openbox Nuclei — 基础感知核的机制治理
updated: '2026-09-20'
---

# Openbox Nuclei — 基础感知核的机制治理

> Use `moss features set-status openbox-nuclei <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

`ghoshell_moss.matrix.openbox.nuclei` 是开箱默认感知核清单（`aside` 改名、`knock` 补齐后为
8 个：input / notify / interrupt / command / knock / aside / cell_event / listener）。

**它们的定位不是取代业务逻辑 nucleus，而是给"通用的、不想开 nucleus 的能力"提供默认通道。**
功能面的语义化机制（multi-tasks / listener / vision 等）是另一层，不在这里竞争。

当前问题不在数量，在**标注层**：名与实脱节，导致最需要读懂它的人（要发信号的调用方）
读不出语义。三个具体症状，是本 workstream 的起点：

1. **`silent` 名字不可读**——它描述听感，机制里没有声音；且丢掉了"会被看到"这半边。
2. **`input` 的 docstring 声称 FIFO，实装是 Buffer**——名实不符从类名挪进了 docstring。
3. **弱提示缺失**——旁路送达的消息进了上下文，但没有任何东西告诉模型"这条不是在叫你做事"。

人类架构师的判断：mindflow 的建模解释力是够的（三根轴能把全部机制说清），
**混乱在标注层，不在模型层**。本 workstream 只治理标注层与两个真实缺口。

## Design Index

- 机制面权威定义: `src/ghoshell_moss/core/blueprint/mindflow.py`（`Nucleus` ABC / `ChallengeMode` / `ImpulsePrimitive`）
- 实现面: `src/ghoshell_moss/core/mindflow/`（8 个 nucleus + `_mindflow.py` 仲裁）
- 清单: `src/ghoshell_moss/matrix/openbox/nuclei.py` → 项目层 `.moss/src/MOSS/manifests/nuclei/default.py`
- 上层规矩: `src/ghoshell_moss/core/blueprint/CLAUDE.md`（code as prompt: 契约留在 docstring，决策史不进 surface）
- 清单去重机制: `moss-openbox-modes`(in-progress) — canonical `__all__` + `import *`
- 仲裁循环现状: `mindflow-interleaved-thinking`(in-progress) — 三循环解耦，本 workstream 的验证基线

## 机制几何（技术要点，勿丢）

### 三根轴

- **win 后**：接管 attention / 旁路注入（不接管）/ 丢弃
- **lose 后**：丢弃 / 注入下一帧 / 保留重试
- **冷却在哪一侧**：无 / lose 后 / win 后

### 现有 7 核在这三根轴上的位置

| nucleus | win 后 | lose 后 | 冷却侧 | 缓存模式 | 具名原语 |
|---|---|---|---|---|---|
| `command` | 接管（执行 logos，不思考） | **丢弃** | 无 | last-wins | `command_only` / `fatal_command` |
| `input` | 接管 | **保留重试** | lose 后 | 聚合重建 | 无（default mode 匿名） |
| `silent`→`aside` | **旁路注入** | 保留重试 | lose 后 | 聚合重建 | 无（只有 `broadcast` = FATAL+silent） |
| `notify` | 接管 | **注入下一帧** | 无 | last-wins | `notify` / `background_notice` |
| `cell_event` | —（BACKGROUND 短路） | 注入下一帧 | 无 | last-wins | `background_notice` |
| `interrupt` | 接管后立即放手 | 丢弃 | **win 后** | last-wins | `interrupt` |
| `listener` | 三相推进（first/clause/tail） | — | — | 逐条保序 | 无 |

### 交付粒度是另一根轴（与 win/lose 正交）

| 粒度 | 现有实例 |
|---|---|
| N → 1（合并快照，priority 取 max） | `input`、`aside` |
| N → 1（只留最新，last-wins） | `command`、`notify`、`interrupt`、`cell_event` |
| 1（逐条，到达序 = FIFO） | **无** |
| 1（逐条，优先级序 = PriorityQueue） | **无** |

**结论：FIFO / PriorityQueue 不是 `input` 的变体，是两种还没有实例的纪律。**

### mindflow 已能承载逐条交付（不需要新机制）

`Impulse.id` + `complete` 就是为"首包抢占注意力、尾包响应"设计的
（`blueprint/mindflow.py:370`），而 `listener_nucleus` 已经在用：`first_packet_interrupt`
是 `complete=False` 高强抢占，`clause_response` 每句一个独立 id，尾包 commit 是同 id
换成 `complete=True`（`listener_nucleus.py:16-22`）。

**FIFO / PQ 是"照着 listener 的形状写一个"，不是"改 mindflow"。**

## Key Decisions

### 1. `silent` 更名为 `aside`（人类手动全量改名）

语义：**闲时才加入上下文，绝不打扰。** 用人类的话写成的自然语言就是它的 docstring：

> Never interrupts the ghost.
> Whenever the ghost is free, the message simply joins what it sees; while the ghost is busy, it waits.

**Why**：`silent` 描述听感，机制里没有声音；且丢了"会被看到"这半边，只剩"不打断"。
名字不承载机制，和 `input` 是同一类毛病。

**改名级联**（不是一处）：`SignalName('silent')`、`new_silent_signal`、
`ChallengeMode.silent`、`openbox/nuclei.py` 的对象名 `silent_nucleus`、
`cli/ghost_run.py:329`、`tests/` 下引用、以及 `silent_nucleus.py:10` 的交叉引用注释。

### 2. 默认 nucleus 的定位 = 通用默认通道

不取代业务逻辑 nucleus，只为"通用能力不想自己开 nucleus"提供入口。功能面是
multi-tasks / listener / vision 这类语义化机制，属于另一层。

**推论（改变了 docstring 的判据）**：句子要写给**要发信号的调用方**看，标准是
"他能不能直接拿去用"，而不是"这格的语义是否正交完备"。

### 3. signal meta 的 docstring = 非机制的自然语言（英文）

**位置在 SignalMeta，不在 Nucleus**——选信号的调用方读的是 SignalMeta。

**判据**：先能用一句不含机制词的自然语言说清它是什么；说不出来 = 语义没定，
不是描述没写。**用机制解释等于没解释。**

禁用词：attention / buffer / preempt / suppress / impulse / mode / challenge / priority。

草稿（行为已逐条核验，可直接落；`aside` 那条在决策 1）：

- **InputSignalMeta** — A message from someone outside, expecting an answer. / If the
  ghost is busy, the message waits — and comes back together with whatever is said next.
- **NotifySignalMeta** — A message that must not be missed. / If the ghost is free it
  becomes its next thought; if the ghost is busy the message is not dropped — it is
  already there the next time the ghost looks.
- **InterruptSignalMeta** — Stop, right now. / The ghost drops what it is doing and
  starts nothing new — it just stops. Asking again right away changes nothing.
- **CommandSignalMeta** — An instruction to act. / The ghost does not think it over —
  it just does it. A newer instruction replaces an older one; one that cannot be
  carried out is dropped.
- **CellEventSignalMeta** — Something in the system came up, became ready, went away,
  or crashed. / How much it matters depends on what happened — a crash should not read
  like a routine start.

### 4. 新增 `inject`：坐实 `BroadcastNucleus` 的空位

`interrupt_nucleus.py:14` 自己写着 `BroadcastNucleus` "未实现, 由
`ImpulsePrimitive.broadcast` 单原语承载"。本 workstream 把它坐实，命名为 `inject`
——与内部调用 `inject_percepts` 同名。

组合 = **FATAL + buffer**（即 `ImpulsePrimitive.broadcast`）。形状照 `CommandNucleus`
（fire-and-forget + last-wins cache，buffered 判决会经 `attended` 清掉），不需要新机制。

**FATAL + buffer 的"诡异"点（记下来，`inject` 之名就是为消解它）**：FATAL 在这里的含义
被从"紧急"挪用成"保证送达"，且它在 `_mindflow.py:626` 短路在 challenge 之前——
**win/lose 根本不发生，整套仲裁在这条路径上空转**。最急的优先级被用来表达最不打扰的送达。
调用方写 `inject` 之后就不必再手写这个组合。

### 5. 新增 `knock`：输后丢弃，配端侧推拉

组合 = `default` mode 的忠实实装（win → 建 attention、lose → 清 cache 即丢弃）
+ last-wins cache，**不带 logos 所以走真实思考**。形状与 `command` 同构，差别只在要不要 logos。

**Why 输后丢弃**：端侧要求拉。模型执行拉动作之后端侧就不要再发，丢弃是协议的一部分。

**2026-09-17 更正（人类）**：

- **名字定为 `knock`**（不是"暂定"）。它逻辑上就是 **pull request**，听感接近手机的消息提示音。
  保留 `knock` 的理由是"没敲开就没了"这条性质必须从名字读出来——这正是它与 `notify` 的唯一分野；
  而 `buzz`/`chime`/`ring` 这类提示音词恰恰表达相反的东西（现代手机通知会留下角标、反复提醒）。
  且 `knock` 已在项目词汇里（`terminal.md` 的"有结果待拉的敲门"）。
- **`knock` 必须有消息体，否则丢弃。** 这条推翻了本 FEATURE 早先"knock 是指针、不携带载荷"的读法
  （那是在为"丢弃无损"找台阶）。正确的台阶不是"没内容可丢"，而是**它是 request**：
  发起方自己持有待取的东西，没敲开就是没敲开。⚠️ 端侧消费方（terminal 等）不在本 workstream 范围。
- 因此 (c) 那条"丢弃对端侧不可观测"按契约消解，不需要新增回执机制——但**要求调用方自己记账**。

**落地形状**：`KnockSignalMeta`（无字段，纯 marker）+ `KnockNucleus`（`CommandNucleus` 的 cache
生命周期原样克隆，`build_impulse` 改判消息体）+ `new_knock_signal`。`build_impulse` 的判据是
`signal.messages` 为空即返回 None，与 `CommandNucleus` 判 `meta.logos` 同形。

**顺带暴露的结构事实**：`default` mode 的忠实实装是 `command` 和 `knock`，
**`input` 反而是偏离 default 的那个**（输侧保留、下一条到来时带旧消息重试）。
`InputSignalMeta` 这个名字读起来像"最基础的输入"，行为却是个特例。

### 6. FIFO / PriorityQueue 立项为独立纪律

它们是"一次交付一条"的两种纪律（到达序 / 优先级序），与 win/lose 正交，与 `input`
的 Buffer 语义不同层。**不要塞进 `input`。**

实现形状：交付队首（带独立 id、`complete=True`）→ `attended` 后出队 → 再 fire 下一个。
照 `listener_nucleus` 的形状写即可。

**实现风险（待处理）**：`attended` 是在 `_challenge_attention` 的调用栈里被回调的，
在其中 `_notify_impulse` 会重入 consuming loop。需确认是"只 set event、下轮再 rank"，
还是会与 attention 创建交错。

### 7. 弱提示：把 `hint` 带过 buffered 路径

**它是已经存在、已经被系统认作载荷的东西，只是在这一条路上被丢了。** 修法方向：
让 `hint`（可能连带 `description`）经 buffered 判决落到注入的 percepts 上。

接口上有两条路，二选一：扩 `inject_percepts` 的签名，或把 hint 折成一条前置 message。
取决于希望 hint 落在 moment 上还是落在消息流里——**见"待定项"**。

### 8. mindflow 只懂 win/lose（对上一版分析的修正）

`attended` 和 `suppress` 对 mindflow 的含义都是"这段时间别再吵我了"。
**"冷却在哪一侧"是 nucleus 的内部事务，不该让 mindflow 理解。**
`InterruptNucleus` 的"反向 suppress"（win 侧冷却放在 `attended` 里）是这个原则的正确应用，
不是 ABC 缺槽位。

同理：一个回调有多种含义不是问题，**关键是 Nucleus 承诺哪一种**。

### 9. `cell_event` 是 log level 设计

`CellEventSignalMeta` 的字段注释写着按 transition 分档（CRASHED 可从 BACKGROUND 提到
NOTICE），**publish event 一开始就是 log level 设计**。上一版草稿里的
"does not need to react" 是写注释时的模型理解错误，已按分档重写（见决策 3）。

**残留的坑**：quiet 系统（无 attention）下 `background_notice` 会创建新 attention
（`ImpulsePrimitive.background_notice` 自己的注意事项写了）。写自然语言时这类边界会浮出来。

## Exploration Paths / 教训

- **一度把 `lose=丢弃` 判为"无意义、实装里不存在"——错。** `CommandNucleus.suppress`
  （`command_nucleus.py:119`）就是它，注释写着 "失败就丢"；`knock` 还要它。
  犯错的路径是只看了 input / silent / notify 三个核就下结论。
- **一度把 `silent` 的语义判为"写不出自然语言"——说重了。** 人类把它讲成"闲时才加入上下文，
  绝不打扰"之后，句子是写得出来的：**语义没问题，是名字的问题。**
- **一度把 `attended`/`suppress` 的多义判为 ABC 缺槽位**——方向反了，见决策 8。
- **一度把默认 nucleus 的层面搞错**：拿"哪些 win/lose 组合无意义"去问基础面。
  基础面上该问的是"这一格能不能用一句自然语言说出来"。
- **一度把 `input` 的"输时等待"当成原始设计**——它是同一天被改出来的，见下。

## Implementation Notes

### 弱提示缺失（已核验，位置具体）

- `_mindflow.py:693` buffered 分支只送 messages：`inject_percepts(*challenger.messages)`，
  `hint` / `description` 全丢。
- `Impulse.hint` 的字段注释自己承认这个缺口：
  "仅在 Impulse 获得 attention 时通过 update_moment 落到 moment.hint"（`blueprint/mindflow.py:391`）。
- 注入后落到 `moment.with_percepts("MomentsInjectedPercepts", ...)`（`moment.py:718`）——
  **唯一的分帧标记就是这个机制名**，没有语义层的"这不是在叫你做事"。
- 而 `_is_useful_frame`（`_mindflow.py:1019`）判定载荷时**把 `hint` 算进去**：
  `bool(messages or dynamic_messages or hint or logos)`。系统别处都认 hint 是载荷，
  只有 buffered 这一条路扔了它。

### buffer 机制（三个待解，均与 inject / knock 相关）

**(a) `_injected_percepts` 无上限、quiet 下无出口。**
`observe()` 只在 attention 生命周期内被调用（`_mindflow.py:1038-1063` 的
`async for attention in self._loop_attention()`）。**安静态（无 attention）下没有任何东西
drain `_injected_percepts`**，它只 `extend` 不消费（`moment.py:808`），全部堆到下一次
思考时一次性出现，且没有 `max_size` 保护。
而 `aside` 的用途（闲时加入）恰好就是这个场景；`inject` 同理。

**(b) `aside` 会重复注入。**
失败侧不清 `_signals`（`silent_nucleus.py:142`），冷静期过后下一条 signal 到达时
`_rebuild_impulse` 会把整个 buffer 重新拼一遍再 fire——
**上一条已经注入过的消息会被再注入一次**。`notify` 无此问题（逐条 last-wins）。

**(c) `knock` 的丢弃对端侧不可观测。**
`suppress` 只回调给 nucleus，signal 的**生产者**收不到任何"你的 knock 被丢了"。
端侧只能从"没等到 pull"反推。若协议是"模型拉过之后就别再发"，这个丢弃需要一条能回到
端侧的回执，否则端侧只能靠超时猜。

— 2026-09-17：**按契约消解，不加回执机制。** `knock` 被定性为 pull request（见决策 5 的更正）：
发起方自己持有待取的东西、自己记账，"没等到 pull"就是它的终态。要新增回执反而会
把"信号只能回执到 nucleus"这条既有边界撑开，代价不成比例。

### `input` 输侧语义的历史（明天要重验的东西）

`input` 的"输时保留（pending）"不是原始设计，是**同一天改出来的**：

- `bbd5f1bf`（2026-07-21 17:39）：suppress 做 `_atomic_clear_buffer()`，
  注释写着 "default 路径 suppress = 信丢" → **那时是输时丢弃**。
- `606e7699`（同一天 18:42）：改成 `_impulse_cache = None`、保留 `_signals`
  → **一小时后变成输时等待**。

两次都引同一个 bug：peek 在冷静期仍可见 → 重排重放。

**但那个 bug 的触发条件在当前代码里已经不成立了**：两个 commit 描述的机制是
"mindflow 的 0.5s 超时循环重排同一个 impulse"，而当前 loop 在超时分支是 `continue`、
**不 rank**（`_mindflow.py:547-549`）；loop 后来被 `41f0cb63` / `0c349f00` 重构过。

**⇒ 当初逼出"输时等待"的约束可能已经不在了。** "input 该 pending 还是该丢"需要在
**当前 loop 上重新验一遍**，而不是沿用当时的结论。

### 其他事实清单

- **FIFO 从没进过代码**：git 全历史搜不到 Fifo 命名的 nucleus，也没有被删的 fifo 文件；
  `InputSignalNucleus` 在 `ca9350e1` 首次出现就是这个类名。FIFO 只活在
  `input_signal_nucleus.py:22,26` 的 docstring 里。那次改名没解决名实不符，
  只是把它从类名挪进了 docstring——从看得见的错换成看不见的错。
- **`silent` 与 `input` 的 `_rebuild_impulse` 逐行相同**，只差一个 `mode=` 字段。
  `silent_nucleus.py:9-13` 声称的"FIFO 离散事件 vs 合并数据流"在代码里**不存在**。
  — 2026-09-17 补：这条假账在改名后原样留在了 `aside_nucleus.py:9-13`（"InputSignalNucleus:
  signal 视为离散事件 (FIFO 保留)"）。`input_signal_nucleus.py` 自己已由 `e7012505` 改对
  （"Not a queue"），`aside` 那边没跟上，**待清**。
- **`silent` 与 `input` 的 suppress 语义也不同**：`input` 清 `_impulse_cache`（606e7699 修的），
  `silent` 不清且 `peek` 不看冷静期（同 commit 明确说是 by design，有它的测试背书）。
  后果：`silent` 的缓存会被**其它 nucleus 的 fire** 触发的 rank 重新挑战——
  这正是 `input` 被修掉的形状。是特性还是漏改，明天一并判。
- **`NotifyNucleus.suppress` 是死代码**：notify 的失败路径走 `buffered`，
  `_challenge_attention` 里 notify 永远不会被判 `suppressed`。
- **notify 的 burst 丢消息**：`_impulse` 逐条覆盖（`notify_nucleus.py:88` 的 TODO 记了）。
- **两个基础 mode 在原语层匿名**：`ImpulsePrimitive` 6 个原语
  （`command_only` / `fatal_command` / `broadcast` / `interrupt` / `notify` / `background_notice`）
  里没有 `silent`，也没有 `default`。`silent` 只作为 `broadcast` 的零件出现，
  而那个名字的语义重心是 FATAL。
- **`status()` 已经把两个词分开了**：`input` 报 `pending: N`，`aside` 报 `buffered: N`。
  "pending" 已经在代码里，只是没被提到名字上。

## 落地清单（人类 2026-09-17 收口）

1. **改名 `silent` → `aside`**（人类手动全量），含决策 1 列出的级联点。 — ✅ `b822bdbc` + `98ee34a8`
2. **落 6 条 SignalMeta docstring**（决策 3 的草稿），`aside` 那条随改名的名字一并落。
   — ✅ 全部 6 条落地并英文化；连带把 meta 上的中文字段描述（command `logos`、
   cell_event `address`/`transition`）与 `CellTransition` 枚举一并英文化（枚举是
   cell_event 信号的载荷面，同一条 surface 半中半英更糟）。nucleus 侧 docstring 未动。
3. **新增 `inject` nucleus**（决策 4）+ 补 `InjectNucleusMeta` 进 `openbox/nuclei.py`。 — ⛔ dropped（人类 2026-09-17 判：有了 `notify` + `next: bool` 后 inject 暂无独立用例）
4. **新增 `knock` nucleus**（决策 5）+ 同上。 — ✅ 名字定为 `knock`；契约见决策 5 的更正
5. **补 `aside` / `default` 的 `ImpulsePrimitive` 具名**，让原语覆盖全部 mode。 — ⛔ dropped（`default` = 不设 mode，具名无意义；`aside` 原语只服务 `add_impulse` 调试路径，AsideNucleus 已直接设 mode）
6. **弱提示落到 buffered 路径**（决策 7）——先定"落在 moment 还是落在消息流"。
   — ⏸ **人类 2026-09-17 决定先不做**：重大决策，先看实际效果再定。
7. **重验 `input` 输侧语义**（历史段），按当前 loop 下结论。 — ✅ **已由 `e7012505` 完成**
   （本清单收口时未与代码对齐）。结论 = 输时等待（pending），且理由换了：不再是当初
   那条"0.5s 超时重排"的 bug，而是 `Nucleus.peek` 契约（suppress 后 impulse 仍可见、
   只是不再主动 fire）+ 挑战闸门（严格更高的聚合权重才破冷静期）。当前 loop 站得住：
   `_mindflow.py:547-549` 超时分支是 `continue`、不 rank，重排只能由别的 nucleus fire 触发。
   另：`input_signal_nucleus.py` 的 docstring 已在同一 commit 改对，"FIFO"假账只剩
   `aside_nucleus.py:9-13` 里抄的一份（见"其他事实清单"末条）。
8. **FIFO / PriorityQueue**：本轮先只立形状与 `attended` 重入风险的判定，
   是否实装另定（避免与 `mindflow-interleaved-thinking` 撞车）。 — 本轮不实现（设计任务，另起）

> 关联决策：讨论中引出的"插队"机制（`ChallengeMode.next` + `ChallengeVerdict.queued` + `NotifySignalMeta.next: bool`）属 mindflow-core mode/verdict 层，不落本清单，随 `mindflow-interleaved-thinking` 走。

### 待定项（三项已定，2026-09-17）

- **`aside` 能不能承诺"会被看到"** — 不承诺。见 `inject`（决策 4）承担"必达"，
  `aside` 的 docstring 保持"闲时才加入、忙时等"的措辞，不写保证送达。
- **`inject` 的 buffer 落在哪** — 未定（该项未开工）。
- **`knock` 的必要性** — **是独立 nucleus**，不做 `command` 的变体。理由：`CommandSignalMeta.logos`
  必填、`CommandNucleus.build_impulse` 无 logos 直接返回 None，`command` 在结构上就载不动
  knock；且一个名字不能同时装"别想直接做"与"来想想"。signal name 是总线路由键，调用方靠名字读意图。

## 收口（2026-09-20）

落地清单 1–8 已全部有终局，逐条核验过代码：

| # | 终局 | 证据 |
|---|---|---|
| 1 改名 `silent`→`aside` | ✅ | `b822bdbc` + `98ee34a8`；级联点（SignalName / helper / ChallengeMode / openbox 清单 / tests）全落 |
| 2 六条 SignalMeta docstring | ✅ | 逐条比对决策 3 草稿一致；meta 字段描述与 `CellTransition` 已英文化 |
| 3 新增 `inject` | ⛔ dropped | 有了 `notify` + `next` 后无独立用例 |
| 4 新增 `knock` | ✅ | `c754c6dd`；`KnockNucleus` + `KnockSignalMeta`（无字段 marker，无消息体即丢） |
| 5 补 `aside` / `default` 原语具名 | ⛔ dropped | `default` = 不设 mode；`aside` 原语只服务调试路径 |
| 6 弱提示落 buffered 路径 | ⏸ 不做 | 人类裁定先看实际效果（`_mindflow.py` 的 `inject_percepts` 仍只送 messages） |
| 7 重验 `input` 输侧语义 | ✅ | `e7012505`；结论 = pending，理由换成 `peek` 契约 + 挑战闸门 |
| 8 FIFO / PriorityQueue | 另起 | 只立了形状，未实装 |

清单外的两处"待清"已清：`aside_nucleus.py` 抄的 "FIFO 保留" 假账已重写为
"聚合 buffer + 优先级提取" 的对称说明；`_is_useful_frame` 与 buffered 路径的
`hint` 缺口保持一致（都是决策 6 的已知留白）。

**随关闭带走（已知问题，不阻塞）**：

1. `_injected_percepts` 无 `max_size`，且 quiet 态（无 attention）没有东西 drain 它 ——
   `observe()` 只在 attention 生命周期内被调用（`moment.py:815` 只 extend）。
2. `aside` 输侧不清 buffer，冷静期过后重拼会把已注入过的消息再注入一次
   （`aside_nucleus.py:142` 的 `suppress` 只设 `_suppress_until`，`_signals` 留到下次
   `add_signal` 的 `_rebuild_impulse` 里重拼；`attended` 路径无此问题）。

`notify` 的 burst 丢消息已由 `6bac6850` 修掉（后到 signal 的 messages 合并进 pending
impulse 而非覆盖），不再是遗留项。

**关联去向**：插队机制（`ChallengeMode.next` + `ChallengeVerdict.queued` +
`NotifySignalMeta.next: bool`）已随 `c754c6dd` 落在 mindflow core，见
`mindflow-interleaved-thinking`。