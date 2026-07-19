---
title: Aurelius Ghost
status: in-progress
status_note: >-
  Aurelius/Memento trajectory is implemented. The 2026-07-19 landing replaces the
  earlier regex Evidence/Claim/Recall/verifier layer with grep-style memory_search +
  memory_show paging, a memory-discipline instruction, and bypass curation; it also
  fixes the CTML worker-thread scheduling crash, adds an in-process write lock, stamps
  folded summaries with note_seq (no fabricated model turns), records failed frames
  honestly, and adds input-side context budgeting: token-estimated window shrinking
  plus provider-overflow retry, folding (never destroying) history per the Memento
  stance. P1 Ground lifecycle/context wiring, default-quiet TUI, and bounded memory
  administration stand, without changing Ghost, Memento, or Desktop core contracts.
  Budget-triggered proactive semantic commit, precise tokenizers, autobiographical
  self-act recall, P2 principal/audience governance remain future work.
priority: P1
created: 2026-07-13
updated: 2026-07-19
depends: [ghost-filesystem-desktop, momento-mori]
milestone: 0.1.0
description: >-
  Aurelius — 第二个 Ghost 原型，取自《沉思录》作者 Marcus Aurelius。相对 Atom 的
  线性内存历史, Aurelius 把"上下文"拆为 Desktop (现在/作业记忆) + Memento (过去/轨迹)
  两层, 作为 moss 实例 (仓库自身的 ghost) 的载体, 并持续承载高级能力迭代.
---

# Aurelius Ghost

> Use `moss features set-status aurelius-ghost <status> -m "note"` to update state.

## Motivation

Atom 是最简参照基线, 它自己在 docstring 里钉死了两个"原型范围外"的欠落:
context window 不裁剪, 历史纯内存重启即丢. 这两个欠落不该由 Atom 补 — 补了
它就不再是任何人能对照的基线.

Aurelius 是补这两个欠落的**高级层原型**, 同时是 `moss` 实例 (这个仓库自身的 ghost)
的载体. 定位是长期迭代母体: 各种高级能力 (反身控制、mindflow、observability)
会持续接进来. 首批集成的两个 P0 能力:

- **Desktop = 现在**: `articulate()` 不再垂直堆 `message_history`, 而是从机面
  (pin 的表面 + open 的场) 组装 context. 接合点是 Ghost ABC 的 `channel()`
  hook — ghost 反身控制的 channel 以 `'ghost'` 名注册, desktop 的
  open/pin/update 动词挂在这里. Desktop 是**作业记忆**, 不与世界自动同步.
  (对 moss 实例而言, 它的 desktop 大概率就是仓库本身.)
- **Memento = 过去**: 纯内存历史换成 commit 轨迹持久化 — 重启不丢、可化身分叉.
  (对 moss 实例而言, memento 甚至可能直接入库.)

## Key Decisions

<!-- Record each meaningful design choice. This is what the next AI incarnation reads first. -->

- **不碰 Atom.** Atom 保持为纯净对照基线 (单轮 articulate + 纯内存线性历史).
  新能力一律落在 Aurelius 上. 这是命名"第二个原型"而非"扩展 Atom"的根本原因.
- **原型 = Aurelius, 实例 = moss.** 原型取自《沉思录》作者 Marcus Aurelius：
  它强调省察、节制与可审计的自我修正，契合 Memento 的反思型人格。实例名 moss —
  这个仓库自身的 ghost, 反身映现整个仓库.
- **上下文双层化 = 本原型的立命之处.** 相对 Atom 的线性 append 历史, Aurelius 的
  context 由 Desktop (现在) + Memento (过去) 组装. 这是 Aurelius 区别于 Atom 的唯一
  硬结构决策, 其余 (mindflow / observability hooks) 都是后续可选迭代.
- **上下文组装不出 runtime.** 信息链路: ghost 的 channel (`Ghost.channel()`) →
  GhostRuntime → shell → 静态面经 MossSystemPrompter 回流 → articulator 带
  moment 回到 ghost. shell 是唯一世界面, ghost 不绕开它另组世界. (否掉过一个
  `assemble_context` hook 提案 — 破坏 shell/ghost 拆分.)
- **一个 ctml tool 统治全部 channel 面.** channel 永不逐个映射为 tool. 真议题
  是反方向: 哪些 tool **不进** channel (判据未决, 见 open problems).
  desktop 修改、bash/mcp/skills 全被此覆盖 — 它们是/将是 channel.
- **1:N articulator:action 原则最优, 本版砍掉.** 理由: 人类和模型都无法颅内
  建模, 需要大家能看懂的方案 (mindflow 已砍过多版). 1:1 保留.
- **think='none' 由 ghost 处理, 不由 runtime 短路.** 现状 ghost_runtime.py:348
  在 effort=='none' 时跳过 articulate — 与 `Impulse.thinking_effort` 字段声明
  ("执行 articulator 的智能体仍有权决定") 矛盾, 且 noop 不进 memento. noop 是
  轨迹事件 ("看见 X, 选择沉默"), Aurelius 必须 witness 它, 否则化身分叉看不见.
- **flash/快响应不进 Ghost API.** 走 Nucleus 侧: 快模型产出 command impulse
  (`Impulse.logos` 反射弧 + `thinking_effort` 建议位已是现成原语). 按需后做,
  不阻塞 Aurelius. 模型配置位现成: `contracts/llms.py` 的
  `DefaultModelTag = 'small_fast_model' | 'flash' | 'pro'`.
- **memento = 标准库件, Ghost 持生命周期 (倾向, 未终决).** 标准实现 ≠ runtime
  拥有: memento 作可复用契约+实现, 各 ghost 在 `__aenter__/__aexit__` 实例化并
  持有. GhostRuntime 对 memento 零感知 (Atom 无, Aurelius 有). 配套: memento channel
  控下轮展示规则 (v1 极简裁剪), 旁路加工做异步精炼 (raw 轨迹全存, 展示走裁剪).
- **首版接线终决: Aurelius 持有薄 AureliusMemory 适配, GhostRuntime 零感知.** 读路径在
  `Aurelius.articulate()` 从 branch window 重建模型历史; 写路径复用
  `Ghost.on_articulate_exit()`，此时完整 logos 已写回 Moment，且正常沉默帧也不会
  丢。失败半帧不入记忆。默认每 4 帧 mechanical commit，初始释义只做限长的
  输入/输出原文摘录索引，不伪造意义；未来旁路可 `reinterpret()`。
- **owner 与存储根终决: 稳定 Ghost 身份, 非 session scope.** 默认 root 是
  `GhostWorkspace.home/memento`，owner 是 ghost name。这样跨进程重启能恢复；
  同 `(root, owner)` 仍严格单写者。并行化身以后用新 owner/branch，不抢跑。
- **首版曾不接 Desktop / CTML memento channel / 反思摘要 / witness。** 先验收单
  branch 退化态的跨重启记忆；随后反思与受限 CTML 控制面在既有 Memento 接口上实现，
  仍不改契约，Desktop 与 witness 留待后续。
- **第二期接线：反思是异步释义旁路，不是写入热路径。** 每个 mechanical commit 先以
  保真摘录冻结，再由 `small_fast_model`（或明确注入的模型）读取已冻结成员并
  `reinterpret()`。反思只能追加 CommitNote，不能修改 Moment；失败不影响对话，启动时
  扫描尚无反思 note 的 mechanical commit 追赶。反思产物只含可见证据上的简短结论，
  不持久化模型私有推理。
- **MemoryConfig 是 Aurelius 的持久策略面。** Window、count-based commit 与 reflection
  参数由 workspace `configs/memory.yml` 提供默认值；AureliusMeta 显式参数只作为宿主/测试
  覆盖。时间阈值与 witness 调度尚无 worker，不在本次伪装成已实现能力。
- **CTML 控制面只暴露本 owner/current branch 的显式动作。** inspect/log/staging/show、
  semantic commit、reinterpret、fork、switch 均经 Ghost.channel() 进入 Shell；不提供
  跨 owner 写和隐式 merge。fork 的出生点仍必须是冻结 commit。
- **thinking 期切片原文不进 Moment.** ghost 自持内存状态, 必要时按 moment
  commit 拆分. `Reaction.executed_logos` ("系统执行的 logos ≠ 模型生成的 logos")
  与 `Reaction.messages` (回声) 已为缝合留好位置, memento 契约
  (contract-frozen) 无需变更.
- **模型层选型: pydantic-ai 现阶段用, 不承诺长期** (对自封装 agent 无兴趣).
  Aurelius 的 `_meta` 不重走 Atom 的 AnthropicModel+环境变量硬编码, 改走
  `contracts/llms.py` 的 LLMConfig 契约.
- **Memento 轨迹不等于可回答事实。** 2026-07-18 的真实回归表明，正确的 commit note 已
  进入模型 history 时，模型仍可将另一字段（`ORBIT-004`）误答为测试代号。Aurelius 的生产级
  路线因此必须在 Memento 之上增加 Evidence/Claim projection、问题驱动 Recall 与答案校验：
  Moment/Commit 是不可变证据，reflection 与 logos 仅是解释/未验证陈述，不能自动升级为
  active Claim。该层借用 Grounds 的“每帧重绘、地址而非快照、读写分离”纪律，但不把长期事实
  错存进 Desktop；`core.memento` 的 payload 不透明契约保持不变。
- **Aurelius 的目标是认知能力，不是长上下文。** 最终形态明确为：`DESKTOP.md` 承载工作场的
  法、长期协作约定和人工策展胶囊；Ground/Pin 承载当前注意力与可观察世界；Memento 承载不可变
  轨迹；Evidence/Claim 承载可问答、可校验知识。Claim 必须区分 fact/preference/opinion/plan/
  procedure/hypothesis，并带证据、置信、状态、subject/owner/branch scope、audience/sensitivity
  和 retention。模型隐藏思维链不属于此系统。
- **存、取、说、忘是四道独立的门。** 用户直接陈述/可信工具结果可以形成候选；logos/reflection
  不能自动提升。Recall 必须先按问题、scope、时效召回，再按 audience/sensitivity 做最小披露，
  最后由 verifier 逐项校验。archive/review/tombstone 是显式治理，不采用静默 LRU 抹掉原始证据。
  同一 Ghost 可记得同一件事，不代表可向儿童、成人、另一用户或另一产品无差别透露。
- **Ground 已可用，Aurelius 尚未接线。** `core.desktop` 的 DefaultGrounds/DESKTOP.md/Pin 是完成的
  通用能力，但当前 Aurelius runtime 未创建 Grounds，也未把 instruction/frame 接入 Agent。本
  workstream 的 P1 是按 Ghost 生命周期和 Channel/CTML 原则接入它；不新造 `Memory.md`/`User.md`
  来混写用户事实与工作规则。
- **Mem0 不在当前实现范围。** 只保留 `MemoryRecallBackend` 的可选 adapter 端口；本地、可重建的
  Claim Recall 始终是权威退化路径。当前不引入 mem0 SDK、API key、配置或网络调用；未来后端只能
  提供候选 ClaimRef，不能成为 Memento/Claim 的真相源。
- **P0 投影首版不另造真相文件。** `MemoryProjection` 每次从当前 branch 的冻结 commit 与 staging
  重建 Evidence/Claim；commit id/note seq 只作稳定引用，Moment 原文仍是唯一证据。规则提取器只
  自动提升明确可解析的用户输入和显式 trusted-tool source；assistant logos 与 reflection 即使能
  解析，也只能成为 rejected candidate。这样先钉死来源、防污染和可重建不变量，再按真实语料扩展
  extractor，不引入第二份会漂移的持久状态。
- **记忆型事实题采用有限 evidence packet + 输出后校验。** Aurelius 只在问题能映射到 canonical
  key 时召回，未知/冲突直接安全回答；有证据时才让主模型组织语言，并在向 Shell yield 前整体校验
  requested key/value、current 状态和结构化干扰值。普通对话保留流式；记忆型事实题为避免先泄露
  后撤回而有意缓冲一轮输出。
- **P1 只做 Ghost 侧适配，不修改 Desktop 抽象。** Aurelius 生命周期持有 owner-scoped
  `DefaultGrounds`，真实 Host 优先以 `Project.root` 为 workspace，测试/嵌入场景退化到
  `GhostWorkspace.home`。Ground instruction 作为本帧附加 instruction，frame 作为临时 user
  context 注入且不写入 Moment；open/pin/unpin/update/frame 继续经 `ghost` channel 的受限命令面。
- **真实 Shell 用户证据 source 是 `input_signal_nucleus`。** Evidence 白名单必须包含该核心
  Mindflow source；`input`/`user` 仅作显式嵌入兼容。测试 helper 与 acceptance 必须默认使用
  `input_signal_nucleus`，避免人工 `user` percept 绕过真实接线而产生假阳性。
- **记忆运维不是普通 Agent 行为。** 完成 Moment 已自动持久化，事实题已有内部 Recall；因此
  `memory_*` 默认对模型不可见且不 `always_observe`，仅保留为人类显式 Shell/CTML 运维面。
  `memory_claims` 默认返回紧凑计数与 Claim 摘要，只有 `detail=true` 才返回最多 `limit` 条完整
  证据/候选。这样阻断“普通陈述 → commit → claims → 再 commit”的自激回路，而不删合法工具帧。
- **默认 UI 与审计轨迹分层。** `moss-run-ghost` normal 模式只显示去除 CTML 后的人类可读回复；
  verbose 显示运行摘要，trace 才显示完整 command-result。Memento 仍保存可审计认知帧，隐藏 UI
  不等于删除轨迹。机械 Note 只摘录可信用户 source 与对应可见回复，单条全局上限 600 字符，
  不再把纯内部控制帧和完整审计 JSON 折回模型 history。

## Future Memory Deliverables (pending)

下列能力对 Aurelius “像一个连续存在的主体”有实际价值，但当前**均未实现**。
Memento 已保存部分所需原始轨迹，不等于已经形成可靠的召回、续说或披露能力。
后续实现必须遵守：

- Memento 继续是不可变事件账本；新能力默认作为 Aurelius 层的可重建投影，不另造真相数据库。
- “Aurelius 曾说过 X”是可验证的行为事件；它不能单独证明“X 是事实”。
- 只持久化可观测输入、生成/实际执行输出、结果与停止原因；不保存模型隐藏思维链。
- 优先不改 Ghost/Memento/Desktop 核心抽象。若精确执行边界缺少可观测交接点，必须单独评审
  最小 hook，不能借本 feature 暗改 core 契约。

### P2-A Autobiographical / SelfAct projection

目标是让 Aurelius 可审计地回答“我曾经说过/做过/拒绝过什么”，不把自己的话污染为外部事实。
候选投影形状：

```text
SelfAct {
  actor, audience, action_kind,
  generated_text_ref, executed_text_ref,
  status, moment_id, created_at, stop_reason
}
```

- `action_kind` 至少区分 utterance/tool-call/decision/refusal/silence；`status` 至少区分
  completed/interrupted/failed。
- `Moment.logos` 只是 generated utterance 证据；`Reaction.executed_logos` 才是世界实际收到的输出证据。
- SelfAct Recall 必须返回 Moment/Reaction 稳定引用；不经转述文本猜测行为是否发生。

完成标准：用确定性 fake stream 产生一次说话、一次拒绝和一次工具行为，跨进程重启后按 actor/
audience/action_kind 召回结果不串类；同时证明该说话的命题未因 logos 来源而进入 active Claim。

### P2-B Interruption / Continuation

目标是让 Aurelius 知道上一次实际说到哪里、为什么停止，并在用户明确要求“继续”时从可验证
边界恢复，而不是根据模型印象重新生成整段。候选短期状态：

```text
ContinuationState {
  source_moment_id, original_request_ref,
  partial_generated_ref, partial_executed_ref,
  stop_reason, resumable, status
}
```

- `status` 至少区分 pending/resumed/abandoned/superseded，且只允许一个当前 active continuation。
- “继续”走 Continuation Recall；新任务、换 principal 或用户明确取消时不得无声续说。
- 当前 `Aurelius.on_articulate_exit(error is not None)` 直接 `skipped_on_error`，因此不能把已有
  `Reaction.stop_reason/executed_logos` 字段当成已完成的持久化续说链。

完成标准：确定性 stream 在 N 个 chunk 后中断，记录原请求、generated/executed 边界与 stop reason；
同进程和重启后的“继续”都不重复已执行前缀；新话题使旧 continuation 变为 abandoned/superseded，
不自动向新对话对象泄露片段。

### P2-C Principal / Audience governance

目标是让“记得”与“此刻可以对谁说”分离。`Memento owner` 是 Ghost 身份，不是当前对话者身份；
principal/audience 必须来自产品接入层的已认证上下文，不允许模型从文本、声音或自称中猜测。

- 每次 Recall 输入至少包含 principal id/role、audience class、workspace/product scope 与授权上下文。
- Claim/SelfAct/Continuation 的召回和生成前都执行 audience/sensitivity 过滤；回答校验同样不得越权。
- principal 缺失、不可验证或 scope 冲突时 fail closed，不把“我不知道你是谁”降级为全量披露。

完成标准：至少两个 principal、两个 product/workspace scope 和 public/private 证据的确定性矩阵；
同一 Ghost 只返回当前 principal 获授权的最小证据，换人/换应用/缺失身份均不串记忆。

### P2-D Memory query router

不同记忆问题不得全部落入事实 Claim extractor。召回路由至少区分：

| 查询意图 | 权威表面 | 例子 |
|---|---|---|
| world/user fact | Claim + EvidenceRef | “我的测试代号是什么？” |
| autobiographical event | Episode/SelfAct + Moment/Reaction ref | “你刚才说过什么？” |
| continuation | ContinuationState + executed boundary | “继续刚才被打断的话” |
| current task/world | Ground/Pin/frame | “现在在处理哪个文件？” |

每类查询使用自己的 scope/filter/verifier；意图不明时安全追问或退化为普通对话，不得把
logos 提升成 fact 来“显得记得”。

完成标准：用同一组包含用户事实、错误 logos、完整行为、中断行为和 Ground 变化的混合轨迹，
确定性证明四类问题只访问对应表面，错误路由安全拒答，不交叉污染。

### Cross-cutting definition of done

- 上述每项都必须有无网络的 fake/TestModel 确定性测试和至少一条真实 TUI 验收路径；
  真实模型偶然答对不算通过。
- 投影删除/重建、进程重启与 branch switch 后语义一致；每个结果能追溯到 Memento 证据。
- 受限 CTML/调试面能检查路由、SelfAct、active continuation 和披露决策，但不提供跨 owner 修改。
- 实现、测试方案、配置说明和本 FEATURE 状态同步更新；未达到本节标准前不得宣称
  Aurelius 已具有可靠的“自传记忆”、“中断续说”或“分对象披露”能力。

## Interleaved Thinking — 候选方案 (未测试, 施工时验证)

thinking 期用 tool 调 moss + 结束后 text block 出 logos. 候选实现形状:

```
ghost.articulate(articulator):
    q = janus.Queue()
    task = articulator.create_task(agent_loop(q))   # 与 attention 同生共死
    # agent_loop: pydantic-ai 多步循环, 携带单帧生成的闭包 tool:
    #   ctml(text) → 送入执行; 采样 Shell.interpretation → 时间切片作 tool_result
    async for delta in q: yield delta               # runtime send_nowait → action 照常
```

- 妙处: tool 不阻塞等结果而返回状态切片 → 返回值桥接被读写拆分消解;
  1:1 articulator:action 保留, 零 mindflow 手术; thinking token 本身是等待时钟;
  长思考不哑 — 一边思考一边经 tool 交互.
- 闭包 tool **每帧生成** (走 janus) 或起点创建 (含动态逻辑), 优于 ghost 长期
  持有裸 shell; shell 从 IoC 取, 可在 `GhostMeta.contracts()` 声明依赖.
- feed 期实时执行已实现 (非假设), 待 moss-as-mcp 实际体验验证切片体感.

## 真问题与可选方案 (open)

- **tool 结果进不进 moment / articulator 是否支持 outcome** — 全场关键点.
  "写路径汇入同一条 logos 流" 是可选方案 (若采纳, executed_logos 缝合白拿),
  **不是决定**. 不假设调研比施工时看到的准.
- **时序对齐** — ghost 自持的 thinking 期内存状态如何按 moment commit 切分,
  模型 (施工者) 会不会做对. 未验证.
- **thinking 期工具的"纯交互"性 / 切片粒度** — thinking 中调用的能力最好是
  纯交互的; shell 命令结果可等待 (非轮询), 查询经 ctml 未必别扭, "哪些 tool
  不进 channel" 的判据悬而未决.
- **desktop 修改动作的落点** — 若 interleaved 方案成立, 经 ctml tool 顺解;
  若不成立, 首版 desktop read-mostly.

## 上下文预算与自动压缩 (2026-07-19 落地 P1)

### 缺口

`articulate` 组装 history 时只按**帧数**切窗 (`window(detail_n=12, summary_m=-1)`), 与
输入 token 无关: `summary_m=-1` 让折叠摘要**无界增长**, 多模态帧 (base64 图) 单帧就可能
顶满窗口。articulate 只组装一次即发, 无 token 计数、无溢出捕获、无重试 — 输入超窗时请求
直接失败, 对话中断, 且 `max_tokens` 修复救不了 (那是输出侧)。

上下文压缩本是记忆的一等能力。行业做法分四层: ①滑窗+头部保留 ②主动触发压缩
(Claude Code `/compact`、Codex auto-compaction, 逼近阈值即摘要替换) ③分层外部化
(CLAUDE.md / RAG, 不常驻窗口) ④溢出兜底重试。Memento 的"折叠而非丢弃、原文
`memory_show` 可缺页取回"就是第③层且**比 RAG 更保真** (无损)。所以我们借鉴 ②④ 的
**触发时机与 token 预算工程**, 但压缩落点用自己的 commit/reinterpret 折叠机制, 不抄
它们的有损摘要替换 — 这正是 momento-mori "对抗行业三种有损方案"的兑现点。

### 落地方案 (本次实现 step 1+2, 收敛为同一机制)

**token 预算 = `context_window - max_output_tokens - context_token_margin`。** 三个值都取
自 `ResolvedModel` 契约 (`context_window`/`max_output_tokens` 已存在), margin 为估算误差
+ 输出 headroom 的安全垫。估算器 `_budget.py` 走 char/CJK 保守除数, base64 图按固定
名义 token 计 (不计其 base64 长度, 否则严重高估)。**保守方向 = 高估输入 → 提前压缩**,
估算不准由第④层兜底。

- **step 2 主动预算 (`_budgeted_history`)**: 先按配置 `detail_n/summary_m` 渲染, 若
  `估算(history)+固定开销 > 预算` 则收敛: **先降 `detail_n`** (明细帧折叠为摘要, 省 token
  同时经 summary 仍可寻址), 再压 `summary_m` (最旧摘要移出上下文, 磁盘原文仍在,
  `memory_show` 可取回), 直到入预算或触底 `min_detail_n`。折叠不销毁任何原文 —
  只缩"渲染进上下文的量", 与 Memento 折叠哲学一致。
- **step 1 溢出兜底 (`articulate` 重试环)**: 若 provider 仍以 `_is_context_overflow`
  文案 (跨 anthropic/openai 归一) 拒绝, 且**尚未 yield 任何 token**, 则进一步腰斩窗口
  重试。**已 yield 后一律上抛** (不能 un-yield, 也不与 attention abort 冲突: abort 非
  overflow 文案, 直接上抛交 driver)。

`inspect_context` 记录实际用的 `detail_n/summary_m/budget/estimated_tokens/shrunk`,
压缩发生与否对人类/模型可见 (报账, 不藏)。`context_budget_enabled=False` 或注入
TestModel 时退化为旧的一次性组装, 不改既有测试语义。

### 仍未做 (留给后续)

- **主动折叠 commit**: 目前压缩只在"渲染"层缩量, 未在逼近阈值时**主动 semantic
  commit** 把最老明细真正冻结成摘要 (Claude Code auto-compact 的完整形态)。当前
  auto_commit_every 已在 staging 侧做机械折叠, 二者合流 (预算触发→提前 commit) 是
  下一步, 涉及"压缩要不要顺手改写轨迹"的产品取舍, 待定。
- **精确 tokenizer**: 现用 char 估算 + 兜底重试, 未接 provider tokenizer。图 token 名义
  值未按尺寸计算。真场景校准 margin/除数后再决定是否值得引精确计数。

## 运行时缺陷 (2026-07-19 人工对话暴露, 已在 Aurelius 侧修正)

真实 `moss-run-ghost aurelius` 连续对话时暴露两个 articulate 阶段缺陷。二者都不是记忆
写入语义的问题 (失败帧仍被 witness 为 `failed`, 记忆哲学未受影响), 而在流式 articulate
的物理收尾与模型配置接线上。两者都继承自 Atom 基线 (`atom/_runtime.py`、`atom/_meta.py`
写法相同), **本次按协作者要求只修 Aurelius, 不碰 Atom / Ghost 基类** — Atom 作为纯净对照
基线保留同款缺陷, 跨原型的统一收敛 (安全流式收尾 helper + 契约 max_tokens 接线放哪一层)
留给后续独立 workstream, 因为它横跨两个原型, 不属于 aurelius-ghost 这条线。

- **打断时未捕获异常逃逸事件循环 (`asynchronous generator is already running`)。**
  新输入抢占当前帧 → attention abort → `AttentionAbortedError` 从 `stream_text()` 的
  `async for` 体内抛出 → 穿过 `run_stream(...)` 的 `__aexit__`, 后者试图 aclose 一个仍在
  运行的 httpx SSE 生成器, 抛出二次异常逃逸到事件循环顶层。第一条
  `❌ articulate error: Attention is already aborted` 是被 driver 正确处理的预期打断;
  逃逸的是第二条。修法: 在 `_runtime.py` 把 `stream_text()` 迭代器放进独立 try/finally,
  abort 传播时先 `aclose()` 文本流并 suppress 其二次 teardown 噪音, 打断本身仍向上传播由
  driver 处理。测试盲区已坐实: Atom/Aurelius 对"流式生成中途被 abort"零覆盖。

- **`max_output_tokens` 未接线 → `token limit (provider default) exceeded`。**
  `_meta._build_configured_model` 构造 `AnthropicModel`/`OpenAICompatibleModel` 时未传
  `max_tokens`, `Agent(...)` 也无 `model_settings`, 于是回退 provider 默认输出上限
  (错误信息里的 `(provider default)` 即此)。与输入长度、模型上下文窗口无关 — 几百字小说
  的 prompt 离 200k 差数量级; 触发点在输出侧预估叠加 thinking 预算在默认上限边缘抖动,
  故时有时无。契约层 `LLMConfig.max_output_tokens` (默认 4096) 早已存在却从未读出。修法:
  仅在走配置构造 (非注入 model) 时以 `ModelSettings(max_tokens=resolved.model.max_output_tokens)`
  设到 Agent `model_settings`; 注入 model 携带自身 settings, 不覆盖。回归
  `test_configured_model_wires_max_output_tokens` 已锁定。

## Implementation Notes

<!-- Gotchas, non-obvious behaviors, reasons for rejecting simpler alternatives. -->

- 参照 `ghosts/atom/` 的分文件形态: `_meta.py` (GhostMeta bootstrapper) +
  `_runtime.py` (Ghost runtime) + `_adapter.py` (Moment↔ModelRequest) + 单测.
- 依赖 desktop (`ghost-filesystem-desktop`) 与 memento (`momento-mori`) 的契约
  就位程度. memento 当前 contract-frozen-pending-review, desktop in-progress
  (channel 落点已定, K14~K18). 起步前先对齐这两条的可用表面.
- 2026-07-17 技术评审落在
  `Docs/MOSS-Ghost-Memory集成技术评审与实施方案.md`（2026-07-19 移入包内
  `src/ghoshell_moss/cli/docs/memory/aurelius-memory-review.md`，随 wheel 发布并入 `moss docs`
  体系；测试方案同迁为 `aurelius-memory-test-plan.md`）。当前分支的 `moss` 根 CLI
  因 `cells_cli.py` 导入已删除的 `CellRegistry` 无法启动，本 workstream 不借机修改
  该相邻重构；feature 状态按同一 frontmatter 契约直接维护。
- 已交付 `ghosts/aurelius`（AureliusMeta/Aurelius/AureliusMemory）、workspace 注册、无网络 acceptance
  script 与人工测试方案。相关回归 138 passed；正式 `tests/ghoshell_moss` 为 1650
  passed、5 failed、2 errors，其中 Mindflow 单项重跑通过，Cell 两项是当前分支旧测试
  与新 ABC 不一致，Zenoh 三项为顺序/关闭超时，均不在本 workstream 路径。
- 2026-07-17 第二阶段已落地：`MemoryConfig` 注册到 workspace config manifest；Aurelius 在
  mechanical commit 后用独立任务反思，在下一次启动时有限追赶；`ghost` channel 只开放
  当前 owner 的 inspect/log/staging/show、semantic commit、reinterpret、fork/switch 与
  手动追赶。反思任务可取消、去重、失败留观测，不进入 articulate 热路径。
- 第二阶段定向回归：`ruff check src/ghoshell_moss/ghosts/aurelius`、Aurelius + Memento 的 pytest
  共 96 passed、`scripts/ghost/aurelius_memory_acceptance.py` 通过，`moss-run-ghost` 成功发现
  `aurelius`。`ghost_runtime.py` 的全文件 ruff 仍有改动前已有的 import/type/line-length
  问题，本次仅在其既有虚拟 channel 机制中加入 Ghost channel 注册，未扩大清理范围。
- 2026-07-18 原型正式更名为 Aurelius：公开包、类型、workspace 注册文件、默认 owner、
  acceptance script、Feature 目录与测试文档全部迁移。旧 `data` 的 Memento root/owner
  不自动改写；为了保真，兼容旧轨迹只能显式传 `memory_root` 与 `memory_owner`，不能复制
  目录伪造迁移。
- `MemoryConfig` 的可编辑模板落在 `.moss/configs/memory.yml`（stub 同步）；它带字段注释。
  YAML 在 Aurelius 初始化时读取，修改后需要重启。集成方案已按最终运行目标重组，并明确了
  `commit → reinterpret → CommitNote` 的追加版本机制、反思 writer、失败追赶和 CTML 边界。
- 2026-07-18 已完成 Lynn 对接技术评审（仅文档，未修改 reachy 项目）：推荐 Lynn 基于
  Aurelius 形成唯一对话主写，保留 SimpleMemory 与 Lynn 的 thinking/flash/取消语义；
  `person_id` 是长期 owner 的候选真相源，但多人 pin/匿名转身份必须经过显式路由，不能
  自动合并。详见 `Docs/Lynn-Aurelius-Memory集成技术评审.md`。
- 测试分层已校正：pytest/acceptance 是不依赖 Zenoh 的 L0；`moss-run-ghost` 的发现和
  TUI 对话属于 L1/L2，必须先 `uv sync --extra host --extra ghost` 并验证 `import zenoh`。
  缺少该 extra 的 traceback 发生在 Host/Matrix 导入期，不能误判为 Aurelius 记忆故障。
- pydantic-ai 2.x 把 `OpenAIModel` 更名为 `OpenAIChatModel`，曾使 Aurelius manifest 在
  discovery 时导入失败并被旧 CLI 静默过滤。`_meta.py` 现兼容 pydantic-ai 1.x/2.x；
  `moss-run-ghost` 也会向 stderr 报告 skipped manifest，避免“未列出但无错误”这一假象。
- 2026-07-18 人工启动验收揭示当前分支相对 `dev` 的通用启动契约断裂：`MossRuntime.logger`
  在 Matrix 尚未启动时承诺回退 `Environment.logger`，但 Environment 重构时该属性被删除。
  这发生在 `GhostRuntime.__aenter__` 的第一个通用步骤，早于 Aurelius factory；`echo` 与
  `aurelius` 都会复现。恢复无运行期依赖的 `Environment.logger` 回退，并让 TUI 同步打印启动
  异常，避免用 `closed / good bye` 掩盖根因。随后暴露第二个同源失配：当前 Matrix 延迟到
  `__aenter__` 才建 IoC Container，而 GhostRuntime 早于 Matrix enter 注册 Ghost provider；dev
  的 Container 则在 Matrix 构造期准备。现恢复“构造期注册、进入期 bootstrap”的两阶段边界。
  两项修复都不使 GhostRuntime 感知 Aurelius，也不改变 Memento 或 LLM 的生命周期。
- 2026-07-18 将生产级事实读取定义为本 workstream 的 P0 后续：实现前 Aurelius 只能称为
  “可审计持久轨迹原型”，不能声称事实型长期记忆可靠。两份验收文档已记录阻断性
  `AMBER-731 / staging` 对 `ORBIT-004` 干扰回归、错误 logos 防污染、反思隔离、来源重建及
  `current/superseded` 校验；实现必须使用 fake/TestModel 做确定性验证，不能以真实模型偶发
  答对作为通过。
- 2026-07-18 P0 首版落地：新增 `_knowledge.py`，从当前 Memento branch 的 commit + staging
  即时重建 Evidence/Claim，不写第二份 projection 真相；规则 extractor 支持显式 canonical
  key 以及已评审的测试/设备/城市/偏好字段。用户与 trusted-tool source 可提升，logos/reflection
  只保留 rejected candidate；Recall 按 key 返回 bounded packet，unknown/conflict fail closed，
  事实题输出在 yield 前校验。顺带修复真实带 history 请求时 pydantic-ai 2 要求
  `UserPromptPart` 的既有适配缺口。
- 2026-07-18 P1 首版落地：新增 `_desktop.py` 薄适配，Aurelius 生命周期持有 DefaultGrounds；
  Host 使用 `Project.root`，测试/嵌入退化到 GhostWorkspace.home。DESKTOP body 作为 run
  instruction、Ground frame 作为临时 request context；CTML 仅开放 workspace-bound 的
  desktop open/close/pin/unpin/update/frame，不提供世界写入。
- 当前定向回归：Aurelius + Memento + Desktop 共 191 passed；ruff 全绿；acceptance 完成
  write → commit → reopen → project → recall → verify，并拒绝 ORBIT 字段替换。P2 principal/
  audience/retention 治理与 P3 backend contract 均未实现，文档已从“目标”改为准确状态。
- 2026-07-18 真实 TUI 验收发现 source 集成缺口：Moment 已将用户输入写入
  `input_signal_nucleus`，而 Evidence 默认值仅含 `input`/`user`，导致已落盘证据被跳过。
  修复只补齐真实 source，并让单测与 acceptance 共用该 source；不改 Memento/Ghost 契约。
  Aurelius + Memento + Desktop + InputSignalNucleus 定向回归 200 passed，且对真实 TUI
  staging 副本重建出 `AMBER-731 / staging`。
- 人工验收的旧数据清理收口为 `scripts/ghost/aurelius_memory_reset.py`：目标从脚本位置
  固定解析，运行中进程、symlink、越界或异常顶层内容均 fail closed，避免文档中的
  裸 `rm -rf` 与手工选路径。
- 2026-07-18 人类协作者明确提出：仅有事实 Claim 不足以支撑有用的持久智能体，还必须
  能区分“外部事实”与“自己曾经的行为”，持久记录可恢复的中断边界，并根据已认证
  principal 做最小披露。本文新增 P2-A~P2-D 和 cross-cutting definition of done，作为后续
  实现与验收的明确轨迹；当前状态仍是 pending。
- 2026-07-18 对真实 `tmp/log` 与 Memento 只读对账：一次普通事实输入被模型放大为 5 个
  Moment、2 次 semantic commit、2 次全量 claims 审计；17 个总 Moment 中大量是无用户 source
  的内部再观察，机械 Note 又摘录了这些 CTML/长回复。修复后 memory 运维面默认隐藏且不触发
  Re-Act，normal/verbose/trace 分层，claims/Note 有界。文档命令回归 198 passed，acceptance 通过。
- 2026-07-19 记忆读取路线转向：删除正则 Evidence/Claim/Recall/Verifier 层（`_knowledge.py`）。
  该层只在极窄的手工 canonical 模板内有效，把语义判断硬编码成脆弱正则并压到契约外层，与
  momento-mori “记忆是主体生产的轨迹、不是管线蒸馏的数据库” 相悖。改为三件套：(1) grep 式
  `memory_search` 在本 owner 全部冻结 commit + staging 上做原文子串扫描，返回稳定地址供
  `memory_show` 缺页展开——承认“按字面找”，把语义判断留给读到证据的模型；(2) `memory_discipline`
  instruction 注入 system prompt，要求无可见依据时先检索、再核对、查不到如实说未找到；(3) 旁路
  curation（`AureliusCurator`）从冻结轨迹重写人类可读笔记并 pin 进 Ground，带出处横幅、可回溯、
  可 unpin，不建第二真相文件。承接的哲学债：正确性不再靠输出后校验，靠不可变证据 + 字面检索 +
  行为纪律。字面检索有诚实边界（同义改写/时间推理覆盖不到），模型应表达不确定而非编造。
- 2026-07-19 修复三个确认缺陷：(1) **CTML 调度崩溃** — CTML 命令在 `asyncio.to_thread` 工作线程
  执行，`memory_reflect`/`memory_curate` 原先直接 `create_task` 因无运行中 loop 抛 `RuntimeError`；
  现由 `Aurelius._spawn()` 用 `__aenter__` 捕获的 loop 经 `call_soon_threadsafe` 编组回主循环。
  (2) **进程内写竞争** — `remember`/反思跑在事件循环，`memory_commit`/`fork`/`switch`/`reinterpret`
  跑在工作线程，两个写者域共享 `staging.jsonl` 与 `self._branch`；`AureliusMemory` 现持 `RLock`，
  所有写方法与读 branch 指针的渲染方法都在锁内，单写者纪律在进程内也成立。(3) **渲染打戳缺失** —
  折叠摘要曾丢 `note_seq`，反思改写 note 后不可归因；现 `<memento commit=... note_seq=... kind=...>`
  满足 FORMAT.md 不变量 13。
- 2026-07-19 落地两个设计项：(1) **折叠摘要不伪造模型回合** — 早期在摘要块后紧跟捏造的
  `ModelResponse("[memento summaries loaded]")`（模型从未说过的话），现将摘要 preamble 折叠进下一条
  真实用户 request，无任何虚构 assistant 轮次；note 正文 `<`/`>` 转义防伪造 `</memento>` 边界。
  (2) **失败帧如实入轨迹** — `on_articulate_exit(error=...)` 不再丢弃失败帧。按 momento-mori “noop 是
  轨迹事件” 的对称推论，“看见 X、尝试、出错” 同样值得 witness；失败帧带 `failed` thread tag 写入，
  永不读作完成回合，`inspect_context['memory_write']` 记 `staged_failed`/`committed_failed`。此外
  多模态 percept 无法转文本/图像时保留占位标记，不静默丢失该轮存在。
- 2026-07-19 实测修复「模型不回复」：`moss-run-ghost aurelius` 问 canonical-key 事实题
  （ORBIT-004）时 ghost 静默。日志显示 CTML `compiled=1 done=1 failed=0 observe=False`——模型
  按记忆纪律先发 `memory_search` 而非散文，但检索命令未标 `always_observe`，命中不回灌下一轮
  Re-Act，回合无人阅读即静默结算。修复：`memory_search`/`memory_show`/`memory_log` 三个「读以
  作答」命令改 `always_observe=True`（写/运维命令保持 False，它们是动作不是答案）。加结构化回归
  断言检索面 observe、写面不 observe，防下个实例静默回退。
- 2026-07-19 定向回归：Aurelius 33 passed（含新增 CTML 工作线程调度、并发 remember/commit、grep
  检索、失败帧、渲染打戳、注入转义、无伪造回合用例），Memento core 79 + host UI 3 passed，ruff 全绿，
  acceptance 完成 write → commit → reopen → search → show 链。两份文档（集成方案、测试方案）已按
  grep 检索 + 纪律 + curation 的真实实现重写，作废 Evidence/Claim 相关 P0 章节与用例。
