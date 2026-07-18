# MOSS Aurelius Ghost Memory 集成技术评审与实施方案

> 状态：已实现 Aurelius/Memento 的**轨迹主路**、P0 可信知识骨架和 P1 Ground 接线。
> 当前实现能在明确支持的 canonical field 范围内重建 Evidence/Claim、按问题召回、隔离
> logos/reflection 污染、校验回答，并把 `DESKTOP.md`/Pin 作为本帧工作场。P2 的 principal/
> audience/retention 治理、通用语义抽取与 P3 外部召回后端尚未实现，因此不能把这一骨架表述为
> 任意领域、任意对话对象下都已达到生产级可靠性的长期记忆产品。
> 2026-07-18 的真实 TUI 采样还修复了默认输出与轨迹放大问题：普通模式只展示用户可读回复，
> 记忆运维命令不再进入普通模型提示或自动触发 Re-Act，机械 Note 有 600 字符全局上限。
>
> 关联文档：[测试方案](MOSS-Ghost-Memory测试方案.md)、[Moshi 集成技术评审](Moshi-长期记忆与进度集成技术评审.md)。

## 1. 结论与命名

第二个 Ghost 原型命名为 **Aurelius**，取自《沉思录》作者 Marcus Aurelius。Atom 仍是
无持久历史的最小基线；Aurelius 是带有可审计 Memento 轨迹、机械锚点和事后反思的持久化
原型。它不是一个泛称为 “Data” 的数据容器。

当前已交付的轨迹闭环如下：

```text
完成的 Moment
  → Memento staging（立即持久化）
  → mechanical commit（默认每 4 个 staged Moment）
  → 非阻塞 reflection（可失败）
  → reinterpret() 追加 CommitNote
  → 下一次模型上下文：早期 note + 近期完整 Moment
```

当前 P0/P1 已把这个闭环扩展为：

```text
完成的 Moment（不可变轨迹证据）
  ├─ Memento staging / commit / branch：保存“发生过什么”
  ├─ Reflection：生成解释、假设、待确认线索；不能直接成为事实
  ├─ Evidence / Claim projection：从可信来源派生可验证知识
  │     → Recall：按当前问题、对象与权限取少量相关 Claim 及来源
  │     → Evidence packet：作为当前帧的有限认知表面
  │     → Answer verifier：拒绝无证据、串字段、越权或陈旧的回答
  └─ Ground / Pin：为当前任务重绘工作场、约定和外部对象的可观察表面
```

原型的公开代码名、运行名与默认 owner 均为 `aurelius`：

| 位置 | 当前名称 |
|---|---|
| Python 包 | `ghoshell_moss.ghosts.aurelius` |
| 类型 | `AureliusMeta`、`Aurelius`、`AureliusMemory`、`AureliusReflector`、`MemoryProjection`、`AureliusDesktop` |
| 工作区注册 | `.moss/src/MOSS/ghosts/aurelius.py` |
| 启动命令 | `.venv/bin/moss-run-ghost aurelius` |
| Ghost home | `.moss/ghosts/aurelius/` |
| 默认 Memento root | `.moss/ghosts/aurelius/memento/` |
| 默认 Memento owner | `aurelius` |
| Feature workstream | `.ai_partners/features/workstreams/2026/07/aurelius-ghost/` |

## 2. 目标、非目标与真相边界

### 2.1 本次目标

1. 每个成功完成的认知帧都可重启恢复；失败或中断帧不伪装成完成记忆。
2. 近期上下文保留完整 Moment，较早内容以可追溯的 commit note 折叠。
3. commit 后能用小模型反思，而不会增加主对话首 token 延迟。
4. 人与 Ghost 能通过受限 CTML 检查、锚定、更正和分叉自己的记忆。
5. 策略在 workspace YAML 中可配置，而不是把轮次阈值写死在代码里。
6. 明确可解析的用户/可信工具事实形成带 Memento 证据的 Claim，模型输出与反思不能自动提升。
7. 记忆型事实题按 canonical key 召回有限证据，并在输出给 Shell 前校验；未知/冲突安全拒答。
8. Ground 的 instruction/frame 只进入当前帧，不因出现在 `DESKTOP.md` 或 Pin 中成为长期事实。
9. 默认 TUI 不暴露 CTML、Moment、System、command-result 或 `Log:`；详细观测必须显式开启。

### 2.2 生产级定位、已修复的 P0 故障与剩余边界

Aurelius 不能停留在“长 prompt 加持久 jsonl”的 Demo 形态。它至少必须是一个可向生产产品
扩宽的有效骨架：原始轨迹可审计、派生状态可重建、来源可追溯、错误输出不会反哺事实、读取
失败宁可明确未知也不能编造。

2026-07-18 的真实验收暴露了原 P0 缺口：`0001.jsonl`、反思 note 和实际模型 history 都含
`AMBER-731 / staging`，但模型仍将另一个字段 `ORBIT-004` 误答为测试代号。故障不是 Memento
丢失数据，而是 Aurelius 把轨迹、反思、控制面结果和历史模型回答混为普通聊天历史，且没有
“查询 → 证据 → 回答校验”的读取链。

当前实现已按下列原则补齐确定性骨架：

1. **Memento 是证据账本，不是事实数据库。** Moment/Commit 保存发生过什么；CommitNote 是
   可变解释，不能因为 last-wins 就成为当前真相。
2. **事实必须是带来源的投影。** active Claim 需要 `moment_id`、`commit_id`、原文 span、
   来源类型、scope 与状态；投影可以重建，不能脱离证据独立漂移。
3. **回答前先检索，回答后再校验。** 对记忆型问题，不能仅依赖 LLM 在完整 history 中“注意到”
   正确文本；输出的每一个事实值必须能映射到本轮检索到的 Claim。
4. **模型输出不自动成为事实。** 成功返回只说明 Moment 可以写入情景轨迹，不能说明其 logos
   可提升为 active Claim。
5. **反思是认知，不是裁决。** 反思可产生候选 Claim、偏好、冲突和待办；它不得直接覆盖
   active Claim 或改变世界执行状态。

### 2.3 明确不做

- 不改 Atom，不把 Memento 逻辑塞进 GhostRuntime 或 `core.memento` 契约。
- 不保存模型隐藏思维链；反思只处理用户可见输入、已完成 logos 与显式执行结果。
- 不在本次轨迹主路中实现向量检索、git witness 调度、时间阈值 commit、自动 branch merge 或
  跨 owner 写。P0 Recall 采用 canonical key + 有界规则抽取 + 原文证据展开；是否引入向量
  召回应在其后以可测的召回质量和成本决定。
- **不接入 Mem0。** P3 仅保留经过专项评审后再定义后端端口的设计位置；当前不实现
  `MemoryRecallBackend`/adapter，不安装 SDK、不增加 API key、配置项、网络调用或外部存储。
  即使未来选择 Mem0，它也只能是 Claim 的候选索引，绝不是 Memento 原始证据或 Claim 真值的替代品。
- 不把反思推断当作用户身份、课程进度、CTML 成功或 TTS 播放位置的权威事实。
- 不修改 moshi App/mode 代码；Moshi 的接入路线见关联技术评审。

### 2.4 最终目标：让 Agent 更聪明，而不是让 prompt 更长

本工作的产品目标不是“给 Aurelius 加一个 Memory 功能”，而是让它形成可扩展的认知能力：既能
连续记住经历，又能将经历结构化为知识；知道什么可靠、什么只是观点、什么尚待验证；知道何时
保存、何时不再主动使用；并能面对不同对话对象只取用有权、相关且适宜的记忆。

这不是保存模型隐藏思维链。Aurelius 应持久化的是可观察、可审计的事实、决定、承诺、执行结果、
显式不确定性与反思结论；“模型内部如何逐 token 思考”既不可作为证据，也不应进入记忆系统。

一套成熟记忆的能力边界如下：

| 人类式能力 | Aurelius 的机制落点 | 禁止的简化 |
|---|---|---|
| 情景记忆 | Memento Moment/Commit，保留发生过什么 | 用摘要覆盖原始轨迹 |
| 工作记忆与注意力 | Ground/Pin 的每帧重绘 | 把已探索的全部文件永久塞进 history |
| 结构化知识 | Evidence 支撑的、带类型与状态的 Claim | 将所有文本都视为等价“事实” |
| 置信与不确定性 | 来源强度、证据数量、冲突状态、显式 confidence | 把模型语言上的自信当作真实性 |
| 自省 | Reflection 产生观点、假设、待确认项 | 让反思直接覆写事实 |
| 遗忘与克制 | retention / archive / tombstone 与召回策略 | 静默 LRU 删除原始证据 |
| 面向对象的表达 | scope、audience、sensitivity 与最小披露策略 | 因同一 Ghost 记得，就向每个对话对象泄露 |

### 2.5 认知状态不能混写

| 状态 | 例子 | 权威来源 | 是否能直接回答事实问题 |
|---|---|---|---|
| 情景记忆 | 用户问过什么、Aurelius 如何答复 | Memento Moment 与 Commit | 否；只能作为证据展开 |
| 工作场 | 当前仓库、打开的任务目录、Pin 的文件片段 | Ground/Pin + `DESKTOP.md` | 否；它是当前可见世界 |
| 事实 | 本轮测试代号、设备字段及其更正状态 | 带 EvidenceRef 的 Claim projection | 是，须通过 scope/状态校验 |
| 观点/偏好 | 用户偏好短回答、某方案风险较高 | 有来源的 Claim，或待确认候选 | 仅能按类型表达，不得伪装成客观事实 |
| 假设/待验证项 | “城市可能已经变更”、尚待读取的文件 | `uncertain` ClaimCandidate / Reflection | 否；应披露不确定性 |
| 世界执行进度 | CTML task 是否完成、TTS 播放 offset | 对应执行组件的事件/进度存储 | 由执行组件回答 |

Memento 可以保存上述状态的“观察证据”，但不能替代它们的权威存储。Claim projection 是
Memento 之上的可重建认知层；它不应替代原始证据，也不应被反思 note 替代。Ground 也不保存
长期事实：它负责将当前任务有关的世界和约定摆在“桌面”上。

## 3. 架构、所有权与上下文

```text
GhostRuntime
  ├─ 调用 Aurelius.articulate() 读取模型上下文
  ├─ 正常结束时调用 Aurelius.on_articulate_exit()
  └─ 将 Aurelius.channel() 注册给 Shell

Aurelius
  ├─ AureliusMemory：Memento 读写、窗口、分支（经历账本）
  ├─ AureliusDesktop（P1）：Ground 生命周期、`DESKTOP.md` 与 Pin 帧
  ├─ AureliusReflector：commit 后的异步释义
  ├─ MemoryProjection（P0）：Evidence / Claim 的可重建投影
  ├─ MemoryRecall（P0）：按问题取得带来源的有限证据包
  ├─ MemoryAnswerVerifier（P0）：检查回答是否可由证据包支持
  ├─ MemoryRecallBackend（P3，未实现）：未来语义/向量候选召回端口
  └─ ghost CTML channel：受限人工/模型控制面
```

`AureliusMemory` 是 Ghost 侧适配器。它只使用 `core.memento` 已有的 `update`、`window`、
`commit`、`reinterpret`、`checkout` 和 `switch` 能力，不扩大 Memento ABC。

一个 `(memento root, owner)` 只有一个写者。不同用户、并行进程或不同产品实例必须分配
不同 owner 或在外层完成单写协调；不能依赖“最后写入者获胜”。当前 CTML 也只允许操作
当前 owner 的 branch。

模型上下文由 `AureliusMemory.model_history()` 每次重建，不维护第二份进程内对话 list：

1. 超出 detail window 的早期 commit 以 `<memento>` note 进入上下文；
2. 最近 `detail_n` 个完整 Moment 以原始 request/response 进入上下文；
3. P0 对被识别为记忆型事实题的当前 Moment 生成有限 evidence packet；
4. P1 把 Ground frame 放进本帧 user context，把 Ground instruction 作为本次 run instruction；
5. 当前 Moment 原始输入最后传给 Agent；系统 prompt/soul 仍由 `AureliusMeta` 和
   `SystemPrompter` 组装。

这使重启恢复、窗口折叠和审计使用同一份持久事实。

当前代码通过薄 `AureliusDesktop` 持有 `core.desktop.DefaultGrounds`：Ghost enter 时自动打开
项目 root（Host 中取 `Project.root`，嵌入/测试退化为 `GhostWorkspace.home`），articulate 每帧读取
instruction/context，Ghost exit 调用 Grounds 的 best-effort sediment。`DESKTOP.md`/Pin 不写入
Moment，也不经过 Claim extractor；P1 没有把 `DESKTOP.md` 变成另一个聊天历史文件。

### 3.1 旧读取路径的失败模式与当前隔离

`model_history()` 仍将早期 CommitNote 与近期完整 Moment 转换为普通 pydantic-ai
history。它同时带入用户输入、历史 `logos`、Shell/CTML 执行结果与 `memory_log` 一类诊断输出。
它负责重放对话，本身不表达“谁是事实来源、哪条是解释、哪条只是旧模型回答”的层级。

因此当前事实读取不再把 history 当权威：`MemoryProjection` 直接遍历 commit/staging，建立
`moment_id + commit_id + note_seq + span + origin + scope` 证据引用。history 只维持对话连续性；
被识别的事实题必须经过 Projection/Recall/Verifier。顺带修复了 pydantic-ai 2 历史适配：
`ModelRequest.parts` 使用正式 `UserPromptPart`，不再把 `TextContent` 误放进历史消息。

以下行为仍不能被接受为正确性，且已成为自动化阻断断言：

- 正确事实已经在 history 中，模型仍选择语义相近但字段错误的值；
- 历史模型回答、反思文本或控制面输出压过用户直接陈述；
- 一条成功但错误的 logos 在后续 commit/反思后成为更显眼的错误上下文；
- 仅靠增加 `detail_n`、`summary_m` 或加强 prompt 试图修复上述问题。

### 3.2 已实现的读取骨架：账本、投影、召回与校验

当前实现保持 `core.memento` 的 payload 不透明与冻结契约，在 Aurelius 上层实现如下链路：

```text
Moment / Commit（不可变事件账本）
  → Evidence extractor（仅识别可信来源及原文 span）
  → Claim projection（active / superseded / uncertain）
  → Recall(query)（候选 Claim + EvidenceRef + MementoRef）
  → 当前帧 evidence packet（有界、按问题重绘）
  → 对话模型
  → Answer verifier（事实值、字段、状态、来源逐项核验）
```

`EvidenceRef` 至少包含 `moment_id`、`commit_id`、原文片段或稳定 span、`origin`（`user`、
`trusted_tool`、`assistant`、`reflection`）与 branch/owner scope。`Claim` 至少包含 canonical
key、value、status（`active`、`superseded`、`uncertain`）、EvidenceRef 列表和可选的
`supersedes` 引用。

例如，以下两项必须是不同 key，而不是仅靠自然语言区分：

```text
test.run.code = AMBER-731
test.run.environment = staging
device.ORBIT-004.validation_phrase = 雪松
```

当用户问测试代号和环境时，Recall 只能返回前两项及其来源；`ORBIT-004` 即使在历史中出现
很多次，也不能通过字段校验。没有证据、候选冲突或 scope 不匹配时，系统必须返回“没有找到”
或显式冲突，而不是让模型补全。

该层应借用 Grounds 的**每帧重绘、地址而非快照、读写分离**思想，但不应把长期事实写入
Desktop/Ground：Ground 负责当前工作表面，Memento 负责过去轨迹；evidence packet 是根据当前
问题从 Memento 投影得到的有限认知表面。

实现细节与客观边界如下：

- `_knowledge.py` 的 `MemoryProjection.snapshot()` 每次从当前 branch 的全部 commit 与 staging
  重建；没有 projection 数据库或缓存文件，重启/切 branch 后直接重建。
- 自动提升只接受 `knowledge_user_sources` 与 `knowledge_trusted_tool_sources` 中的 percept。
  assistant logos 和 `by=memento-reflection` 的 note 即使解析出同 key，也只保留为 rejected
  `ClaimCandidate`。
- 首版 extractor 有意收紧：支持显式 `canonical.key = value`，以及测试代号/环境、设备校验词/
  颜色、用户城市与回答偏好等已评审模板。无法映射到 canonical key 的自由叙述仍只在 Memento，
  不会被模型“聪明地猜成”事实。
- Recall 只在当前输入同时具有问题形态和已支持 key 时进入事实门；字段缺失返回 `unknown`，
  同 key 未显式更正的多值返回 `conflict`。显式更正把旧 Claim 标为 `superseded` 并保留引用。
- `render_packet()` 保持 JSON 完整并按字符预算逐级收紧 quote；无法在预算内完整表达时转为
  `unknown`，不截断出非法 JSON。
- 事实题的模型输出在 yield 前缓冲；verifier 要求每个 requested key 的 active value 出现，
  拒绝投影中未召回的值、额外结构化 token 和错误环境枚举。失败只输出安全拒答，错误模型文本
  不会先流给用户。普通对话仍保持原流式路径。
- 这不是通用自然语言事实证明器：未识别领域、同义改写、复杂时间/实体关系和任意自由文本输出
  的语义蕴含仍需要后续 extractor/verifier 扩展。当前正确性承诺只覆盖已识别 canonical field。

#### 3.2.1 Claim 的类型、置信与状态

`Claim` 不是只有 `key=value` 的 KV 表。至少应有下列字段，以使“知道”“认为”“等待确认”在
数据层可区分，而不依赖回答措辞：

```text
Claim {
  key, value, kind, status,
  subject_scope, owner_scope, branch_scope,
  audience_policy, sensitivity, retention,
  evidence_refs[], confidence, supersedes[]
}
```

| 字段 | 含义 | 关键规则 |
|---|---|---|
| `kind` | `fact`、`preference`、`opinion`、`plan`、`procedure`、`hypothesis` | 不同 kind 不可用同一断言口吻回答 |
| `status` | `active`、`superseded`、`uncertain`、`archived`、`tombstoned` | `active` 也不等于对所有 audience 都可见 |
| `confidence` | 对证据充分度的可解释分级/数值 | 由来源、相互印证、时间与冲突计算；不是 LLM 自评分 |
| `subject_scope` | 该信息关于谁/什么，例如用户、设备、项目 | 不能把甲的属性答到乙身上 |
| `owner_scope` / `branch_scope` | 哪个 Ghost 身份、轨迹分支可读取 | 保持 Memento 单写者和 branch 语义 |
| `audience_policy` / `sensitivity` | 可向谁、以何种粒度披露 | 召回后、生成前都必须检查 |
| `retention` | keep / archive / review-at / delete-on-request 等 | “不再主动记起”不等于静默抹掉证据 |

Evidence 的可信度必须与 Claim 的置信度分开：用户直接陈述和已认证工具结果可以是强证据；
assistant logos 与 reflection 只是弱观察。多个独立强 Evidence 可提高 confidence；更正、时效
到期或彼此矛盾应降低 confidence 或把 Claim 标为 `uncertain`，而不是由“最后一条文本”获胜。

当前 P0 已承载上述字段形状和基础状态转换：用户初始证据置信为 `0.9`、trusted tool 为 `0.98`，
无显式更正的异值冲突降为 `uncertain/0.5`。`audience_policy`、`sensitivity`、`retention` 目前只是
不让数据模型再次破坏兼容性的保守字段；尚无 principal policy engine，不能据此声称 P2 已完成。

#### 3.2.2 记忆管理、遗忘与选择性披露

“应该存什么”与“此刻应该告诉谁”是两道不同的决策，必须分离：

```text
保存资格（是否形成 Candidate）
  → 提升资格（是否成为 Claim）
  → 保留策略（keep/archive/review/tombstone）
  → 查询召回（相关性 + scope + 时效）
  → 披露裁决（audience + sensitivity + 当前任务）
  → 回答校验
```

保存的正面条件是：将来可复用、来源可说明、scope 可判定、用户或产品策略允许。短暂寒暄、重复
内容、无来源的模型猜测、机密/凭据和不应持久化的敏感信息默认不提升。这里借用 MOSS
Grounds 的 K20 原则：系统不得以 LRU 静默替主体决定“忘掉什么”。

遗忘也不能一概等同删除：`archive` 表示不参与默认 Recall 但仍可审计；`review-at` 使暂时性
计划在到期后降级；`tombstoned` 使被撤回或按删除请求处理的信息不再被召回。原始 Memento
证据的保留/清除必须服从显式隐私与合规策略；在没有该策略前，系统不能假装已经实现“安全删除”。

面向儿童和成人、不同用户或不同应用说话时，Aurelius 可以具有同一条内部证据，但绝不能默认
共享。每次 Recall 至少以 `owner_scope`、会话 principal/audience、workspace/product scope 和
sensitivity 做过滤；披露层再按任务需要做最小化呈现。拒绝披露、答“没有可共享的信息”与“系统
不应告诉你该信息”是正确结果，不能为了显得聪明而绕过边界。

本节是 P2 约束，当前未落地。现有 Recall 只强制当前 Memento owner/branch，尚未接收产品层
principal、audience、授权上下文或 review-at 时钟；在这些输入契约完成前，不得用于跨用户或
儿童/成人差异化披露场景。

#### 3.2.3 Ground / Pin / `DESKTOP.md` 的正式分工

`DESKTOP.md` 是 Ground 的 L0 持久化载体，不是用户画像数据库。它包含三类相互独立的信息：

| 载体部分 | 应存内容 | 不应存内容 |
|---|---|---|
| frontmatter | GroundConvention、帧预算/目录等结构策略 | 用户事实、聊天摘要 |
| body（法） | 身份、长期协作约定、项目规约、人工 promote 的工作胶囊 | 未验证的人物属性、随意累积的对话历史 |
| `desktop:pins` | 当前关注的地址、观察 hash、note | 文件内容快照、长期知识库副本 |

Ground/Pin 的职责是每帧为 Agent 重绘“现在看得见什么”：打开哪个目录、关注哪段文件、哪些磁盘
对象发生变化、当前工作场有哪些约定。`Pin` 保存地址和观察，不保存世界快照；`update()` 是主体
明确承认变化，而非系统悄悄同步。这正适合智能体做任务、调试和协作，不适合承载“用户住在哪里”
这类跨场景事实。

P1 已按以下顺序接线：Ghost enter 以明确 workspace root 打开 Ground；每个 articulate 帧将
Ground instruction 置入 run instruction，将 `context()` 放入有界 current-ground context；Ghost
exit 只 sediment Pin 清单。`desktop_open/close/pin/unpin/update/frame` 经 Ghost Channel/CTML 暴露，
目录与 Pin 均做 workspace/Ground root 边界校验。将 Pin promote 为 body 胶囊仍应是显式、可审计
动作；若它还要成为长期可问答知识，必须另走 Evidence/Claim 提升链，而非因出现在
`DESKTOP.md` 就自动变成事实。

#### 3.2.4 可选语义召回后端：P3 设计位置，不接 Mem0

P0 已实现本地、可重建、可解释的 Claim Recall。为了不把未来的规模化需求硬编码进 Aurelius，
P3 经独立评审后可以在其外定义一个可选端口，例如：

```text
MemoryRecallBackend
  index(claims, scope) -> IndexReceipt
  recall(query, scope, limit) -> CandidateClaimRefs
  health() -> BackendHealth
```

该端口只能返回候选 `ClaimRef` 与排序理由，不能返回无 EvidenceRef 的“真相”，也不能写 Memento、
改 Claim 状态或越过 audience filter。本地 Recall 是权威退化路径：后端不可用、超时或返回异常时，
事实题使用本地索引/安全拒答，普通对话不被外部依赖阻塞。

Mem0 是未来可评估的一个 adapter 候选，而非本设计的前置依赖。当前既不提供
`MemoryRecallBackend` 的仓促公共契约，也不提供 `Mem0Backend`、`mem0ai` 或任何 Mem0 配置；
是否定义端口、是否选择它，都应在数据出域、召回质量、成本、时延和删除/审计语义的专项评审后决定。

#### 3.2.5 实施优先级与可迁移边界

| 阶段 | 状态 | 要交付的能力 | 验收重点 | 不做什么 |
|---|---|---|---|---|
| P0：可信知识 | 已实现首版 | Evidence/Claim、基础类型/状态/置信、候选提升、本地 Recall、回答校验、污染隔离 | AMBER/staging 与 ORBIT 干扰必定答对或安全拒答 | 不接向量库；只承诺已支持 canonical field |
| P1：认知场 | 已实现接线 | Aurelius 生命周期接 Ground/Pin、`DESKTOP.md` 规则面、CTML 桌面动作、每帧 context 重绘 | 文件变化、Pin 更新和工作约定进入当前帧但不污染长期事实 | 不把 Desktop 当作 User.md 或历史摘要 |
| P2：记忆治理 | 未实现 | audience/sensitivity/retention 策略、archive/tombstone、对象隔离与最小披露 | 同一 Ghost 对不同 principal 不越权披露 | 不做静默遗忘或无策略的物理删除 |
| P3：可选规模化 | 未实现 | 经评审的 `MemoryRecallBackend` 契约与可选 adapter | 接口故障可退化、本地证据仍可审计 | 不让外部后端成为事实真相源 |
| P4：可复用能力包 | 未实现 | 将通用 contracts/测试夹具提取给 Lynn 等 Ghost 使用 | owner、subject、世界执行权威仍由应用决定 | 不把 Aurelius 的人格或用户数据迁移给别的 Ghost |

抽象提取的方向应遵守 MOSS 的层次：`core.memento` 继续保持 payload-opaque 的轨迹地基，
`core.desktop` 继续保持 owner-scoped 的认知场；Evidence/Claim/Recall/Verifier 应成为二者之上
的可复用 capability。Aurelius 是第一个完整参考实现，不应把它的具体提示词、记忆 root 或用户
数据硬编码成 Lynn、Moshi 或其他 Ghost 的公共契约。

### 3.3 写入与提升规则

| 输入来源 | 是否写入 Moment | 是否可生成 ClaimCandidate | 是否可直接成为 active Claim |
|---|---:|---:|---:|
| 用户直接陈述 | 是 | 是 | 仅在明确、可解析且策略允许时 |
| 可信执行组件结果 | 是 | 是 | 可以，仍需来源与 scope |
| Ghost logos | 是 | 可以标注为未验证候选 | 否 |
| Reflection | 是（作为 note 的来源） | 可以 | 否 |
| Shell/CTML 诊断输出 | 是（审计需要时） | 默认否 | 否 |

“提升”不是模型自由改写历史：它是可审计状态转换。用户更正产生新的 Claim，并以
`supersedes` 标记旧 Claim；不删除原 Moment。反思只能建议提升，不能执行提升。这样既保留
Aurelius 的省察能力，又阻断“答错一次 → 自动记成真相”的污染环。

当前实现中，用户与可信工具来源由 `MemoryConfig.knowledge_user_sources` /
`knowledge_trusted_tool_sources` 精确列举；未列举 percept source 默认不参与抽取。显式更正可
完成 `active → superseded`，未显式更正的同 key 异值进入 `uncertain/conflict`，不使用静默 last-wins。

### 3.4 上下文与能力分发原则

记忆读取应复用 MOSS 的能力分发原则：一千万个 Channel 不会全量进入上下文，同样也不能将
全部历史记忆送给模型。Memory capability 应提供目录/Recall/Show 等渐进披露能力：

1. 普通闲聊保留有限近期 Moment，以维持对话连续性；
2. 显式记忆问答或检测到事实型问题时，先运行 Recall；
3. 只把当前问题相关的 Claim、EvidenceRef 和必要的原文片段作为 evidence packet 注入；
4. `memory_show` 仍用于人工或模型按稳定 `MementoRef` 缺页展开，而不是替代 Recall；
5. `memory_log`、完整控制面结果与反思长文不得无限回流为普通对话 history。

当前 runtime 的具体行为是：普通对话继续使用历史窗口，模型 logos 仍流式送入 Shell 执行；
normal TUI 为避免 CTML 泄露，在 articulation 边界统一输出净化后的人类文本。只有能识别
canonical key 的问题进入 Recall。`unknown/conflict` 不调用主模型，直接安全回答；`ok` 才注入
packet，并为校验缓冲该轮模型输出。Ground/evidence 都是临时 request part，不修改当前 Moment。

记忆管理面不应成为普通对话的自激工具环。完成 Moment 已自动写 staging，事实题也已有内部
Recall，因此 `memory_commit`、`memory_recall`、`memory_claims` 等 `memory_*` 命令默认
`visible=False`、`always_observe=False`：它们仍可由人类在 Shell/CTML 调试面显式执行，但不进入
模型日常能力提示，也不会因返回值自动制造下一帧。`desktop_*` 仍是 Aurelius 正常工作的可见
能力，不受此限制。

现有 `ResourceStorage.recall(query) -> Recollection` 是可优先评估的契约方向；若它不适合
owner/branch/MementoRef 语义，也应在同等抽象层增加专用 Recall 接口，而不是将检索规则隐入
prompt 或 Agent 私有状态。

### 3.5 运行时启动边界

`AureliusMeta.factory()` 不是命令启动的第一步。正确链路是：CLI 封存 `Environment` →
`Host` 发现 Ghost manifest 并构造 `GhostRuntime` → Matrix 在构造期准备同一个未 bootstrap 的
IoC Container → GhostRuntime 注册 Ghost providers 并校验 contracts → Matrix enter/bootstrap →
`AureliusMeta.factory()` 创建 Agent、Memory 与反思器 → Ghost enter/启动追赶 → Mindflow 三循环。

这个边界有两条通用契约：

- Matrix 启动前读取 `MossRuntime.logger` 必须有无依赖的 `Environment.logger` 回退；不能等待
  ConfigStore、Matrix provider 或 Aurelius 才能记录启动日志。
- Ghost providers 必须在 Matrix bootstrap 前注册到同一个 Container；因此 Container 可在 Matrix
  构造期取得，但只能在 Matrix enter 时 bootstrap。

它们与 Aurelius 无关，`echo` 与任何未来 Ghost 都共享。TUI 在 Runtime enter 失败时必须同步
打印 traceback；`closed / good bye` 只是正常收尾文案，不能作为启动成功的信号。

## 4. Moment、Stage 与 Commit 的写入机制

### 4.1 成功帧的写入规则

`GhostRuntime` 在模型输出完整写回 `Moment.logos` 后调用 `Aurelius.on_articulate_exit()`。
当 `error is None` 时，Aurelius 调用 `AureliusMemory.remember(moment)`：

1. `update_moment()` 把完整 Moment 写入 owner 的 pool，并把 id 放入 staging；
2. 若 `len(staging) < auto_commit_every`，流程结束；staging 已经落盘，进程退出不会丢；
3. 若达到阈值，`branch.commit(..., kind="mechanical", by="aurelius")` 冻结全部 staging；
4. 初始正文是可确定复现的 extractive index：只摘录配置认可的用户 source 及对应可见回复，
   跳过纯内部控制子帧；单条输入/回复分别最多 140 字符，整条机械 Note 最多 600 字符；
5. commit 成功后才安排反思任务。

因此默认值 `auto_commit_every: 4` 的意思是“每四个完成帧形成一个认知锚点”，**不是**
“每四轮才保存一次”。第 1 到第 3 帧已经在 staging 中持久化。`0` 禁用自动 commit，
但仍保留 staging，直到使用 `memory_commit` 进行显式 semantic commit。

Moment 是认知帧，不等于人类问题数。合法工具结果可能触发后续观察并形成额外 Moment；但普通
记忆陈述不应触发 `commit → claims → commit → claims` 的自审计链。该链会同时放大终端输出、
Moment 数、mechanical/semantic commit 计数和 Note 内容，属于行为/展示缺陷，不应通过删除内部
Moment 或修改 Memento 核心语义掩盖。

当 articulate 出错时，Aurelius 标记 `memory_write=skipped_on_error`，不写入未完成的
Moment。正常沉默帧仍是完整轨迹事件，可以保存。

这里的“成功帧”仅是**情景轨迹写入**条件，不是事实正确性判定。生产级 Claim projection
不得把 `error is None` 解释为“本轮 logos 已被证实”；回答正确性必须由第 3.2 节的证据和
校验链另行决定。

### 4.2 Commit 的不可变成员

一个 Memento commit 包含冻结的 Moment id 列表；冻结后不能修改其成员原文。commit 的
`kind` 当前有：

- `mechanical`：由阈值触发，初始 note 为原文索引；
- `semantic`：由 `memory_commit(summary=...)` 显式触发，summary 不能为空。

一个 commit 只能从 staging 产生。不能从尚未冻结的 staging 创建 fork，也不能通过
反思修改某个 Moment 的输入或 logos。

## 5. Commit 如何追加 Note：完整流程

### 5.1 Memento 的 note 版本模型

`branch.commit(text, kind=..., by=...)` 在同一原子写入中记录两件事：

1. commit 成员行：固定 seq 和被冻结的 Moment id；
2. 初始 `CommitNote`：正文、`Kind` trailer 和写入者 `by`。

之后调用 `branch.reinterpret(commit_id, body, by=...)` **不会修改**成员行或覆盖旧 note；
它在该 commit 的记录流末尾追加一个新的 `CommitNote` 版本。`CommitView.summary()` 返回
最新 note 的正文，`branch.notes(commit_id)` 可以审计所有旧版本。换言之，note 是可再巩固
的解释层，Moment 是不可变的证据层。

```text
commit c1
  ├─ note #0  [extractive mechanical index] + Kind: mechanical
  ├─ note #1  用户偏好短回答。 + Kind: mechanical + Reflection: llm
  └─ note #2  人工更正后的释义 + Kind: mechanical
```

### 5.2 自动反思追加 note

自动路径由 `AureliusReflector` 执行：

1. `Aurelius` 收到新 mechanical commit，或启动时发现待处理 commit；
2. `reflection_candidates()` 选择本 owner 尚无 `by=memento-reflection` note 的 mechanical
   commit；为了修复旧数据，也选择“初始 note 正文为空”的 legacy commit；
3. `commit_transcript()` 只从冻结成员读取可见 `percepts` 与 `logos`，并截断为
   `reflection_max_source_chars`；
4. 背景 Agent 使用 `reflection_model_tag`（默认 `small_fast_model`）生成不超过
   `reflection_max_summary_chars` 的简短中文结论；
5. `apply_reflection()` 保留原 `Kind`，追加 `Reflection: llm` trailer，并以
   `by=memento-reflection` 调用 `reinterpret()`；
6. 后续上下文读取新的 note，原 Moment 和原始机械索引仍可审计。

反思任务在 `asyncio` 后台运行：自动 commit 不等待它；同一 commit 通过 inflight set
去重；失败只记入 `inspect_state()['reflection']['errors']`，下次启动仍可重试。退出时正在
执行的任务会取消，未写出 reflection note 的 commit 留待下次启动追赶。

生产级约束补充：反思 note 的权威级别必须低于 Evidence/Claim。它可以生成“建议确认用户
偏好简洁”“发现城市更正关系”等候选，但不可以直接写 active Claim、覆盖 `superseded` 状态，
也不可以把 Ghost 过去的错误回答重新概括成事实。

### 5.3 人工/CTML 追加 note

`memory_reinterpret(commit="<seq 或唯一 id 前缀>", summary="...")` 是显式人工运维动作：

1. `find_commit()` 解析稳定 seq 或无歧义 commit id 前缀；含糊、不存在的 token 失败；
2. `join_trailers()` 生成新正文并保留原 commit 的 `Kind`；
3. 用当前 owner 作为 `by` 追加新 note；
4. `memory_show` 可展开被冻结 Moment，`memory_log` 可查看最新释义。

人工重释义不是 reflection 成功标记；自动反思仍只以是否存在
`by=memento-reflection` 为准。这样“人的修正”和“模型的反思”在审计上可区分。

## 6. 配置：路径、注释、优先级与生效时机

### 6.1 精确配置路径

当前仓库 workspace 的配置文件是：

```text
/Users/lipeng/TraeProject/MOSShell/.moss/configs/memory.yml
```

对其他 MOSS workspace，等价路径是：

```text
<workspace root>/configs/memory.yml
```

它是 Matrix 级配置，不在 `.moss/ghosts/aurelius/` 下；多个 Ghost 可以读到它，但目前
只有 Aurelius 消费 `MemoryConfig`。模板已随仓库提供，并且由
`.moss/src/MOSS/manifests/configs/__init__.py` 注册。配置文件修改后需**重启 Aurelius**；
运行中的实例会持有初始化时读取到的策略，不支持热更新。

`AureliusMeta(...)` 的显式参数优先于 YAML，主要用于宿主嵌入和测试：
`memory_detail_n`、`memory_summary_m`、`auto_commit_every`、`reflection_enabled`、
`knowledge_enabled`、`desktop_enabled`、`desktop_root`。

### 6.2 配置项与选择建议

| 字段 | 默认 | 含义与建议 |
|---|---:|---|
| `detail_n` | `12` | 最近完整 Moment 的数量；大些更精确但增加 prompt。必须至少 1。 |
| `summary_m` | `-1` | 早期 commit note 数；`-1` 表示全部。长会话宜设置正数以控制 prompt。 |
| `auto_commit_every` | `4` | staged Moment 数达到此值时 mechanical commit；`0` 仅禁用自动冻结。 |
| `reflection_enabled` | `true` | 是否调度后台反思；模型/凭据尚未可用时可先设 `false`。 |
| `reflection_model_tag` | `small_fast_model` | 在 `LLMConfig` 中解析的模型 tag；应选择低成本、低延迟模型。 |
| `reflection_max_summary_chars` | `360` | 每个反思 note 的最大字符数；它不是 token 限额。 |
| `reflection_max_source_chars` | `12000` | 单次送入反思模型的冻结原文上限。 |
| `reflection_startup_limit` | `16` | 单次启动最多追赶的待反思 commit；`0` 暂停启动追赶。 |
| `knowledge_enabled` | `true` | 启用可重建 Claim、按问题 Recall 与事实回答校验；关闭不影响 Memento 轨迹。 |
| `knowledge_user_sources` | `[input_signal_nucleus, input, user]` | `input_signal_nucleus` 是真实 Shell 的默认用户输入 source；`input`/`user` 供显式嵌入；不要加入 logos/command。 |
| `knowledge_trusted_tool_sources` | `[trusted_tool]` | 产品接入层已认证的工具 source；普通 CTML 输出不属于此类。 |
| `knowledge_recall_limit` | `8` | 单帧 evidence packet 的 Claim 上限。 |
| `knowledge_evidence_max_chars` | `6000` | packet 字符预算；超预算收紧 quote，仍超出则安全未知。 |
| `desktop_enabled` | `true` | 启动时自动打开项目 Ground，并按帧注入 instruction/context。 |

完整的带注释样例见 `.moss/configs/memory.yml`。若设 `reflection_enabled: true`，还必须让
`LLMConfig` 能解析 `small_fast_model`（或把 tag 改为已配置的 tag）；否则应先关闭反思，
主记忆写入仍可正常运行。

## 7. CTML 控制面与分支规则

Aurelius 通过 `Ghost.channel()` 注册虚拟 channel `ghost`。`memory_*` 是人类显式运维/审计面，
默认对模型隐藏且不自动触发观察；`desktop_*` 仍对模型可见。两类能力都不提供跨 owner 读写。

| CTML 命令 | 作用 | 关键约束 |
|---|---|---|
| `memory_inspect` | 查看 root、owner、branch、staging、commit 与反思 pending | 不泄露其他 owner |
| `memory_log` / `memory_staging` | 查看锚点或未冻结 Moment | staging 仍是持久化状态 |
| `memory_show` | 按 seq/唯一 id 展开冻结原文 | 只读 |
| `memory_commit` | 将 staging 形成 semantic commit | summary 非空，staging 非空 |
| `memory_reinterpret` | 追加当前 owner 的新 note | 不改 Moment/旧 note |
| `memory_reflect` | 请求后台追赶待反思项 | 不阻塞当前 CTML 回合 |
| `memory_branches` | 列出当前 owner 的 branch | 不显示跨 owner branch |
| `memory_fork` | 从已冻结 commit 创建并切换新 branch | 不能从 staging fork |
| `memory_switch` | 按唯一 branch id 前缀切换 | 含糊或不存在即失败 |
| `memory_recall` | 按问题审计本地 Claim 召回结果 | 只读；不把未支持自由文本猜成 key |
| `memory_claims` | 查看当前 branch 重建的 Claim | 默认紧凑摘要；`detail=true` 才返回有界证据/候选，`limit` 为 1..100 |
| `desktop_open` / `desktop_close` | 在项目 workspace 内开关 Ground | 不能越过 workspace root |
| `desktop_pin` / `desktop_unpin` | 管理 Ground 的地址 Pin | Pin 是地址与观察，不是文件快照 |
| `desktop_update` / `desktop_frame` | 承认外部变化并重绘当前场 | 只读世界内容，不提供写文件动作 |

`fork` 是有祖先关系的时间线分叉，不是复制后自动合并。当前没有真 branch merge；
`make_merge_message()` 只能把 commit 引用带入后续对话，不能解决冲突、来源和人格主权。

### 7.1 TUI 输出级别

`moss-run-ghost` 默认 `--output-mode normal`：仅显示人类可读 logos、显式 command-output 和错误；
CTML 标签、`MOMENT`、`SYSTEM`、模型专用 `COMMAND-RESULT`、操作 start/done 与 `Log:` 均隐藏。
普通回复在一个 articulation 完成后去除 CTML 再输出，避免控制标签泄露。

- `--output-mode verbose` 或运行中 `/verbose`：显示 Moment/System 等运行摘要，但仍隐藏完整
  command-result；
- `--output-mode trace` 或 `/trace`：显式显示完整内部结果，适用于排障；
- `/normal`：恢复默认精简模式。

这只是展示策略；内部 Session 事件和 Memento 轨迹仍按各自契约保存。不能以“界面隐藏”为由
丢弃错误、合法工具观察或可审计轨迹。

## 8. 存储、升级与观测

### 8.1 文件位置

默认实例的 Memento 位于：

```text
.moss/ghosts/aurelius/memento/
```

其中包含 owner-scoped Moment pool、branch、commit 和 note 记录。直接编辑这些记录会破坏
Memento 格式与审计性；排查使用 `memory_show`、`memory_log`，或只读搜索。

Claim projection **没有独立磁盘目录**。`memory_claims`/`memory_recall` 每次从当前 branch 的
commit 与 staging 重建，因而不存在“删 projection 文件”的运维动作；若以后为性能加入缓存，
缓存必须保持可删除、可重建语义。

### 8.2 从旧 `data` 原型升级

旧原型的默认路径 `.moss/ghosts/data/memento/` 和 owner `data` 不会被 Aurelius 默认读取。
不要直接把目录移动到新位置：branch 的 owner 归属仍是 `data`。安全选择有两种：

1. 保留旧目录为只读历史，Aurelius 从新 owner 开始；
2. 在宿主代码中显式构造
   `AureliusMeta(memory_root="../data/memento", memory_owner="data")`，以 Aurelius 运行时
   兼容读取旧 owner 的轨迹；同一 root/owner 仍只能单写。

跨 owner 的真正迁移需要导出、校验并按新 owner 重放 Moment；当前没有自动迁移器，不能通过
复制目录假装完成迁移。迁移前先备份整个旧 Memento root。

### 8.3 可观测字段

`memory_inspect` 返回 `staging_count`、`commit_count`、`head_commit_id`、窗口/commit 策略、
`reflection_pending` 以及 Claim/candidate 计数。`Aurelius.inspect_state()` 还报告反思任务、
knowledge projection 与 Desktop active Ground/Pin 状态；`inspect_context()` 记录本帧 Recall、
verifier 结果和 Ground context 字符数。待反思不等于记忆丢失：机械 commit 与原始 Moment 已落盘。

## 9. 安全、准确性与失败退化

### 9.1 生产级事实不变量

1. **来源不变量**：每一个进入事实型回答的 value 都必须能映射到本轮 evidence packet 中的
   active Claim 及至少一个 EvidenceRef。
2. **字段不变量**：`test.run.code`、`test.run.environment`、设备属性等 canonical key 不可
   仅因自然语言相似而互相替换。
3. **时间不变量**：被标记为 `superseded` 的 Claim 不得作为 current 答案；存在多项 active
   候选时必须报告冲突。
4. **污染隔离不变量**：assistant logos、反思 note、控制面输出在未经来源验证前不得提升为
   active Claim。
5. **可重建不变量**：删除或重建 Claim projection 后，应能从 Memento 原始证据和显式提升/
   更正事件重新得到同一状态；投影不是唯一真相源。
6. **安全拒答不变量**：Recall 空、证据不足、校验失败或 scope 不匹配时，回答“没有找到”或
   “记录冲突”，不得依据常识或历史模型措辞补全。

上述 1/2/3/4/5/6 已在**首版支持的 canonical field 范围内**形成确定性测试。P2 的 audience/
sensitivity/retention enforcement 和任意自然语言事实的完整语义校验不在该承诺内；数据模型有字段
不等于策略已经执行。

### 9.2 防御层次

Prompt 只属于最后一层防御，可明确要求“仅使用 evidence packet，给出来源，不得推断”，但
不能替代下列结构性约束：

```text
可信来源过滤 → Claim 状态机 → 问题驱动 Recall → 有界 evidence packet
→ 生成 → 字段/值/状态/来源校验 → 失败时重试或安全拒答
```

`summary_m`、`detail_n` 和 prompt 文案只影响模型注意力与成本，不能作为事实完整性机制。

### 9.3 退化策略

- 反思 prompt 不接收隐藏 reasoning；输出只做可见证据上的解释。
- 反思模型、网络或凭据失败时，机械轨迹和现有 active Claim 仍可读取；不阻断对话。
- Recall 索引或 projection 损坏时，可从 Memento 重建；重建期间事实问答安全拒答，普通对话
  可退化为有限近期 history，但不得宣称长期事实。
- `summary_m=-1` 会让全部早期 note 进入 prompt；机械 Note 已有 600 字符单条上限，生产配置仍应
  有总 token 预算，且不能用“全量 history”替代 Recall。
- 对话同时写同一 owner、直接改 jsonl、把模型猜测当作执行进度，都会破坏保真性，必须避免。

## 10. 测试与验收入口

自动化与人工对话测试在 [MOSS-Ghost-Memory测试方案.md](MOSS-Ghost-Memory测试方案.md)。
最低回归命令：

```bash
.venv/bin/ruff check src/ghoshell_moss/ghosts/aurelius src/ghoshell_moss/host/tui_entries/ghost_ui.py
.venv/bin/pytest -q src/ghoshell_moss/ghosts/aurelius tests/ghoshell_moss/default/core/memento tests/ghoshell_moss/core/desktop tests/ghoshell_moss/host/test_ghost_ui_output.py
.venv/bin/python scripts/ghost/aurelius_memory_acceptance.py
.venv/bin/moss-run-ghost
```

验收应覆盖：成功帧写入、失败帧跳过、跨重启、窗口折叠、note 追加版本、反思失败与启动
追赶、配置生效、owner 隔离、fork 边界以及 CTML 的错误输入。P0/P1 自动化现已覆盖：
Evidence/Claim 重建、按问题 Recall、干扰字段隔离、错误 logos/reflection 防污染、
current/superseded/conflict、答案校验、Ground instruction/frame 注入、Pin changed/update、路径越界、
默认精简输出、CTML 隐藏、运维命令不触发 Re-Act、Note 全局限长与内部帧排除。
2026-07-18 本轮文档命令回归为 `198 passed`，acceptance 完成 write → commit → reopen → project →
recall → verify。P2/P3 与真实 LLM/Host 手工验收仍按测试方案单列；Moshi 的用户模型和世界执行
进度属于下一层产品集成，不能以 Aurelius 的反思 note 代替。
