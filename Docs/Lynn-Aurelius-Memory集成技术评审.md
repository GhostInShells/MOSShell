# Lynn 对接 Aurelius/Memento 记忆技术评审

> 状态：技术评审；**不修改** `/Users/lipeng/TraeProject/moss-in-reachy-mini` 的代码、配置或运行数据。
>
> 依据：`moss-in-reachy-mini/.moss_ws/src/MOSS/ghosts/lynn.py`、
> `src/framework/apps/conversation/local_store.py`、
> `src/framework/apps/memory/storage_memory.py`、当前 Aurelius 实现，以及 Lynn 的身份服务。
>
> **2026-07-19 修订**：对齐 Aurelius 最新实现。要点：(1) 事实读取是 grep 式
> `memory_search`/`memory_show`（`always_observe` 回灌 Re-Act）+ 记忆纪律 instruction +
> 旁路 curation，早期的正则 Evidence/Claim 层已删除，Lynn 不得重建同类层；(2) Aurelius
> 新增输入侧 token 预算（`context_budget_*` 配置）与溢出折半重试，Lynn 的 `max_turns=5`
> 顾虑由此有了运行时替代；(3) **失败帧语义变化**：articulate 出错的帧现以 `failed` thread
> 如实入轨迹（不再丢弃）——Lynn 的"取消不保存"要求需按 §5.2 修订后的对齐方案处理；
> (4) **owner 命名修正**：owner 必须匹配 `^[A-Za-z0-9._\-]{1,64}$`（FORMAT.md §1），
> 旧稿冒号命名非法，本稿改为点号分隔；(5) 进程内并发写已由 AureliusMemory 的 RLock
> 保护，CTML 工作线程调度崩溃已修复。

## 1. 结论：Lynn 应接入哪一种记忆

Lynn 不应简单“再加一个摘要文件”，也不应让 `LocalConversationStore` 和 Memento 同时成为
可写的对话真相源。推荐目标是：

```text
Lynn 的人格/指令记忆（保留） ──────────────── SimpleMemory Markdown
Lynn 的完成对话与长期轨迹（替换） ──────────── AureliusMemory / Memento
身份与关系资料（保留，证据可关联） ────────── IdentityService / ProfileStore
设备、动作、TTS 的真实执行进度（另建，不由记忆猜） ─ Execution Trace / Progress
```

换言之，**Lynn 需要成为带 Lynn 人格和思考能力的 Aurelius 子类/适配器**：复用
Aurelius 的 Moment → staging → commit → note/reinterpret 机制；保留 Lynn 特有的
Anthropic thinking、flash 路由、流式取消保护与 `moss_execute_ctml` 工具策略。现有
`LocalConversationStore` 退出“在线上下文的主写路径”，只作为历史导入源和迁移期只读备份。

这是一次依赖升级和运行时接线工程，不是向 `soul.md` 或 `memory:refresh` 多写几段文字。

## 2. 当前 Lynn 的事实基线

### 2.1 当前对话路径

Lynn 位于：

```text
/Users/lipeng/TraeProject/moss-in-reachy-mini/.moss_ws/src/MOSS/ghosts/lynn.py
```

其当前链路是：

```text
Moment → LynnGhost.articulate()
       → LocalConversationStore.get_current()
       → 最近 max_turns 条 records → pydantic-ai history
       → 完整流结束后 save_model_request()
       → append_with_logos(moment, logos) + conversation.save()
```

关键事实：

| 项目 | 当前行为 | 影响 |
|---|---|---|
| 基类 | `LynnGhost(Atom)` | 没有 Memento 窗口、commit、note、fork 或反思能力 |
| 对话存储 | `LocalConversationStore` | 每个 `session_scope` 下有 current conversation 和 JSON 文件 |
| 上下文预算 | `.moss_ws/configs/conversation/conversation.yml` 的 `max_turns: 5` | 第 6 轮及以前仍在磁盘，但不进入模型上下文 |
| 写入时机 | `stream_text()` 正常结束后保存 | 中断流不会被保存为完成回合，这一保真语义必须保留 |
| 当前会话 | `.convo_index.yml` 的 `current_id` | 可 create/switch 平行 conversation，但不带 Memento 祖先/commit 语义 |
| 特有能力 | thinking、flash、interleaved tool、取消清理 | 迁移时绝不能退化或改写其时序 |

当前 `LocalConversationStore` 的数据在：

```text
.moss_ws/runtime/conversations/<session_scope>/<conversation_id>.convo.json
```

一个 record 包含 `user_input`、`moment_json` 与 `logos`。它足以作为历史导入的来源，但
没有 commit、不可变 note 版本、反思、长期折叠策略或 owner 主权。

### 2.2 现有 SimpleMemory 不是要替换的对象

当前主 channel 已挂载 `memory_chan`。其 `SimpleMemory` 按 `session_scope` 保存：

```text
.moss_ws/runtime/memory/<session_scope>/
├── personality.md
├── behavior_preference.md
├── mood_base.md
├── autobiographical_memory.md
├── summary_memory.md
└── consciousness_memory.md
```

这些 Markdown 会作为 channel instruction 注入 prompt，承载人格、行为偏好和人工维护的
长期背景。它们不是逐回合的可审计事件流，因此迁移中应：

- **保留** `SimpleMemory` 和 `<memory:refresh />` 的既有语义；
- 不把它们自动转换为 Memento Moment；
- 不让反思 worker 自动重写人格或 mood 文件；
- 可把其中内容作为系统/静态上下文保留，与 Memento 对话记忆并列。

### 2.3 已有身份能力是正确的 owner 来源，但还缺路由策略

项目已有 `IdentityService`、稳定 `person_id`、ProfileStore 与 pin 池。`person_id` 是可持久
目录名，优于昵称、摄像头 track id 或 `session_scope`。但当前 pin 池允许多个在场人，
没有“本轮私人对话唯一属于谁”的显式主权选择。

因此不能简单设 `memory_owner=session_scope`：它会让同一场地、同一直播或同一设备上的
不同人共享私密历史。也不能在检测到人脸后静默把匿名对话并入某人。

## 3. 目标架构与数据主权

### 3.1 推荐运行形态

```text
                         ┌──────────────────────┐
                         │ LynnMeta / Lynn Ghost │
                         │ persona + thinking    │
                         └──────────┬───────────┘
                                    │
                  ┌─────────────────┼──────────────────┐
                  │                 │                  │
                  ▼                 ▼                  ▼
           SimpleMemory       AureliusMemory      IdentityService
           静态人格/人工背景    对话 Moment/Commit    person_id/Profile
                  │                 │                  │
                  └───────────── prompt assembly ──────┘
                                    │
                                    ▼
                         Execution Trace / Progress
                       （独立权威事实，按需作为感知）
```

推荐的 Memento root：

```text
.moss_ws/ghosts/lynn/memento/
```

一个 root 可有多个 owner；owner 决定谁可以读写其 branch。Lynn 原型名保持 `lynn`，
不应把人物身份混进 Ghost 名或直接复用默认 `owner="lynn"`。

### 3.2 owner 命名与选择策略

owner 必须匹配 `^[A-Za-z0-9._\-]{1,64}$`（FORMAT.md §1）；用点号分层，各段 id 先归一为
合法字符集，超长截断加短 hash：

| 场景 | 推荐 owner | 规则 |
|---|---|---|
| 已明确的一对一对话 | `lynn.p-{person_id}` | 仅当 identity 选择器明确确认唯一人 |
| 未识别的临时对话 | `lynn.anon.{scope}-{conversation_id}` | 会话隔离，不自动归属任何人 |
| 多人场景的公共互动 | `lynn.group.{scope}-{conversation_id}` | 只保存可公开共享的群体互动 |
| Lynn 自身连续叙事/运维 | `lynn.self` | 与人的个人记忆绝对分离 |

建议新增一个**显式的 MemoryOwnerResolver 概念**（实现留待后续）：它在创建/切换 Lynn
记忆会话前读取 identity 选择结果并返回 immutable owner。其输入应是明确的
`person_id`、conversation id、session scope 与模式；输出应包含 `owner`、`kind`、
`reason` 和 `identity_evidence`。不允许 LLM 自己猜 owner。

身份从匿名变为已识别时，第一版规则是：创建一个新的 personal branch/owner，并以显式的
“adopt anonymous history”操作选择性引用或重放历史。**禁止自动 merge**；否则误识别人脸会
导致不可逆隐私泄漏。

### 3.3 四个真相源的边界

| 数据 | 写入者 | 读取者 | 不允许的替代 |
|---|---|---|---|
| Moment/Commit/Note | Lynn 的完成回合与 Aurelius 反思器 | Lynn 上下文、审计 CTML | 用 Markdown 摘要覆盖原文 |
| SimpleMemory Markdown | 人工或既有 `memory:refresh` | 主 channel instruction | 让反思自动改人格 |
| Identity Profile | IdentityService/人工识别流 | owner resolver、身份上下文 | 用对话推断直接改 `person_id` |
| 执行进度 | TTS/CTML/设备事件 | Lynn 下一轮感知 | 用模型“我已经做完”替代确认事件 |

## 4. 依赖与兼容性门槛

当前 `moss-in-reachy-mini` 固定依赖：

```text
dependencies/ghoshell_moss-0.1.0b0-py3-none-any.whl
```

该 wheel 没有当前分支的 `ghoshell_moss.ghosts.aurelius` 与现代 `core.memento` 实现；
Lynn 当前还从旧 API `ghoshell_moss.core.blueprint.memento` 导入 `Moment`。因此，**先升级
依赖是硬前置，不能先编辑 Lynn 文件赌 import 能工作。**

上线前的兼容性 gate：

1. 构建并安装包含 Aurelius、Memento、MemoryConfig 与 Ghost channel 注册的 MOSS 发行物；
2. 在 reachy 项目独立虚拟环境中执行 import smoke：Aurelius、Memento、Lynn 现有
   `Moment`/Mindflow/Session API 都能加载；
3. 比对 pydantic-ai 和 Anthropic SDK 版本，确认 Lynn 的
   `run_stream`、`event_stream_handler`、`AnthropicModelSettings`、thinking beta 字段仍兼容；
4. 用 TestModel 或本地假模型跑一轮 Lynn 的 producer/cancel 逻辑；
5. 只有 gate 通过后，才开始改写存储路径。

若当前 MOSS 版本与 Lynn 的旧 `Moment` JSON 格式不兼容，先写只读转换验证器，不要直接把
旧 JSON 写进新 Memento pool。

## 5. Lynn 运行时的对接设计

### 5.1 推荐继承/组合关系

推荐结果是 `LynnGhost` 基于 `Aurelius`，而不是继续继承 `Atom` 后在旁边塞一个 Memento：

```text
Aurelius
  ├─ Memento 生命周期、model_history、remember、commit、reflection、ghost channel
  └─ LynnGhost
       ├─ Lynn 的 soul/persona
       ├─ flash model 路由
       ├─ producer queue + cancellation cleanup
       ├─ thinking 可见化
       └─ interleaved moss_execute_ctml policy
```

理由：Aurelius 已定义“什么时候把 completed Moment 变成持久轨迹”的唯一主路径；让 Lynn
复制一次 `remember()`、窗口或反思逻辑，会立刻产生两个不一致的历史实现。

### 5.2 必须保留的 Lynn 时序

Lynn 当前的 producer queue 解决流式 SSE 被中断时的异常传播和 async generator 清理。这些
行为不是可有可无的 UI 优化。迁移后的定义应保持：

1. producer 成功完整消费 `stream_text()` 后才视为成功完成；
2. 外层取消时 cancel 并 await producer；
3. `CancelledError` 不转化为已完成对话；
4. **失败帧语义对齐（本次修订的关键差异）**：当前 Aurelius 的 `on_articulate_exit` 在
   `error is not None` 时不再丢弃帧，而是带 `failed` thread 如实写入轨迹（"看见 X、尝试、
   出错"是轨迹事件）。这与 Lynn 旧的"取消不保存"要求并不冲突，但语义要对齐：failed 帧
   **永远不会被渲染为已完成回合**进入模型历史，`memory_write` 记为 `staged_failed`。
   Lynn 侧需要决定的只是：语音场景的用户主动打断（abort）是否也值得 witness——建议保留
   Aurelius 默认（witness 为 failed），因为"用户打断了我"本身是社交信号；若产品确认某类
   打断完全无记忆价值，应在 Lynn 适配层显式过滤，而不是改 Aurelius 的默认；
5. thinking part 只用于实时展示，不写入 Moment logos、Memento 或反思 prompt；
6. `moss_execute_ctml` 的 fire-and-forget 结果仍在后续 Moment 作为感知回流，不能被
   当作本轮已确认事实。

现有 `save_model_request()` 是旧 ConversationStore 的写入口。接入后必须删除其作为主写入的
职责；否则每轮会有两份可变历史，恢复后模型可能读到不同版本。

### 5.3 上下文组装顺序

Lynn 每轮应按以下优先级组装上下文：

1. 系统安全、Lynn soul 与当前 mode instruction；
2. 现有 SimpleMemory 的人格/人工维护背景；
3. 当前明确 owner 的 Identity Profile（有来源、有限长度）；
4. 当前设备/任务的权威 Execution Trace 快照（如果有）；
5. Aurelius 的早期 CommitNote 与最近完整 Moment；
6. 当前感知与用户输入。

不能把所有 Profile、所有群体人的历史、所有 SimpleMemory 文件和所有 Memento 原文一次性
塞进 prompt。多主体场景必须先由 owner resolver 选择可读取的 namespace。

## 6. Commit、反思与 Lynn 的人格边界

### 6.1 推荐初始策略

在 `.moss_ws/configs/memory.yml` 注册并配置 `MemoryConfig`。建议 Lynn 的首个线上策略：

```yaml
detail_n: 8
summary_m: 12
auto_commit_every: 3
reflection_enabled: true
reflection_model_tag: small_fast_model
reflection_max_summary_chars: 300
reflection_max_source_chars: 9000
reflection_startup_limit: 8
# 输入侧上下文预算（Aurelius 2026-07-19 起自带；语音实时场景建议全开）
context_budget_enabled: true
context_token_margin: 4096
context_min_detail_n: 2
context_fixed_overhead_tokens: 2048   # Lynn 的 SimpleMemory/Profile 注入大时相应调大
# 旁路 curation（可选：把"对该用户的稳定观察"沉淀为带出处笔记并 pin 进 Ground）
curation_enabled: true
curation_model_tag: small_fast_model
```

同时在 `llms.yml` 为 Lynn 实际使用的模型填准 `context_window` 与 `max_output_tokens`——
预算直接从模型契约推导，这两个值不准，主动收缩的触发时机就不准。

这是建议起点而非硬编码。Lynn 的实时互动比纯文本问答更频繁，`3` 可较快形成锚点；但应先
通过真实 token、延迟和对话质量数据校准。反思必须使用低成本模型并且永不阻塞 Lynn 的语音
首 token。

### 6.2 note 追加的 Lynn 语义

完成 Moment 先进入 staging，达到阈值后形成 mechanical commit。初始 note 是可验证的
输入/logos 摘录；反思器只读冻结内容，再以 `reinterpret()` 追加新 note：

```text
冻结 Moment ── commit c17 ── note #0: extractive index
                                  └─ note #1: 反思结论（by=memento-reflection）
```

反思可提取：用户偏好的表达节奏、明确更正、未结束线索、重复问题、情绪支持边界。但不得把
“Lynn 觉得对方可能低落”写成 Identity Profile 的权威事实，也不得自动修改 `soul.md`、
`personality.md` 或 `mood_base.md`。

当反思失败时，mechanical index 与原始 Moment 已存在；启动时可追赶。人工如需更正，使用
`memory_reinterpret` 追加新版本 note，保留模型反思和旧 note 的审计链。

### 6.3 控制面暴露策略

Lynn 可继承 Aurelius 的 `ghost` CTML memory channel，但分层授权：

Aurelius 当前的可见性/观察语义是：`memory_search`/`memory_show`/`memory_log` 对模型
可见且 `always_observe=True`（"读以作答"必须回灌下一轮 Re-Act，否则模型检索后静默）；
`memory_commit`/`memory_reinterpret`/`memory_reflect`/`memory_curate`/`memory_fork`/
`memory_switch`/`memory_inspect` 对模型隐藏（`visible=False`），仅供人工 Shell/CTML 运维。
Lynn 继承此默认即可，无需另行分层：

| 命令类别 | 开发/运维 | Lynn 模型默认权限 | 说明 |
|---|---:|---:|---|
| search/show/log | 允许 | 可见，结果自动回灌 Re-Act | 只读，受 owner 过滤；是模型自证回忆的主路 |
| inspect/staging | 允许 | 隐藏 | 人工诊断面 |
| semantic commit | 允许 | 隐藏 | 防止模型把每句话都锚定 |
| reinterpret | 允许 | 隐藏 | 人工更正优先，避免自我改写叙事 |
| reflect/curate | 允许 | 隐藏 | 自动旁路/启动追赶优先 |
| fork/switch | 允许 | 隐藏 | 分叉会改变后续上下文，需要明确产品动作 |

`moss_execute_ctml` 在 thinking 内仍应拒绝会返回 Observation 的命令。Memento 的 inspect/
show 等只读命令是否能进入 thinking，必须根据其真实返回/observe 语义测试后决定；不要仅因
“它是记忆命令”就绕过 Lynn 已有的 dry-run gate。

## 7. 历史 Conversation 的迁移方案

### 7.1 原则

- 迁移是一次**可重复、可审计、可回滚**的导入，不是移动目录；
- 导入前停止 Lynn，备份 `.moss_ws/runtime/conversations/`；
- 原始 `.convo.json` 永久保留为 source of record，至少保留到验收完成；
- 一个 source conversation 只导入到一个明确 target owner/branch，不能按昵称猜归属；
- 历史 record 只作为完成回合导入；损坏、无法解析或身份不明的文件进入隔离报告，不静默丢弃。

### 7.2 导入器应有的输入/输出契约

后续实现一个离线 `lynn-conversation-import` 工具，而不是让启动过程自动导入。它应支持：

| 能力 | 要求 |
|---|---|
| dry-run | 只解析、校验和生成映射计划，不写 Memento |
| owner 映射 | 必须显式提供 conversation → owner；未映射项拒绝或写匿名隔离空间 |
| 格式校验 | 校验 `moment_json`、`logos`、source record 序号与 checksum |
| 幂等 | 记录 source file checksum + record index；重复运行不产生重复 Moment |
| 批次锚点 | 每个导入批次创建带 source 信息的 semantic commit，便于审计/回滚 |
| 报告 | 输出成功、跳过、损坏、未映射、目标 commit id 的 JSON/Markdown 报告 |
| 回滚 | 回滚通过切换/删除新建 owner root 的受控运维操作，不改原 conversation |

历史导入后的初始 note 应明确标记来源，例如 `Source: lynn-conversation-import` 与 source
conversation id。不要对导入的全部历史立即并发反思；先限速、分批并检查质量。

### 7.3 灰度切换

1. **备份与只读核对**：导出 conversation count、turn count、最后 logos、checksum；
2. **影子转换**：对副本导入 Memento，只比较历史渲染，不让 Lynn 读取它；
3. **影子读取**：同一新输入分别构造旧 5-turn history 与 Memento history，人工审查差异；
4. **单 scope / 匿名 owner 灰度**：只让一个测试 scope 使用 Memento 主写；
5. **身份 owner 灰度**：仅对明确单人身份启用 personal owner；
6. **正式切换**：停止 ConversationStore 写入，保留只读浏览/导出；
7. **回滚窗口**：若发现人格、延迟或隔离问题，切回旧 store，不把新 Memento 反写成旧 JSON。

## 8. 实施阶段、交付物与验收门

| 阶段 | 目标 | 主要交付物 | 退出条件 |
|---|---|---|---|
| 0. 基线冻结 | 知道当前行为 | 存储备份、版本清单、回归对话集 | 可复现 Lynn 当前 5-turn 行为 |
| 1. 依赖升级 | 让 Aurelius 可被 reachy 项目使用 | 新 wheel、compatibility report | import + TestModel + thinking smoke 通过 |
| 2. 运行时适配 | 只有一个对话主写路径 | Lynn-on-Aurelius 适配设计/实现 | 成功写入、取消不写入、thinking 不入记忆 |
| 3. 身份路由 | 个人/匿名/群体隔离 | OwnerResolver、显式 adopt policy | 误识别不自动合并，owner 不串读 |
| 4. 历史迁移 | 安全导入旧会话 | 离线导入器、mapping/report | 幂等、可审计、原文件未改 |
| 5. 反思灰度 | 得到长期语义而不影响实时 | memory.yml、反思监控 | 首 token 无回归、pending 可追赶 |
| 6. 产品化 | 运营可控 | CTML 权限、指标、runbook | 压测、故障演练、人工验收通过 |

阶段 1–3 未完成前，不应把任何生产 Lynn 会话切到 Memento 主写。

## 9. 测试矩阵

### 9.1 必须自动化的契约

| 测试 | 断言 |
|---|---|
| history 等价 | 指定旧 conversation 的最后 N 回合与 Memento 近期窗口渲染符合预期 |
| 正常完成 | 一个完整 Lynn 输出恰好形成一个 Moment；达到阈值恰好形成一个 commit |
| 取消/断流 | producer 被取消、HTTP 流异常时**不产生完成回合**；若按 Aurelius 默认 witness，帧必须带 `failed` thread 且不进入模型历史渲染 |
| thinking 隔离 | ThinkingPart 不出现在 logos、Moment、note 或 reflection 输入 |
| SimpleMemory 共存 | personality/behavior instruction 仍进入 prompt，未被 Memento 覆盖 |
| 反思 | 只追加 note；原始 Moment、初始 note 可审计；失败后可追赶 |
| owner 隔离 | 两个 person_id 和匿名 owner 互相不能召回内容 |
| 多人保护 | 多个 pin 时 resolver 不自动选择个人 owner |
| import 幂等 | 同一 source record 导入两次，目标 Moment 数不增加 |
| config | `memory.yml` 的阈值、关闭反思、启动追赶上限重启后生效 |

### 9.2 人工场景

1. **长时陪伴**：超过旧 5 轮后，Lynn 能准确回忆早先明确事实，并能给出可审计来源；
2. **情绪边界**：反思可总结“需要温和回应”，但不会把瞬时情绪永久写成人格标签；
3. **打断恢复**：说话/动作中断后，Lynn 不宣称已经说完或已经执行；
4. **两人轮流**：A 说过的私密内容不被 B 的 owner 读取；
5. **匿名转识别**：先匿名再认出 A，不自动把匿名历史合并到 A；人工 adopt 后才可见；
6. **群体演示**：多人 pin 下只使用 group owner，个人 profile 不泄露给群体上下文；
7. **反思服务故障**：关闭/破坏反思模型后，Lynn 正常对话；修复后 pending 被有限追赶；
8. **旧历史导入**：抽样核对 source `.convo.json` 的输入/logos 与 target Moment 完全一致。

### 9.3 观测指标与回滚阈值

上线需至少记录：写入成功率、失败帧误渲染率、commit/turn 比、反思 pending age、反思失败率、
反思 P50/P95、模型 history token 估算、owner 越权拒绝数、旧/新 history 回归差异数。

以下任一项应停止灰度并回滚到旧 ConversationStore 主读写：跨 owner 泄漏、取消/失败帧被
渲染为完成回合、
Lynn thinking/logos 混写、首 token 延迟显著回归、导入出现非幂等重复、身份不明确却被自动
归入个人 owner。

## 10. 风险与需要产品决策的事项

| 风险/决策 | 当前答案 | 后续必须确认 |
|---|---|---|
| 谁拥有记忆 | `person_id` 优先，匿名/群体独立 | UI/CTML 如何显式选择 owner |
| 多人正在场 | 不自动建个人记忆 | 哪些互动可以写 group owner |
| 旧记录归属 | 不猜测 | 哪些 conversation 可映射到哪个 person_id |
| Profile 与反思 | Profile 是身份资料，反思是证据化推断 | 哪些高置信观察允许人工确认后进入 profile |
| 分叉 | 仅运维/明确用户动作 | “探索另一种回应”是否作为产品能力开放 |
| 删除/遗忘 | 当前 Memento 保真优先 | 隐私合规下的删除、加密、保留期与导出策略 |
| 高并发 | 同 owner 单写 | 多设备/多个 Lynn 实例的租约或队列机制 |

## 11. 推荐的首个可交付切片

最小可上线切片应刻意保守：

1. 升级 reachy 项目的 MOSS 依赖到包含 Aurelius 的已验证发行物；
2. 将 Lynn 的 ConversationStore 主读写替换为 AureliusMemory，但保留 Lynn 全部流式、
   thinking、flash 和 tool 行为；
3. 先只使用 `lynn.anon.{scope}-{conversation_id}` owner（注意字符集限制），不接自动
   identity 路由；
4. 保留 SimpleMemory；不做旧 conversation 自动导入；
5. `auto_commit_every=3`、后台反思开启、无 fork/merge 的模型权限；
6. 使用专门测试 scope 连续运行、重启、断流和反思故障验收。

这样先验证“Lynn 能连续记住且不破坏实时人格/流式行为”，再逐步引入 `person_id` 的私人
记忆、历史导入与更复杂的长期用户模型。最危险的做法恰好相反：在首版同时迁移旧数据、
自动认人、自动合并和自动修改人格。
