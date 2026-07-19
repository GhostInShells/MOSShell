# Moshi 长期记忆与进度集成技术评审

> 状态：技术评审；不修改 `moss-in-reachy-mini`、不修改 moshi mode 或 App 代码。
>
> 依据：当前 MOSS Aurelius Ghost/Memento 实现、`Docs/moshi支持讨论.md`（仓库根 Docs/ 下），以及
> `moss-in-reachy-mini/.moss_ws/apps/ui/moshi` 的现状代码。
>
> **2026-07-19 修订**：对齐 Aurelius 最新实现。要点：(1) Aurelius 已删除正则
> Evidence/Claim/Recall/verifier 层，事实读取改为 grep 式 `memory_search`/`memory_show`
> + 记忆纪律 instruction + 旁路 curation（详见《Aurelius Memory 集成技术评审与实施方案》）——
> 本文的"用户模型"设计必须以此为参照，不得走回正则抽取；(2) Aurelius 新增输入侧上下文
> token 预算与溢出重试，§5 的上下文预算约束已有运行时兜底；(3) 失败/中断帧现以 `failed`
> thread 如实入轨迹，不再静默丢弃；(4) **owner 命名修正**：Memento owner 必须匹配
> `^[A-Za-z0-9._\-]{1,64}$`（FORMAT.md §1），旧稿的冒号分隔命名非法，本稿改为点号分隔。

## 1. 结论

moshi 要实现"会按不同人、不同理解进度讲课，能回到旧问题并自然续讲"，不能只接一个
对话摘要。它需要三条相互独立、但在 Moment 汇合的链路：

```text
用户交互 ───────────────┐
课程/CTML/TTS 执行事件 ──┼─→ Moment ─→ Memento ─→ commit 反思/curation ─→ 用户模型
MoshiProgress 权威状态 ─┘       │
                               └─→ 下一轮 Ghost 上下文
```

- **Memento** 保存"发生过什么"：用户问答、Ghost 回答、可展开的 commit 原文；模型可经
  `memory_search`（grep 式字面检索）+ `memory_show`（缺页展开）逐字查回，检索结果
  `always_observe` 回灌下一轮 Re-Act；
- **反思/curation 派生物** 表达"这对该用户意味着什么"：已理解、未理解、重复问题、偏好。
  反思以 `reinterpret()` 追加 CommitNote；curation 以小模型异步重写带出处的可读笔记并
  pin 进 Ground。二者都是可回溯的解释层，不是权威事实；
- **MoshiProgress/Trace** 保存"世界实际执行到哪里"：章节、段落、CTML 任务与 TTS 播放
  offset。它是权威状态，绝不能由 LLM 摘要覆盖。

当前 moshi/Lynn 仅有"同一 session 下最近 `max_turns=5` 的线性持久化对话"和进程内
`course/current_id`。它尚不满足上述目标；本评审的目标是定义可实施的迁移顺序，而不是
把现有能力说成已经具备。

## 2. 当前事实与差距

| 项目 | 当前 moshi/Lynn | Aurelius/Memento 可提供 | 集成后仍需新增 |
|---|---|---|---|
| 对话历史 | `LocalConversationStore` 保存完整回合，但模型只见最近 5 轮 | commit、折叠、原文展开、跨重启 | 从旧 store 的迁移策略 |
| 课程导航 | `next_chapter`/`jump_chapter`，状态仅进程内 | 可把导航经过保留为 Moment | 持久化权威课程进度 |
| 用户差异 | 无 memory namespace 路由 | owner/branch 隔离 | 稳定 person id、匿名 session 身份 |
| 重复讲解 | 无计数或理解模型 | 反思能提供证据和观察 | 可查询的用户模型投影 |
| 中断续讲 | 无精确恢复点 | 可保留已收到的世界反馈 | CTML/TTS execution trace |
| 分叉 | 无 | 从冻结 commit checkout | 产品化分叉时机与主权规则 |
| 合并 | 无 | commit 引用，不是真 merge | 冲突、来源、授权与合并语义 |

另一个硬前提是依赖版本：`moss-in-reachy-mini` 当前固定本地
`ghoshell_moss-0.1.0b0` wheel，其中没有此分支新增的 Aurelius Ghost 与 `core.memento`。
集成前必须先用包含本分支提交的 wheel/发行物替换它；不能只改 moshi 的 Python 文件。

## 3. 推荐的身份与记忆命名空间

不要以"所有观众共用一个 Ghost owner"起步。推荐把身份分为两层（owner 必须匹配
`^[A-Za-z0-9._\-]{1,64}$`，用点号分隔层级）：

| 层 | 键 | 用途 | 生命周期 |
|---|---|---|---|
| 课程主体 | `moshi.{course_id}` | 课程公共知识、导演人格、可共享课程进度模板 | 长期 |
| 受众会话 | `moshi.{course_id}.p-{person_id}` | 一个人的问答、理解观察、讲解偏好 | 长期 |
| 匿名退化 | `moshi.{course_id}.s-{session_id}` | 未识别用户的临时隔离记忆 | 会话结束后可清理/归档 |

`course_id`/`person_id`/`session_id` 自身也必须先归一为合法字符集（字母数字、点、下划线、
连字符），总长不超 64；超长时截断加短 hash。`person_id` 必须来自身份服务的稳定内部 ID，
而不是昵称、说话人序号或模型猜测的人名。身份未知时只能进入匿名 owner；身份后来确认时，
应显式创建可审计的"迁移/引用"操作，不能静默把两个陌生人的记忆合并。

同一 owner 同时只能一个写者。Aurelius 已在**进程内**用 RLock 保证事件循环与 CTML 工作
线程不交错写；**跨进程**仍靠部署纪律——多设备、多个 app 或重连场景先按 owner 加单写
协调，不要靠"最后写入者获胜"处理 Memento staging。

## 4. 数据模型：三层不可混淆

### 4.1 Memento：情景记忆

每个完成 Moment 可包含：用户输入、语音转写来源、当前章节、Ghost logos、上一帧的
CTML/世界反馈。达到策略阈值后冻结为 commit；反思只向该 commit 追加释义。

适用问题：

- "用户在第一章问过什么？"
- "这件事是否讲过，原话是什么？"
- "当时 Ghost 是怎样回答的？"

以上问题的读取路径是 Aurelius 的 grep 式检索：`memory_search` 命中稳定地址（commit_id/
moment_id）→ `memory_show` 展开冻结原文核对 → 按记忆纪律作答（查不到就说没找到，不猜）。
检索命令 `always_observe`，结果回灌下一轮 Re-Act。**不要**为 moshi 另建正则/canonical-key
抽取层——Aurelius 已试过并删除了该方案（脆性模板 + 系统越权替模型判断，见主评审文档）。

### 4.2 UserModel：可撤销的派生认知

用户模型不是原始事实仓。它应该由反思/curation 从有证据的 Moment 派生，并保留来源
commit id 与置信度。例如：

```yaml
person_id: chen_a3f8
course_id: peach_blossom
understood:
  - concept: 渔人进入桃花源的因果
    confidence: 0.72
    evidence_commits: [cmt_xxx]
uncertain:
  - concept: 太守寻访为何失败
repeated_questions:
  - topic: 第一章的地理路线
    count: 3
style_preference:
  pace: slow
  explanation: concrete_examples
updated_at: "..."
```

该对象可存为独立投影，也可在 v1 直接复用 Aurelius 的 curation 旁路：小模型异步读冻结
commit，把"该用户的理解状态"写成带出处横幅的可读笔记（每条结论标注 evidence commit id），
pin 进该 owner 的 Ground。无论哪种载体，必须允许根据 commit 证据重新计算；不能把模型的
推断误当作用户的永久属性。

### 4.3 MoshiProgress/Trace：权威执行事实

长程讲解的恢复依据不是“模型刚才计划讲到哪”，而是执行系统确认的状态。例如：

```yaml
course_id: peach_blossom
chapter_id: chapter_04
segment_id: paragraph_04_03
state: speaking                 # idle | speaking | paused | completed | failed
ctml_task_id: task_xxx
tts:
  utterance_id: utt_xxx
  confirmed_offset_chars: 183
  state: playing                # queued | playing | completed | interrupted
updated_at: "..."
```

最低要求是为每个状态变迁记录：`event_id`、时间、来源组件、前后状态、关联 Moment/CTML
task id。只有 TTS/设备确认后的 offset 才能作为“已播到这里”；模型生成文本、CTML 编译
通过、任务提交成功都不足以替代它。

## 5. 上下文组装规则

每轮提供给 moshi Ghost 的上下文应有稳定优先级：

1. 系统与课程安全/表演纪律；
2. 当前 `MoshiProgress` 的权威快照；
3. 当前用户的简短 UserModel 投影，并附来源与置信度；
4. 当前课程概览与当前章节/相关段落；
5. Memento 的早期 commit 摘要与近期完整 Moment；
6. 本轮用户输入与实时感知。

不要把整门课程的全文、全部历史原文、所有用户模型同时常驻。讨论中"剧本展平"的正确
含义是**可在同一检索空间定位任意章节**，而不是无预算地把所有章节塞入 prompt。当前
moshi 已有章节索引和 `jump_chapter`，可以先把"按需取相关章节/段落"做成明确动作。

Aurelius 现已自带输入侧 token 预算兜底（`context_budget_enabled`）：按模型契约
`context_window - max_output_tokens - margin` 主动收缩历史渲染窗口（先折明细帧、再压
早期摘要，原文不销毁），provider 仍溢出且未 yield 时折半重试。moshi 集成后课程正文、
Progress 快照与 UserModel 属于"当前输入/ground 注入"一侧，计入
`context_fixed_overhead_tokens`；若课程注入显著大于默认估算，需相应调大该配置，
不要指望预算层替产品侧管理课程正文的体积。

## 6. 推荐实施顺序

### Phase 0：依赖与回归基线

1. 将 `moss-in-reachy-mini` 升级到包含 Aurelius/Memento 的 MOSS 发行物；
2. 保留 Lynn/`LocalConversationStore` 只读，导出一份可回滚的历史备份；
3. 用同一课程和同一输入录制当前输出、章节跳转和 TTS 行为，作为回归基线。

验收：不改变 moshi 行为时，Aurelius 能在独立 Ghost 中完成跨重启记忆、机械 commit 与反思。

### Phase 1：先接 Aurelius，不接个性化

1. 将 moshi 使用的 Ghost 原型从 Lynn/Atom 迁移到 Aurelius，或让 Lynn 复用 AureliusMemory；
   二者选其一，不能保留两份同时写的历史；
2. 用 `memory_owner=moshi.{course_id}.s-{session_id}` 先做匿名单会话（注意 owner 字符集限制）；
3. 用 `MemoryConfig` 设置较小的 `auto_commit_every`（建议 2–4）和有限的
   `reflection_startup_limit`；`context_budget_enabled` 保持开启；语音场景若课程注入大，
   调大 `context_fixed_overhead_tokens`；
4. 为运营/调试保留 Aurelius 的 `memory_inspect`、`memory_log`、`memory_show`；模型侧的
   `memory_search`/`memory_show`/`memory_log` 可见且 `always_observe`，是模型自证回忆的主路。

验收：重启后能回答此前问题；旧问题可回到对应章节；反思失败不影响讲课。

### Phase 2：引入 MoshiProgress/Trace

1. 把 `load_course`、`next_chapter`、`jump_chapter` 的状态迁移为持久化 Progress 事件；
2. 为 CTML 执行结果和 TTS 播放回调写入 Trace；
3. 中断时以 Trace 恢复：先确认最后已播放/已执行的位置，再决定是否继续、重说或询问用户；
4. 将 Progress 快照作为 Moment 的感知输入，而不是让反思猜测它。

验收：在章节跳转、CTML 失败、TTS 中断、进程重启四种情况下，恢复点都由程序事件复现。

### Phase 3：身份路由与用户模型

1. 接入稳定 person id；
2. 为每个 person 创建隔离 owner，匿名用户保持 session owner；
3. 将反思输出写入可查询的 UserModel 投影，并保留 evidence commit；
4. 只把有限、置信度足够的用户模型注入 prompt；低置信推断以问题形式验证，不直接当事实。

验收：两个观众的事实、偏好和重复问题互不串扰；同一观众在跨会话后获得一致但可更正的
讲解适配。

### Phase 4：受控分叉，不做自动合并

适用场景是“同一用户要沿旧锚点探索另一条讲解路线”。只允许从冻结 commit fork，并把
fork 的理由、覆盖指令和 owner 写入审计。第一版应使用 commit 引用把有价值结论显式带回
主线；不要实现自动 merge。

## 7. 反思与 curation 策略建议

- 使用 `small_fast_model`，并将反思/curation 留在后台；主讲模型的实时首 token 不应等待它们；
- 输入只含已冻结的可见用户输入、logos 和执行结果；不保存隐藏思维链；
- 反思输出长度固定并结构化，至少给出：摘要、理解/疑点、重复问题、开放线索、证据 commit；
- curation 笔记每条结论标注 evidence commit id，带出处横幅，pin 进对应 owner 的 Ground；
- 将"用户已掌握 X"视为概率观察。后续互动反驳时以新的 evidence 覆盖，而不是删除历史；
- 讲课被打断/模型出错的帧会以 `failed` thread 如实入轨迹（Aurelius 当前语义），不会被读作
  完成回合——恢复决策仍以 Progress/Trace 为准，不以 failed 帧的半截 logos 为准；
- 当反思/curation 不可用时，系统退化为机械摘录 + 原文检索展开，不能停止课程服务。

## 8. 测试矩阵

| 测试 | 目的 | 通过标准 |
|---|---|---|
| 跨重启问答 | 验证 Memento 主路 | 历史事实准确且不串用户 |
| 反复问第一章 | 验证重复问题与相关章节回取 | UserModel 计数增加，回答仍基于课程原文 |
| 成人/儿童两人 | 验证隔离与表达适配 | 两人 owner 不串，风格差异有证据来源 |
| 跳章再返回 | 验证 Progress | 返回的章节、段落和画面状态准确 |
| TTS 中断 | 验证 Trace | 只从确认 offset 恢复，不重复宣称已播内容 |
| 反思服务故障 | 验证退化 | 主讲不中断，pending 任务可在下次启动追赶 |
| fork 探索 | 验证时间线边界 | 子线看不到 fork 点后的父线内容 |

## 9. 不应做的事

- 不在 moshi App 内维护第二份无版本的 conversation list；
- 不把 LLM 摘要作为课程进度、TTS offset 或 CTML 成功的真相源；
- 不用昵称作为长期记忆 owner；
- 不允许模型无约束地跨 owner 读取或写入；
- 不在没有冲突模型之前实现“自动合并用户分支”；
- 不因引入记忆而把整门课程全文和全部历史塞进每轮上下文。

这条路线的关键不是把 moshi 变成“记得更多”的聊天机器人，而是让它能分清：自己记得
什么、对某位用户推断了什么，以及世界实际已经执行到什么位置。
