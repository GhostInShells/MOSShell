---
title: Aurelius Memory — 集成技术评审与实施方案
description: Aurelius Ghost 的 Memento 轨迹记忆、grep 检索、反思/curation 旁路、上下文预算的当前实现评审与边界。含配置、CTML 控制面与失败退化策略
---

# MOSS Aurelius Ghost Memory 集成技术评审与实施方案

> 状态：已实现 Aurelius/Memento 的**轨迹主路**、grep 式记忆检索、事后反思、旁路 curation
> 与 P1 Ground 接线。当前实现能跨重启恢复认知帧、把较早轨迹折叠为可追溯的 commit note、
> 用小模型异步反思与策展、把 `DESKTOP.md`/Pin 作为本帧工作场，并通过一条记忆纪律 instruction
> 要求模型在无可见依据时先 `memory_search`/`memory_show` 自证。P2 的 principal/audience/
> retention 治理与 P3 外部召回后端尚未实现，因此不能把它表述为任意领域、任意对话对象下都已
> 达到生产级可靠性的长期记忆产品。
>
> **2026-07-19 修订**：本轮删除了早期的正则 Evidence/Claim/Recall/Verifier 知识层（`_knowledge.py`），
> 改为 grep 式 `memory_search` + 记忆纪律 + 旁路 curation 的组合。同时修复了三个缺陷并把两个
> 设计项落地：(1) CTML `memory_reflect`/`memory_curate` 在工作线程调度崩溃；(2) 进程内两个写者
> 域（事件循环的 `remember`/反思 与 `to_thread` 的 CTML 写命令）无序竞争；(3) 折叠摘要丢失
> `note_seq` 渲染打戳。设计项：折叠摘要不再伪造模型回合；失败帧不再丢弃，改为带 `failed`
> thread 如实写入。详见 §2.2、§5、§9。
>
> 关联文档：[测试方案](aurelius-memory-test-plan.md)、Moshi 集成技术评审（仓库 `Docs/Moshi-长期记忆与进度集成技术评审.md`，不随包发布）。

## 1. 结论与命名

第二个 Ghost 原型命名为 **Aurelius**，取自《沉思录》作者 Marcus Aurelius。Atom 仍是
无持久历史的最小基线；Aurelius 是带有可审计 Memento 轨迹、机械锚点、事后反思与旁路策展的
持久化原型。它不是一个泛称为 “Data” 的数据容器。

当前已交付的轨迹闭环如下：

```text
完成的 Moment（不可变轨迹证据）
  → Memento staging（立即持久化）
  → mechanical commit（默认每 4 个 staged Moment；初始 note 为保真原文索引）
  → 非阻塞 reflection（可失败）：reinterpret() 追加解释性 CommitNote
  → 旁路 curation（可失败）：小模型从冻结轨迹重写记忆笔记并 pin 进 Ground
  → 下一次模型上下文：早期 note 折叠 + 近期完整 Moment + 当前 Ground 表面
```

模型对过去的**主动查询**不走独立的事实数据库，而走 grep 式检索：`memory_search` 在本 owner
的全部冻结 commit 加当前 staging 上做大小写不敏感子串扫描，返回稳定地址（commit_id/moment_id）
供 `memory_show` 缺页展开。这与 momento-mori 的 “`cat`/`grep` 即记忆查询语言” 一脉相承。

原型的公开代码名、运行名与默认 owner 均为 `aurelius`：

| 位置 | 当前名称 |
|---|---|
| Python 包 | `ghoshell_moss.ghosts.aurelius` |
| 类型 | `AureliusMeta`、`Aurelius`、`AureliusMemory`、`AureliusReflector`、`AureliusCurator`、`AureliusDesktop` |
| 工作区注册 | `.moss/src/MOSS/ghosts/aurelius.py` |
| 启动命令 | `.venv/bin/moss-run-ghost aurelius` |
| Ghost home | `.moss/ghosts/aurelius/` |
| 默认 Memento root | `.moss/ghosts/aurelius/memento/` |
| 默认 Memento owner | `aurelius` |
| Feature workstream | `.ai_partners/features/workstreams/2026/07/aurelius-ghost/` |

## 2. 目标、非目标与真相边界

### 2.1 本次目标

1. 每个成功完成的认知帧都可重启恢复；失败或中断帧不伪装成完成记忆，但也不静默丢弃。
2. 近期上下文保留完整 Moment，较早内容以可追溯的 commit note 折叠，折叠摘要带渲染打戳。
3. commit 后能用小模型反思，而不会增加主对话首 token 延迟。
4. 旁路 curation 从冻结轨迹重写人类可读的记忆笔记文件，并作为 Pin 进入 Ground。
5. 模型在无可见依据时，由记忆纪律 instruction 要求先 `memory_search` 检索、`memory_show` 核对，
   查不到就明确说“没有找到记忆证据”，而不是编造。
6. 人与 Ghost 能通过受限 CTML 检查、检索、锚定、更正和分叉自己的记忆。
7. 策略在 workspace YAML 中可配置，而不是把轮次阈值写死在代码里。
8. Ground 的 instruction/frame 只进入当前帧，不因出现在 `DESKTOP.md` 或 Pin 中成为长期事实。
9. 默认 TUI 不暴露 CTML、Moment、System、command-result 或 `Log:`；详细观测必须显式开启。

### 2.2 定位、本轮修复与剩余边界

Aurelius 的目标是认知能力，不是长上下文；但它也不试图用一套正则事实证明器把“记住”变成
可判定的数据库查询。早期版本曾实现 Evidence/Claim/Recall/Verifier 层（`_knowledge.py`），
用 canonical key 抽取、问题驱动召回和输出后校验来对抗“模型答错字段”。本轮评审后该层被删除，
原因是：它只在极窄的手工模板内有效，把大量正则维护成本压到契约外层，且与“记忆是主体生产的
轨迹、不是管线蒸馏的数据库”这一先前哲学相悖。取而代之的是更诚实的三件套：

1. **grep 式检索**：`memory_search` 不做语义解析、不建 canonical key，只做原文子串扫描并返回
   稳定地址。它承认“我按字面找”，把语义判断留给读到证据的模型，而非伪装成事实裁决。
2. **记忆纪律 instruction**：系统提示明确要求模型在缺乏可见依据时先检索、再核对、查不到就
   如实说未找到。这是 prompt 层的行为约束，不是正确性保证——它替代了脆弱的输出后校验。
3. **旁路 curation**：小模型异步从冻结轨迹重写人类可读笔记并 pin 进 Ground，让“稳定事实”以
   可审计、带出处横幅的形式进入工作场，而不是塞进一个会漂移的第二真相文件。

本轮同时修复了三个确认缺陷：

- **CTML 调度崩溃**：CTML 命令在 `asyncio.to_thread` 工作线程执行，`memory_reflect`/`memory_curate`
  原先直接 `asyncio.create_task` 会因无运行中事件循环而抛 `RuntimeError`。现由 `Aurelius._spawn()`
  用 `__aenter__` 捕获的 loop 经 `call_soon_threadsafe` 编组回主循环。
- **进程内写竞争**：`remember`/反思跑在事件循环，`memory_commit`/`fork`/`switch`/`reinterpret`
  跑在工作线程——两个写者域共享 `staging.jsonl` 与 `self._branch` 指针。`AureliusMemory` 现持有一把
  `threading.RLock`，所有写方法与读 branch 指针的渲染方法都在锁内，保证单写者纪律在进程内也成立。
- **渲染打戳缺失**：折叠摘要渲染 `<memento commit=...>` 时曾丢弃 `note_seq`，反思改写 note 后
  “上一轮模型读的是哪个版本”不可归因。现在标签同时带 `commit` 与 `note_seq`，满足 FORMAT.md
  不变量 13。

两个设计项落地：

- **折叠摘要不伪造模型回合**：早期做法在摘要块后紧跟一条捏造的 `ModelResponse("[memento summaries
  loaded]")`——一句模型从未说过的话。现在摘要 preamble 折叠进下一条真实的用户 `ModelRequest`，
  没有任何虚构的 assistant 轮次。
- **失败帧如实入轨迹**：`on_articulate_exit(error=...)` 不再丢弃失败帧。哲学上“看见 X、尝试、
  出错”与“看见 X、选择沉默”同样是轨迹事件。失败帧带 `failed` thread tag 写入 staging，
  永远不会读作已完成回合，`inspect_context()['memory_write']` 记为 `staged_failed`/`committed_failed`。

剩余边界仍需明确：`memory_search` 是字面检索，不解决同义改写、时间/实体推理；记忆纪律是行为
约束不是校验；audience/sensitivity/retention 的 principal 治理未实现。这些不能表述为已达生产级。

### 2.3 明确不做

- 不改 Atom，不把 Memento 逻辑塞进 GhostRuntime 或 `core.memento` 契约。
- 不保存模型隐藏思维链；反思与 curation 只处理用户可见输入、已完成 logos 与显式执行结果。
- 不在本次轨迹主路中实现向量检索、git witness 调度、时间阈值 commit、自动 branch merge 或
  跨 owner 写。检索采用 grep 式原文子串扫描；是否引入向量召回应在其后以可测的召回质量和成本决定。
- **不再实现正则 Evidence/Claim/Recall/Verifier 层**：该层已删除，因为它把语义判断硬编码成
  脆弱的 canonical 模板，违背“记忆是主体生产的轨迹而非管线蒸馏的数据库”这一原则。
- **不接入 Mem0。** 未来若需规模化语义召回，应在专项评审后于 Aurelius 之外定义可选端口，
  且它只能返回候选地址，绝不能成为 Memento 原始证据或笔记的真相源。当前不实现任何 adapter、
  不安装 SDK、不增加 API key、配置项、网络调用或外部存储。
- 不把反思推断或 curation 笔记当作用户身份、课程进度、CTML 成功或 TTS 播放位置的权威事实。
- 不修改 moshi App/mode 代码；Moshi 的接入路线见关联技术评审。

### 2.4 最终目标：让 Agent 更聪明，而不是让 prompt 更长

本工作的产品目标不是“给 Aurelius 加一个 Memory 功能”，而是让它形成可扩展的认知能力：既能
连续记住经历，又能在需要时按字面检索自己的轨迹、核对原文再回答；知道何时保存、何时不再主动
使用；并能面对不同工作场只取用相关的记忆表面。

这不是保存模型隐藏思维链。Aurelius 应持久化的是可观察、可审计的事实、决定、承诺、执行结果、
显式不确定性与反思结论；“模型内部如何逐 token 思考”既不可作为证据，也不应进入记忆系统。

### 2.5 认知状态不能混写

| 状态 | 例子 | 权威来源 | 模型如何使用 |
|---|---|---|---|
| 情景记忆 | 用户问过什么、Aurelius 如何答复 | Memento Moment 与 Commit（不可变） | 检索 + 展开原文核对后回答 |
| 工作场 | 当前仓库、打开的任务目录、Pin 的文件片段 | Ground/Pin + `DESKTOP.md` | 作为当前可见世界 |
| 反思解释 | 用户偏好短回答、发现字段更正 | `by=memento-reflection` 的 CommitNote | 弱解释层；不得当作权威事实裁决 |
| 策展笔记 | curation 从轨迹重写的稳定事实清单 | Ground 中被 pin 的笔记文件（带出处横幅） | 参考；可回溯到冻结 commit 核对 |
| 世界执行进度 | CTML task 是否完成、TTS 播放 offset | 对应执行组件的事件/进度存储 | 由执行组件回答 |

Memento 保存上述状态的“观察证据”，但不替代它们的权威存储。反思 note 与 curation 笔记都是
可再巩固的解释层，永远可回溯到冻结的 Moment 证据；它们不因 last-wins 就成为不可质疑的真相。
Ground 也不保存长期事实：它负责把当前任务有关的世界和约定摆在“桌面”上。

## 3. 架构、所有权与上下文

```text
GhostRuntime
  ├─ 调用 Aurelius.articulate() 读取模型上下文
  ├─ 结束时调用 Aurelius.on_articulate_exit()（成功与失败帧都记录）
  └─ 将 Aurelius.channel() 注册给 Shell

Aurelius
  ├─ AureliusMemory：Memento 读写、窗口、分支、grep 检索（经历账本）
  ├─ AureliusDesktop（P1）：Ground 生命周期、`DESKTOP.md` 与 Pin 帧
  ├─ AureliusReflector：commit 后的异步释义（追加 CommitNote）
  ├─ AureliusCurator：从冻结轨迹重写记忆笔记并 pin 进 Ground（旁路）
  └─ ghost CTML channel：受限人工/模型控制面
```

`AureliusMemory` 是 Ghost 侧适配器。它只使用 `core.memento` 已有的 `update`、`window`、
`commit`、`reinterpret`、`checkout` 和 `switch` 能力，不扩大 Memento ABC。本轮对 `core.memento`
零改动。

一个 `(memento root, owner)` 只有一个写者。跨进程单写靠部署纪律（不同 owner 或外层协调，
不能依赖“最后写入者获胜”）；**进程内**单写由 `AureliusMemory` 的 `RLock` 强制：事件循环上的
`remember`/反思与工作线程上的 CTML 写命令不会交错破坏 staging。当前 CTML 也只允许操作
当前 owner 的 branch。

模型上下文由 `AureliusMemory.model_history()` 每次重建，不维护第二份进程内对话 list：

1. 超出 detail window 的早期 commit 以带 `commit`/`note_seq` 打戳的 `<memento>` note 折叠进上下文，
   折叠 preamble 附着在下一条真实用户 request 上，不生成虚构的模型回合；
2. 最近 `detail_n` 个完整 Moment 以原始 request/response 进入上下文；
3. 多模态 percept（音频等）无法转为文本/图像时保留占位标记，不静默丢失该轮存在的事实；
4. P1 把 Ground frame 放进本帧 user context，把 Ground instruction 作为本次 run instruction；
5. 当前 Moment 原始输入最后传给 Agent；系统 prompt/soul 由 `AureliusMeta` 和 `SystemPrompter`
   组装，并注入记忆纪律 instruction。

这使重启恢复、窗口折叠和审计使用同一份持久事实。

当前代码通过薄 `AureliusDesktop` 持有 `core.desktop.DefaultGrounds`：Ghost enter 时自动打开
项目 root（Host 中取 `Project.root`，嵌入/测试退化为 `GhostWorkspace.home`），articulate 每帧读取
instruction/context，Ghost exit 调用 Grounds 的 best-effort sediment。`DESKTOP.md`/Pin 不写入
Moment；curation 笔记作为独立文件写入并被 pin，不混进聊天历史。

### 3.1 记忆检索：grep 而非事实数据库

模型对过去的主动查询走 `AureliusMemory.search()`：

- 大小写不敏感的原文子串扫描，覆盖本 owner 的全部冻结 commit（新→旧）加当前 staging（最近优先），
  每个 Moment 只命中一次；
- 命中返回 `SearchHit`：`moment_id`、`commit_id`（staging 命中为 `None`）、`commit_seq`、
  `frozen`、`role`（input/logos）与带上下文窗口的 `snippet`；
- 结果按 `limit` 截断，供 `memory_show` 用稳定地址缺页展开。

它有意不做语义解析：没有 canonical key、没有同义归一、没有输出后校验。正确性由记忆纪律
instruction 驱动模型“检索→核对→查不到就说没有”，而不是由一层脆弱正则替模型判断。这与
momento-mori “文件系统即查询语言、stable id 优于模糊召回” 的立场一致，也承认了字面检索的
诚实边界：对“模糊地记得说过什么”这类查询，本地检索没有语义答案，模型应如实表达不确定。

### 3.2 写入与提升规则

| 输入来源 | 是否写入 Moment | 是否进入 mechanical 索引/curation 摘录 |
|---|---:|---:|
| 用户直接陈述（`curation_index_sources`） | 是 | 是 |
| 可信执行组件结果 | 是 | 按 source 配置 |
| Ghost logos | 是 | 是（作为可见回复摘录，不伪造意义） |
| Reflection | 追加为 CommitNote | 不进 Moment，只作解释层 |
| Shell/CTML 内部控制子帧 | 是（审计需要时） | 否（机械索引显式跳过内部控制帧） |

反思与 curation 都不改写 Moment：反思只 `reinterpret()` 追加 note，curation 只写外部笔记文件。
用户更正通过新的 semantic commit 或 reinterpret 表达，旧 note 版本永远可审计。这样保留 Aurelius
的省察能力，又阻断“答错一次 → 自动记成真相”的污染环。

### 3.3 上下文与能力分发原则

记忆读取复用 MOSS 的能力分发原则：不把全部历史送给模型。记忆管理面也不应成为普通对话的
自激工具环。完成 Moment 已自动写 staging，因此 `memory_commit`、`memory_reflect`、`memory_curate`
等运维/旁路命令默认 `visible=False`、`always_observe=False`：它们仍可由人类在 Shell/CTML 显式
执行，但不进入模型日常能力提示，也不会因返回值自动制造下一帧。

`memory_search`/`memory_show`/`memory_log` 对模型可见，是模型自证的检索面，且必须
`always_observe=True`：模型按记忆纪律「先检索、再作答」时，会先发出 `memory_search` 而非直接
输出散文；若检索结果不回灌下一轮 Re-Act，命中就无人阅读，回合静默结算——这正是 canonical-key
问题（如 `ORBIT-004`）上「模型不回复」的根因（2026-07-19 实测修复）。因此「读以作答」的命令观察
其结果，「写/运维」的命令不观察（它们是动作不是答案）。`memory_inspect` 仍是 `visible=False` 的
人类运维面。`desktop_*` 仍是 Aurelius 正常工作的可见能力。

### 3.4 运行时启动边界

`AureliusMeta.factory()` 不是命令启动的第一步。正确链路是：CLI 封存 `Environment` →
`Host` 发现 Ghost manifest 并构造 `GhostRuntime` → Matrix 准备同一个未 bootstrap 的 IoC Container
→ GhostRuntime 注册 Ghost providers 并校验 contracts → Matrix enter/bootstrap →
`AureliusMeta.factory()` 创建 Agent、Memory、反思器与 curator → Ghost enter/启动追赶 → Mindflow 三循环。

启动追赶：enter 时扫描尚无 `by=memento-reflection` note 的 mechanical commit（含 legacy 空 note），
在 `reflection_startup_limit` 内补反思；`_spawn` 保证从任意线程调度都编组回 ghost loop。
TUI 在 Runtime enter 失败时必须同步打印 traceback；`closed / good bye` 只是正常收尾文案。

## 4. Moment、Stage 与 Commit 的写入机制

### 4.1 帧的写入规则

`GhostRuntime` 在模型输出完整写回 `Moment.logos` 后调用 `Aurelius.on_articulate_exit()`：

- **成功帧**（`error is None`）：`remember(moment)` 把完整 Moment 写入 owner pool 并放入 staging。
  正常沉默帧也是完整轨迹事件，照常保存。
- **失败帧**（`error is not None`）：不再丢弃。`remember(moment, threads=("failed",))` 如实写入，
  带 `failed` thread tag 标明这是一次未完成的尝试，不会被读作已完成回合。哲学依据：momento-mori
  “noop 是轨迹事件” 的对称推论——“看见 X、尝试、出错”同样值得 witness，否则下次实例对用户
  那句话毫无痕迹。`inspect_context()['memory_write']` 记为 `staged_failed`/`committed_failed`。

写入后的 commit 规则对两类帧一致：

1. `update_moment()` 把完整 Moment 写入 owner pool，把 id 放入 staging；
2. 若 `len(staging) < auto_commit_every`，流程结束；staging 已落盘，进程退出不丢；
3. 达到阈值时 `branch.commit(..., kind="mechanical", by="aurelius")` 冻结全部 staging；
4. 初始正文是可确定复现的 extractive index：只摘录 `curation_index_sources` 认可的用户 source
   及对应可见回复，跳过纯内部控制子帧，逐条限长、整条机械 Note 有全局字符上限；
5. commit 成功后才安排反思与 curation 任务。

因此 `auto_commit_every: 4` 意为“每四个完成帧形成一个认知锚点”，不是“每四轮才保存一次”。
`0` 禁用自动 commit 但仍保留 staging，直到显式 `memory_commit`。

“成功帧”仅是**情景轨迹写入**条件，不是事实正确性判定：`error is None` 不代表本轮 logos 已被
证实。正确性由模型经 `memory_search`/`memory_show` 自证，以及记忆纪律 instruction 约束。

### 4.2 Commit 的不可变成员

一个 Memento commit 含冻结的 Moment id 列表；冻结后不能修改成员原文。`kind`：

- `mechanical`：阈值触发，初始 note 为原文索引；
- `semantic`：`memory_commit(summary=...)` 显式触发，summary 不能为空。

commit 只能从 staging 产生：不能从未冻结的 staging fork，也不能通过反思修改 Moment。

## 5. Commit 如何追加 Note 与 curation

### 5.1 note 版本模型与渲染打戳

`branch.commit(text, kind=..., by=...)` 在同一原子写入中记录成员行与初始 `CommitNote`。此后
`branch.reinterpret(commit_id, body, by=...)` **不修改**成员行、不覆盖旧 note，只在记录流末尾
追加新 `CommitNote` 版本。`CommitView.summary()` 返回最新 note，`branch.notes(commit_id)` 审计所有版本。

**渲染打戳（FORMAT.md 不变量 13）**：`model_history()` 把折叠 note 渲染为
`<memento commit="{commit_id}" note_seq="{seq}" kind="{kind}">…</memento>`。带上 `note_seq` 使
“上一轮模型读的是哪个释义版本”在反思改写后仍可归因。note 正文中的 `<`/`>` 做转义，防止内容
伪造 `</memento>` 边界（注入面）。折叠 preamble 附着到下一条真实用户 request，**不生成**任何
虚构的 `ModelResponse`。

### 5.2 自动反思追加 note

`AureliusReflector` 异步执行：选择本 owner 尚无 `by=memento-reflection` note 的 mechanical commit
（含 legacy 空 note）；`commit_transcript()` 从冻结成员读取可见 percept 与 logos 并截断；小模型
（`reflection_model_tag`，默认 `small_fast_model`）生成不超过 `reflection_max_summary_chars` 的简短
结论；`apply_reflection()` 保留原 `Kind`、追加 `Reflection: llm` trailer、以 `by=memento-reflection`
调用 `reinterpret()`。任务后台运行，自动 commit 不等待；inflight set 去重；失败只记入
`inspect_state()['reflection']['errors']`，下次启动仍可追赶。反思 note 权威级别低于原始 Moment，
只能生成解释，不能重写 Moment 或把过去的错误回答概括成事实。

### 5.3 旁路 curation

`AureliusCurator` 从冻结轨迹重写一份人类可读的记忆笔记文件（`curation_notes_name`，默认写入
Ground 域），并把它作为 Pin 呈现给下一帧。它读取本 owner 可见 commit log（截断为
`curation_max_source_chars`），用 `curation_model_tag` 小模型生成不超过 `curation_max_notes_chars`
的笔记，笔记带出处横幅指回冻结 commit，供模型回溯核对。curation 是 best-effort 旁路：失败不
阻断对话，原始轨迹仍可检索。它不改 Moment、不建第二真相数据库——笔记只是 Ground 上一个可
重写、可回溯、可 unpin 的工作场表面。

### 5.4 人工/CTML 追加 note

`memory_reinterpret(commit="<seq 或唯一 id 前缀>", summary="...")` 是显式人工运维动作：解析稳定
seq 或无歧义 id 前缀（含糊/不存在即失败），保留原 `Kind`，以当前 owner 作为 `by` 追加新 note。
人工重释义不是 reflection 成功标记；自动反思仍只以是否存在 `by=memento-reflection` 为准。

## 6. 配置：路径、优先级与生效时机

### 6.1 配置路径

workspace 配置文件为 `<workspace root>/configs/memory.yml`（本仓库即 `.moss/configs/memory.yml`）。
它是 Matrix 级配置，不在 `.moss/ghosts/aurelius/` 下；目前只有 Aurelius 消费 `MemoryConfig`。
由 `.moss/src/MOSS/manifests/configs/__init__.py` 注册。**修改后需重启 Aurelius**，不支持热更新。

`AureliusMeta(...)` 的显式参数优先于 YAML，主要用于宿主嵌入和测试。

### 6.2 配置项

| 字段 | 默认 | 含义 |
|---|---:|---|
| `detail_n` | `12` | 最近完整 Moment 数量；至少 1。 |
| `summary_m` | `-1` | 早期 commit note 数；`-1` 全部。长会话宜设正数控 prompt。 |
| `auto_commit_every` | `4` | staged Moment 达此值 mechanical commit；`0` 禁用自动冻结。 |
| `reflection_enabled` | `true` | 是否调度后台反思；模型/凭据未就绪时可先 `false`。 |
| `reflection_model_tag` | `small_fast_model` | 反思模型 tag；应选低成本低延迟。 |
| `reflection_max_summary_chars` | `360` | 单条反思 note 最大字符数。 |
| `reflection_max_source_chars` | `12000` | 单次送入反思模型的冻结原文上限。 |
| `reflection_startup_limit` | `16` | 单次启动最多追赶的待反思 commit；`0` 暂停追赶。 |
| `curation_enabled` | `true` | 是否启用旁路 curation。 |
| `curation_model_tag` | `small_fast_model` | curation 模型 tag。 |
| `curation_index_sources` | 见 yml | mechanical 索引与 curation 摘录视为用户输入的 percept source。 |
| `curation_max_source_chars` | `12000` | 单次发给 curation 模型的可见 commit log 上限。 |
| `curation_max_notes_chars` | 见 yml | curation 笔记最大字符数。 |
| `curation_notes_name` | 见 yml | curation 笔记文件名。 |
| `memory_discipline` | 见 yml | 注入系统 prompt 的记忆纪律 instruction：无可见依据时先 `memory_search`、再 `memory_show` 核对、查不到如实说未找到。 |
| `desktop_enabled` | `true` | 启动时自动打开项目 Ground 并按帧注入 instruction/context。 |
| `context_budget_enabled` | `true` | 输入侧 token 预算：按模型契约主动收缩渲染窗口 + 溢出兜底重试。 |
| `context_token_margin` | `4096` | 预算安全垫；从 `context_window - max_output_tokens` 再扣除。 |
| `context_min_detail_n` | `2` | 主动收缩时明细帧窗口下限；触底仍超则交兜底重试。 |
| `context_fixed_overhead_tokens` | `2048` | system + 当前输入 + ground 注入的固定开销估算。 |

带注释样例见 `.moss/configs/memory.yml`。若 `reflection_enabled`/`curation_enabled` 为 true，
`LLMConfig` 必须能解析对应 tag；否则先关闭旁路，主记忆写入仍正常。

### 6.3 上下文预算与自动压缩

`detail_n`/`summary_m` 只按**帧数**切窗，与 token 无关：`summary_m=-1` 的折叠摘要会无界增长，
多模态帧单帧即可顶满窗口。Aurelius 因此在 articulate 侧实现两级防护（`_budget.py` + `_runtime.py`）：

1. **主动预算收缩**：输入预算 = `context_window - max_output_tokens - context_token_margin`
   （前两个值来自 `ResolvedModel` 契约）。渲染 history 后用保守估算器（char/CJK 除数、图按固定
   名义 token 计，方向=宁可高估提前压缩）核对；超预算则先折半 `detail_n`（明细帧折叠为已有摘要），
   再压 `summary_m`（最旧摘要移出上下文），直到入预算或触底 `context_min_detail_n`。
2. **溢出兜底重试**：provider 仍以溢出文案拒绝时（跨 anthropic/openai 归一，见
   `_is_context_overflow`；不匹配输出侧 max_tokens 错误与 attention abort），且尚未 yield 任何
   token，则进一步腰斩窗口重试（至多 3 次）。已开始流式输出则一律上抛——不能 un-yield。

关键哲学约束：**压缩只减少"渲染进上下文的量"，不销毁任何 Memento 原文**。被移出窗口的
明细/摘要仍在磁盘上，`memory_search`/`memory_show` 永远可取回——这就是 Memento
"折叠而非丢弃"在上下文管理上的兑现，与 Claude Code/Codex 的有损 auto-compact（摘要替换原文）
的本质区别。压缩发生与否记录在 `inspect_context()['context_budget']`（含实际 `detail_n`/
`summary_m`/`estimated_tokens`/`shrunk`/`overflow_retry`），报账不藏。

尚未做（见 FEATURE.md）：逼近阈值时主动 semantic commit 折叠最老明细（auto-compact 完整
形态）；接 provider 精确 tokenizer。注入 model（测试）时预算自动禁用，兜底重试仍在。

## 7. CTML 控制面与分支规则

Aurelius 通过 `Ghost.channel()` 注册虚拟 channel `ghost`。运维/旁路命令默认对模型隐藏且不自动
触发观察；检索/展开与 `desktop_*` 对模型可见。两类能力都不提供跨 owner 读写。

| CTML 命令 | 作用 | 模型可见 | 关键约束 |
|---|---|:---:|---|
| `memory_inspect` | 查看 root/owner/branch/staging/commit/反思 pending | 是 | 不泄露其他 owner |
| `memory_search` | grep 式原文子串检索，返回稳定地址 | 是 | 字面匹配；不做语义归一 |
| `memory_show` | 按 seq/唯一 id 展开冻结原文 | 是 | 只读 |
| `memory_log` | 查看锚点最新释义 | 是 | 只读 |
| `memory_staging` | 查看未冻结 Moment | 是 | staging 仍是持久化状态 |
| `memory_commit` | 将 staging 形成 semantic commit | 否 | summary 非空、staging 非空 |
| `memory_reinterpret` | 追加当前 owner 的新 note | 否 | 不改 Moment/旧 note |
| `memory_reflect` | 请求后台追赶待反思项 | 否 | 不阻塞当前回合；经 loop 编组调度 |
| `memory_curate` | 请求旁路重写记忆笔记 | 否 | 不阻塞；失败不影响对话 |
| `memory_branches` | 列出当前 owner 的 branch | 否 | 不显示跨 owner branch |
| `memory_fork` | 从已冻结 commit 创建并切换新 branch | 否 | 不能从 staging fork |
| `memory_switch` | 按唯一 branch id 前缀切换 | 否 | 含糊/不存在即失败 |
| `desktop_open` / `desktop_close` | 在项目 workspace 内开关 Ground | 是 | 不能越过 workspace root |
| `desktop_pin` / `desktop_unpin` | 管理 Ground 的地址 Pin | 是 | Pin 是地址与观察，不是文件快照 |
| `desktop_update` / `desktop_frame` | 承认外部变化并重绘当前场 | 是 | 只读世界内容 |

`fork` 是有祖先关系的时间线分叉，不是复制后自动合并。当前没有真 branch merge。反思与 fork/switch
的交互：反思作用于调度时的 branch，fork 后祖先 commit 的反思由祖先 owner 负责，子 branch 不
重复追赶祖先（避免跨 owner reinterpret 抛错）。

### 7.1 TUI 输出级别

`moss-run-ghost` 默认 `--output-mode normal`：仅显示人类可读 logos、显式 command-output 和错误；
CTML 标签、`MOMENT`、`SYSTEM`、`COMMAND-RESULT`、操作 start/done 与 `Log:` 均隐藏。`--output-mode
verbose`/`/verbose` 显示运行摘要；`trace`/`/trace` 显示完整内部结果；`/normal` 恢复默认。这只是
展示策略；内部 Session 事件与 Memento 轨迹仍按契约保存，不能以“界面隐藏”为由丢弃错误或轨迹。

## 8. 存储、升级与观测

### 8.1 文件位置

默认实例 Memento 位于 `.moss/ghosts/aurelius/memento/`，含 owner-scoped Moment pool、branch、
commit 和 note 记录。直接编辑会破坏格式与审计性；排查用 `memory_search`/`memory_show`/`memory_log`
或只读扫描。curation 笔记是 Ground 域下的独立可重写文件，删除后下次 curation 会重建。

### 8.2 从旧 `data` 原型升级

旧原型默认路径 `.moss/ghosts/data/memento/` 和 owner `data` 不会被默认读取。不要直接移动目录：
branch owner 归属仍是 `data`。安全选择：(1) 保留旧目录为只读历史，Aurelius 从新 owner 开始；
(2) 宿主显式构造 `AureliusMeta(memory_root="../data/memento", memory_owner="data")` 兼容读取。
跨 owner 真正迁移需导出、校验并按新 owner 重放 Moment；当前无自动迁移器，不能靠复制目录假装
完成迁移。迁移前先备份整个旧 Memento root。

### 8.3 可观测字段

`memory_inspect` 返回 `staging_count`、`commit_count`、`head_commit_id`、窗口/commit 策略与
`reflection_pending`。`Aurelius.inspect_state()` 报告反思任务、curation 状态与 Desktop active
Ground/Pin；`inspect_context()` 记录本帧 `memory_write`、history 消息数与 Ground context 字符数。
待反思不等于记忆丢失：机械 commit 与原始 Moment 已落盘。

## 9. 安全、准确性与失败退化

### 9.1 正确性模型

Aurelius 不承诺一层正则能证明任意自然语言事实。正确性依赖三条协同约束：

1. **不可变证据**：Moment/Commit 冻结、append-only，是唯一权威证据源；反思/curation 是可回溯的
   解释层，不能覆盖它。
2. **字面检索 + 缺页展开**：`memory_search` 按原文匹配返回稳定地址，`memory_show` 展开冻结原文，
   让模型核对而非凭记忆。
3. **记忆纪律 instruction**：无可见依据时先检索、再核对、查不到如实说未找到，不用常识补全。

Prompt 是最后一层，不能替代前两条结构性约束。字面检索有诚实边界：同义改写、时间/实体推理它
覆盖不到，模型应表达不确定而非编造。

### 9.2 并发与进程边界

进程内两个写者域（事件循环的 `remember`/反思，工作线程的 CTML 写命令）由 `AureliusMemory` 的
`RLock` 串行化，读 branch 指针的渲染也在锁内。跨进程单写靠部署纪律：同一 `(root, owner)` 不能
被两个进程同时写，并行化身应分配新 owner/branch。CTML 调度经 `_spawn` 从任意线程编组回 ghost
loop，杜绝工作线程 `create_task` 崩溃。

### 9.3 退化策略

- 反思/curation prompt 不接收隐藏 reasoning；只做可见证据上的解释。
- 反思/curation 模型、网络或凭据失败时，机械轨迹和检索仍可用，不阻断对话。
- 失败帧带 `failed` tag 如实入轨迹，不伪装成完成记忆，也不静默丢弃。
- `summary_m=-1` 的无界摘要与多模态帧由 §6.3 的 token 预算主动收缩兜住；预算估算失准时
  溢出重试折半窗口。两级都失效才请求失败，且失败帧仍被 witness。
- 对话同时写同一 owner、直接改 jsonl、把模型猜测当作执行进度，都会破坏保真性，必须避免。

## 10. 测试与验收入口

自动化与人工对话测试在 [aurelius-memory-test-plan.md](aurelius-memory-test-plan.md)。
最低回归命令：

```bash
.venv/bin/ruff check src/ghoshell_moss/ghosts/aurelius
.venv/bin/pytest -q src/ghoshell_moss/ghosts/aurelius tests/ghoshell_moss/default/core/memento tests/ghoshell_moss/host/test_ghost_ui_output.py
.venv/bin/python scripts/ghost/aurelius_memory_acceptance.py
.venv/bin/moss-run-ghost aurelius
```

自动化现已覆盖：成功帧写入、**失败帧如实入轨迹（带 failed tag）**、跨重启、窗口折叠、
**折叠摘要渲染打戳（commit + note_seq）且不伪造模型回合**、note 追加版本、反思失败与启动追赶、
**CTML `memory_reflect` 在工作线程调度不崩溃**、**并发 remember/commit 不损坏 staging**、
grep 检索命中与地址、注入转义、多模态占位、owner 隔离、fork 边界、默认精简输出与 CTML 隐藏。
P2（audience/sensitivity/retention 治理）、P3（外部召回后端）与真实 LLM/Host 手工验收仍按测试
方案单列；Moshi 的用户模型和世界执行进度属于下一层产品集成，不能以 Aurelius 的反思/curation
笔记代替。
