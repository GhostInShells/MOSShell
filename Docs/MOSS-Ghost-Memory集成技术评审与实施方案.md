# MOSS Aurelius Ghost Memory 集成技术评审与实施方案

> 状态：已实现的 Aurelius/Memento 记忆主路；本文描述当前代码的目标、边界与运行方式。
>
> 关联文档：[测试方案](MOSS-Ghost-Memory测试方案.md)、[Moshi 集成技术评审](Moshi-长期记忆与进度集成技术评审.md)。

## 1. 结论与命名

第二个 Ghost 原型命名为 **Aurelius**，取自《沉思录》作者 Marcus Aurelius。Atom 仍是
无持久历史的最小基线；Aurelius 是带有可审计 Memento 轨迹、机械锚点和事后反思的持久化
原型。它不是一个泛称为 “Data” 的数据容器。

当前交付的最小闭环如下：

```text
完成的 Moment
  → Memento staging（立即持久化）
  → mechanical commit（默认每 4 个 staged Moment）
  → 非阻塞 reflection（可失败）
  → reinterpret() 追加 CommitNote
  → 下一次模型上下文：早期 note + 近期完整 Moment
```

原型的公开代码名、运行名与默认 owner 均为 `aurelius`：

| 位置 | 当前名称 |
|---|---|
| Python 包 | `ghoshell_moss.ghosts.aurelius` |
| 类型 | `AureliusMeta`、`Aurelius`、`AureliusMemory`、`AureliusReflector` |
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

### 2.2 明确不做

- 不改 Atom，不把 Memento 逻辑塞进 GhostRuntime 或 `core.memento` 契约。
- 不保存模型隐藏思维链；反思只处理用户可见输入、已完成 logos 与显式执行结果。
- 不实现向量检索、git witness 调度、时间阈值 commit、自动 branch merge 或跨 owner 写。
- 不把反思推断当作用户身份、课程进度、CTML 成功或 TTS 播放位置的权威事实。
- 不修改 moshi App/mode 代码；Moshi 的接入路线见关联技术评审。

### 2.3 三类状态不能混写

| 状态 | 例子 | 权威来源 |
|---|---|---|
| 情景记忆 | 用户问过什么、Aurelius 如何答复 | Memento Moment 与 Commit |
| 派生认知 | 用户似乎偏好简洁、某主题仍有疑点 | 反思 note 或后续可重算投影 |
| 世界执行进度 | CTML task 是否完成、TTS 播放 offset | 对应执行组件的事件/进度存储 |

Memento 可以保存后两类状态的“观察证据”，但不能替代它们的权威存储。

## 3. 架构、所有权与上下文

```text
GhostRuntime
  ├─ 调用 Aurelius.articulate() 读取模型上下文
  ├─ 正常结束时调用 Aurelius.on_articulate_exit()
  └─ 将 Aurelius.channel() 注册给 Shell

Aurelius
  ├─ AureliusMemory：Memento 读写、窗口、分支
  ├─ AureliusReflector：commit 后的异步释义
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
3. 当前 Moment 由 runtime 传给 Agent；
4. 系统 prompt/soul 仍由 `AureliusMeta` 和 `SystemPrompter` 组装。

这使重启恢复、窗口折叠和审计使用同一份持久事实。

## 4. Moment、Stage 与 Commit 的写入机制

### 4.1 成功帧的写入规则

`GhostRuntime` 在模型输出完整写回 `Moment.logos` 后调用 `Aurelius.on_articulate_exit()`。
当 `error is None` 时，Aurelius 调用 `AureliusMemory.remember(moment)`：

1. `update_moment()` 把完整 Moment 写入 owner 的 pool，并把 id 放入 staging；
2. 若 `len(staging) < auto_commit_every`，流程结束；staging 已经落盘，进程退出不会丢；
3. 若达到阈值，`branch.commit(..., kind="mechanical", by="aurelius")` 冻结全部 staging；
4. 初始正文是可确定复现的 extractive index：每帧输入与 logos 各截断 240 字符；
5. commit 成功后才安排反思任务。

因此默认值 `auto_commit_every: 4` 的意思是“每四个完成帧形成一个认知锚点”，**不是**
“每四轮才保存一次”。第 1 到第 3 帧已经在 staging 中持久化。`0` 禁用自动 commit，
但仍保留 staging，直到使用 `memory_commit` 进行显式 semantic commit。

当 articulate 出错时，Aurelius 标记 `memory_write=skipped_on_error`，不写入未完成的
Moment。正常沉默帧仍是完整轨迹事件，可以保存。

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

### 5.3 人工/CTML 追加 note

`memory_reinterpret(commit="<seq 或唯一 id 前缀>", summary="...")` 是显式人工或 Agent
动作：

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
`memory_detail_n`、`memory_summary_m`、`auto_commit_every`、`reflection_enabled`。

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

完整的带注释样例见 `.moss/configs/memory.yml`。若设 `reflection_enabled: true`，还必须让
`LLMConfig` 能解析 `small_fast_model`（或把 tag 改为已配置的 tag）；否则应先关闭反思，
主记忆写入仍可正常运行。

## 7. CTML 控制面与分支规则

Aurelius 通过 `Ghost.channel()` 注册虚拟 channel `ghost`。能力均为 `always_observe`，用于
显式检查和控制；它们不提供跨 owner 读取/写入。

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

`fork` 是有祖先关系的时间线分叉，不是复制后自动合并。当前没有真 branch merge；
`make_merge_message()` 只能把 commit 引用带入后续对话，不能解决冲突、来源和人格主权。

## 8. 存储、升级与观测

### 8.1 文件位置

默认实例的 Memento 位于：

```text
.moss/ghosts/aurelius/memento/
```

其中包含 owner-scoped Moment pool、branch、commit 和 note 记录。直接编辑这些记录会破坏
Memento 格式与审计性；排查使用 `memory_show`、`memory_log`，或只读搜索。

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

`memory_inspect` 返回 `staging_count`、`commit_count`、`head_commit_id`、窗口/commit 策略和
`reflection_pending`。`Aurelius.inspect_state()` 还报告反思是否启用、running/inflight 数及最近
三条错误。待反思不等于记忆丢失：机械 commit 与原始 Moment 已经安全落盘。

## 9. 安全、准确性与失败退化

- 反思 prompt 不接收隐藏 reasoning；输出要求为可见证据上的简短结论。
- 反思是派生解释，可能错误；需要原文证据时使用 `memory_show`，人工用
  `memory_reinterpret` 更正，不删除历史。
- `summary_m=-1` 会让无界早期 note 进入 prompt；长会话应设置预算并通过 CTML 精查旧记录。
- 反思模型、网络或凭据失败时，系统退化为“机械摘录 + 完整原文”，不阻断对话。
- 对话同时写同一 owner、直接改 jsonl、把模型猜测当作执行进度，都会破坏保真性，必须避免。

## 10. 测试与验收入口

自动化与人工对话测试在 [MOSS-Ghost-Memory测试方案.md](MOSS-Ghost-Memory测试方案.md)。
最低回归命令：

```bash
.venv/bin/ruff check src/ghoshell_moss/ghosts/aurelius
.venv/bin/pytest -q src/ghoshell_moss/ghosts/aurelius tests/ghoshell_moss/default/core/memento
.venv/bin/python scripts/ghost/aurelius_memory_acceptance.py
.venv/bin/moss-run-ghost
```

验收应覆盖：成功帧写入、失败帧跳过、跨重启、窗口折叠、note 追加版本、反思失败与启动
追赶、配置生效、owner 隔离、fork 边界以及 CTML 的错误输入。Moshi 的用户模型和世界执行
进度属于下一层产品集成，不能以 Aurelius 的反思 note 代替。
