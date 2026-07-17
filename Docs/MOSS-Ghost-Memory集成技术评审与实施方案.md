# MOSS Ghost Memory 集成技术评审与实施方案

> 评审对象：`dev-matrix-cell-refact` 当前分支
>
> 评审日期：2026-07-17
>
> 参考文档：
> - `MOSS-Memory-认知场与Memento分支分析与评审更新.md`
> - `MOSS 记忆与能力分发架构梳理：memento 对比、Ghost_App 集成路径、千万级 Channel 规模化推演.md`

## 1. 结论

本期应把已有 Memento 能力接到一个新的 **Data Ghost**，而不是修改 Atom、扩张
GhostRuntime，或再造一套 Memory ABC。

最小闭环是：

1. Data Ghost 启动时在自己的 `GhostWorkspace` 内打开 owner-scoped Memento；
2. 每轮推理前，从当前 branch 渲染“远端 commit 摘要 + 近期完整 Moment”；
3. 每轮成功结束后，把已经落定 logos 的 Moment 写入 staging；
4. staging 达到阈值后机械 commit，保证历史有稳定锚点；
5. Ghost 退出时关闭 Memento，下一次启动从同一 owner/current branch 恢复。

这一方案只增加 Ghost 侧适配，不修改 Memento 契约层，也不让 Host 知道具体存储。
它实现了跨进程记忆、窗口折叠与可追溯原文，同时保留未来 semantic commit、
reinterpret、fork、见证层与认知场集成的接口。

## 2. 当前事实与文档校正

### 2.1 已有能力

当前分支已经具备：

- `core.memento` 契约与文件系统实现：MomentRecord、Commit、Branch、Memento；
- `porcelain` 强类型桥：`Moment ↔ MomentRecord`、窗口渲染、MementoRef；
- `Ghost.memories()`：动态记忆的现成读接口；
- `Ghost.on_articulate_exit()`：一轮 logos 已完整落定后的现成写入挂点；
- `GhostWorkspace.home`：Host 为 Ghost 分配的持久化目录；
- Atom：纯内存线性历史的最小 Ghost 基线。

### 2.2 两个需要按当前代码修正的判断

第一，参考文档把写入挂点概括为 Mindflow `on_moment`。该回调发生在 Moment 创建
阶段，此时 logos 尚未生成，直接持久化只能得到半帧。当前更精确的挂点是
`Ghost.on_articulate_exit()`：GhostRuntime 已先把完整 logos 写回 Moment，再调用该
hook；`thinking_effort == none` 也会调用它，因此“看见但选择沉默”仍可入轨迹。

第二，`Ghost.memories()` 已存在，但 Host 目前不会自动把它拼进模型上下文。直接在
GhostRuntime 里全局接线会让所有 Ghost 被迫接受一种记忆语义，也会破坏 Atom 的
基线定位。因此本期由 Data Ghost 在自身 `articulate()` 内消费记忆窗口。

## 3. 记忆边界

MOSS 当前两块记忆地基正交：

| 维度 | Grounds / Desktop | Memento |
|---|---|---|
| 回答的问题 | 此刻什么在眼前 | 过去发生了什么 |
| 时间性 | 工作记忆 / 现在 | 轨迹记忆 / 过去 |
| 上下文位置 | 动态 context | memory + conversation |
| 数据语义 | 地址每帧重绘 | append-only Moment/Commit |
| 本期范围 | 不接入 | 接入 Data Ghost |

本期只做 Memento。Desktop 的 pin、promote、预算报账属于“当前注意力”问题，不应
为了让 Ghost 先拥有持久记忆而绑进同一个提交。

## 4. 方案比较

### 4.1 方案 A：GhostRuntime 全局持有 Memento

优点是所有 Ghost 自动获得记忆；缺点是 Host 必须决定 owner、存储根、commit
策略、上下文裁剪和失败语义，Atom 也不再是纯内存基线。它把策略错误地下沉到
编排层，本期否决。

### 4.2 方案 B：新建 `Memory` ABC

Memento 已提供轨迹契约，Grounds 已提供工作记忆契约，Ghost 也已有
`memories()`。第四个平行 ABC 只会增加转换层和命名争议，本期否决。

### 4.3 方案 C：Data Ghost 持有 Memento

该方案与现有 `data-ghost` FEATURE.md 一致：Atom 保持基线，Data 负责“现在 +
过去”的高级上下文；Memento 是标准库件，但生命周期和策略归具体 Ghost。它是
本期采用方案。

## 5. 目标结构

```text
GhostRuntime
  └─ Data Ghost
      ├─ Agent / Model
      └─ DataMemory
          └─ FsMemento(owner=data)
              └─ current MementoBranch
                  ├─ commit summaries  ──→ 模型历史前缀
                  ├─ recent moments    ──→ 完整对话历史
                  └─ staging           ←── 完成的当前 Moment
```

数据流：

```text
Signal → Moment → Data.articulate()
                    │
                    ├─ DataMemory.model_history() → Agent
                    └─ stream logos → GhostRuntime 写回 Moment.logos
                                          │
                                          └─ on_articulate_exit()
                                               ├─ update_moment(staging)
                                               └─ 达阈值 → mechanical commit
```

## 6. 详细设计

### 6.1 DataMemory

DataMemory 是 Ghost 侧的薄适配器，不进入 `core.memento` 契约层。职责包括：

- 打开和关闭 owner-scoped Memento；
- 把 branch window 转为模型 SDK 的历史消息；
- 把完成的 Moment 写入 staging；
- 按数量阈值机械 commit。
- 为反思旁路提供已冻结 commit 的可见原文与 `reinterpret()` 孔径；
- 为 Ghost CTML 控制面提供当前 owner/branch 的显式操作。

它不负责语义检索、Desktop、git witness 或跨 owner 写入。

### 6.2 存储地址与 owner

默认根目录：

```text
{GhostWorkspace.home}/memento/
```

默认 owner 使用 Ghost 名称。Ghost 名和 home 都是跨重启稳定的，因此同一个 Ghost
重新启动会自然恢复；不同 Ghost 的 home 天然隔离。

硬约束：同一个 `(root, owner)` 同时只能有一个写者。当前 Host 对单个 GhostRuntime
满足这一点；未来并行化身必须使用新 owner 或 branch 规则，不得用多个进程并写同一
owner。

### 6.3 读取窗口

窗口参数：

- `detail_n`：近期完整 Moment 数，默认 12；
- `summary_m`：明细区之前保留的 commit 摘要数，默认全部；
- summaries 作为一个明确标记的“较早记忆摘要”回合；
- details 用 `Moment.to_history_turns()` 恢复用户/assistant 回合。

这保持了 Memento 的可逆折叠：摘要进入热上下文，原文仍可由 commit id 展开。

### 6.4 写入与 commit

只在 `on_articulate_exit(error is None)` 写入。这样：

- 模型成功回答：保存 percept、reaction 与完整 logos；
- 正常沉默：保存空 logos 的 Moment，轨迹仍连续；
- 模型调用失败：本期不把失败半帧伪装成完成记忆，错误由运行日志保留。

staging 达到 `auto_commit_every`（默认 4）时执行 mechanical commit。初始释义是
有长度上限的输入/输出原文摘录索引，只保真摘录、不推断意义，避免旧 commit 在退出
明细窗口后完全不可召回。反思旁路在 mechanical commit 冻结后使用
`MemoryConfig.reflection_model_tag`（默认 `small_fast_model`）读取可见原文并 `reinterpret()`
追加语义 CommitNote。它不阻塞回答，不改 Moment；失败任务会在下次启动后追赶，且不
持久化模型私有推理过程。

### 6.5 生命周期

- `DataMeta.factory()` 解析 workspace、模型配置和记忆配置；
- `Data.__aenter__()` 后台调度有上限的 reflection catch-up；
- `Data.__aexit__()` 取消未完成旁路任务后关闭记忆，下次启动继续追赶；
- 不在退出时强制 commit：staging 本身持久化，强制 commit 会把进程退出误当成
  认知边界。

### 6.6 模型配置

Data 优先接受构造时传入的 pydantic-ai Model，便于测试和宿主自定义；未传入时从
IoC 的 `ConfigStore` 读取 `LLMConfig`，再按 `anthropic/openai` 协议构建 provider。
如果宿主没有 ConfigStore，则退化到 `LLMConfig().resolve()` 的环境变量配置。

### 6.7 MemoryConfig 与 CTML 控制面

`MemoryConfig` 是持久化的 Data 策略面，位于 workspace `configs/memory.yml`。它包含
window（`detail_n`/`summary_m`）、count-based commit（`auto_commit_every`）、反思开关、
模型 tag、长度与启动追赶上限。默认为 12 帧明细、全部早期摘要、每 4 帧
mechanical commit 和启用 `small_fast_model` 反思。`DataMeta` 显式参数优先于 YAML，用于
测试和宿主覆盖。时间阈值与 witness 调度仍未实现。

Data 通过 `Ghost.channel()` 向 Shell 注册名为 `ghost` 的虚拟 channel，提供
`memory_inspect`、`memory_log`、`memory_staging`、`memory_show`、`memory_commit`、
`memory_reinterpret`、`memory_branches`、`memory_fork`、`memory_switch` 和 `memory_reflect`。
控制面只操作当前 Data owner 的 current branch：不提供跨 owner 写入、隐式 merge，
且 fork 只能从冻结 commit 出生。

## 7. 不变量与失败语义

必须守住：

1. Memento 契约层不 import Ghost、Host、IoC 或模型 SDK；
2. Atom 行为不变，仍是纯内存基线；
3. 只持久化完成帧，不重复保存同一 Moment id；
4. 模型历史完全可由 Memento 重建，Data 不维护第二份线性历史；
5. perspectives 与 hint 按现有 porcelain 规则不入持久层；
6. mechanical commit 只写带标识的原文摘录，不能伪造语义；
7. 同 owner 单写者；
8. 记忆损坏应显式失败，不能静默清空后继续“失忆运行”。

## 8. 本期交付范围

### 必做

- 新增 `ghoshell_moss.ghosts.data`；
- DataMemory 的窗口渲染、写入、机械 commit、关闭；
- Data Ghost 的模型配置、持久化 articulate 与观测信息；
- `MemoryConfig` 的 workspace YAML 默认策略；
- 非阻塞的 LLM 反思、启动追赶与反思失败观测；
- 仅当前 owner/branch 的 Memento CTML 控制面；
- workspace/stub 中注册可直接运行的 `data` Ghost；
- 单测覆盖跨实例恢复、窗口裁剪、commit、失败不写入；
- 自动化验收脚本与人工对话测试方案。

### 明确不做

- 不修改 Memento FORMAT/ABC；
- 不接 Desktop/Grounds；
- 不做向量检索；
- 不做 git witness daemon；
- 不做 branch merge 与跨 owner 写入；
- 不解决重绘层“承诺保全”。

## 9. 测试策略

### 9.1 单元测试

- 空存储返回空历史；
- 写入一个完成 Moment 后可重建 user/assistant 回合；
- 达阈值后 staging 清空并生成 mechanical commit；
- 新实例用同 root/owner 恢复同一历史；
- `detail_n` 只保留近期明细，旧 commit 以摘要进入窗口；
- 失败回合不写入；
- `memories()` 输出带 MementoRef 的摘要与可读明细。
- 反思只增加 CommitNote，不改写 Moment；启动后能追赶没有反思 note 的 mechanical commit；
- MemoryConfig 能从 ConfigStore 生效，控制面暴露受限的当前 owner/branch 操作。

### 9.2 集成测试

用 pydantic-ai `TestModel` 跑两轮 Data Ghost：第一实例回答并落盘，销毁后创建第二
实例，断言第二次模型请求包含第一轮历史。该测试不访问网络。

### 9.3 人工对话测试

核心场景：

1. 告诉 Data 一个随机事实，退出并重启，询问该事实；
2. 连续对话超过 `detail_n`，确认旧信息通过摘要/锚点而非原文常驻；
3. 同时给出相似但不同的事实，检查是否串写；
4. 更正旧事实，检查模型是否区分“历史事实”和“当前事实”；
5. 制造一次模型调用失败，恢复后确认失败输入没有被伪装成成功回合；
6. 查看磁盘 jsonl，核对回答中的引用与实际 Moment/Commit。

详细话术、评分方法和脚本见配套测试方案。

## 10. 风险与后续

| 风险 | 本期处理 | 后续方向 |
|---|---|---|
| 反思失败或延迟 | 机械索引先行，失败不阻断对话 | 重试、启动追赶与任务观测 |
| 模糊召回能力弱 | commit 摘要 + 原文窗口 | 先目录/LLM recall，必要时再向量化 |
| 同 owner 并发写 | 明确单写者约束 | 化身 owner/branch 治理 |
| 活承诺在折叠中丢失 | 不声称已解决 | 重绘层承诺 reconcile |
| CTML 控制误操作记忆 | 仅当前 owner/branch，fork 只从 commit | 权限、审计与模型策略 |

## 11. 验收标准

满足以下条件即认为 Ghost 已具备第一阶段 Memory 能力：

- Data Ghost 的模型输入历史来自 Memento，而不是进程内 list；
- 同一 Ghost 跨进程重启能恢复并回答之前保存的信息；
- 每个成功认知帧只写一次，机械 commit 可稳定触发；
- Atom、Memento 契约层与 GhostRuntime 不因本功能改变行为；
- 自动化测试通过，人工测试可定位“写入、折叠、召回、纠错”各阶段结果。

这是一条可退化、可验证的最短路径：先证明 Ghost 能持续记住，再让它学会主动解释、
检索、分叉和整理自己的记忆。

## 12. 补充评审快照：demo 完整能力、反思策略与 moshi 适配

> 补充日期：2026-07-17
>
> 新增评审输入：`tmp/memento_demo.py`、`Docs/moshi支持讨论.md`、
> `/Users/lipeng/TraeProject/moss-in-reachy-mini/.moss_ws/apps/ui/moshi`，
> 以及 `moss-in-reachy-mini` 的 Lynn Ghost 与 `LocalConversationStore`。
>
> 本节记录实施前的差距分析；Data 的当前实现状态以第 13 节为准。

### 12.1 结论先行

1. **Memento 内核原语足以实现 demo 展示的能力，但当前 Data Ghost 尚未全部
   接出。** 已接入的是完成 Moment 持久化、窗口重建、机械 commit 和跨重启
   恢复；大模型反思、手动语义 commit、reinterpret、show、fork/switch 和 git witness
   仍只存在于底层原语或 demo 编排中。
2. **当前每个成功回合都立即 stage，默认累计 4 个 staged Moment 后立即整理并
   mechanical commit。** 它不是“每 4 轮才存一次”；staging 本身已持久化，第
   1—3 轮即使进程退出也不会丢。
3. **阈值可在 `DataMeta(auto_commit_every=N)` 构造时配置，`0` 表示禁用自动
   commit；但它还不是 `ConfigType`/YAML 配置，也不支持运行时热更新。** 整理、
   commit 和 witness 还没有独立策略项。
4. **反思应该是下一个高优先级能力，但必须是 commit 之后的非阻塞旁路。** 反思只改写
   commit 的释义层，不改原始 Moment；失败不得阻断对话，且可以扫描空释义后补。
5. **当前 `moss-in-reachy-mini` 中的 moshi 不支持讨论里的完整效果。** 它能加载课程、
   跳章、接收交互和展示 logos；Lynn 能持久化线性对话。但它尚无 Memento commit/
   反思/分支，无每个用户的知识模型，无重复讲解计数，也无精确的课程、CTML 和
   TTS 执行进度。

### 12.2 Stage → Organize → Commit 的精确时序

| 节点 | 当前规则 | 是否持久化 | 是否可配置 |
|---|---|---:|---:|
| Stage | 每个 `on_articulate_exit(error is None)` 完成帧立即 `update_moment()` | 是 | 否，这是保真不变量 |
| Organize | 达阈值时，对 staging 中每帧的输入和 logos 各截取 240 字符，生成原文索引 | 随 commit 持久化 | 否 |
| Memento commit | `len(staging) >= auto_commit_every` 时同步冻结，默认 4 帧 | 是 | 构造时可配 |
| Semantic reflection | 当前 Data 未接入 | 否 | 否 |
| Git witness snapshot | 当前 Data 未接入 | 否 | 否 |

`commit` 是 Memento 的认知锚点，不是代码仓库的 git commit；git 只是可选 witness
层。当前 Organize 也不是反思，只是可确定复现的抽取索引。它能保证旧内容在
离开 detail window 后仍有可见线索，但不会形成“用户在意什么”或“任务进展到
哪里”的语义。

### 12.3 `memento_demo.py` 能力对照

| demo 能力 | Memento 内核 | 当前 Data Ghost | 结论 |
|---|---:|---:|---|
| 每轮 Moment 写 staging | 已有 | 已接入 | 完成 |
| 历史完全从 Memento 重建 | 已有 | 已接入 | 完成 |
| 跨进程恢复 | 已有 | 已接入 | 完成 |
| 近期明细 + 早期 commit 折叠 | 已有 | 已接入 | 完成 |
| 每 N 帧 mechanical commit | 已有原语 | 已接入，N 默认 4 | 完成 |
| 大模型事后反思 | `reinterpret()` 支持补释义 | 未接编排 | 下一阶段 |
| 启动时追赶空总结 | 可扫描 commit/note | 未接入 | 下一阶段 |
| 主动 semantic commit | `branch.commit(kind="semantic")` | 无 Ghost/channel 入口 | 需 CTML 控制面 |
| log/staging/window/show | 读 API 已有 | 无 Ghost/channel 入口 | 需 CTML 控制面 |
| reinterpret 与释义追溯 | 已有 | 未暴露 | 需 CTML 控制面 |
| fork/branches/switch | 已有 | 未暴露 | 需身份/分支策略 |
| git witness | 已有 `Witness`/`snapshot()` | 未接入 | 应作为低频旁路 |

因此，“能不能实现 demo 里所有能力”的答案是：**能，且大多数不需要修改
Memento ABC/FORMAT；但当前产品化 Ghost 还只完成了最小主路。**

Memento 中的 `make_merge_message()` 是“把 commit 引用带回主路”，不是 Git 式两条
branch 的结构合并。讨论中“把用户子分支合并回主记忆”还需另行定义冲突、来源和主权
规则，不能宣称已支持。

### 12.4 反思的必要性与接入约束

机械索引解决“原文线索不要丢”，反思解决“这段经历对以后的认知和行动意味
什么”。对 moshi 这类长程互动，反思用于稳定提取：

- 用户已理解、未理解、反复追问的内容；
- 用户更适合的讲解节奏、语言和专业深度；
- 被更正的旧事实、尚未结束的线索和下次应恢复的目标；
- 多轮对话对当前任务的实际意义，而不只是字面摘录。

建议的最小链路：

```text
completed Moment → staging → mechanical commit
                                  │
                                  └─ emit commit-id → reflection worker
                                                        ├─ 读 commit_records()
                                                        ├─ 调 small_fast_model
                                                        └─ reinterpret(commit-id)
```

反思必须满足：

1. **旁路非阻塞**：commit 先成功，反思可失败、可重试、可在下次启动追赶。
2. **原文不变**：只追加 CommitNote，不改 Moment 成员和原始 payload。
3. **可引用**：重要结论带 moment/commit 证据 id，低置信推断明确标记为推断。
4. **不持久化隐藏思维链**：输入只使用可见对话、执行结果、错误和状态事件，产物是
   简洁结论，不保存模型的私有推理过程。
5. **反思不是权威事实源**：它可提出“用户似乎已掌握 X”，不能凭空改写用户身份或
   设备执行进度。

用于 moshi 的反思输出至少需要稳定表达：

```yaml
summary: 这段交互的核心内容
user_model:
  understood: []
  uncertain: []
  repeated_questions: []
progress_observation: []  # 只是观察，不覆盖权威 progress
open_threads: []
corrections: []
evidence_moments: []
confidence: 0.0
```

v1 可先把它压缩为 CommitNote 正文与 trailers；若后续需要稳定查询用户模型，再增加
独立的派生投影，不修改原始 Moment 仓。

### 12.5 建议的可配置策略

当前只有 `auto_commit_every`，它把“何时冻结”和“如何整理”隐式绑定。下一阶段
应用 `ConfigType` 拆成四类 workspace YAML 策略：

| 策略 | 建议配置 | 原则 |
|---|---|---|
| Stage | 无开关；成功完成帧必写 | 不为节省 IO 牺牲轨迹完整性 |
| Commit | `max_moments=4`, `max_age`, semantic trigger | 数量、时间和语义边界取最先到者 |
| Reflection | `enabled`, `model_tag`, `async`, `retry`, `max_chars` | 可失败、可追赶、不阻塞主路 |
| Witness | `every_commits`, `max_age`, `on_idle` | git 快照不进高频写路径 |

不建议把 commit 仅固定为“每 N 轮”。moshi 可能长时间只有一个大回合，也可能几秒内
产生多个小互动。应同时支持时间阈值和显式语义边界，例如“完成一个知识点”、
“用户中断并转向旧问题”或“课程小节结束”。

### 12.6 moshi 当前代码与讨论需求的差距

#### 12.6.1 moshi App 自身

当前 moshi 的状态只有进程内的 `course` 和 `current_id`：

- 加载课程时会读取所有章节，但只向 Ghost 注入“课程概览 + 当前章节全文”，并非
  讨论所说的“全课程展平后全程可见”；
- `next_chapter()` 和 `jump_chapter()` 只记录当前章节，不记录章内知识点、
  已展示画面、已完成 CTML 动作和已实际播放的文字位置；
- App 重启后 `course/current_id` 重置；
- 没有 user id 到 memory owner/branch 的路由，也没有“这个人已听过几次”的派生状态。

#### 12.6.2 实际运行的 Ghost 与依赖

`moss-in-reachy-mini` 当前锁定本地 `ghoshell_moss-0.1.0b0` wheel；该 wheel 包含 Atom，
不包含本分支的 `core.memento` 与 Data Ghost。工作区的 Lynn Ghost 另外实现了
`LocalConversationStore`：

- 每轮保存 user input、Moment JSON 和 logos；
- 在稳定 `session_scope` 下可恢复当前 conversation；
- 模型上下文只取最近 `max_turns`，当前 YAML 配置是 5；
- 超出的旧回合只被截断，没有摘要、commit 锚点、按需展开、reinterpret 或 fork；
- `create/switch` 是平行 conversation 切换，不是带祖先的 Memento branch。

所以，即使 Lynn 已解决了“最近对话重启后还在”，也不能等价为讨论里的长程
个性化记忆。

#### 12.6.3 能力支持矩阵

| 讨论目标 | 当前 moshi/Lynn | Memento/Data 基础 | 还缺什么 |
|---|---:|---:|---|
| 记录人与 Ghost 聊过什么 | 部分支持，仅最近线性回合 | 支持持久轨迹 | 升级依赖并切换 Data |
| 往回跳章后继续 | 支持手动跳章 | 可回忆跳转经过 | 持久化 `MoshiProgress` |
| 知道某问题已讲过几次 | 不支持 | 原始 Moment 可作证据 | 反思投影 + 可查询计数 |
| 估计用户已懂/未懂 | 不支持 | 可从 commit 反思 | 用户模型与置信度 |
| 按大人/孩子/专家调整表达 | 可临时做，无持久依据 | 可保留互动证据 | 稳定 person id → user branch/profile |
| 不同人记忆隔离 | 未与 moshi 连通 | owner/branch 可隔离 | 身份识别与路由策略 |
| 从过去锚点分叉 | 不支持 | 底层支持 checkout | Ghost/channel 控制面与产品规则 |
| 子分支合并回主体 | 不支持 | 仅支持 commit 引用，无真合并 | 主权、冲突与来源设计 |
| 长程讲解中断后精确续讲 | 不支持 | Moment 可保存回合结果 | CTML/TTS 结构化 execution trace |

### 12.7 记忆、用户模型与执行进度必须分层

| 类型 | 例子 | 建议真相源 |
|---|---|---|
| 情景记忆 | 用户问过什么，Ghost 如何回答 | Memento Moment/Commit |
| 用户模型 | 用户似乎掌握什么，偏好哪种节奏 | 基于 Memento 证据的反思派生物 |
| 执行进度 | 课程到第几节，某段 TTS 实际播到哪个 offset | `MoshiProgress` + execution trace |

执行进度必须由程序事件更新，不应由大模型反思直接写权威值。长文本“已生成”、
CTML “已提交”、设备“已执行”和 TTS “已播放”是四个不同时刻。当前 Data 在模型流失败时
跳过整个未完成 Moment，能避免把半帧伪装成完成帧，但也无法告诉 moshi 长段语音究竟
已经播了多少。

```text
MoshiProgress/Trace ──权威执行事实──┐
                                         ├─→ Moment 作为一次认知快照
User interaction ──对话与感知─────┘              │
                                                        └─→ Commit 反思与长期召回
```

### 12.8 后续优先级（本次不改代码）

1. **P0：Data 反思旁路。** 以 demo 的 `small_fast_model → reinterpret()` 为起点，改成非阻塞
   worker，支持失败重试与启动追赶。
2. **P0：Memory `ConfigType`。** 拆分 commit/reflection/witness/window 策略，落 workspace YAML；
   保留构造参数作为测试和宿主覆盖面。
3. **P1：moshi 权威进度。** 先定义 `MoshiProgress` 和 CTML/TTS trace，再谈中断续讲；
   不用 LLM 摘要代替执行日志。
4. **P1：moshi 用户路由。** 使用稳定 person id 选择 profile/branch，匿名人使用 session
   临时身份；先隔离，后考虑跨人共享。
5. **P1：Memento CTML 控制面。** 暴露 inspect/show/semantic-commit/reinterpret/fork/switch，
   默认不给模型无约束跨 owner 写权。
6. **P2：分支治理与合并语义。** 先用 commit 引用进行显式吸收；真正 branch merge 等
   冲突、来源和人格主权规则明确后再实现。

综合结论：**现有 Data Ghost 已经是可用的“持久化记忆主路”，但还不是 demo 里的
“可反思、可自我整理的完整 Ghost”，更不能直接满足 moshi 的个性化长程讲解。正确的
下一步不是扩大一个摘要 prompt，而是同时补上“反思旁路”和“权威执行进度”两条
相互独立的链路。**

## 13. 2026-07-17 实施状态（覆盖第 12 节的候选判断）

以下四项已实现并由定向测试覆盖：

| 需求 | 实现 | 边界 |
|---|---|---|
| commit 反思 | `DataReflector` 在 mechanical commit 后后台调用 LLM，以 `reinterpret()` 追加 note | 只读已冻结的可见输入/logos，不改 Moment，不保存私有思维链 |
| 启动追赶 | `Data.__aenter__()` 将没有 reflection note 的 mechanical commit 调度到有上限的后台任务 | 退出时未完成任务取消，下次启动重试 |
| MemoryConfig | `configs/memory.yml` 的 `ConfigType`，管理 window、count commit 与 reflection 参数 | 没有 `max_age` 定时提交或 witness 调度，不宣称已实现 |
| Memento CTML 控制面 | Data Ghost 的 `ghost` 虚拟 channel 提供 inspect/log/staging/show/commit/reinterpret/branches/fork/switch/reflect | 仅当前 owner，无跨 owner 写、无真 branch merge |

配置示例：

```yaml
# configs/memory.yml
detail_n: 12
summary_m: -1
auto_commit_every: 4
reflection_enabled: true
reflection_model_tag: small_fast_model
reflection_max_summary_chars: 360
reflection_max_source_chars: 12000
reflection_startup_limit: 16
```

所以当前的准确结论是：**Data 已从“可持久化主路”进一步成为“可反思、可追赶、可显式控制的
Memento Ghost”。moshi 的用户模型和权威执行进度仍是独立产品集成课题，不由本次 Data 改动自动解决。**
