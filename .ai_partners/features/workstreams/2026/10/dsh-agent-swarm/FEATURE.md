---
created: 2026-10-17
depends:
- dsh-fusion
description: Ghost 主导的 dsh agent 蜂群 — 任务树治理 + D3 关系图观测 + 文件夹持久化. 是通用 agent swarm 的原型,
  不是通用层本身.
milestone: null
priority: P3
status: in-progress
status_note: Subtask index established (D28); 00-methodology records the alignment,
  01-04 are skeletons awaiting further unloading. No implementation yet.
title: DSH Agent Swarm
updated: '2026-10-28'
---

# DSH Agent Swarm

> Use `moss features set-status dsh-agent-swarm <status> -m "note"` to update state.
> Stage: 计划为 stage3 的核心 feature (2026-08-v0.1.0 是 stage2)。D28 起进入 in-progress：
> 先立子任务索引与方法论，实现尚未开始。

## Subtask Index

主文档只维护动机、决策（KD，只追加）与索引；每个子任务是同目录下的一份 `.md`。
索引第一条是方法论——它不属于任何领域，是其余四个共用的前提。

| # | 子任务 | 文档 | 关系 |
|---|---|---|---|
| 00 | 方法论：验收集群用真任务、facts 为地基、不先卸载再动手 | `00-methodology.md` | 全部的前提 |
| 01 | 集群运行时可视化界面风格（面向人类） | `01-visualization-style.md` | 独立 |
| 02 | 集群运行时设计（规划/控制/数据/呈现/容错/故障发现） | `02-runtime-design.md` | **依赖 03** |
| 03 | 默认协作体系 | `03-collaboration-system.md` | 先于 02 |
| 04 | dsh 具体版本的依赖解耦（`packages/` + 版本线） | `04-dsh-version-decoupling.md` | 独立，可穿插 |

拆分标准是**按并行阻塞**（D28, thirdgerb）：现在没有集群，多个管线（claude code 等）并行推进，
所以切的是"谁不挡谁"，不是"谁先落地"。

## Motivation

最终用户故事：人类与 MOSS the Ghost **线性交流**，建立 n 个任务；Ghost 作为这 n 个任务的
**L2 架构师 + 监管者**；人类通过一张可视化的关系图看见所有任务的运行状态与任务状态。

这是一个 **agent 蜂群 (swarm)**，不是 ghost 集群：只有一个 Ghost 在 L2，下面是大量的
执行型 agent。每个 agent 本质上是一个 dsh session。

**为什么现在只立 draft**：思路的八成细节已在人类脑中完成建模，但当前 stage 的迭代计划
不能被它打乱。先固化已对齐的边界与决策，等 stage3 再逐域推进。

## Position in the Ladder

```
attention engineering
  └─ agent swarm
       └─ dsh-agent-swarm   ← this workstream
```

- **attention engineering**：把 N 个有状态信号源压缩到人类可接受的决策工作集（例如 100+ 源 → 10 决策）。
- **agent swarm**：产生并压缩这些信号的、runtime-agnostic 的层。
- **dsh-agent-swarm**：第一个具体原型，故意不通用；它的价值是让上层命题在真实负载下被校准。

这不是一条很长的推测链：下层的 features / moss CLI / mindflow / dsh 都已稳定运行，每一级是上一级的
地板。整体阶梯规划在 2019–2020 年已完成；2026-05-07 引入 features 是其地基。原型可丢弃，不等于
这一级白做——它是通用层校准的必经实例，且 KD6 保证丢弃时状态不丢。

## Key Decisions

<!-- 2026-10-17 讨论对齐的结论; 后续推进时只追加, 不回改. -->

**KD1 — 这是 `dsh-agent-swarm`，不是通用 `agent-swarm`。**
- 名字一旦叫 agent swarm，就隐含承诺支持 dsh 之外的 agent；而通用层设计要求更高。
- 项目惯例通常是"先做通用设计，再具体化成用例"，但此处的通用层心智成本远高于单一实现。
- 因此明确：**dsh agent swarm 是 agent swarm 的原型**。通用抽象是未来独立 feature，
  不由本 workstream 顺手长出。
- **本机制未来是可丢弃的**：新的 general agent swarm 不必在 dsh agent swarm 基础上生长。
  前提是 KD6 成立——上下文与产物独立于 session 保存。原型被丢弃时，状态不丢。

**KD2 — 树负责治理，图负责看见。**
- 可视化用 D3 关系图呈现（中心 Ghost，向外第 2、3 层），但治理结构必须是树：单父、无环、可归档。
- 跨任务依赖（A 等 B）是独立边类型（类似 features 的 `depends`），不是父子关系；否则文件夹
  归属与归档会失去唯一性。

**KD3 — 两条状态轴，两个真相源。**
- **运行态**：来自 dsh session 事件，是观测，读穿，不落库。
- **任务态**：来自规划，是意图，落盘，由 Ghost（人类参与）写。
- 运行态 `idle` 永远不等于任务态 `done`。UI 把两条轴叠在同一节点上。
- **归档由任务态进入终态触发**，不由 session 停下触发。

**KD4 — 四个可分离的层。**
1. 规划层：任务记录、父子关系、任务态、负责人、依赖边。
2. 绑定层：节点 ↔ dsh session、cwd/worktree、agent preset。绑定是坐标，不是身份；
   节点 id 必须独立于 sessionId（session 可 fork、重建、重连）。
3. 观测层：复用 `DshRuntime` 的 ledger / watch 三档 / receipts，不新建第二套运行态采集。
4. 呈现层：D3 图 + 节点详情。只订阅，不计算真相。

**KD5 — 每个节点两个文件夹。**
- **node home**：治理面（任务记录、状态、绑定坐标、日志、artifacts、ground 摘要），归 MOSS 管，
  类比 `.moss/ghosts/<ghost>/`。
- **worker workspace**：被 agent 改的目录，通常是 git worktree，归 agent 管，是 session 的 cwd。
- 铁律：node home 不进 worker 的可见范围，或只读挂载——执行者不能改写自己的考核表。

**KD6 — 文件系统优先（铁律）。**
- 不让 dsh session 成为开发上下文的状态。产物是真相，文档是上下文，session 日志只是证据。
- 上下文靠 **moss 工具拉取**，不靠 prompt 推送。这绕开了 dsh "没有不进历史的动态上下文槽位"
  的硬约束，也避免 ground 注入烧 token。
- session 因此可丢弃：fork、重建、换 cluster 都只是换绑定，任务本身不动。
- 多个 agent 写同一份上下文必须有 owner 或锁；路径寻址稳定，不能依赖 sessionId。

**KD7 — 注册发现是投影，不是新真相。**
- 能力来自 manifests/nodes，任务来自文件，存活态来自 dsh ledger。
- 中央索引必须**可从这些源重建**；索引挂了不丢事实。
- liveness 与 task state 不能混进同一个 store——一个重启就会写坏任务状态。
- 能力发现 / 任务发现 / 感知管理是独立注册发现模块，集中索引管理，但保持源与索引分离。

**KD8 — 实体边界与数据流不变量。**
- **Cluster**：一个 dsh 进程族（DSH_HOME + 端口 + 版本 + 容量），是故障域与升级单位。
- **Agent space**：节点的治理文件夹（任务记录、ground、上下文），是认知边界。
- **Workspace**：产物文件夹（worktree/目录），是写入边界。
- **Task node**：规划实体，持久、可归档、可被观察。
- **不变量**：worker 之间不通过聊天通讯，只通过文件和显式契约通讯。控制从上往下
  (ghost → node → session)，观测从下往上 (session → ledger → 索引/UI)，产物落在文件系统、
  由父节点读取。横向通信若确需，也走文件或显式代码接口。
- 明确否决：LangGraph / Coze 那种绘图连线式工作流。通信链路一律纯代码交互体系。

**KD9 — dsh 集群隔离从 launcher 走。**
- 多个目录、多个 DSH_HOME、多端口、多版本，由 launcher 管理；避免有状态 session 挤在单一进程。
- 新原语 **rebind**：cluster 挂了，任务不死，重新绑一个 session。
- 必须提前处理：cluster 容量与背压、任务到 cluster 的粘性与版本钉定、token/权限按 cluster 隔离。

**KD10 — 解耦方法论：先立薄脊，再让领域长肉。**
- 五个领域不是并列：KD6 是横切铁律，KD8 是本体论，可视化 / 注册发现 / 集群隔离才是可并行的
  三个工作流。
- **薄脊**只包含：节点 id 与路径布局、两条状态的枚举、事件信封、实体词表。
- 每个领域开工前先签脊的契约；缝合检查点前置，统一性是脊的责任，不是最后一道工序。

**KD11 — 注意力交换率（压缩比）是核心指标，不是任务数。**
- 100+ 是信号源数量，10 是决策数量；体系的核心指标是二者的压缩比。
- 两层都必须亚线性：Ghost 把 O(n) 源压成 O(k) 决策给人，同时它自己的工作集只保留
  O(k') 的异常；否则只是把人的瓶颈换成 Ghost 的瓶颈。
- 人的“10”不是常数，要从行为校准（响应延迟、决定质量、打断后恢复时间），不能假设固定值。
- 接入人现有的注意力表面（终端/聊天/屏幕），不造第二个 inbox；图是透镜，不是新的通知流。

**KD12 — 互信是可校准的机制，不是态度。**
- 分级自治：observe → advise → act with approval → act autonomously，按任务类型与风险逐级放开。
- 升级契约：不可逆、跨边界、低置信、目标漂移必须到达人；其余可静默。
- 审计与金丝雀：自动批准/升级/拒绝留痕并可抽检；偶尔让一个已知错误穿过，检验监督是否还在工作。
- **放手让 L2 自驱的门槛是指标**：升级精确率/召回率、每任务人类决策分钟数、返工率、
  集成冲突率、离开后重建时间——不是“跑到了 100 个任务”。

## Scope — 五个可解耦的领域 (superseded 2026-10-28)

> **Superseded.** 下面这套五分法（2026-10-17）已被 `## Subtask Index` 的四领域取代：
> 它自己就说明"五个不是并列"（KD6 横切、KD8 本体论），本来就不是并行单元。
> 保留原文，按本 workstream 的规矩只追加、不回改。


1. **可视化界面** — 用前端 mock 子页面在零实现背景下逼出读模型；fixture 形状即未来 API 形状；
   讨论产出是接口契约而非截图；必须覆盖边缘态与上百节点的图性能。
2. **注册发现与控制面** — ghost + project level 的独立注册发现模块；能力 / 任务 / 感知集中索引。
3. **文件系统优先** — 横切铁律（KD6），不单独作为可并行模块。
4. **实体边界** — 本体论（KD8），最先钉死词表。
5. **dsh 集群隔离** — 多实例 supervisor、容量背压、版本钉定、rebind（KD9）。

## Open Questions

- "空感知管理" 的确切指代：空闲感知（谁可接活）还是空间感知（工位/目录占用）？决定调度接口形状。
- 单个 cluster 内多 session，还是每节点独立 dsh？故障域与成本需要权衡。
- 任务的 ground 注入方式：user/message（进历史）vs plugin/system prompt vs 工具拉取。
- 每个任务内部是否默认立 features 体系，还是只在任务长成跨会话工作流时才立？需避免第二套平行真相。
- 人类点击节点后允许哪些动作：只读、interrupt、approve、改任务态？
- dsh approval / user-questions 的 `agentId→session` 映射与人类/模型消息判别，尚未对活 dsh 验证。

## Not In Scope

- 通用 agent 抽象与通用 agent swarm 层（未来独立 feature，需人类与模型对齐设计）。
- LangGraph / Coze 式可视化连线编排。
- 在 dsh-fusion 内部扩展；本 feature 是 dsh-fusion 具体面之上的消费者。

## Design Index

- 上游：`workstreams/2026/08/dsh-fusion/FEATURE.md`（传输、session 坐标、控制面、注意力铁律）
- 边界判断：`dsh-fusion/research/research_2026-09-09_agent_platform_vs_generic_facade.md`
- 单轮原语：`dsh-fusion/research/research_2026-09-11_session_run_single_turn.md`

## Exploration paths

<!-- 尚未推进实现; 本 workstream 目前只有讨论对齐, 没有走过的死路. -->

## Methods

<!-- 待实现阶段补充. -->

## Implementation Notes

<!-- 待 stage3. 当前不写实现, 不排队优先级. -->