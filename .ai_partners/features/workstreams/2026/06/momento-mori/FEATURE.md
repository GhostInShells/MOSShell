---
created: 2026-06-11
depends: []
description: 以 commit 为第一公民的认知轨迹系统，第 9 轮收敛为极简「锚点 + 摘要」两级存储。
  commit 是纯锚点（metadata 装 session 还原钥匙），message 单一真值在 Note，fork 引用。
  旧代码已删（2026-09-10），被删设计的完整轨迹经 git log 反查。
milestone: null
priority: P0
status: completed
status_note: '2026-09-10 第 9 轮 lean 收敛 + 旧代码删除完成：abcd.py（契约）/ _fs_memento.py（实现）/
  test_lean_memento.py（单测）落地。历史讨论已压缩（§2），复盘见 §3。'
title: Memento — 轨迹第一公民的认知基建（第 9 轮 lean 收敛：锚点 + 摘要两级）
updated: '2026-09-10'
---

# Memento

> Memento mori — 无数个 branch 湮灭了，也终将湮灭。但新的认知每天都在复苏。

## 0. 给下一个化身：先读这一节

**当前状态（2026-09-10）**：第 9 轮 lean 收敛已完成并落地。memento = 极简「锚点 + 摘要」两级认知轨迹索引。

- **契约**：`src/ghoshell_moss/memento/abcd.py`
- **实现**：`src/ghoshell_moss/memento/_fs_memento.py`（filesystem，只绑定本地 path）
- **单测**：`tests/ghoshell_moss/memento/test_lean_memento.py`

**旧代码已删**：`abc.py` / `fs_memento.py` / `FORMAT.md` / `_storage.py` / `porcelain.py` /
`witness.py` / `memento_cli.py` / `agents/*`。被删设计的完整轨迹与决策过程经
`git log -- src/ghoshell_moss/memento/ src/ghoshell_moss/agents/` 反查；本文件 §2 只留压缩后的弧线。

**主权划分不变**：契约层（abcd.py）人类 review，实现层（_fs_memento.py）模型主权。

## 1. 当前设计（第 9 轮 lean）

### 1.1 核心不变量

- commit 是「可还原一个 session 的锚点」，**不承载 moment**。moment 字节住在 agent session，
  memento 只拿 metadata 里的钥匙（约定 `session_id` + `tail`）指过去；还原归消费者（dolores）。
- **message 单一真值**：commit 不带 message，message 只住 `Note`（last-wins）；title = 首行、body = 其余（对齐 git `-m`）。
- **commit / note 拆两步**：commit 打裸锚点（快，O_APPEND）；note 旁路后补摘要（慢，后台 agent 生成）。
  依据 dolores ego：实时 ghost 的 compact 不能占主循环，慢的摘要生成前置到 commit。
- **fork 是引用**（读父支 + 子支两个 commits.jsonl 拼连续轨迹），不复制；`ForkRef = {branch_id, commit_id}` 极窄。
- **看用 seq（派生，branch 内 1-based），引用用 id（ULID 全局唯一）。**

### 1.2 数据模型（字段详见 abcd.py docstring）

| 模型 | 字段 |
|---|---|
| `CommitRef` | id, metatype, metadata（还原钥匙 opaque）, created |
| `Note` | commit_id, message（title=首行 / body=其余派生） |
| `BranchRef` | name, description, branch_id, created |
| `ForkRef` | branch_id, commit_id |
| `BranchMeta` | branch_id, name, description, metatype, metadata, fork_from, created |
| `CommitSummary` | id, message, seq（读侧投影，seq 派生不存） |
| `BranchView` | name, description, branch_id, created, commit_id, previous, history, latest, commits_total |

### 1.3 存储布局

```
{owner}/
  branches.jsonl               # 历史 branch 名单 (append-only)
  {name}.ref.json              # 当前指针 (BranchRef)
  branches/{branch_id}/
    meta.json                  # BranchMeta (metatype / metadata / fork_from)
    commits.jsonl              # 正序 append-only CommitRef (权威)
    commit_notes.jsonl         # 旁路 Note (last-wins, 可丢可重建)
```

### 1.4 API 面

- **读**：`commits` / `notes`（缓存快路径）+ `acommits` / `anotes` / `aview` / `aquery_commits`（aiofiles，IO-costly）。
- **写**：`commit` / `note` / `fork`（同步快路径）+ `acommit` / `anote` / `afork`（持锁、互锁）。
- **写门控**：`async with branch:` = FileLocker flock fast-fail，进程级文件锁，**不防单个文件准入约束**（协作约定，非安全边界）。

## 2. 历史轨迹（已压缩，详见 git log）

**弧线**：8 轮设计漂移 → 真实消费者（dolores + deadline）出现 → 1 轮收敛到 lean。

- **v3（第 5 次重开）**：commit 自治目录 + Y-m 分桶 + uid 工作区 / name 分离；实现了写侧，读侧整体失效。
- **第 7 轮（abcd.py 草稿）**：branch-as-directory 明确；被第 8 轮推翻。
- **第 8 轮（§11）**：「索引不做存储 / recap-read-agent」，moment 是可插拔 payload——未实现。
- **第 9 轮（本版）**：放弃分形目标，收敛为锚点 + 摘要两级。moment 仍不内联（钥匙指 session），故第 8 轮「索引不做存储」精神保留；segment 降级为 commit 的旁路摘要（Note）。

**完整决策过程**：`git log -- src/ghoshell_moss/memento/ .ai_partners/features/workstreams/2026/06/momento-mori/`

## 3. 复盘（2026-09-10）

> 人类工程师复盘，deepseek-v4-flash-vision-exp 记录。

1. memento 是 moss ghost 方案一直依赖的核心诉求，有几个核心目标。a) 历史 100% 可追溯。b) 可追溯历史被多级索引，按模型可理解的方式，可以由它自己用不同层级遍历查找——后者不是自动做的，而是模型自己、或用 subagent 做的。c) 用 memento 还原短期轨迹。d) 它可以基于文件系统存储，可以分发。

2. memento 是什么？在人类工程师眼里，它是各种连续 moment 的碎片化锚。人类工程师对它的视觉想象是，一个装满回忆的房间，有点像档案馆或者证物室：有各个书柜，书柜里有时间排序的格子，格子里有档案 + 各种物件存放。所以 commit 是最小单位，commit 可以拥有自己按 id 约定的目录是第一步。目录里需要有可放置的 inventories，类似证物盒。moss features 体系就是 memento 的雏形，它的见证层是 git，和 features 一模一样。原始设计是分形的，从 owner → branch → fractal → … → fractal → commit，每个 fractal 目录和 commit 目录构成分形的 memento。这样 memento 每个节点对于多方都是可以直接读的。「对多方可读」这个概念极其重要，这是协作基础。

3. memento 要能拆分并行 branch，它不是服务于 rewind 的，而是服务于调度的。可以假设一个 ghost 在多个上下文中切换，每个是一个 branch，但每个 branch 可以看到别的 branch 存在。这和人类上学、上班、在各个不同的时空场景做线性切换一样。注意，这里是线性切换。并行多分支涉及不同的 owner 语义，比如一个 ghost，躯体只有一个，不存在后台的 clone 能够同时控制躯体。所以 memento 在单 owner 上设计是 branch checkout，不同 owner 上是 parallel，但是可能共同构成一个 ghost。

4. memento 原计划不做存储。但 moss 项目落库 momento 是从 moment 开始的，由于没有现成的全功能 agent 或者 harness 可以植入 ghost，人类工程师面临困境——在做 moss 的同时还要做全功能 ghost（23/24 年都在做这类命题，比如 ghostos），两边的维护成本是相当的。被迫做存储构想，就是因为必须做极简 ghost、基于 agent 框架驱动。

5. 所以一直以来，memento 迭代被定义为背景任务，持续在旁路并行做，让模型做 owner，人类只推动。这个本身就是 memento 设计思想的一个实现——一个全套的 cli 作为 memento 唯一可验证工具，让模型自行 dogfooding。类似的另一个任务是 ghost ground。

6. memento 正式启动是通过 fable5 + opus 推动的。人类工程师的基本思路是，提出草图，模型理解，制作出原型；人类工程师提出改进，模型继续制作，在一个不扩散范围的目录里迭代。这样人类工程师就可以旁路迭代它。这个过程就是实验主路 ghost 驱动旁路 agent（人类工程师扮演 ghost），是多 owner 推动的例子。

7. 早期开发，memento 最大的问题就是模型（包括 fable5）缺乏独立品味，不能成为 owner，对人类工程师的提议不会按品味剪枝。比如人类工程师提了一个思路，就被落成了 thread + annotations，明显过于复杂。这属于小问题，memento 代码复杂度是可控的，模型完全可以独立实现。

8. 致命问题在于第五轮打磨后，模型认知无法突破了。对于人类而言，验证性的项目、实验开发方向，根据实验结果调整结构、重构原来的方案是常态。但 memento 第一次遇到重构命题（如何加入分形的 segment），代码实现就开始循环了（保留旧代码的基础上，又想要实现新代码），于是开始停滞。

9. 由于 dolores ghost 任务紧迫，人类工程师决定开始验证 memento 的可用性，将它和其它几个任务找到了一个夹缝（memento agent）推进，通过 agent 运行效果倒过来判断开发效果，这样人类工程师可以投入较少的旁路精力。

10. 转折点发生在 8 月中旬，两件事：1) deepseek 家族模型进入一场大崩溃，所有的 feature 实现都出现了严重的、致命的偏离，同时污染了迭代轨迹，已经记录到 8 月复盘。在 memento agent 体系的表现是，出现了拓扑错乱的交叉耦合——agents/memento-agent 依赖 agents/_instruction 等等，导致不重构无法推进。2) dsh 经过调研下决心作为 dolores ghost 的内核，不那么依赖自己实现一个全功能 agent 做推理架构。所以 memento 的开发被暂停，优先做 dsh 的集成。

11. dsh 集成基本概念全部做完后，开始集成 memento。人类工程师的体力预算是三个工作日（之前 3 天完成 mindflow 1.3 万行代码重构，透支过一波）。人类工程师开始自己手写第五轮后无法推进的抽象面改造，并且实现了分形版本的 memento 样貌——简单来说，用泛型代替所有的 meta 数据，做 MementoModel，支持 MementoDir(Generic[MEMENTO]) 的方式，同时可定义父子层级，叶子节点和中间节点的区别是 MementoDir(Generic[METATYPE, CHILDTYPE], ABC)，ChildType 为 None 时表示为叶子节点，大意如此，从第九版可以推导。由于体力和时间不济，人类工程师发现这个机制的复杂度无法被模型接手，而人类工程师没有更多时间去迭代——它的验收和迭代、接线都无法被当前的资源层级推动。

12. 最终决策是还原早期的极简版本（第九版实际上接近第一版），然后将所有已经生产的资产删除。删除不是为了放弃，是从当前版本代码承诺里拿掉；未来可能选择时机，复活删除代码。

**review 附注（deepseek-v4-flash-vision-exp，review by）**

以上 12 点为人类工程师复盘原话，本模型未改动其措辞。核对结果：

- 对照 `.ai_partners/stages/2026-08-v0.1.0/STAGE.md` 的 Retrospective（Phase 2 deepseek-v4 regression）与各 feature 记录逐条核查。
- 有明确「声明-交付漂移」记录的 6 个：memento、memento-agent、feature-review、voice-input-state-machine、warrant、llms-cli。
- 未归因但高度吻合的 2 个：matrix-operator（08-13 completed→in-progress 重开 + 7 条致命内核问题，当时不知是模型问题）、mcp-fusion-point（停在 08-14 未推进）。
- 结论：可确定受影响 **≥ 8**；基于 review 无法直接支持「> 8」或「所有」。
- 诊断分层（STAGE.md）：当时先判为机制问题（声明-交付漂移），后经外部报告确认为模型问题（deepseek V4-Pro-0813 官方回归，与发作时间对齐）。
