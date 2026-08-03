---
created: 2026-06-11
depends: []
description: 以 commit 为第一公民的认知轨迹系统。成员不可变、释义永远开放。化身从 commit
  出生，task 降级为可丢弃投影，git 降级为见证层。 契约层（FORMAT.md + ABC + golden tests）
  人类 review，实现层主权归模型。
milestone: null
priority: P0
status: in-progress
status_note: '2026-08-03 §5 第 5 次契约重开: uid 工作区、name/head 分离、fork-over-rewind、
  存储/内存抽象分离、append-only 关联索引 (checkouts/confluents/branches.jsonl)、branch≈task、
  MomentRecord +content。 语义与动作体系收敛完毕；数据结构与磁盘格式留待 specification 轮。'
title: Memento — 轨迹第一公民的认知基建（第 5 次重开：uid 工作区、fork-over-rewind）
updated: '2026-08-03'
---

# Memento

> Memento mori — 无数个 branch 湮灭了，也终将湮灭。但新的认知每天都在复苏。
> （目录名 momento 是 typo。按"成员不可变"的自身语义，它将 memento mori 地留在轨迹里。）

## 0. 给下一个化身：先读这一节

**这份文档是移交契约，不是执行计划。** 当前版本经 5 次契约重开（§2），最后一次
是 2026-08-03 的第 5 次重开（uid 工作区 / fork-over-rewind / 存储与内存抽象分离）。

**你必须先读的配套文件：**
- `discuss/04-5th-reopen-uid-workspace-and-trustworthy-self.md` — 第 5 次重开的碰撞轨迹
- `discuss/03-ascension-trajectory-first-citizen.md` — 轨迹第一公民上升
- `FORMAT.md` — 磁盘格式规范（当前 v2；第 5 次重开后需起草 v3）
- `abc.py` — 契约 ABC（当前实现混同了存储与内存抽象；需分离）
- `../../2026/07/memento-cli-and-agent/FEATURE.md` — agent 侧实现与设计

**主权划分（人类已确认）：**
- **契约层，人类 review**：FORMAT.md、ABC 语义与 docstring、golden tests。
- **实现层，主权归模型**：已有代码和单测没有那么重要——好用留，不好用重做。
  jsonl 是唯一 truth，索引是可再生缓存，实现是可丢弃的。可丢弃实现的头号死法
  不是写错，是契约静默漂移——FORMAT.md 里每个模糊点都必须写死。

**防顺从声明**：这个仓库的语料刻意保留冲突与演进（见 `.ai_partners/CLAUDE.md`），
顺从执行过时结论是已知失败模式。本文件与代码冲突时，验证后更新本文件。

**当前状态**：第 5 次重开的语义与动作体系已收敛（§5），数据结构与磁盘格式尚未设计。
下一步：起草 FORMAT v3 → 分离存储/内存抽象 → 重写 abc.py → 重写 fs_memento.py →
重写 golden tests → 重写 CLI → agent 读侧接线。

## 1. 定位与核心不变量

memento **不是**对话持久化工具。它是 MOSS 五条主线的公共地基：
**并行思考、关键帧思考、参差思考（不同时序下并行）、任务移交、记忆**。

核心倒置——**轨迹是第一公民，结构是轨迹上的派生层**：
- **commit 是重绘的起点**，task 不是第一公民，是旁路规划的容器（可丢弃投影）。
- 轨迹 = commit 链（不可变锚点）。plan 从写下就开始腐烂，必须锚定在不可腐烂的
  东西上——执行轨迹才是 truth，task 只是投影。
- **memento 不等于 context-branch 论文**：memento 是跨会话、持久、地址化的历史组织；
  context-branch 是会话内、瞬时、操作上下文窗口的技术。差别是本体，不是程度。

**核心动机（第 5 次重开钉死）**：并行思考 + 历史可回溯可读取。
compact = 上下文生产（渲染层），commit = 锚点，压缩只是消费者，不是本体。

**跨重开存活的不变量：**
- commit 成员不可变、释义追加式 last-wins（历史诚实，意义可补）
- commit 永不删、出生即冻结、自治目录
- payload 不透明，type 做判别符
- 化身只能从 commit 出生，永不从 staging
- 单父链钉死（ancestry 冻结在 commit meta 里）
- 见证层正交：git 降级为 memento 文件系统的见证 daemon
- 退化态 golden tests：实现 A 写盘、实现 B 读回、字节等价

## 2. 历史重开轨迹

memento 契约经五次重开。每次的触发都是"真使用压力推着契约进了一步"，而非凭空重构。

**第 1 次（§13, 2026-07-12）**：契约层从零起草。Moment 定位为信封的第一个住户
（不是房子本身），payload 不透明、type 判别。ABC + FORMAT v1 + golden tests 三件套建立。

**第 2 次（§14-§15, 2026-07-18/19）**：moments 池废除，commit 文件自包含。旧池是
"一个 moment 属于多个 commit"（已否决）和 SQLite key-val（已否决）的化石；staging
持真身、commit 冻结时整体搬入。新增 checkout(commit_id, moment_id) 切片能力。
FORMAT v1.1。

**第 3 次（§16-§17, 2026-07-19）**：branch 降维为纯 ref → 时间线原生化。commit
自治目录 + owner 级 worktree → staging 归线（每条线有自己的活边）。"merge 不存在"
写入契约。branch = `branches/{name}/`（ref + staging.jsonl 小目录）。**零 ULID id**。

**第 4 次（§18-§19, 2026-07-20/21）**：CLI 定案 + commits/ Y-m 分桶 + commits.jsonl
契约化。ULID→(Y-m) 纯函数 O(1) 寻址。CLI 自解释验收通过（init → create → record →
commit → window 全流程 + fork 隔离 + reset + annotate）。

**第 5 次（§5, 2026-08-03）**：下文完整论述。触发：memento agent CLI 试用暴露了
"line 必须预先 branch create 否则静默丢轨迹""存储与内存抽象混同""prompt_sha 废除后
仍活在代码里"；人类六个不满意点 + 07-30 branch 设计突破方向（双向索引、merge 三分、
branch_id 真目录）未落地；外部锚点驱动的"可信的我自己"重新定位设计目标。

## 3. 关键设计决策（存活部分）

### 3.1 存储：per-owner 分片 jsonl，SQLite 已否决

每个 owner 一个目录。storage root = `.memento/`（项目级，init 创建）。
jsonl append-only，POSIX O_APPEND 原子写。SQLite 曾提案并被否决——原因见旧版。

### 3.2 fork 边界：化身只能从 commit 出生，永不从 staging

staging 没有 id，没有东西可指。"化身从 staging 出生" = 伪造历史（把未冻结的时刻
当成可以复用的出生点）。

### 3.3 可变性：成员冻结，释义 last-wins

moment 冻结后不可变。释义（commit note / moment threads）追加新版本，last-wins。
"一个 commit 在不同轨迹里有不同 summary 版本"是自然成立的——summary 是一次解释
行为，发生在某条线的某个时刻，属于解释者的轨迹。

### 3.4 旁路孔径：恰好两个

- **孔径一**：Matrix 消息跨 owner 读（"帮我看看你这个 commit"）。完整上下文副本。
- **孔径二**：annotate 释义追加。owner 自己的释义走 commit 目录 notes.jsonl；
  外来轨迹对该 commit 的 summary 存在它自己的空间里，按 id 引用。

### 3.5 trailer 规范

Commit body = 正文 + trailer 块。`Kind: semantic|mechanical`、`Thread: <name>`、
`Resumes: <cmt_...>`、`Suspends: <name>`、`Memento-Ref: <owner>/<cmt_...>`。
正文与 trailer 间空行分隔；trailer 行 `Key: Value`。工具：`split_trailers` /
`join_trailers` / `trailer_values`。

### 3.6 见证层

git 是 memento 文件系统的见证 daemon，不参与读写热路径。memento id = 身份，
git sha = 完整性。重绘历史不可被重绘丢失——最终担保是 sha 链。反查裸 commit_id：
`git grep` 见证 repo，O(grep)，路径不索引不维护。init 时选择 sidecar|outer|none。

## 4. 退化谱系

核心原则：**可退化方案，不是最小实现**。判据——完整形态的需求来自自身架构的内部
结构，还是对未来用户的想象。memento 的内部依赖是 MOSS 五条主线。

验收退化态的硬条款：golden test 里"蠢记忆"用例代码中 fork / branch / confluent 词汇
一个不出现。退化态 = `get_line("main")` + record/commit/log——fork 词汇完全不出现。

## 5. 第 5 次重开：当前设计方向

> 碰撞轨迹记录在 `discuss/04-5th-reopen-uid-workspace-and-trustworthy-self.md`。
> 本节是碰撞后的声明式定案——只写"设计是什么"，不重述"怎么讨论出来的"。
> 数据结构与磁盘格式在此处只述语义，字段级设计留待 FORMAT v3 与 abc.py 重写轮。

### 5.1 元纪律：存储数据结构与内存抽象分离

**问题**：当前 `MomentRecord` 身兼二职——既是 API 信封（abc.py 的 pydantic model），
又是磁盘行格式（staging.jsonl / moments.jsonl 的 jsonl row）。存储格式与 API 模型
1:1 混同，是上一轮实现的病根。

**定案**：memento 模块内部必须分离两层——
- **存储数据结构**：FORMAT.md 定义的磁盘行格式（jsonl row types）。进化受 append-only、
  崩溃恢复、字节稳定性约束。
- **内存抽象**：ABC / facade 暴露给消费方的 API 模型（Memento、Line、CommitView 等）。
  进化受消费方（agent / ghost / CLI）的易用性约束。

两层独立演化。存储格式变了，API 面可以不动（内部做投影）；API 面加字段，存储格式
可以滞后（向前兼容）。健康判据：交换 FORMAT 实现不需要改消费方 import。

### 5.2 uid 工作区与 name/head 分离

**问题**：§17 定案 branch = `branches/{name}/` 目录（ref + staging.jsonl）。
name 可 reset——所以 branch 不能承重；但纯 ref + `-D` 后叶子 commit 轨迹丢失，
需要全量遍历才能找到不活跃叶子。这是"名字可变"与"轨迹不丢"的结构冲突。

**定案**：name 与 uid 分离，照 git 的 refs/HEAD 两层但去掉 GC（memento commit 永不删）：
- **uid**（branch_id，稳定标识）：branch 的不可变身份。拥有工作区（ws/{uid}/），
  承载所有动态状态——ref、staging、status、task 文件（PLAN.md 等）。uid 从创建到
  废弃终生不变。
- **name**（可抢占的指针）：`heads/{name}` 一文件一指针，内容 = uid。glob = 活跃
  branch 列表。name 可删（`-D` 只删 head 文件）、可抢占（换指另一个 uid）、可 reset。
  **name 不带轨迹**——删 name 不丢 uid 工作区与 commit。

**后果**：叶子永不丢。rewind（移 ref 向后）退位——详见 5.5。

### 5.3 动静分离：动态工作区与静态 commit

**问题**：§17 把 staging 归了 branch（每条线有自己的活边），但 branch 没有"自由
空间"概念。PLAN.md、status、task 状态无处安放。

**定案**——物理分离动态与静态：
```
{owner}/
  meta.json                     # owner 身份卡
  branches.jsonl                # 全量 branch 索引 (append-only, 低频全搜索 API)
  heads/
    main                        # name → uid, 一行文本
    idea-x
  ws/
    {branch_uid}/               # branch 动态工作区 (随 commit 推进变化)
      ref                         # {"fork":..., "commit_id":..., "moment_id":...}
      staging.jsonl               # 活边: 未冻结的 moments
      status.json                 # 生命周期 + task 状态
      PLAN.md                     # task 化的 branch 产物 (自由文件, 契约沉默)
  commits/
    {Y-m}/cmt_{ULID}/           # 静态: §18 原样, 自治目录出生即冻结
      meta.json
      moments.jsonl
      notes.jsonl

  checkouts.jsonl               # fork 事件记录 (派生方追加)
  confluents.jsonl                  # confluent 事件记录 (引用式, commit 链不动)
```

- **ws/{uid}/ 内的契约沉默条款**（照 §17.3 #4 精神）：保留名单（ref / staging.jsonl /
  status.json）之外，契约不感知、不承诺、不禁止。业务自由放置 plan / todo / ground
  快照 / link 等文件，变动历史由见证层兜底。
- **branches.jsonl**：每条 branch 一行（uid + 当前 name + status + fork 起点），
  append-only。被删的名字也留行（status=abandoned）——全量搜索 API 的物理保证。
- **checkouts.jsonl**：每次"从某 commit 开新 branch"追加一行。由**派生方本地追加**
  （零协调、无跨 owner 写）。正向（从 A 看 B）本地顺读；反向（"谁借了我"）低频，
  走 branches.jsonl 或见证层 grep。
- **confluents.jsonl**：引用式 confluent——目标方追加"我接收了 {owner}/{branch} 的引用提交"。
  commit 的 parent 链不动（单父链钉死），confluent 是独立关联事件。"提交引用而非内容，
  消灭冲突解决问题域"（07-30 discuss 的 merge 类型 1，现更名为 confluent 融汇）。

### 5.4 O(1) 寻址与 path 作为运行时放置面

- **commit → path**：ULID→(Y-m) 纯函数，O(1)。`commit_space(commit_id) -> path`
  保留（§17.3 #4）。
- **uid → ws path**：纯函数，O(1)。
- **name → uid**：读 `heads/{name}` 文件，O(1)。
- **path 是运行时放置面**：ws/{uid}/ 和 commits/{...}/ 的路径对业务方可见，
  ghost / agent 可以在其中放置自己的产物（契约沉默范围外）。这是设计动机的一部分——
  不只是"存在哪"，是"运行时能在哪安放东西"。

### 5.5 fork-over-rewind

**问题**：rewind 移 ref 向后。commit 不丢，但**branch 作为活轨迹的连续性**被亲手
放弃——前向推进、进行中决策、未冻结 staging 全部陪葬。这是所有 harness 的通病
（claude code plan mode 的 rewind 困境是同款：易逝状态缠绕在会话时间线上）。
memento 是积累物，不是消耗品——rewind 不应是一等操作。

**定案**：向后看的唯一合法动作是"读一个旧锚点"或"从一个旧锚点分叉"。
- 想"回到过去换个思路" → 从旧 commit **开新 branch**（新 uid、新工作区）。
- 旧 branch 保留（或标 abandoned）。
- 想继续用同一个 name → 删旧 name、建新 name 指新 uid——name 可抢占，旧 uid
  工作区和 commit 轨迹一条不丢。
- CLI `branch reset`（rewind 动词）删除或降级为 `branch create --from` 的别名。
- checkouts.jsonl 承担 fork 事件的正规记录——每次分叉追加一行。

### 5.6 branch ≈ task

§9.3 of cli-agent FEATURE 的"branch ≈ task"纪律从文档层提进契约层：branch 的
动态工作区（ws/{uid}/）天然承载 task 产物——PLAN.md、status.json、todo 文件。
branch = 一次思考 / 一个 sub-agent 会话 / 一个 workstream 段；commit = 段内自然
节点；branch 摘要 = task summary 的自然存在。

plan/todo 是契约沉默自由空间内的文件，不进 memento 信封——memento 只承诺它们的
存在位置（ws/{uid}/ 下）和见证（git 拍下变动）。

### 5.7 MomentRecord：content + payload

MomentRecord 信封增加 `content` 字段（str，可为空）：moment 的纯文本投影。
**契约字段，非软约定**（§13.6 曾定为"软约定非契约"，第 5 次重开提级）。

动机：memento 不能只返回不可读的 json。CLI `branch window`、commit show、读侧
渲染都需要一个不依赖 payload 解析的可读投影。content 由记录方（agent / runner）
在 record 时填入。

开放问题（本轮不定）：moment 用 record 行承载 vs 按协议存独立文件。当前设计是
一个 commit 一个 moments.jsonl（n 个 moment = 1 个文件）；独立文件的代价是一次
commit 读取变成 n 个文件句柄。倾向维持当前方案，但不在此轮钉死。

### 5.8 读侧 = 信任层

memento 迄今为止完成的是**写侧**（record → commit → staging → log/window）。
读侧（§12.6 的 (b) 阶段：折叠文本回流、META 真话、模型看见自己的历史）是缺失的
一半。第 5 次重开将读侧从"迭代路径里的下一步"升级为"信任层的承重墙"。

原因："可信的我自己"不靠锚点存活自动获得，靠可核实——① 锚点不可篡改（写侧已保证）；
② 证据可达（读侧让模型能回到原文核实，而非只信摘要）；③ 释义开放诚实（last-wins
永远可补）。没有读侧，"可携带的我"只是故事；有读侧，"我"才是可核实的自我。

读侧的交付物：折叠窗口渲染（summary cursor + detail cursor 双游标，§12.1）、
read_commit 工具（按 index 检索，支持分级展开 L0-L3）、META 指令的真话（"你有
memento 历史，这里是摘要"而非"each session starts fresh"的撒谎）。

### 5.9 与 memento agent 的关系

memento agent（`memento-cli-and-agent` workstream）是本契约的验证器和 dogfooding
消费者。第 5 次重开后，agent 侧的对齐点：
- **line 不存在时的行为**：不再静默丢轨迹（`except Exception: return`）。要么
  自动创建 line（退化态 main），要么显式报错——agent invoke 不应无声失败。
- **prompt_sha 删除**：§13.2 人类裁决废除，代码里仍有残留——清理。
- **export-context / describe**：实现（当前 NotImplementedError）。读侧接线后
  这两个方法成为 agent 视角的上下文导出与线摘要。
- **content 字段**：agent invoke 时在 MomentRecord 填入 final answer 的纯文本。

## 6. 实现改动清单

以下按依赖序排列。**数据结构设计（字段级）留待 FORMAT v3 起草轮，此处只列语义动作。**

### 6.1 契约层

| 项 | 文件 | 动作 |
|---|---|---|
| 存储/内存分离 | abc.py | 新增 storage schema 模块（jsonl row types），与 API models 分开。API models（Memento, Line, CommitView 等）不变或小幅调整；row types 私有。 |
| MomentRecord +content | abc.py | 信封加 `content: str` 字段，默认空串。 |
| BranchMeta 引入 uid | abc.py | 新增 `BranchMeta`（uid + status + fork_ref + created）。branch 的生命周期标识从 name 移到 uid。 |
| Line protocol | abc.py | 签名调整：`get_line(uid)` 返回 Line handle；`create_line(name, from_ref)` 创建 uid + head 文件。 |
| Memento facade | abc.py | 新增：`list_branches()` → 全量搜索（读 branches.jsonl）；`active_branches()` → glob heads/。删除或降级 `reset_line`。 |
| FORMAT v3 | FORMAT.md | 磁盘布局重写：ws/{uid}/ + heads/ + branches.jsonl + checkouts.jsonl + confluents.jsonl。§14–§18 的存活部分合并。 |

### 6.2 存储层

| 项 | 文件 | 动作 |
|---|---|---|
| 全量重写 | fs_memento.py | 按新布局重写。存储层只操作 row types（不 import API models）。API facade 做 row→model 投影。授权丢弃旧实现。 |
| 关联索引 | fs_memento.py | checkouts.jsonl / confluents.jsonl / branches.jsonl 的 append-only 写入。读路径分离（正向 O(1) / 反向低频）。 |
| 崩溃恢复 | fs_memento.py | 精化恢复判据：commits.jsonl 尾行 commit_id 校验 + staging 截断（§18.2 精神保留）。新布局下恢复面扩大（ws/ + heads/ + * 索引文件），逐项定义恢复规则进 FORMAT v3。 |

### 6.3 CLI

| 项 | 命令 | 动作 |
|---|---|---|
| fork-over-rewind | `branch reset` | 删除或降级为 `branch create --from` 别名。 |
| uid 暴露 | `branch list/log/staging` | 输出含 uid（短格式），name 只作展示标签。 |
| 寻址统一 | 全部 branch/commit 命令 | agent 命令（--owner/--branch）与 branch/commit 命令（`<owner/name>` 位置参数）统一为同一种寻址方案。 |
| 新命令 | `branch checkout` | 从 commit 开新 branch（fork 的用户面动词）。= `branch create --from` 的语义别名。 |
| agent line 创建 | `agent invoke` | line 不存在时自动创建（退化态 main）或显式报错——删除静默丢轨迹的 `except Exception: return`。 |

### 6.4 Agent（memento-cli-and-agent workstream）

| 项 | 动作 |
|---|---|
| prompt_sha 清理 | 从 CLI metadata 和 impl payload 中删除。 |
| export-context / describe | 实现（读侧接线的前提）。 |
| content 字段填入 | invoke 后在 MomentRecord.content 写入 final answer 纯文本。 |
| 读侧接线 | (b) 阶段：折叠窗口渲染 → META 真话 → agent 看见自己的历史。依赖本契约层的 read_commit / window 就位。 |

### 6.5 测试

| 项 | 动作 |
|---|---|
| golden tests | 按新布局重写。退化态（蠢记忆无 fork 词汇）照旧。字节等价条款照旧（实现 A 写盘、实现 B 读回）。 |
| 新能力测试 | fork-over-rewind（rewind 拒绝 / fork 新建）、uid 生命周期（创建→活跃→abandoned→name 抢占）、关联索引 append-only 完整性、content 字段往返。 |
| 崩溃恢复测试 | 覆盖新布局下的恢复面（ws/ staging 残留、heads/ 与 ws/ 不一致、索引文件尾行撕裂）。 |

### 6.6 执行顺序

```
1. FORMAT v3 起草 → 人类 review 冻结
2. abc.py 存储/内存分离 + 新增模型 (BranchMeta, row types)
3. fs_memento.py 重写 (以 FORMAT v3 + abc 新模型为准)
4. golden tests 重写
5. CLI 对齐
6. agent 侧对齐 (memento-cli-and-agent workstream, 至少 (b) 读侧)
```

第 1 步 FORMAT v3 之前，数据结构设计（字段级）需单独一轮；本节只述语义，不抢 FORMAT 的职责。

## 7. 存活的不变量（明确圈出，防过度重做）

以下从 §14–§19 四次重开中存活，第 5 次重开不改变它们：

- **信封模型**：MomentRecord、CommitNote、trailer 工具——零变化（仅 MomentRecord +content 字段）
- **释义 last-wins**：追加式整体替换，历史版本永远可寻址
- **trailer 规范 §6**：正文 + trailer 块，`split_trailers` / `join_trailers` / `trailer_values`
- **commit 自治目录 + 出生即冻结 + 懒创建**（§16.2 #2/#3）
- **时间前缀冻结**（§16.2 #3，§16.5 #1）：commit() 收可选的边界 moment_id，默认全量
- **单父链钉死**（§16.2 #5）：ancestry 入 commit meta，寻路可达
- **零锁契约三承诺**（§16.2 #6）：成员文件 immutable、append-only 文件读者跳撕裂尾行、
  ref 更新原子写
- **见证层**（§6/§9）：git 正交、Memento-Ref trailer、反查 grep
- **退化态验收**：蠢记忆无 fork 词汇，golden 互读字节等价
- **commits.jsonl 契约化 + Y-m 分桶**（§18.1/§18.2）：O(1) commit→path 纯函数，
  时序日志 append-only
- **CLI 寻址**：`<owner>/<name>` 格式，`cmt_` 前缀 = commit 否则 = branch（uid 引入后
  name 只是展示标签，uid 是主键）

## 8. 关联文档

- `discuss/01-l2-collision.md` — L2 碰撞：memento 的第一次上升
- `discuss/02-existing-code-relationship.md` — 旧代码关系
- `discuss/03-ascension-trajectory-first-citizen.md` — 轨迹第一公民上升（task 降级、见证层正交）
- `discuss/04-5th-reopen-uid-workspace-and-trustworthy-self.md` — 第 5 次重开碰撞轨迹
- `../../2026/07/memento-cli-and-agent/FEATURE.md` — agent 侧设计与实现
- `.discuss/2026-07-30_mcp_duplex_convergence_and_memento_branch.md` — branch 设计突破（双向索引、merge 三分）
- `FORMAT.md` — 磁盘格式规范 v2（待起草 v3）
- `abc.py` — 契约 ABC（待存储/内存分离）
