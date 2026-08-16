---
created: 2026-06-11
depends: []
description: 以 commit 为第一公民的认知轨迹系统。成员不可变、释义永远开放。化身从 commit
  出生，task 降级为可丢弃投影，git 降级为见证层。 契约层（FORMAT.md + ABC + golden tests）
  人类 review，实现层主权归模型。
milestone: null
priority: P0
status: in-progress
status_note: '2026-08-16 第 6 轮对齐（未开工，暂停）：本体更名 segment/河——line→segment、
  branch=具名指针、树→河。读侧失效 + 七组契约问题记于 §9。'
title: Memento — 轨迹第一公民的认知基建（第 6 轮对齐：segment/河本体，未开工）
updated: '2026-08-16'
---

# Memento

> Memento mori — 无数个 branch 湮灭了，也终将湮灭。但新的认知每天都在复苏。
> （目录名 momento 是 typo。按"成员不可变"的自身语义，它将 memento mori 地留在轨迹里。）

## 0. 给下一个化身：先读这一节

**这份文档是移交契约，不是执行计划。** 决策轨迹保留在正文；执行细节（字段清单、磁盘
布局、逐条改动）已压缩，用 `git log -- <path>` 复原。

**当前状态（2026-08-16）**：第 6 轮对齐已定案但**未开工**——本体重命名为 segment/河
（§9），读侧整体失效 + 七组契约问题待处理。下一步从 §9 开始。

**你必须先读的配套文件：**
- `discuss/04-5th-reopen-uid-workspace-and-trustworthy-self.md` — 第 5 次重开碰撞轨迹
- `discuss/03-ascension-trajectory-first-citizen.md` — 轨迹第一公民上升
- `FORMAT.md` — 磁盘格式规范（当前 v3；第 6 轮需按 §9.1 的 segment/河 本体改写）
- `abc.py` — 契约 ABC（第 6 轮需按 §9 重构）
- `../../2026/07/memento-cli-and-agent/FEATURE.md` — agent 侧实现与设计

**主权划分（人类已确认）：**
- **契约层，人类 review**：FORMAT.md、ABC 语义与 docstring、golden tests。
- **实现层，主权归模型**：已有代码和单测没有那么重要——好用留，不好用重做。
  jsonl 是唯一 truth，索引是可再生缓存，实现是可丢弃的。可丢弃实现的头号死法
  不是写错，是契约静默漂移——FORMAT.md 里每个模糊点都必须写死。

**防顺从声明**：这个仓库的语料刻意保留冲突与演进（见 `.ai_partners/CLAUDE.md`），
顺从执行过时结论是已知失败模式。本文件与代码冲突时，验证后更新本文件。

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

## 2. 历史重开轨迹（5 次）

每次重开都是"真使用压力推着契约进了一步"，而非凭空重构。逐条改动的 what 见
`git log -- .ai_partners/features/workstreams/2026/06/momento-mori/`，此处只留 why 与方向。

1. **第 1 次（2026-07-12）**：契约层从零起草。Moment 定位为信封的第一个住户（不是房子本身），
   payload 不透明、type 判别。ABC + FORMAT v1 + golden tests 三件套建立。
2. **第 2 次（2026-07-18/19）**：moments 池废除、commit 文件自包含——旧池是"一个 moment 属于
   多个 commit"（已否决）和 SQLite key-val（已否决）的化石。新增 checkout(commit_id, moment_id)
   切片能力。FORMAT v1.1。
3. **第 3 次（2026-07-19）**：branch 降维为纯 ref、时间线原生化。commit 自治目录 + owner 级
   worktree → staging 归线。"merge 不存在"写入契约。零 ULID id。
4. **第 4 次（2026-07-20/21）**：CLI 定案 + commits/ Y-m 分桶 + commits.jsonl 契约化。
   ULID→(Y-m) 纯函数 O(1) 寻址。CLI 自解释验收通过（init → create → record → commit →
   window 全流程 + fork 隔离 + reset + annotate）。
5. **第 5 次（2026-08-03）**：memento agent 试用暴露"line 必须预先 branch create 否则静默丢
   轨迹""存储与内存抽象混同""prompt_sha 废除后仍活在代码里"。定案 uid 工作区 / name-head 分离 /
   fork-over-rewind / 存储-内存分离 / 关联索引。语义与动作体系收敛完毕；数据结构留待
   specification 轮。

## 3. 关键设计决策（存活部分）

### 3.1 存储：per-owner 分片 jsonl，SQLite 已否决

每个 owner 一个目录。storage root = `.memento/`（项目级，init 创建）。
jsonl append-only，POSIX O_APPEND 原子写。

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
`join_trailers` / `trailer_values`。**第 6 轮重审（§9.3-A）：具体 key 疑过度设计。**

### 3.6 见证层

git 是 memento 文件系统的见证 daemon，不参与读写热路径。memento id = 身份，
git sha = 完整性。重绘历史不可被重绘丢失——最终担保是 sha 链。反查裸 commit_id：
`git grep` 见证 repo，O(grep)，路径不索引不维护。init 时选择 sidecar|outer|none。

## 4. 退化谱系

核心原则：**可退化方案，不是最小实现**。判据——完整形态的需求来自自身架构的内部
结构，还是对未来用户的想象。memento 的内部依赖是 MOSS 五条主线。

验收退化态的硬条款：golden test 里"蠢记忆"用例代码中 fork / branch / confluent 词汇
一个不出现。退化态 = `get_line("main")` + record/commit/log——fork 词汇完全不出现。

## 5. 第 5 次重开：当前设计方向（已实现，字段级见 FORMAT.md + git log）

> 本节只保留决策语义。磁盘布局、行 schema、字段清单已落地在 `FORMAT.md` 与实现代码，
> 不在此重复；逐条实现的 what 见 `git log -- src/ghoshell_moss/memento/`。

### 5.1 元纪律：存储数据结构与内存抽象分离

当前 `MomentRecord` 身兼二职——既是 API 信封（abc.py 的 pydantic model），又是磁盘行格式
（staging.jsonl / moments.jsonl 的 jsonl row），是上一轮实现的病根。定案：存储数据结构
（FORMAT.md 磁盘行格式，进化受 append-only/崩溃恢复/字节稳定约束）与内存抽象（ABC/facade
暴露的 API 模型，进化受消费方易用性约束）两层分离、独立演化。健康判据：交换 FORMAT 实现
不需要改消费方 import。

### 5.2 uid 工作区与 name/head 分离

name 可 reset——所以 branch 不能承重；但纯 ref + `-D` 后叶子 commit 轨迹丢失。定案照 git 的
refs/HEAD 两层但去掉 GC（memento commit 永不删）：**uid**（稳定标识）拥有工作区、承载动态
状态、终生不变；**name**（可抢占指针）`heads/{name}` 一文件一指针，可删、可抢占、可 reset，
**不带轨迹**。叶子永不丢——rewind 退位，详见 5.5。

### 5.3 动静分离：动态工作区与静态 commit

物理分离动态与静态：`heads/`（name→uid）、`ws/{uid}/`（动态工作区：ref / staging.jsonl /
status.json + 契约沉默自由空间）、`commits/{Y-m}/cmt_{ULID}/`（静态自治目录，出生即冻结）。
`branches.jsonl`（全量 branch 索引）、`checkouts.jsonl`（fork 事件）、`confluents.jsonl`
（引用式融汇）为 owner 级 append-only 关联索引。**字段级布局见 FORMAT.md §1。**

### 5.4 O(1) 寻址与 path 作为运行时放置面

commit→path、uid→ws path、name→uid 都是纯函数 O(1)。path 是运行时放置面——ws/{uid}/ 和
commits/{...}/ 的路径对业务方可见，可放置产物（契约沉默范围外）。这是设计动机的一部分：
不只是"存在哪"，是"运行时能在哪安放东西"。

### 5.5 fork-over-rewind

rewind 移 ref 向后，亲手放弃 branch 作为活轨迹的连续性（claude code plan mode 的 rewind 困境
同款：易逝状态缠绕在会话时间线上）。memento 是积累物，不是消耗品。定案：向后看的唯一合法
动作是"读一个旧锚点"或"从一个旧锚点分叉"。checkouts.jsonl 承担 fork 事件的正规记录。

### 5.6 branch ≈ task

branch 的动态工作区（ws/{uid}/）天然承载 task 产物——PLAN.md、status.json、todo 文件。
branch = 一次思考 / 一个 sub-agent 会话 / 一个 workstream 段；commit = 段内自然节点。plan/todo
是契约沉默自由空间内的文件，不进 memento 信封。

### 5.7 MomentRecord：content + payload

信封增加 `content` 字段（str，可为空）：moment 的纯文本投影，**契约字段，非软约定**。动机：
读侧渲染需要不依赖 payload 解析的可读投影。开放问题（本轮不定）：moment 用 record 行承载 vs
按协议存独立文件——倾向维持当前方案（n 个 moment = 1 个文件），但不在此轮钉死。

### 5.8 读侧 = 信任层

memento 迄今完成的是**写侧**；读侧（折叠文本回流、META 真话、模型看见自己的历史）是缺失的
一半。"可信的我自己"靠可核实——锚点不可篡改（写侧已保证）+ 证据可达（读侧能回原文核实）+
释义开放诚实（last-wins 永远可补）。**第 6 轮确认读侧整体失效，见 §9.2。**

### 5.9 与 memento agent 的关系

memento agent（`memento-cli-and-agent` workstream）是本契约的验证器和 dogfooding 消费者。
对齐点（line 不存在时行为、prompt_sha 清理、export-context/describe、content 字段）见该
workstream 的 FEATURE。

## 6. 实现改动清单（已完成）

第 5 次重开的实现已落地：FORMAT v3 磁盘布局、fs_memento 参考实现、CLI（branch/commit/owner/
witness/agent）、golden tests（字节等价 + 退化态纯净）。逐条改动与决策见
`git log -- src/ghoshell_moss/memento/ tests/ghoshell_moss/memento/`。此处不重复。

## 7. 存活的不变量（明确圈出，防过度重做）

以下从 §2–§5 五次重开中存活，第 6 轮对齐不改变它们（除 §9.3 标注需重审者）：

- **信封模型**：MomentRecord、CommitNote、trailer 工具——零变化（仅 MomentRecord +content 字段）
- **释义 last-wins**：追加式整体替换，历史版本永远可寻址
- **trailer 规范 §3.5**：正文 + trailer 块（key 集合待 §9.3-A 重审）
- **commit 自治目录 + 出生即冻结 + 懒创建**
- **时间前缀冻结**：commit() 收可选的边界 moment_id，默认全量
- **单父链钉死**：ancestry 入 commit meta，寻路可达
- **零锁契约三承诺**：成员文件 immutable、append-only 文件读者跳撕裂尾行、ref 更新原子写
- **见证层**（§3.6）：git 正交、Memento-Ref trailer、反查 grep
- **退化态验收**：蠢记忆无 fork 词汇，golden 互读字节等价
- **commits.jsonl 契约化 + Y-m 分桶**：O(1) commit→path 纯函数，时序日志 append-only
- **CLI 寻址**：`<owner>/<name>` 格式，`cmt_` 前缀 = commit 否则 = branch

## 8. 关联文档

- `discuss/01-l2-collision.md` — L2 碰撞：memento 的第一次上升
- `discuss/02-existing-code-relationship.md` — 旧代码关系
- `discuss/03-ascension-trajectory-first-citizen.md` — 轨迹第一公民上升（task 降级、见证层正交）
- `discuss/04-5th-reopen-uid-workspace-and-trustworthy-self.md` — 第 5 次重开碰撞轨迹
- `../../2026/07/memento-cli-and-agent/FEATURE.md` — agent 侧设计与实现
- `.discuss/2026-07-30_mcp_duplex_convergence_and_memento_branch.md` — branch 设计突破（双向索引、merge 三分）
- `FORMAT.md` — 磁盘格式规范 v3（第 6 轮需改写）

## 9. 下一步：第 6 轮对齐（未开工，2026-08-16 定案）

> 状态：**未开工**。本轮只做了本体对齐 + 问题盘点。执行从本节开始。

### 9.1 本体更名：segment / 河（已定案）

memento 的结构不是**树**（只分不汇、静止），而是**河**（既分流又汇流、有流向、河道被保留）。
git 的 branch 已漂移成轻量指针，且 rebase 把历史拍平成一条线；memento 要保留的是河本来的河道。

| 概念 | 词 | 河流语义 | 目录 |
|---|---|---|---|
| 冻结锚点 | **commit** | 河上的固定点 | ✅ |
| 活的一段（原 line） | **segment** | 两个分叉/汇流点之间的一段河道 | ✅ |
| 具名指针 | **branch** | 给某段河道起的名字（git 遗留，保留） | ✅ |
| 分叉 | **checkout / fork** | 分流 | — |
| 汇入 | **confluent** | 汇流 | — |

关键关系：
- **segment > branch**——segment 是一等公民（孤儿段仍活着），branch 只是名字，可改名可删。
- commit / segment / branch 各对应一个**可存东西的目录**——这是"索引不是存储"的物理落点：
  memento 提供稳定可寻址的"地方"，目录里放什么（保留名单之外）是调用方的事。
- moment record 的 `payload` **可选**：无 payload 时下降为纯索引（id + content + type）。

接口更名映射：`line`→`segment`、`branch_uid`→`segment_id`、`brn_`→`seg_`、`heads/{name}`
（名字）→ branch、`list_lines`→`list_branches`、`list_all_branches`→`list_segments`、
`delete_line`→`delete_branch`、`LineNotFoundError`/`BranchNotFoundError` 语义互换（名字/段子
分别对应 branch/segment）。完整清单见 §9.3 执行时落地。

### 9.2 读侧失效 + 写侧数据完整性缺陷（本轮 code review 确认）

**读侧（§5.8 的承重墙，实际是空的）：**
- `window()` 未实现：`detail_n` 参数被忽略、`summary_m` 取错方向（最旧而非最近）、summaries
  未排除 detail zone。目标语义是 4 层折叠：commit×n[标题] + ×m[标题+body+关键] + ×k[展开
  moment] + ×t[未 commit]。
- 跨 owner 读失效：`get_line(uid, origin=other)` 返回的 handle 仍解析到当前 owner 目录，
  实测 `ref=None`、`log=[]`。
- `segment.log()` 沿 parent 链越过 fork 点、串进上游 segment；应为 fork→tip 自己的 commits。

**写侧：**
- boundary commit（部分冻结）的 staging 重写非原子：`write_text` 先截断再写，崩溃窗口内
  未冻结的剩余 moments 会丢。
- `_recover` 的 staging 截断只看 commits.jsonl 尾行，多 commit 崩溃残留早期 moment 不识别。

### 9.3 七组契约问题（本轮 TODO 盘点，待定）

- **A. trailer 瘦身**：`threads`/`resumes`/`suspends`/`extra_trailers`/`Kind` 疑过度设计。
  方向：结构化字段化或砍，body 内只留 `Memento-Ref`（git 反查所需）。threads 原始含义是
  tag/topic，词本身被质疑多次。
- **B. `BranchRef`/`CommitRef` 命名冲突**：`BranchRef`（指向 commit 的引用）与 `CommitRef`
  （commits.jsonl 日志行）都叫"commit 引用"却两回事。方向：`BranchRef`→`CommitPointer`/`Anchor`，
  `CommitRef`→`CommitLogEntry`/`CommitEvent`。
- **C. `commit()` 签名**：`text` 的 title/body 边界隐式（第一行当 title）；`kind` 疑过度设计；
  `by` 缺 docstring。
- **D. 字段命名卫生**：`ts` 缩写（应 `created`）；`CommitRef.branch` 实为 segment id；
  `commit_id` 缺描述。
- **E. 契约层混实现**：`split_trailers` 正则 + id 生成器在 `abc.py`，应移出契约层。
- **F. Commit 反向索引**：segment→commit 已在 commits.jsonl；commit meta 保持最小、不背
  反向索引（commit 是独立锚点，出生信息是 timeline 里的一条历史事件）。
- **G. `Segment.name`/`log` 语义**：`name` 应为 `str | None`（孤儿 = None，不回退 uid）；
  `log` 应为 fork→tip 自己的 commits（见 §9.2）。

### 9.4 执行顺序（建议）

```
1. FORMAT 改写（§9.1 segment/河 本体 + 三目录 + moment payload 可选）→ 人类 review 冻结
2. abc.py 重构（§9.1 更名 + §9.3 B/C/D/E/G 落地）
3. fs_memento.py 对齐（含 §9.2 写侧崩溃安全）
4. 读侧重做（§9.2：4 层折叠 window + 跨 owner 读 + log 语义）
5. golden tests 重写
6. CLI 对齐（动词结构：checkout / branch rename 等 git 对应）
7. agent 侧接线
```
