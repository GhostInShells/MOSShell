---
title: Memento — 可折叠、可分叉、可追溯的对话历史
status: in-progress
priority: P1
created: 2026-06-11
updated: 2026-06-16
depends: []
milestone:
description: >-
  以 commit_id 为稳定锚点的对话历史数据结构。Moment (原子) → Commit (不可变快照) → Branch
  (生产序列) → Fork (命名空间)。第一版落在 core.memento，ABC 不耦合任何外部体系，
  通过 hook 屏蔽 session 事件，构造函数注入 Storage/owner。
---

# Memento

> Memento mori — 无数个 branch 湮灭了，也终将湮灭。但新的认知每天都在复苏。

## Motivation

memento 不只服务一个目标，而是同时填四个空：

1. **持久化**：当前 ghost 原型 (`Atom`) 把对话历史抱在 `_history: list[ModelMessage]` 里，
   重启即丢。一切生产化路径都过不去。
2. **跨 cell 可见性**：Matrix 多 cell 并行思考时，主 cell 需要看见子 cell 的轨迹，
   子 cell 需要 fork 主 cell 的某个检查点。当前没有任何抽象承担。
3. **Vendor 中立**：`Atom._history` 用的是 pydantic-ai 的 `ModelMessage`。memento 以
   `Moment` / `Reaction` 为载体，让不同 ghost prototype 共享同一份认知轨迹，重放、
   迁移、debug 都可行。
4. **Commit_id 作为对话稳定锚点**：当前对话系统普遍用 `messages[-N:]` 切片或
   embedding similarity 当 ad hoc anchor，都不稳定。Commit_id 是 content-addressable
   的对话锚点，模型可以说"在 commit abc123 我们决定用 X 不用 Y"，跨 session 引用
   不靠 RAG 模糊匹配，靠 stable id。

(4) 是这套设计真正的杠杆。git 在代码世界发生过的事——commit_id 一旦稳定，bisect、
blame、cherry-pick、CI hook、审计 工具链涌现——在对话系统目前是真空地带。memento
把这一原语落地。

## Four-layer Model

| 层 | 模型 | 角色 | 不变性 |
|----|------|------|--------|
| 数据 | `Moment` | 单帧关键帧 (已有，不动) | 内容更新通过新对象表示 |
| 快照 | `Commit` | 一段 staging 冻结的快照 | **不可变 (no amend)** |
| 生产 | `MementoBranch` | Commit 有序序列 + base pointer | branch_id 不可变，commits 仅追加 |
| 命名 | `Memento` (Fork) | owner-scoped 命名空间 (name → branch_id) | name 可改，branch_id 不可改 |

`MementoBranch` 是单一生产序列，owner 可写，其他人只读。

`Fork = owner` 是构造期参数（一个字符串，比如 cell address，但 ABC 不知道），不是
ABC 概念。

## 关键决策

### 1. commit_id 是 primary anchor

每个 Commit 持有：
- `id: str` — 全局稳定 (unique_id)，**对外引用都用它**
- `seq: int` — branch 内有序编号，给目录列表用

存储结构可能演进（文件 → SQLite → 远端），但 commit_id 不变。所有跨 branch、跨
session、跨进程的 commit 引用一律走 id。seq 只是"在这个 branch 当前的本地序号"，
不是身份。

### 2. Base pointer 回溯（方案 A），不复制

Branch B fork 自 Branch A 的某个 commit。B 的完整历史 = B 自己的 commits + 沿
`BasePointer{fork, branch_id, commit_id, commit_seq}` 回溯 A。Commit 不带
parent_commit_id，跨 branch 关系由 branch 层面的 base pointer 表达。

### 3. Staging 显式维护

`staging.jsonl` 记录"上次 commit 之后的 moment_ids"。commit 时冻结进快照 + 清空。
不通过 `Moment.previous.moment_id` 链反推——多 owner 写入下反推有歧义。

### 4. Owner-isolated writing 杜绝并发

只有 branch 的 owner 能写自己的 staging 和 commits。其他 owner 只读。不存在并发
写 commit，不需要锁。Moment 池是 SQLite WAL，多 owner 写入由 SQLite 兜底（写入
频次 ≤1Hz，无压力，已被 Cache / Parameter 验证）。

### 5. Merge ≡ 带引用的 Message

子 owner 思考结论 → 产生 Commit → 主链路收到一条 Message：

```
Message {
    content: commit.summary,
    additional: { 'memento.ref': {fork, branch_id, commit_id, commit_seq} }
}
```

主链路看到一条消息。背后挂一条完整思考轨迹。不需要 MergeRequest 模型。
`Memento.make_merge_message(branch_id, commit_id)` 是显式 API，调用者只需要把它
作为普通 Message 喂回主链路。

### 6. Cache 三段分层 — Prompt Cache 边界与 commit 对齐

Anthropic prompt cache 的 cache_control breakpoint 经济学：cache hit ≈ 10x 便宜、
5x 快，但任何动态内容打破 cache。memento 的 commit 边界恰好是一个自然的"应用语义
对齐 infra 边界"的锚点：

```
[cache_control] instruction (system + ghost identity)
[cache_control] history to last commit   ← 不可变快照，cache 最大化
                ↓
                current staging (这部分变化)
```

- Commit 之前的 token 序列冻结 → cache 稳定。
- Staging 在 commit 后清空 + 新 commit 重新成为 cache 边界。
- 模型可以决策"我现在 commit 是不是能省一轮 cache miss"——cache 经济进入应用层
  决策空间，而不是 framework 黑盒。

物理限制（不试图消除）：standard cache 5-min TTL，premium 1-hour TTL。memento 给的是
**决策杠杆**，不是无限优化。

### 7. 折叠语义 — git log + show

History 默认展示：最近 N 个 Moment 全量 + 之前 M 个 Commit summary + base chain 摘要。
原始 Moment 在 SQLite 池**仍可寻址**——memento channel 未来提供 `show <commit_id>`
能展开任意历史 commit 的完整 moment 序列。这是行业里没有的形态：

- 截断：忘了就是忘了（forgetful）
- 摘要替换：summary 顶掉原文（lossy + irreversible）
- RAG：embedding 检索（lossy + 失序）
- **memento：层级折叠 + commit_id 作 anchor，原文仍可寻址**

### 8. Hook 屏蔽 session 事件

memento 实现不依赖 Session / Matrix / IoC。所有"事件外溢"通过 `MementoHooks`
Protocol 暴露：

```python
class MementoHooks(Protocol):
    def on_moment_updated(self, branch_id: str, moment: Moment) -> None: ...
    def on_commit(self, branch_id: str, commit: Commit) -> None: ...
    def on_branch_created(self, meta: BranchMeta) -> None: ...
    def on_branch_switched(self, branch_id: str) -> None: ...
```

Wire 阶段（Phase 2/3）才把 hook 接到 Session 的 output / topic / parameter 上。
单测里 hook 直接是个 list collector。这让 ABC 干净到可以作为独立包剥离的程度。

## 第一版位置：`core.memento`

第一版整体落在 `ghoshell_moss.core.memento.*`，**不动 blueprint**：
- 旧 `core/blueprint/memento.py` 的注释代码暂不删（人类 review 时一次性迁移）。
- 验收 OK 后由人类用 IDE 整体回迁。

ABC 通过构造函数注入外部依赖：

```python
def new_filesystem_memento(
    *,
    moment_store: MomentStore,   # SQLite 实现注入
    branches_storage: Storage,    # 文件系统根注入
    owner: str,                   # cell address 或任何字符串 (ABC 不知道)
    hooks: MementoHooks | None = None,
) -> Memento: ...
```

任何外部体系（Session/Matrix/Ghost/IoC）都不出现在 ABC 文件里。

## Moment / Reaction — 不动

`Moment` 和 `Reaction` 经 mindflow 验证，结构稳定，本次不改造。

Owner 字段也**不**塞 Moment。SQLite store 单独维护 `owner` 列，写入由
`MementoBranch.update(moment)` 注入。Moment 保持纯净，可作为独立包剥离时无负担。

## 迭代步骤（每段都可独立验收）

```
1. abc                   core/memento/abc.py — 干净契约，零外部体系依赖
2. 底层实现              core/memento/_sqlite_moment_store.py + _fs_branch.py + _fs_memento.py
3. 单测                  tests/core/memento/ — 不依赖 Session/Cell/Ghost
   ──────── 第一版交付边界 ────────
4. session wire          MementoHooks 实装为 session.output / topic fan-out
5. ghost runtime 集成    on_articulate_exit hook → memento.current().update(moment)
6. memento channel       read / chat / commit / log / show / diff — 最终验收
   ──────── 回迁到 blueprint ────────
```

第一版（步骤 1-3）的 acceptance：
- 单 owner 完整生命周期：update → commit → fork → checkout → history。
- 多 owner readonly 边界正确：非 owner 调 update/commit 抛错。
- Persistence round-trip：写一轮，新建实例读回来等价。
- Base pointer 链回溯：A→B→C 三层 fork 的 history 正确性。
- Hook 触发：list collector 断言事件 fan-out 数量与顺序。

## Storage Layout

```
{branches_storage.root}/
  moments.db                              # SQLite WAL — 全局 Moment 池
  branches/
    {owner}/                              # owner 命名空间 (cell address or whatever)
      _branches.json                      # {name → branch_id, current: branch_id}
      {branch_id}/
        meta.json                         # BranchMeta (id, fork, name, base, ...)
        staging.jsonl                     # 未提交 moment_ids
        commits/
          0001.json                       # Commit 不可变快照
          0002.json
```

人类可以用 `ls` / `cat` 直接浏览，与 features 体系同哲学。

## Industry Note

Commit_id 作为 stable conversation anchor，是当前对话系统里缺失的原语。LangChain
ConversationSummaryBuffer 不可回引；LangGraph checkpoint 是 thread-local DAG，没有
"语义锚点 vs 存储"的分离；OpenAI Assistants thread_id 是会话级 anchor 但单条
message id 不被设计为长期锚点；mem0/letta/zep 在 episodic memory 抽象上，不是
conversation cache 层。

memento 第一版交付一个干净底座；真正展开会是 memento channel + commit-as-cache-
breakpoint 落到 Atom 之后。第一个可 demo 的 wow moment 预计是 **"模型自己 commit
自己的对话"**——`<memento:commit summary="..."/>` 作为 CTML 命令，模型自决何时打
cache 边界、何时打认知锚点。

## Open Points for L3 Review

- **Commit summary 的生成策略**：人类 commit / 模型 CTML commit / 规则触发 commit
  三选一或共存？第一版 API 不预设，`commit(summary: str)` 接受字符串，由调用者决定。
- **history() sliding window 默认参数**：当前 `detail_n=最近全量, summary_m=全部
  summary`。未来可能加 token budget 模式。第一版不上。
- **跨 fork 引用一致性**：B fork 自 A 的 commit C，若 A 后删了 commit C 怎么办？
  第一版策略：**commit 一旦写入永不删**，branch 可以被 archive 但 commits/ 目录
  保留。GC 由人类显式介入。
- **Moment 包剥离**：未来想让 LangChain / pydantic-ai 直接 adopt memento，需要
  `ghoshell_moss.message` 也能独立。第一版 ABC 刻意只依赖 pydantic + 该 message
  包，预留剥离窗口。

---

历史 design 碰撞记录保留：
- `discuss/01-l2-collision.md` — L2 推演与人类方案碰撞
- `discuss/02-existing-code-relationship.md` — 与旧 momento.py 的迁移路径
  (注：旧 ABC 已注释，第一版位置改在 core.memento，迁移待回迁)
