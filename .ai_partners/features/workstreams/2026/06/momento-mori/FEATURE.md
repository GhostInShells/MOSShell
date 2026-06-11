---
title: Momento Mori — 多脑并行思维的版本化对话历史
status: draft
priority: P1
created: 2026-06-11
updated: 2026-06-11
depends: []
milestone:
description: >-
  极简 git-like 对话历史数据结构：Moment (原子) → Commit (不可变快照) → Branch (生产序列) → Fork (命名空间)。
  服务多 cell 并行思考，文件即认知，SQLite + JSONL + 目录结构。
---

# Momento Mori

> Momento mori — 无数个 branch 湮灭了，也终将湮灭。但新的认知每天都在复苏。

## Motivation

MOSS 的 Matrix 体系下同时存在多个 cell 进程，每个 cell 可以并行思考。需要一套极简的
对话历史数据结构，让一个意识可以同时沿多条路径思考：

- 主对话在 host/main 上推进
- 子 cell 可以 checkout 主对话的某个 commit，分叉出旁路思考
- 旁路得出结论后，以消息（携带 branch/commit 引用）的形式传回主链路
- branch 可以消亡，moment 永存，新认知随时从旧检查点复苏

这不是版本控制。git 追踪文件变更，momento 追踪思维轨迹。merge 的本质不是内容合并，
而是洞察传递——通过消息引用完成。

## Design Index

- L2 设计碰撞记录：`discuss/01-l2-collision.md`
- 与现有 `momento.py` (blueprint) 的关系：`discuss/02-existing-code-relationship.md`

## Storage Layout (converged)

```
momento/
  moments.db                              # SQLite: 全局 moment 池
  branches/
    {fork}/                               # 命名空间 = cell address
      _branches.json                      # name → branch_id 映射
      {branch_id}/
        meta.json                         # 身份 + base_pointer
        staging.jsonl                     # 未提交的 moment_ids
        commits/
          0001.json                       # 不可变快照
          0002.json
```

## Key Decisions

### 1. 四层模型解耦

| 层 | 模型 | 存储 | 职责 |
|----|------|------|------|
| 数据 | Moment | SQLite | 全局原子，按 id 寻址 |
| 快照 | Commit | `commits/{seq}.json` | 不可变，属 Branch 不属全局池 |
| 生产 | Branch | `{branch_id}/` 目录 | Commit 有序序列，base pointer 回溯 |
| 命名 | Fork | `_branches.json` | name → branch_id 映射，owner = cell address |

Branch name 不是 Branch 的根本——name 是语义标签，branch_id 是生产序列实体。
同一生产序列可被 rename，数据不受影响。

### 2. Commit 不是全局池

Commit 属于且只属于创建它的 Branch。其他 Branch 通过 base pointer 链回溯触及，
不直接引用 Commit ID。文件系统目录列表即 commit 索引——seq 编号定义顺序。

和 git 的关键区别：git 的 branch 是 commit 的 view，commit 有 parent 形成 DAG。
Momento 的 branch 显式持有 commit 序列，commit 的 parent 关系由 branch 层面的
base pointer 表达，而非 commit 之间的 DAG 边。

### 3. Base pointer 回溯 (方案 A)

Branch B 从 Branch A 的 commit_X 分叉。B 的完整历史 = B 自己的 commits + 沿
`base_branch_id` 回溯 A，截至 commit_X。Commit 不自带 parent_commit_id——
顺序由 Branch 的目录列表提供，跨 Branch 追溯由 base pointer 提供。

选择方案 A 而非方案 B（自包含复制）的原因：Commit 全局唯一，不应在每个 Branch 里
重复列出共享部分的引用。

### 4. Staging 显式维护

staging.jsonl 记录"上次 commit 之后的 moment_ids"。commit 时冻结进快照 + 清空。

为什么不用 Moment 的 previous_id 链反推边界：多进程下 moment 可能跨越不同 cell 的
写入边界，反推需要多次 SQLite 查询。维护 staging 文件 O(1) 清空，无歧义。

### 5. 写隔离从根源杜绝并发

只有 Branch owner cell 能写自己的 staging 和 commits。其他 cell 只读。
不存在 Commit 的并发写问题，不需要锁。

### 6. Merge 作为消息引用

子 cell 得出思考结论 → 产生 Commit → 以 Message 形态发回主链路：

```
Message {
    content: summary,              // 摘要，快速扫读
    meta: { branch_id, commit_id } // 引用，可潜入对话
}
```

主链路上看到的是一条消息。区别是它背后连着另一个 Branch 的完整思考轨迹。
不需要 MergeRequest 模型——merge 就是一条携带引用的 Message。

### 7. 文件即认知

与 MOSS features 体系一致的哲学：目录即索引，文件即数据，不需要额外的元数据库。
Commit 的 seq 编号从目录列表推导，Branch 的 name 映射从 `_branches.json` 读取。
人类可以直接用 ls/cat 浏览完整历史。

## Relationship with Existing `momento.py`

现有 `ghoshell_moss.core.blueprint.momento` 定义了 `Moment`, `Reaction`, 
`MomentBranch` (ABC), `Momento` (ABC), `MomentoIndex`, `MomentoMetadata`。

`Moment` + `Reaction` 对象保持不变。`MomentBranch` 和 `Momento` 的 ABC 接口
可能需要在 Commit/Branch 概念引入后调整。`MomentoIndex` 的 `from_moment_id` /
`from_branch_id` 需要迁移到 base pointer 模型。

具体改造方案在 `discuss/02-existing-code-relationship.md` 中展开。

## Implementation Notes

- 第一期：Moment (SQLite) + Commit + Branch + Fork 的核心存储和 CRUD
- Moment 对象先收敛支持 mindflow 的当前需求（`Moment` + `Reaction` 不变）
- `compact()` (现有) vs `commit` 的关系待明确：compact 是破坏性截断 + 新 branch，
  commit 是非破坏性快照。可能共存
- 具体开发时间点由人类另行决定
