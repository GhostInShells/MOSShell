# 2026-06-11 — 与现有 `momento.py` 的关系

## 上下文

`ghoshell_moss.core.blueprint.momento` 是已有的 momento 抽象。L2 碰撞结束后，
需要理清现有代码哪些保留、哪些需要改造、哪些是新增。

## 保留：Moment + Reaction

`Moment` 和 `Reaction` 的设计经过 mindflow 验证，结构稳定，不改造。

```
Moment
  id: str
  previous: Reaction | None      # 隐式链表，previous.moment_id 是父指针
  perspectives: dict[str, list[Message]]
  compacted_perspectives: list[Message] | None
  percepts: list[Message]
  hint: str
  created: AwareDatetime

Reaction
  moment_id: str                 # 上一轮 Moment id
  logos: str                     # 模型输出的 logos
  messages: list[Message]        # 躯体反馈
  stop_reason: str
```

Moment 通过 `previous.moment_id` 形成隐式链表。这个链表在新模型中仍然是
moment 之间的连接方式——staging 里的 moment_ids 可以通过 previous_id 链验证
连续性。

## 可改造：MomentoIndex → Fork + Branch

`MomentoIndex` 当前承载的概念混合了 Fork 和 Branch：

| MomentoIndex 字段 | 新模型映射 |
|-------------------|-----------|
| `name` | `_branches.json` 中的 key |
| `branch_id` | `Branch.id` |
| `session_id` | 保留（session scope） |
| `root_id` | 保留（整个 momento tree 的 root） |
| `from_branch_id` | `Branch.base_branch_id` |
| `from_moment_id` | 应由 `Branch.base_commit_seq` + 追溯还原 |

关键变化：`from_moment_id` 不再直接存。新模型的分叉点是 commit（seq 编号），
不是 moment。要找到对应 moment，遍历 source branch 的 commits/ 到 base_commit_seq。

## 可改造：MomentoMetadata

`MomentoMetadata` 是 Branch 级别的可变属性：

| MomentoMetadata 字段 | 新模型映射 |
|----------------------|-----------|
| `title` | `Branch.meta.title` |
| `description` | `Branch.meta.description` |
| `recap` | 前情提要，fork 时写入新 Branch 的 `meta.recap` |
| `summary` | 当前最新 commit 的 summary（可从 commits/ 目录读） |
| `updated` | `Branch.meta.updated` |

其中 `summary` 不应再是独立字段——它是"最新 commit 的 summary"，从 commits/
目录最后一份文件读取。

## 需新增：Commit

`Commit` 是新概念，现有代码完全没有：

```python
class Commit(BaseModel):
    id: str
    summary: str
    moment_ids: list[str]            # 显式有序列表
    created: AwareDatetime
```

存储为 `commits/{seq}.json`。不可变，不 amend。

## 需调整：MomentBranch (ABC)

`MomentBranch` 是 Branch 的接口抽象。新模型引入后，其接口契约需要调整：

| 现有方法 | 调整 |
|---------|------|
| `update(moment)` | 保留 —— INSERT moments.db + APPEND staging.jsonl |
| `update_meta(meta)` | 保留 —— 写 meta.json |
| `moments(reverse, limit)` | 语义变化 —— 不是从内存 list 取，是从 staging + commits 组合 |
| `fork(moment_id, ...)` | 参数变化 —— `moment_id` → `commit_seq`（或从 head commit 分叉） |
| `compact(moment_id, recap)` | **待定** —— compact 是破坏性截断，commit 是非破坏性。可能保留为独立操作 |

新增方法：
- `commit(summary)` —— 冻结 staging → commits/{seq}.json，清空 staging
- `commits()` —— 列出当前 Branch 的所有 commit（含 base 链回溯）
- `context(detail_n, summary_m)` —— sliding window 上下文窗口

## 需调整：Momento (ABC)

`Momento` 是跨 Branch 的存储管理。调整点：

| 现有方法 | 调整 |
|---------|------|
| `main()` | 保留 —— 返回当前 session 的 main Branch |
| `history(reverse, limit)` | 语义变化 —— 返回 `_branches.json` 中的分支列表 + latest commit summary |
| `get_branch(branch_id, readonly)` | 参数变化 —— 需要 `fork` (cell address) 参数做权限检查 |
| `switch(branch)` | 保留 —— 切换 main branch 指向 |

新增方法：
- `checkout(fork, branch_name, base_fork, base_branch_id, base_commit_seq)` —— fork 新 Branch
- `list_branches(fork)` —— 读取 `_branches.json`，展示面板

## 迁移策略

第一期：Moment 不变，MomentoIndex 和 MomentoMetadata 维持现有用法，
新增 Commit 写入而不改 MomentBranch 接口。旧接口和新存储共存。

第二期：改造 MomentBranch 和 Momento 的 ABC，统一到新模型。旧 `compact()`
如果确认可被 commit 替代，废弃掉。

第三期：清理 MomentoIndex 和 MomentoMetadata 中的冗余字段。
