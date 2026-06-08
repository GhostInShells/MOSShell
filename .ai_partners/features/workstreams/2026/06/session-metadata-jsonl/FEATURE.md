---
created: 2026-06-02
depends:
- storage-typed-protocols
- storage-scope-governance
- session-communication-bus
description: Session 获得人类可读身份与持久化元信息。三层结构： ScopeMeta（scope 级发现文件，host 写，PID 验活）、SessionRecord（scope
  级 JSONL 索引，不可变追加）、SessionMetadata（session 级 YAML，matrix 写，含完整运行时现场）。 发现流：scope meta
  + PID check 替代 meta.yaml 存在性检查。
milestone: null
priority: P0
status: completed
title: Session Metadata & JSONL Storage — session 元信息持久化
updated: '2026-06-07'
---

# Session Metadata & JSONL Storage

> 2026-06-07 设计重对齐。原版 scope meta 隐式耦合在 meta.yaml 存在性中，
> 讨论后显式分离：scope meta 是 discovery 文件（host 创建/删除，PID 验活），
> session metadata 是 matrix 运行时现场记录。两文件生命周期不同，职责不同。
>
> 原版 SessionMetadata 无运行时现场字段——补入 matrix.scopes() 内容，去掉 status
> 字段（状态由 PID 推导，不维护标记）。

## Motivation

Session 目前无身份——每次启动生成随机 session_id，无法区分"上周调试机械臂的 session"
和"刚才测试语音的 session"。同时，scope 作为认知隔离边界，scope 内的 session 是
同一 Ghost 的连续"生命片段"，应该有索引可回溯。

多 cell 组网场景：新 cell 通过 CLI 加入时只有 `workspace + scope`，无法知道
当前 scope 是否有活着的 session、session_id 是什么、host 主节点是谁。
需要一个 scope 级发现文件。

三个问题现已解决：
1. **Storage 无追加写** → `Storage.append_model`/`read_models` 已解决
2. **Session 无人类可读身份** → 本 feature 补 ScopeMeta + SessionRecord + SessionMetadata
3. **组网发现** → scope meta 文件，PID 验活

## Philosophical Foundation

**Scope = 认知隔离，不是通讯隔离。**

虽然 ghost 和 mode 是用户可见的隔离维度，但真正的隔离边界是 session scope。
未来 ghost/mode 运行时自动生成不同 scope，物理上不同 scope 的 session 互相不可见。

这带来语义一致性：
- **Ghost 看到自己的历史**：同一 scope 下 `sessions.jsonl` 列出"我上次怎么了"
- **不同 Ghost 不可见**：不同 scope 物理隔离，JSONL 文件不同
- **Ghost/mode 不在 record 中**：scope 本身隐含了认知归属，不需要冗余字段
- **恢复语义清晰**："恢复上次" = 当前 scope 内找最近一条记录

## Data Structures

三个模型，三层职责：

### ScopeMeta — scope 级发现文件（host 创建/删除）

位置：`workspace/runtime/scopes/scope-{scope}.yml`

写者唯一：host 主进程。创建于进程锁之后、session 创建之前。
host 正常退出时删除。残留文件通过 PID 验尸。

```python
class ScopeMeta(BaseModel):
    """scope 级发现文件 — host 写，PID 验活."""
    session_id: str           # 当前 scope 的 active session — 新 cell 加入时用
    host_pid: int             # 存活验证
    host_cell_address: str    # 主节点地址
    created_at: str           # ISO 8601
```

Environment 提供读写接口：
- `restore_from_scope_meta(scope_storage)` → 找到 + PID 存活 → 设置 session_id
- `write_scope_meta(scope_storage)` → 创建/覆盖

### SessionRecord — JSONL 索引行（不可变，scope 级）

位置：`scope_storage/sessions.jsonl`，`Storage.append_model` 追加。

```python
class SessionRecord(BaseModel):
    """scope 级 JSONL 索引行 — append-only, 不可变."""
    session_id: str           # uuid
    created_at: str           # ISO 8601
    # ghost/mode 不在 record 里 — scope 隐含了认知归属
```

只追加，不修改，不删除。scope 内所有 session 的创建事实记录。
列出 scope 下有哪些 session 时，读这个文件即可——不需遍历子目录。

### SessionMetadata — YAML 详情（matrix 写，session 级）

位置：`session.storage/meta.yaml`，`Storage.read_yaml`/`write_yaml` 读写。

写者：matrix（`_is_main` 才写），session 创建后写入。
读者：所有 cell 的 matrix，通过 `Session.meta` 只读 property 访问。

```python
class SessionMetadata(BaseModel):
    """session 运行时现场记录 — matrix 写."""
    # 运行时现场（来自 matrix.scopes()）
    session_id: str           # uuid
    session_scope: str        # 认知隔离 scope
    mode_name: str            # 当前 mode
    ghost_name: str           # ghost 名，"None" 表示无 ghost
    host_cell_address: str    # 主节点 cell address
    host_pid: int             # host 进程 PID
    created_at: str           # ISO 8601
    # 人类可读可变字段
    title: str = ""
    description: str = ""
    updated_at: str = ""      # 最后修改时间

    # 无 status 字段 — "活着"由 host_pid 推导，"已关闭"由 scope meta 被删除推导，
    # "崩溃"由 scope meta 残留但 PID 不存活推导。不维护可推导的状态标记。
```

## Key Decisions

### 1. 三层结构，不同生命周期

| 层 | 生命周期 | 写者 | 格式 |
|----|---------|------|------|
| ScopeMeta | host 创建，退出删除 | host 主进程 | YAML |
| SessionRecord | 不可变追加，永久保留 | matrix (main) | JSONL |
| SessionMetadata | 创建后可变 | matrix (main) | YAML |

**接受**：三个模型，两个文件（scope meta + session meta），一个 JSONL 索引。
**拒绝**：合并 scope meta 和 session metadata——生命周期不同（删 vs 留），合并后无法独立清理。

### 2. 发现流：scope meta + PID，非 meta.yaml 存在性

原版用 `meta.yaml` 存在性判断创建/恢复。但 meta.yaml 在 session 子目录下，
新 cell 不知道 session_id 时连路径都拼不出来。scope meta 在固定位置，只依赖
`workspace + scope` 就能定位。

**启动发现流**：
1. 进程锁获取后
2. 读 `workspace/runtime/scopes/scope-{scope}.yml`
3. 文件存在 + host_pid 存活 → 恢复 session_id，加入已有 session
4. 文件不存在 OR host_pid 已死 → 创建新 session，写 scope meta
5. 非 main cell 读不到 scope meta 不拒绝启动——允许无主进程测试

**接受**：scope meta 作为 discovery 约定。
**拒绝**：遍历 session 子目录找 meta.yaml——不知道 session_id 时无法定位。

### 3. 无 status 字段

"active"/"closed"/"crashed" 是派生状态，不是本原事实：
- **active**：scope meta 存在且 host_pid 存活
- **closed**：scope meta 被 host 正常退出时删除
- **crashed**：scope meta 残留但 host_pid 不存活

维护一个 status 字段 = 在多个地方同步同一事实 = 不一致风险。
session metadata 只记录创建时的运行时现场，不做状态标记。

**接受**：状态由 PID 推导。
**拒绝**：`status: Literal["active", "closed", "crashed"]` 字段。

### 4. Ghost/mode 不在 SessionRecord 中

Scope 即认知隔离——scope 本身隐含了 ghost + mode 的归属信息。
在 record 中冗余存储违反单一事实源原则。

**接受**：record 只含 session_id + created_at。
**拒绝**：ghost_name / mode_name 字段——scope 已定义认知边界，重复存储会产生不一致风险。

### 5. 写者分离：host 写 scope meta，matrix 写 session metadata

Scope meta 是进程级发现文件，由启动流程（host 侧）在进程锁后写入。
Session metadata 是 matrix 运行时现场，由 matrix（`_is_main`）在 session
创建后写入。Session ABC 暴露 `meta` property 为只读接口。

**接受**：两个写点，不同层级。Session 不承担写入职责。
**拒绝**：Session 自己写 meta.yaml——session 不应感知自己的持久化生命周期。

### 6. 基建已就绪，不需新工具类

- `append_model` = JSONL 追加
- `read_models` = JSONL 全量读
- `read_yaml` / `write_yaml` = YAML 读写

三个模型只是 pydantic BaseModel，配合现有方法即可。不需要额外封装。

## File Path Convention

```
workspace/runtime/
  scopes/
    scope-{scope}.yml                          ← scope 级发现文件 (YAML, host 创建/删除)

  sessions/                                     ← sessions_root_storage
    scope-{scope}/                              ← scope_storage
      sessions.jsonl                             ← scope 级 session 索引 (JSONL)
      session-{session_id}/                     ← storage (单个 session)
        meta.yaml                                ← session 运行时现场 (YAML)
        ...

  sessions-tmp/                                 ← sessions_tmp_root_storage
    {scope}-{session_id}/                       ← tmp_storage
      cache.db                                   ← sqlite3 cache
```

## Relation to session-communication-bus

- `session-communication-bus` 讨论的 meta index（`sessions/meta.jsonl`，matrix 管理）
  记录 `{created, closed, crashed, reclaimed}` —— 矩阵用它做生命周期治理（tmp 回收等）。
  本 feature 的 `sessions.jsonl` 是 scope 级 session 创建索引。
- 两者可以是同一文件，也可以是不同层级的两个文件。实现时再定。

## Environment 方法

```python
class Environment:
    def restore_from_scope_meta(self, scope_storage: Storage) -> bool:
        """从 scope meta 恢复。返回 True 表示成功恢复（文件存在 + PID 存活）。"""
        ...

    def write_scope_meta(self, scope_storage: Storage) -> None:
        """写入 scope meta（host 进程锁后调用）。"""
        ...
```

## Implementation

实现路径：
1. `ScopeMeta` + `SessionRecord` + `SessionMetadata` 模型定义（`core/blueprint/session.py`）
2. `Environment.restore_from_scope_meta` / `write_scope_meta`
3. `MatrixImpl.__aenter__`：进程锁后执行 scope meta 发现，`_is_main` 时写 scope meta + session metadata
4. `Session.meta` 只读 property（`storage.read_yaml("meta", SessionMetadata)`）
5. `MatrixImpl.__aexit__`：host 正常退出时删除 scope meta
6. 单测：mock storage 验证 create/recover/append 流程

全部基于已有 Storage typed methods，零新基建。