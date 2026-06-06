---
title: Session Metadata & JSONL Storage — session 元信息持久化
status: draft
priority: P0
created: 2026-06-02
updated: 2026-06-06
depends:
  - storage-typed-protocols
  - storage-scope-governance
  - session-communication-bus
milestone:
description: >-
  Session 获得人类可读身份与持久化元信息。两个数据结构：SessionRecord（scope 级 JSONL
  索引，不可变追加）和 SessionMetadata（session 级 YAML，可变）。Scope 即认知隔离边界。
  基建已就绪（Storage typed methods），对齐决策后暂不实现，等待 Actor/Future 完成后启动。
---

# Session Metadata & JSONL Storage

> 2026-06-06 设计重对齐。原版计划 JsonlFile 工具类 + HostSessionProvider 启动检查。
> 基建演进后重新评估：Storage typed methods 已覆盖 JSONL 读写，scope 语义已明确为
> 认知隔离，矩阵负责 session 生命周期治理。此版本收敛为两个数据模型 + 最小写入逻辑。

## Motivation

Session 目前无身份——每次启动生成随机 session_id，无法区分"上周调试机械臂的 session"
和"刚才测试语音的 session"。同时，scope 作为认知隔离边界，scope 内的 session 是
同一 Ghost 的连续"生命片段"，应该有索引可回溯。

三个问题现已部分解决：
1. **Storage 无追加写** → `Storage.append_model`/`read_models` 已解决
2. **Session 无人类可读身份** → 本 feature 补 SessionRecord + SessionMetadata
3. **启动时无法判断创建/恢复** → `meta.yaml` 存在即恢复，不存在即新建

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

两个模型，职责分离：

### SessionRecord — JSONL 索引行（不可变，scope 级）

位置：`scope_storage/sessions.jsonl`，`Storage.append_model` 追加。

```python
class SessionRecord(BaseModel):
    """scope 级 JSONL 索引行 — append-only, 不可变."""
    session_id: str          # uuid
    created_at: str          # ISO 8601
    # ghost/mode 不在 record 里 — scope 隐含了认知归属
```

只追加，不修改，不删除。scope 内所有 session 的创建事实记录。
列出 scope 下有哪些 session 时，读这个文件即可——不需遍历子目录。

### SessionMetadata — YAML 详情（可变，session 级）

位置：`session.storage/meta.yaml`，`Storage.read_yaml`/`write_yaml` 读写。

```python
class SessionMetadata(BaseModel):
    """session 自己存储空间里的可变元信息."""
    title: str = ""
    description: str = ""
    updated_at: str = ""       # ISO 8601, 最后修改时间
    status: Literal["active", "closed", "crashed"] = "active"
```

`meta.yaml` 存在即代表 session 已创建。矩阵主节点在 session 实例化时：
- `meta.yaml` 不存在 → 创建新 session → `write_yaml` + `append_model` 写 JSONL
- `meta.yaml` 存在 → 恢复已有 session → `read_yaml` 加载

未来可在此文件追加更多可变字段（Ghost 人格快照、最后对话摘要等），
但 SessionRecord 保持不变（只记录创建事实）。

## Key Decisions

### 1. 两个结构，不可变与可变分离

SessionRecord 是事实——创建了就记录，永不修改。SessionMetadata 是状态——
title 变了就改写。这和 Journal（append-only）与 ParameterStore（可变 KV）
的分工一致。

**接受**：两个模型，不同存储格式（JSONL vs YAML），不同写语义（append vs overwrite）。
**拒绝**：合并为一个模型——不可变索引和可变详情语义矛盾，合并后更新 title 需要
重写 JSONL 整文件（append-only 不支持原地修改）。

### 2. Ghost/mode 不在 SessionRecord 中

Scope 即认知隔离——scope 本身隐含了 ghost + mode 的归属信息。
在 record 中冗余存储违反单一事实源原则。

**接受**：record 只含 session_id + created_at。
**拒绝**：ghost_name / mode_name 字段——scope 已定义认知边界，重复存储会产生
不一致风险。

### 3. 矩阵主节点执行写入，非 Session 自身

Session 是通讯总线，不是自己的生命周期管理者。meta.yaml 的创建/更新
由矩阵（HostSessionProvider 或等价节点）在 session 实例化前后完成。

**接受**：矩阵写 meta.yaml + append_model。Session ABC 暴露 `meta` property
为只读接口（从 meta.yaml 加载），不承担写入职责。
**拒绝**：Session 自己写 meta.yaml——session 不应感知自己的持久化生命周期。

### 4. 基建已就绪，不需新工具类

原版计划的 `JsonlFile` 被 Storage typed methods 替代：
- `append_model` = JSONL 追加
- `read_models` = JSONL 全量读
- `read_yaml` / `write_yaml` = YAML 读写

SessionRecord 和 SessionMetadata 只是两个 pydantic 模型，配合现有方法即可。
不需要额外封装。

### 5. 启动判断：meta.yaml 存在性

`Session.storage`（`scope_storage/sub/session-{id}`）目录由 `sub_storage` 自动创建，
但 `meta.yaml` 文件只在矩阵写入后才存在。

- `storage/meta.yaml` 不存在 → 新 session → 矩阵写 meta.yaml + append sessions.jsonl
- `storage/meta.yaml` 存在 → 恢复 → `read_yaml` 加载，继续运行

简单、无竞态（同一 session_id 只被一个矩阵节点管理）。

## File Path Convention

```
workspace/runtime/sessions/                  ← sessions_root_storage
  scope-{scope}/                             ← scope_storage
    sessions.jsonl                            ← scope 级 session 索引 (JSONL)
    session-{session_id}/                    ← storage (单个 session)
      meta.yaml                               ← session 元信息详情 (YAML)
      ...

workspace/runtime/sessions-tmp/              ← sessions_tmp_root_storage
  {scope}-{session_id}/                      ← tmp_storage
    cache.db                                  ← sqlite3 cache
```

## Relation to session-communication-bus

- `session-communication-bus` 讨论的 meta index（`sessions/meta.jsonl`，matrix 管理）
  是本 feature `sessions.jsonl` 的上层视角。meta.jsonl 记录 `{created, closed, crashed,
  reclaimed}` —— 矩阵用它做生命周期治理（tmp 回收等）。
- 本 feature 的 `sessions.jsonl` 是 scope 级 session 创建索引，人类/模型可读，
  用于 "列出这个 scope 下的 session"。
- 两者可以是同一文件（matrix 写 created 事件 = SessionRecord append），
  也可以是不同层级的两个文件。实现时再定。

## Implementation (暂不启动)

等待 `session-communication-bus` 的 Actor + Future 完成后启动。

实现路径：
1. `SessionRecord` + `SessionMetadata` 模型定义（`core/blueprint/session.py` 或独立文件）
2. Session ABC 补 `meta: SessionMetadata` 只读 property
3. 矩阵在 session 实例化时执行写入逻辑（`HostSessionProvider` 或等价处）
4. 单测：mock storage 验证 create/recover/append 流程

全部基于已有 Storage typed methods，零新基建。
