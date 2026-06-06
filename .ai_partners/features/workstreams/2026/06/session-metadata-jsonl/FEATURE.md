---
title: Session Metadata & JSONL Storage — session 元信息持久化与 JSONL 存储机制
status: draft
priority: P0
created: 2026-06-02
updated: 2026-06-02
depends: [storage-typed-protocols]
milestone:
description: >-
  Storage 层增加 JSONL 追加读写机制；Session 增加可运行时修改的元信息（标题、描述）；启动时
  基于持久化元信息判断 session 创建/恢复策略。
---

# Session Metadata & JSONL Storage

> Session 目前是无身份的——每次启动生成随机 session_id，没有标题、描述等人类可读的元信息。
> 同时 Storage 只有裸 bytes 读写，缺乏结构化追加写能力。此 feature 补上这两块，并为
> `session-communication-bus` 的 Journal 原语提供 JSONL 基础。

## Motivation

三个紧密相关的缺口：

1. **Storage 只有 get/put**：没有追加写（append），每次写都要读-改-写全量数据。JSONL 一行一条记录，天然适合日志、事件、索引等追加场景。`local_image.py` 已经用手工 JSONL 证明了模式可行，现在是抽出通用机制的时候。

2. **Session 没有人类可读身份**：`session_id` 是随机 uuid，在 REPL 或 TUI 中无法区分"上周那个调试机械臂的 session"和"刚才测试语音的 session"。给 session 加上 title + description 元信息，Matrix 启动时即可展示"可恢复的 session 列表"。

3. **启动时无法判断创建/恢复**：当前 `HostSessionProvider.factory()` 每次都创建新 session（或从 env 拿 session_id）。有了持久化的 session 元信息索引，启动时可以：检查 scope 下有哪些 session → 展示列表 → 决定新建还是恢复。

三个部分互为依赖：JSONL 是存储基础，SessionMeta 是数据模型，启动检查是消费场景。

## Design Index

- Session 蓝图：`ghoshell_moss.core.blueprint.session:Session`
- Session 实现：`ghoshell_moss.host.session.zenoh_session:MossSessionWithZenoh`
- Session 创建：`ghoshell_moss.host.providers.moss_session_provider:HostSessionProvider`
- Storage 契约：`ghoshell_moss.contracts.workspace:Storage`
- 已有 JSONL 先例：`ghoshell_moss.core.resources.local_image:LocalImageStorage`
- 下游消费 feature：`session-communication-bus`（Journal 原语依赖 JSONL）
- Session 存储约定：`Session.storage` / `Session.scope_storage`

## Key Decisions

### 1. JsonlFile 作为独立工具类，不侵入 Storage 接口

`Storage` 保持最小接口（get/put/remove/exists/sub_storage）。在其上构建 `JsonlFile`：

```python
class JsonlFile:
    """Storage 上的 JSONL 追加读写工具。"""
    def __init__(self, storage: Storage, file_path: str): ...
    def append(self, obj: dict | BaseModel) -> None: ...    # 追加一行
    def tail(self, n: int = 1) -> list[dict]: ...            # 读最后 n 行
    def read_all(self) -> list[dict]: ...                    # 读全部行
    def count(self) -> int: ...                              # 行数
    def find(self, **kwargs) -> list[dict]: ...              # 简单字段匹配遍历
```

位置：`ghoshell_moss.core.session.jsonl` 或 `ghoshell_moss.core.helpers.jsonl_utils`。

**接受**：薄封装，不引入 sqlite/第三方依赖，百级数据量足够。大量数据场景换 sqlite 时接口不变。
**拒绝**：把 append 方法直接加到 Storage 接口上——Storage 是通用文件抽象，JSONL 是特定格式约定；直接加会污染接口。

### 2. SessionMeta：pydantic 模型，存于 scope 级 JSONL 索引 + session 级 YAML

```python
class SessionMeta(BaseModel):
    session_id: str                              # 对应 Session.session_id
    title: str = ""                              # 人类可读标题
    description: str = ""                        # 内容描述
    status: Literal["active", "closed", "crashed"] = "active"
    created_at: str                              # ISO 8601
    updated_at: str                              # ISO 8601
```

存储两层：
- **scope 索引**：`scope_storage/sessions.jsonl`，每个 session 一行，append-only。用于列出 scope 下所有 session。
- **session 级详情**：`storage/meta.yaml`，包含完整元信息，可运行时读写。用于单个 session 的元信息管理。

**接受**：双写（JSONL 索引 + YAML 详情）。JSONL 索引让你不必扫描所有 session 子目录就能列出列表；YAML 详情提供人类可编辑的完整信息。
**拒绝**：只存 JSONL 索引——列出全部 session 时快，但单个 session 的元信息更新需要重写整个 JSONL 文件（append-only 不支持原地修改）。只存 YAML——列出 session 列表需要遍历所有子目录读 YAML，session 多了变慢。

### 3. JSONL 索引不可变追加，YAML 详情可原地修改

JSONL 索引行是 **不可变的 session 创建记录**。title/description 变更只写 YAML 详情，不更新 JSONL（append-only 无法更新已有行）。

如果需要"最新的 session 元信息"：读 `storage/meta.yaml`。
如果需要"列出 scope 下有哪些 session"：读 `scope_storage/sessions.jsonl`，然后用每行的 session_id 找到对应 storage 读 meta.yaml 获取最新 title。

**接受**：JSONL 作为 append-only 事实记录 + YAML 作为可变状态。这和 Journal（append-only）与 ParameterStore（可变 KV）的分工一致。
**拒绝**：用 SQLite 统一管理——过度设计，百级 session 不需要数据库。

### 4. Session 抽象增加 meta 相关方法

在 `Session` ABC 中增加：

```python
@property
@abstractmethod
def meta(self) -> SessionMeta:
    """当前 session 的元信息（内存态）"""
    pass

@abstractmethod
def update_meta(self, *, title: str | None = None, description: str | None = None) -> None:
    """运行时修改元信息，同时更新内存和持久化"""
    pass

@abstractmethod
def save_meta(self) -> None:
    """显式持久化当前元信息到 YAML"""
    pass
```

`MossSessionWithZenoh` 实现时，在 `__init__` 中从 `storage/meta.yaml` 加载已有 meta（如果存在），否则创建默认 meta。

### 5. 启动检查流程在 HostSessionProvider 中实现

`HostSessionProvider.factory()` 的启动逻辑：

1. 创建 `MossSessionWithZenoh` 实例（此时 session 从 `storage/meta.yaml` 加载或创建默认 meta）
2. 检查 `scope_storage/sessions.jsonl` 是否存在该 session_id 的记录
3. 如果不存在：这是新 session → `append` 到 JSONL 索引 + `save_meta()` 写 YAML
4. 如果存在且 status=active：这是恢复已有 session → 从 YAML 加载最新 meta
5. 如果存在且 status=closed/crashed：根据策略决定（默认创建新 session？还是恢复？）

**当前阶段**（scope 内单一 session 运行）：启动时自动处理，不需要交互选择。
**未来阶段**（scope 内多 session 并存）：REPL/TUI 在启动时展示 JSONL 索引中的 session 列表，让人类选择新建或恢复哪个。

### 6. 命名与文件路径约定

```
{workspace}/runtime/sessions/                 ← sessions_root_storage
  scope-{scope}/                              ← scope_storage
    sessions.jsonl                             ← scope 级 session 索引
    session-{session_id}/                     ← storage (单个 session)
      meta.yaml                                ← session 元信息详情
      ...

{workspace}/runtime/sessions-tmp/             ← sessions_tmp_root_storage
  {scope}-{session_id}/                       ← tmp_storage
```

**接受**：meta.yaml 放在 session 自己的 storage 根下，与 session 生命周期绑定。sessions.jsonl 放在 scope 级，作为 scope 内所有 session 的索引。

## Implementation Notes

### 实现顺序

1. **JsonlFile** — 纯工具类，无外部依赖，可先独立实现并单测
2. **SessionMeta + Session ABC 扩展** — 在 session blueprint 中加模型和抽象方法
3. **MossSessionWithZenoh 实现** — 在 `__init__` 中加 meta 加载逻辑，实现 `update_meta` / `save_meta`
4. **HostSessionProvider 启动检查** — 在 `factory()` 中加 JSONL 索引检查与写入

### JSONL 格式细节

- 每行一个完整 JSON object，`json.dumps(ensure_ascii=False)` + `\n`
- 写入时用 `storage.put()` 全量覆盖（当前 Storage 不支持 append 模式），追加场景下 JsonlFile 内部做 read-append-write
- 如果未来 Storage 支持流式写入，JsonlFile 可以优化为真正的 append

### 与 session-communication-bus 的关系

此 feature 是 `session-communication-bus` 的前置依赖：
- `JsonlFile` 被 Journal 原语直接复用
- `SessionMeta` 的 JSONL 索引模式被 `sessions/meta.jsonl`（matrix 管理的 meta index）参考
- `session-communication-bus` 中讨论的 Cabinet 可以封装 JsonlFile 的使用

### 单测策略

- `JsonlFile`: 用 mock Storage (in-memory dict) 测试 append/tail/read_all/count/find
- `SessionMeta`: pydantic 序列化/反序列化
- `MossSessionWithZenoh.meta`: 用 mock storage 测试加载/保存/更新流程
