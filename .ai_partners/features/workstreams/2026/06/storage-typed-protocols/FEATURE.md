---
created: 2026-06-05
depends: []
description: Storage 类型化协议扩展 — SharedFile[BaseModel] 和 ListFile[BaseModel] 作为 Storage
  的一等 companion 类。
milestone: null
priority: P0
status: in-progress
status_note: SharedFile + ListFile 设计完成。companion 类，frontmatter serialization，不侵入
  Storage。
title: Storage Typed Protocols
updated: '2026-06-05'
---

# Storage Typed Protocols

## Motivation

`ConfigStore` 解决静态引导配置（启动读一次，缓存在内存，运行时不变）。MCP Hub 的 server 列表、
session metadata、Ghost 记忆等场景需要**运行时状态持久化**——add/remove/write 是常态，跨进程可见。
`Storage.put()/get()` 给 bytes，但缺少类型安全的 BaseModel 读写协议。

这是第三个撞上同一问题的场景（MCP Hub、session-metadata-jsonl、Ghost 记忆），抽象时机已到。

## Key Decisions

### 1. Companion 类，不侵入 Storage 协议

`Storage` 保持 minimal contract（bytes in/out）。`SharedFile` 和 `ListFile` 是 contracts 包里的独立类，
构造时接收 `Storage` + `name` + `model_type`。

- `moss codex contracts` 可发现
- `Storage` 协议不受影响
- 心智模型：`SharedFile(storage, "config", MyModel)`

### 2. SharedFile[BaseModel]

一个强类型对象的持久化。序列化标准为 **frontmatter**（YAML metadata + markdown content）。

API:
```python
shared = SharedFile[T](storage, name, model_type)
obj, content = shared.read()   # tuple[T, str] | None
shared.write(obj, content="")  # 加锁写
shared.modify(fn)              # read-modify-write 原子
```

选择 frontmatter 而非纯 YAML：meta 是类型化的，content 是人类/模型可读的注释。
MCP Hub config、Ghost 配置等场景——模型能看到并编辑正文中的使用说明。

### 3. ListFile[BaseModel]

append-only JSONL 持久化。适合 event log、session record、audit trail。

API:
```python
lst = ListFile[T](storage, name, model_type)
lst.append(item)                     # 追加一行
await lst.aread()                    # async 惰性读取，返回 AsyncIterator[T]
lst.watch(callback)                  # 可选：监听变化，默认走 stat 轮询
```

- JSONL：append-only，惰性读取避免大文件 OOM
- 写时加锁（复用 `workspace.lock()`）
- watchdog 是可选能力，默认用 `os.stat` mtime 轮询。高性能场景换 OS 原生事件

### 4. 位置

`ghoshell_moss/contracts/storage_protocols.py`，与 `workspace.py`（Storage 定义）同级。
companion 类 + Storage → 合约组合。

### 5. 被拒绝的方案

- **扩展 Storage 协议**：破坏 minimal contract，所有 Storage 实现都要改。
- **ConfigType/ConfigStore 复用**：ConfigStore 假设静态、有缓存，不适合动态运行时状态。
- **纯 YAML 序列化**：丢掉了人类可读的 content 层，frontmatter 就是为"meta + comment"设计的。

## Dependents

- `session-metadata-jsonl`（P0）— session 元信息用 SharedFile，事件日志用 ListFile
- `mcp-hub-channel`（P0）— MCP server 列表用 SharedFile 持久化

## Implementation Notes

- `ListFile` 的 async read 需要包装线程 I/O，用 `asyncio.to_thread` 或 `run_in_executor`
- 文件锁复用 `workspace.lock(name)` — 已经是进程级文件锁
- frontmatter 依赖已在项目中使用（`mode.py`, `app.py`）