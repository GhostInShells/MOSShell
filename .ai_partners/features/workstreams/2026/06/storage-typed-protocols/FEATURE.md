---
created: 2026-06-05
depends: []
description: Storage typed methods — frontmatter / JSONL / YAML model 读写直接挂在 Storage
  协议上，不建 companion 对象。
milestone: null
priority: P0
status: completed
status_note: 2026-06-05 重新设计：方法归于 Storage 协议，弃 companion 类。
title: Storage Typed Protocols
updated: '2026-06-05'
---

# Storage Typed Protocols

## Motivation

`ConfigStore` 解决静态引导配置。MCP Hub 的 server 列表、session metadata、Ghost
记忆等场景需要**运行时状态持久化**——add/remove/write 是常态，跨进程可见。
`Storage.put()/get()` 给 bytes，缺少类型安全的 BaseModel 读写。

## Key Decisions

### 1. 方法归于 Storage 协议，不建 companion 对象

之前设计 `SharedFile` 和 `ListFile` 作为独立 companion 类。讨论后确认：
类本身无状态（只持 storage + name + model_type），本质是 `Storage` 的方法组合。
直接挂在 `Storage` 协议上更简洁，避免不必要的对象层级。

### 2. append — OS 级 "a" 模式写入

`Storage.append(path, content)` 走 `open(path, 'a')` 原子追加。
JSONL 的 append 天然跨进程安全，不需要锁。

### 3. 六种 typed 方法

| 方法 | 格式 | 后缀 | 签名 |
|------|------|------|------|
| `read_model` | frontmatter | `.md` | `(name, model_type) -> tuple[T, str] \| None` |
| `write_model` | frontmatter | `.md` | `(name, obj, content="") -> None` |
| `read_models` | JSONL | `.jsonl` | `(name, model_type) -> Iterator[T]` |
| `append_model` | JSONL | `.jsonl` | `(name, item) -> None` |
| `read_yaml` | YAML | `.yml` | `(name, model_type) -> T \| None` |
| `write_yaml` | YAML | `.yml` | `(name, obj) -> None` |

`name` 是逻辑 key（不带后缀），内部自动补 `.md` / `.jsonl` / `.yml`。
如果传入的 name 已带对应后缀，跳过补；带了不匹配的后缀则报错。

YAML 序列化复用 `ghoshell_common.helpers.yaml_pretty_dump`，
并在文件头部插入 import path 注释（参照 `configs.py` 的模式）。

### 4. 全部同步 + async_ 代理

所有 typed 方法同步。`Storage.async_` 返回 `AsyncStorageProxy`，
把每个 typed 方法包一层 `asyncio.to_thread`。

唯一意义：code as prompt 防蠢。模型写 CTML 时不用判断 "这个是不是 IO"，
统一走 async 就行。

### 5. 无锁

`Workspace.lock()` 已在进程级提供文件锁。需要 read-modify-write 原子性的调用者
自己拿锁。Storage 不内置锁机制。

### 6. 无 watch

JSONL 数据量小（1MB ~ 5000-10000 条），全量读毫秒级。mtime 轮询不划算。

### 7. 被拒绝的方案

- **Companion 类 SharedFile/ListFile**：无状态 wrapper，多一层对象无收益
- **扩展 Storage 协议为 async**：破坏同步简洁性
- **内置文件锁**：锁语义在 Workspace 层，不在 Storage 层
- **纯 YAML 序列化**：丢掉 human-readable content 层

## Dependents

- `session-metadata-jsonl`（P0）— session 元信息用 frontmatter，事件日志用 JSONL
- `mcp-hub-channel`（P0）— MCP server 列表用 frontmatter 持久化

## Implementation

### Storage 协议扩展（contracts/workspace.py）

新增 `append` 抽象方法，六个 typed 方法使用默认实现（基于 get/put/append/exists + serialization）。

### LocalStorage 实现

`append` 走 `open(path, 'ab')`。

### AsyncStorageProxy（contracts/workspace.py）

轻量 proxy，对 typed 方法做 `to_thread` 包装。

### 位置

全部在 `ghoshell_moss/contracts/workspace.py` 中，与现有 Storage 定义同文件。
YAML 序列化复用 `ghoshell_common.helpers.yaml_pretty_dump`。