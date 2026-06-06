---
title: Use Cache
description: 在 Matrix 体系中使用 Cache 协议做跨进程 KV 存储、Hash map 和分布式锁。通过 session.cache 获取，支持 TTL 过期、惰性删除、跨进程并发安全。面向 app 开发者和 Ghost 开发者。
---

# Use Cache

## 背景

Matrix 提供了多条通讯路径。cache 是其中**跨进程共享 + TTL 过期 + 含分布式锁**的临时存储路径：

```bash
moss codex get-interface ghoshell_moss.contracts.cache:Cache
```

| 路径 | 持久化 | 强类型 | 变更通知 | 何时用 |
|------|--------|--------|----------|--------|
| `cache` | 半 (TTL) | 否 | 无 | 临时 KV、分布式锁、跨进程共享小数据 |
| `parameter` | 是 | 是 | invalidation | 共享配置、持久状态、需要乐观锁 |
| `topic` | 否 | 是 | pub/sub | 结构化事件广播 |
| `stream` | 否 | 否 | pub/sub | 高频实时流 |

满足以下**任意一条**用 cache：

- 多个进程需要读写同一份临时数据（如计算中间结果、状态标记）
- 需要跨进程分布式锁做互斥（如任务队列调度）
- 数据有 TTL 语义（如 token 缓存、会话临时标记）
- 需要 hash map 结构存储（一个 key 下多个字段）

不适合 cache 的场景：

- 需要强类型 schema → 用 parameter
- 需要变更通知 → 用 topic
- 需要持久化不丢 → 用 parameter（persistent 模式）
- 高频传感器数据 → 用 stream

Cache 位于 Session 上，是 Session 通讯协议的一部分：

```bash
moss codex get-interface ghoshell_moss.core.blueprint.session:Session
```

## KV 基础用法

通过 Session 获取 Cache 实例。Session 退出时 db 文件随 tmp 目录清理：

```python
# session = ...  # 从 Matrix 或 IoC 获取
cache = session.cache

# 写
cache.set("my_key", "hello")

# 读
val = cache.get("my_key")  # → "hello" 或 None

# 覆盖
cache.set("my_key", "world")

# 删除
cache.remove("my_key")  # → 1 (deleted count)
```

## TTL 过期

`exp` 参数控制过期时间（秒）。`exp=0`（默认）表示永不过期：

```python
# 60 秒后过期
cache.set("token", "abc123", exp=60)

# 永不过期
cache.set("constant", "value", exp=0)
cache.set("constant", "value")          # 同上

# 更新过期时间
cache.expire("token", exp=120)          # 续期到 120 秒
```

过期是惰性的——`get()` 时检查 TTL，过期返回 None。不会主动清理。

## Hash Map

一个 key 可以同时有 KV 值和多个 hash 成员：

```python
# 设置成员
cache.set_member("robot", "joint_count", "6")
cache.set_member("robot", "firmware", "v2.1")

# 读取成员
cache.get_member("robot", "joint_count")  # → "6"
cache.get_member("robot", "missing")      # → None

# 删除成员
cache.remove_member("robot", "firmware")  # → 1

# remove() 同时清理 KV + hash + lock
cache.remove("robot")                     # → 1
```

注意：hash map 不继承 key 的 TTL。成员独立存在，需手动清理。

## 分布式锁

### lock / unlock

```python
if cache.lock("task_queue"):
    try:
        # 临界区 — 只有一个进程能进来
        do_work()
    finally:
        cache.unlock("task_queue")
```

`overdue` 参数设置锁的超时（秒），防止死锁。`overdue=0` 永不过期：

```python
# 10 秒后自动过期，防止崩溃导致的死锁
cache.lock("task_queue", overdue=10)
```

### locked() context manager

推荐使用 `locked()`，退出时自动释放，异常安全：

```python
from ghoshell_moss.contracts.cache import Cache

try:
    with cache.locked("task_queue", overdue=10):
        do_work()
except RuntimeError:
    # 锁被其他进程持有
    pass
```

同一个 key 在同一时刻只能被一个进程锁定。其他进程获取失败返回 False（`lock()`）或抛 RuntimeError（`locked()`）。

## 跨进程安全

Cache 底层是单文件 SQLite（WAL 模式）。所有进程 connect 同一个 `.db` 文件，读写并发安全：

- 多进程同时读 — 无锁竞争
- 多进程同时写 — WAL + `busy_timeout=3000` 保证等待
- 跨进程锁仲裁 — `BEGIN IMMEDIATE` 事务保证原子性

无需额外配置。只要各进程拿到同一个 Session instance（通过 Matrix 的 session scope），就自动共享同一份 cache。

## 常见问题

### 问题：get() 返回 None 但 key 明明 set 过

检查 TTL。`exp` 参数单位是秒，`exp=1` 表示 1 秒后过期。如果确认不应该过期，确认 `exp=0` 或是够大的值。

### 问题：lock() 一直返回 False

检查是否有僵尸进程持有锁。如果锁是 `overdue=0`（永不过期）且进程崩溃未释放，锁不会自动清理。此时用 `remove(key)` 手动清理锁。建议生产环境始终设 overdue。

### 问题：cache 数据在 Session 重启后还在吗

不在。Cache 数据库在 `session.tmp_storage` 下，Session 退出时随 tmp 目录清理。需要持久化数据用 parameter。

### 问题：hash map 的 member 可以 TTL 吗

不支持。TTL 是 key 级别的（`strings` 表），member 独立于 TTL。如果需要 member 级过期，用多个 key 组织，或在应用层管理。

## 探索路径

```bash
moss codex get-interface ghoshell_moss.contracts.cache:Cache
moss codex get-source ghoshell_moss.core.cache
```

参考代码见 `tests/ghoshell_moss/core/cache/test_sqlite_cache.py`。

## 文档目标

读者按照本文档操作，应该能够：
1. 通过 `session.cache` 获取 Cache 实例并执行 set/get/expire/remove
2. 理解 `exp=0` 的语义，正确设置 TTL
3. 使用 `set_member` / `get_member` / `remove_member` 操作 hash map
4. 用 `locked()` context manager 做跨进程互斥，处理获取失败
5. 在测试文件中找到可运行的参考代码
