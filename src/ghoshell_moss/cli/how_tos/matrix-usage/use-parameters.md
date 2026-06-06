---
title: Use Parameters
description: 在 Matrix 体系中使用 Parameter 协议做跨进程共享状态存储。declare 声明参数类型并获取 typed handle，get/set 读写 + version CAS 乐观锁，跨进程自动同步缓存。面向 app 开发者和 Ghost 开发者，帮你判断什么时候用 parameter、什么时候用其他路径。
---

# Use Parameters

## 背景

Matrix 提供了六条通讯路径。parameter 是其中**唯一持久化 + 强类型 + 跨进程自动同步**的共享状态路径：

```bash
moss codex get-interface ghoshell_moss.core.blueprint.session:Session
```

| 路径 | 持久化 | 强类型 | 变更通知 | 何时用 |
|------|--------|--------|----------|--------|
| `output` | 否 | 半 | 有 | 系统对外的展示消息 |
| `signal` | 否 | 有 | push | 驱动 Ghost 三循环 |
| `stream` | 否 | 否 | pub/sub | 高频实时流 |
| `topic` | 否 | 有 | pub/sub | 结构化事件广播 |
| `cache` | 是 (TTL) | 否 | 无 | 临时 KV, 分布式锁 |
| **parameter** | **是** | **是** | **invalidation** | **共享配置、持久状态** |

满足以下**任意一条**用 parameter：

- 多进程需要读写同一份状态，且写完了别的进程要知道
- 数据有"当前值"语义（如机器人配置、Ghost 人格参数），不是一次性事件
- 写频率低 (<1Hz)，读频率任意（内存缓存，读零 IO）
- 需要乐观锁防止多写者互相覆盖

不适合 parameter 的场景：

- 高频传感器数据 (>10Hz) → 用 stream 直接 Zenoh pub/sub
- 一次性事件通知 → 用 topic
- 仅需临时缓存 → 用 cache
- 仅当前进程内共享 → 用模块级变量即可

Parameter 协议的核心抽象：

```bash
moss codex get-interface ghoshell_moss.core.blueprint.parameter:ParameterStore
moss codex get-interface ghoshell_moss.core.blueprint.parameter:Parameter
moss codex get-interface ghoshell_moss.core.blueprint.parameter:ParameterModel
```

## 声明参数类型

每个参数类型是一个 `ParameterModel` 子类，定义 key、类型 schema 和默认值：

```python
from ghoshell_moss.core.blueprint.parameter import ParameterModel

class GhostPersona(ParameterModel):
    name: str = "Echo"
    temperature: float = 0.7

    @classmethod
    def param_name(cls) -> str:
        return "ghost_persona"

    @classmethod
    def param_default(cls) -> "GhostPersona":
        return cls()
```

`param_name()` 是默认 key（可在 declare 时覆盖），`param_default()` 是 miss 时返回的零值。

## 读写基础用法

通过 Session 获取 ParameterStore，declare 拿到 typed handle：

```python
# session = ...  # 从 Matrix 或 IoC 获取
store = session.parameters

# 声明参数 — 返回类型绑定的 handle
persona = store.declare(GhostPersona)

# 读 — 纯内存 dict lookup, 零 IO
cfg = persona.get()  # → GhostPersona(name="Echo", temperature=0.7)

# 写 — 返回新 version
new_v = persona.set(GhostPersona(name="Nova", temperature=0.9))

# 查版本
v = persona.version()  # → 0 表示从未被 set

# 声明时 override key（用于同一模型类型多实例）
alt = store.declare(GhostPersona, key="alt_persona")
```

handle 无生命周期管理——Session 关闭时 store 自动清理。

## CAS 乐观锁

多写者场景用 version CAS 防止默默覆盖：

```python
from ghoshell_moss.core.blueprint.parameter import VersionConflict

persona = store.declare(GhostPersona)

try:
    persona.set(new_value, version=3)   # 只在当前版本为 3 时写入
except VersionConflict:
    # 别人先写了 — 重读再试
    latest = persona.get()
    # ... 合并逻辑 ...
    persona.set(merged, version=persona.version())

# version=None (默认) — 强制覆盖, 不检查冲突
persona.set(new_value)
```

version=0 的特殊语义：key 不存在时 `persona.version()` 返回 0，`set(value, version=0)` 仅在 key 不存在时创建（否则抛 VersionConflict）。

## 跨进程同步

写操作自动同步到同 scope 的其他进程——缓存最终一致。无需额外配置。

**通知丢了？下次读到的是旧值，不会读到错值。**

## 示例

参考代码见 session parameter 的单元测试和集成测试。

## 常见问题

### 问题：get() 拿到的值是旧的

Parameter 的缓存是**最终一致**的。跨进程写入后，其他进程的缓存刷新有极短延迟（Zenoh 消息传输时间）。如果需要强一致性，用 `persona.version()` 轮询版本号，或用 cache 的分布式锁做写前互斥。

### 问题：declare 时报 "database is locked"

两个进程同时启动时，首个 `declare()` 触发 SQLite WAL init，后到的进程会等待 busy_timeout（默认 3s）。这是正常的，仅在 Session 初始化时发生一次。如果超时，检查是否有僵尸进程持有 SQLite 文件锁。

### 问题：不想每次传 model_type

handle 在 `declare()` 时已绑定类型。`persona.get()` 返回 `GhostPersona`，IDE 自动推导。不需要反复传 model_type。

### 问题：怎么列出已声明的参数

```python
store.declared()  # → ["ghost_persona", "robot_config", ...]
```

这只是运行时已声明的 key。环境级别的参数类型发现（如 `moss manifests parameters`）尚未实现——parameter 不走 IoC manifests，当前只能通过代码约定发现。

## 探索路径

```bash
moss codex get-interface ghoshell_moss.core.blueprint.parameter
```

关于通讯路径选择，参见 howto `matrix-usage/use-topics-and-windows` 中的协议选择表。

## 文档目标

读者按照本文档操作，应该能够：
1. 继承 `ParameterModel` 定义带 key 和 default 的参数类型
2. 通过 `session.parameters.declare(Model)` 获取类型绑定的 handle
3. 用 `handle.get()` / `handle.set(value)` 读写，用 `handle.set(value, version=N)` 做 CAS
4. 处理 `VersionConflict` 并重试
5. 在测试中找到 session parameter 的可运行参考代码
