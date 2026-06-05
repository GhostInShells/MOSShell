---
created: 2026-05-29
depends:
- ghost-playground
- matrix-channel-hub
description: Session 通讯总线从纯 ephemeral 向 stateful 演进——补 KV/Journal/Cache/Lock/Actor/Future
  六种跨进程原语，统一文件存储治理，为 Ghost 和应用开发提供开箱通讯基线。
milestone: null
priority: P0
status: in-progress
status_note: "Parameter implemented — SessionParameterStore on Session.parameters, lazy init via thread pool. Actor + Future pending."
title: Session Communication Bus — 跨进程通讯基线演进
updated: '2026-06-05'
---

# Session Communication Bus

> Session 是跨进程模块的通讯基线。当前只有 ephemeral 通讯（pub/sub/topic/signal），
> 迟到者追不上任何历史状态。此 feature 把 session 从"消息总线"升级为"通讯 + 存储 + 协调"的完整基线。

## Motivation

Session 协议（`ghoshell_moss.core.blueprint.session`）已定义 `sub_stream`/`pub_stream_delta`/`output`/`topics`/`signal`，
全是无状态的——组件启动晚了，之前的数据全部丢失。同时 `Session.storage` 是裸文件操作，缺乏结构化治理。

两个核心缺口：
1. **有状态通讯原语缺失**：没有 KV、Journal、Cache、Lock、Actor、Future 等跨进程原语
2. **文件存储治理混乱**：持久化数据和临时数据没有分层，没有 session 生命周期索引

最终目标：通讯体系完备时，模型可快速开发各种应用。每个原语有 1-2 个 fewshot，开箱即用。

## Architecture Overview

```
Session (通讯总线)
├── ephemeral (zenoh)                    ← 现有，不变
│   ├── pub_stream / sub_stream
│   ├── output / output_buffer
│   ├── topics (TopicService)
│   └── signal (Mindflow)
│
├── .cabinet                             ← 新增，可复用文件模块
│   │                                     （传入 Storage 即可复用，workspace/session/GhostPlayground 通用）
│   ├── .persist                         ← workspace/sessions/<sid>/
│   │   ├── journal/                     ← 关键行为日志 (JSONL)
│   │   ├── parameters/                  ← 持久化参数
│   │   ├── resources/                   ← session 级资源
│   │   └── data/                        ← 通用持久存储 (现有 Storage)
│   └── .tmp                              ← workspace/runtime/sessions/<sid>/
│       ├── cache.db                     ← sqlite3
│       └── files/                       ← 临时文件
│
└── meta index                           ← sessions/meta.jsonl (matrix 管理)
    └── {created, closed, crashed, reclaimed} 事件

新增通讯原语:
├── ObservableStorage                    ← Storage + zenoh 变更通知
├── Journal (JSONL)                     ← 追加/tail/offset
├── KVCache                              ← diskcache + TTL + remember
├── ParameterStore                       ← 版本化 KV + 乐观锁 + watch
├── ActorQueue                           ← 单消费者队列 + 锁竞争
└── FutureManager                        ← 跨进程 Future + 审批/超时/cancel
```

## Design Index

- Session 协议定义: `ghoshell_moss.core.blueprint.session:Session`
- 文件存储抽象: `ghoshell_moss.contracts.workspace:Storage`
- Matrix 级资源路由: `ghoshell_moss.contracts.resource:ResourceRegistry`
- Future 原子内核: `ghoshell_moss.core.helpers.asyncio_utils:ThreadSafeFuture`
- 文件治理先例: `.ai_partners/features/workstreams/2026/05/ghost-playground/`

## Key Decisions

### 1. 文件模块统一为可复用结构

名称暂定 `SessionCabinet`。核心逻辑：传入 `Storage`，返回结构化文件能力（JSONL 读写、临时文件、子目录管理）。
**接受**：轻量级结构化对象，类似 Workspace 模式，workspace / session / GhostPlayground 均可复用。
**拒绝**：把 JSONL、cache 等功能直接写在 Session 上——workspace 等需要复用相同能力时重写。

### 2. 目录分层：persist 与 tmp 物理分离

```
workspace/sessions/<sid>/          ← 持久化，matrix 重启不丢
workspace/runtime/sessions/<sid>/  ← 临时，可被集中 rm -rf
```

**接受**：物理分离，清理 runtime 不影响持久数据。
**拒绝**：系统临时文件——跨平台清理策略不一致（macOS 3 天清、Linux 重启清、Windows 手动清），且路径分散 matrix 自检找不到。

### 3. Tmp 回收：matrix 启动自检

Matrix 启动时遍历 `sessions/meta.jsonl`，找到 crashed 但未 reclaimed 的 session，清理其 `runtime/sessions/<sid>/`。
Matrix 负责 meta index 的写入（它是 session 生命周期的管理者），session 自身写 `journal/`。
Meta index 是 matrix 的治理边界，session 不感知它。

### 4. diskcache 作为 Cache 和 Lock 的底层

优势：纯 Python + sqlite3 stdlib、零编译、三平台兼容、SQLite WAL mode 提供进程安全读写。
Lock 不独立暴露，基于 `cache.add` 语义被 Actor/Parameter 内部使用。
**拒绝**：手写文件 cache（Windows 原子 rename 行为有差异）、Redis（单机场景不需要额外服务进程）。

### 5. JSONL 作为 Journal 和 Meta Index 的格式

一行一个 JSON，追加写，逐行读。屏蔽实现细节，对外暴露 `append()` / `tail(offset)`。
与 stream 分工：Journal = 持久化记录（迟到者可回溯），Stream = 实时通知（"events offset=N"）。

### 6. Actor vs Future 的并发模型分离

| | Actor | Future |
|---|---|---|
| 消费模式 | 1 key → 互斥消费 → 单 handler | 1 key → 广播 → N 个观察者 |
| 并发控制 | lock（diskcache.add 原子操作） | 状态机广播（zenoh pub） |
| 状态 | pending → locked → processing → done | pending → resolved/rejected/cancelled/timed_out |
| 场景 | 任务队列、模型调用调度 | 审批流程、请求-响应 |

两者共用底层（Journal + diskcache + zenoh），并发模型完全相反，API 分开。

### 7. ParameterStore 的乐观锁

持久化参数：`get/set` 带 version counter，version 不匹配拒绝写入，防止默默覆盖。
非持久参数：diskcache 原子操作，不需要 version。

### 8. Lock 不独立暴露

基于 sqlite3 INSERT OR IGNORE 的原子操作做分布式锁，已有 SqliteCache.lock/unlock 实现。
跨平台一致，有过期机制防死锁。被 Actor 和 Parameter 内部使用，不作为 Session 的独立原语。

### 9. 底层从 diskcache 切换为 raw sqlite3

与 SqliteCache 同模式——每个进程打开同一个 WAL .db 文件。
diskcache 是额外依赖；项目已有 SqliteCache 验证了 raw sqlite3 模式的三平台兼容性。
**拒绝**：diskcache（零依赖优先，已有 SqliteCache 先例）。

### 10. Parameter：SQLite 真值 + Zenoh 轻量失效通知

两个关键决策：

**(a) 写频率决定物理通道，不是 Parameter 承担所有频率**

| 写频率 | 物理通道 | 真相源 | 零值 | 用途 |
|--------|---------|--------|------|------|
| >10Hz | Zenoh raw sub (stream) | 无，latest is truth | 无零值 | 机器人关节姿态、传感器 |
| <1Hz | SQLite + Zenoh inval | SQLite | 有零值 (default) | 配置、conversation 状态 |

高频场景不新建概念——Session.stream 已有 pub/sub/key-expr。只需加 `latest()` 语义的薄封装，读时一次反序列化，热路径零浪费。

**(b) Zenoh 信号只传 (key, version)，不传 value**

- SQLite 是唯一真相源
- 写：`UPDATE ... WHERE version = ?` → 成功 → Zenoh pub `(key, new_version)`
- 读：从 SQLite 读，可选本地内存缓存 + version 校验
- 版本策略：integer 自增（CAS 语义适合"基于前值的部分更新"）
  ULID 适合 LWW 全量覆盖，但 parameter 的写语义更接近 CAS，用整数 version
- 失效信号丢失？最多延迟读到新值，不会读到错值——最终一致

### 11. Parameter 命名对齐 ROS2

`declare_parameter` / `get_parameter` / `set_parameter` / `on_set_parameters_callback`。
降低开发者心智负担——ROS2 开发者直接理解语义。
MOSS 版本：强类型 BaseModel（非 ROS2 的基础类型 descriptor），version 级乐观锁。

### 12. 跨网协议未来独立定义

当前实现针对本地 Matrix（同机多进程）。未来云端 Matrix hub 重新定义 parameter 抽象——底层可换 etcd/consul/nats KV，实现同一个接口。
Storage 挂 S3 时，SQLite 不经过 Storage 协议读写——Storage 只提供 db 文件路径。

### 13. Parameter 不带 TTL

TTL 语义已有 Cache 承担。Parameter 是持久化状态——为 null 时返回 default，不过期。
**拒绝**：parameter 引入 TTL（Cache 做这件事，职责不重叠）。

### 14. Actor 用 Zenoh queryable，Future 跨进程协调

- **Actor**：Zenoh queryable 实现请求-响应，单消费者处理
- **Future**：跨进程异步结果追踪，支持审批/超时/cancel。底层 sqlite3 存状态 + Zenoh 状态变更通知

## Primitives Summary

已有（ephemeral，不变）：

| 原语 | 底层 | 用途 |
|------|------|------|
| pub_stream / sub_stream | zenoh | 字节流 pub/sub |
| output / output_buffer | 内存 + 回调 | 结构化消息输出 |
| topics | zenoh | 强类型广播 |
| signal | zenoh | Mindflow 感知信号 |

新增（stateful，待实现）：

| 原语 | 持久化? | 底层 | 关键 API |
|------|---------|------|----------|
| ObservableStorage | 是 | Storage + zenoh | `put` 自动发变更通知 |
| Journal | 是 | JSONL + zenoh | `append` / `tail(offset)` / `on_append` |
| KVCache | 否 | sqlite3 | `get` / `set` / `remember(key, ttl, cb)` |
| ParameterStore | 混合 | sqlite3 + zenoh | `store.declare(Model)` → `Parameter.get/set/version` ✅ |
| ActorQueue | 是 | journal + lock | `enqueue` / `dequeue` / `ack` |
| FutureManager | 否 | sqlite3 + zenoh | `create` / `next` / `resolve` / `cancel` |

## Exploration Paths

讨论过但否定的方向：

- **共享内存替代文件**：mmap 零依赖可行，但 resize 限制 + 需额外同步。当前场景不需要共享内存性能，留到具身智能体 sensor 高频通道时再评估。
- **watchdog 替代 zenoh 通知**：watchdog 基于 OS 事件性能低开销，但 zenoh 已有 pub/sub，引入 watchdog 冗余。文件存储 + zenoh 通知 = 最优组合。
- **Actor 和 Future 合并 API**：并发模型相反（互斥 vs 广播），合并导致一个 API 承担两种语义。底层共用，API 分开。

## Planned Fewshots

每个原语的目标验证场景：

| 原语 | Fewshot 1 | Fewshot 2 |
|------|-----------|-----------|
| ObservableStorage | SystemInfo 模块（写 KV，UI 实时渲染） | 配置热更新监听 |
| Journal | Ghost 关键行为日志 | 跨进程事件溯源 |
| KVCache | 模型推理结果 memoization | 环境发现快照 |
| ParameterStore | Ghost 人格参数 | 运行时配置共享 |
| ActorQueue | 模型调用队列（token 预算） | 文件处理任务分发 |
| FutureManager | 审批模块（跨进程等审批） | 模型发起的异步任务追踪 |

## Open Questions

1. **SessionCabinet 最终命名？** 待定。候选：`SessionFS`、`FileCabinet`、直接用结构化 Storage 方法。
2. **Resources session 级范围？** 是资源文件存在 session storage + registry 做索引，还是更轻量？
3. **Parameter 持久/非持久：同一 key 空间还是 flag？** 倾向 `parameter.set(key, val, persistent=True/False)`。

## Implementation Progress

### Parameter ✅ (2026-06-06, coding by deepseek-v4-pro)

实现文件：

| 层 | 文件 | 内容 |
|----|------|------|
| ABC | `core/blueprint/parameter.py` | `ParameterModel`, `Parameter[T]`, `ParameterStore`, `VersionConflict` |
| 实现 | `core/parameter/session_parameter.py` | `SessionParameterStore` — Session 驱动, 读走 dict 缓存, 写走 SQLite + Zenoh |
| Session 集成 | `core/blueprint/session.py` | 新增 `parameters` 抽象属性 |
| Session 实现 | `host/session/zenoh_session.py` | lazy property + `__aenter__` 线程池 init |
| Mock | `core/session/mock_session.py` | MockSession 补 `parameters` |

关键设计决策（实现阶段新增/修正）：

- **不走 IoC, Session lazy property**。讨论过走 provider + manifests, 结论是 ParameterStore 没有独立消费场景（需要它的人一定有 Session）。Cache 已建立此先例。
- **线程池 eager init**。`Session.__aenter__` 中 `asyncio.to_thread` 完成 SQLite WAL init + Zenoh sub, 避免首次调用 `session.parameters` 阻塞事件循环。
- **读路径纯 dict lookup**。每进程内存缓存, 跨进程一致性靠 Zenoh invalidation stream（只传 `{key, version}` 不传 value）。
- **写路径 CAS**。`version=None` force-write, `version=N` CAS, 冲突抛 `VersionConflict`。

测试：33 条（16 参数单测含跨进程 CAS + 15 原有 mock session + 2 集测 `session.parameters` 端到端）。

### 待实现

- ActorQueue / FutureManager
- ObservableStorage / Journal / KVCache

参见 [discuss/parameter-design-collision.md](discuss/parameter-design-collision.md) — 设计碰撞记录。

## Original Implementation Notes

- 此 feature 是**锚点文档**——后续 feature 验证、更新、对比，直到目标阶段性完成或舍弃
- Cabinet 的实现可参考 GhostPlayground 的模式（树形约定 + 薄封装）