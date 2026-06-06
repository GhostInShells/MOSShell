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
status_note: "Parameter + Journal 完成。KVCache/ObservableStorage 移除。2026-06-07 决策：Zenoh queryable 暂不引入，ActorQueue 简化为 cell 级 declare (学 ROS2)，FutureManager 设计收敛 (单 Zenoh 路径 + sqlite3 真相源 + issuer/receiver 双视图)。待实现。" 
title: Session Communication Bus — 跨进程通讯基线演进
updated: '2026-06-07'
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
├── Journal (JSONL)                     ← 追加/tail/offset ✅ 由 Storage.append_model/read_models 实现
├── ParameterStore                       ← 版本化 KV + 乐观锁 + watch ✅
├── ActorQueue                           ← 单消费者队列 + 锁竞争 → 简化为 cell 级 declare (学 ROS2)
└── FutureManager                        ← 跨进程 Future: sqlite3 单一真相源 + 单 Zenoh 通知路径, issuer/receiver 双视图

移除项:
├── ObservableStorage                    ← 移除: Storage + notify 是应用层组合，非通讯原语
└── KVCache                              ← 移除: 模型推理缓存不归 MOSS 管; Cache(KV+TTL) 已覆盖 session 级缓存
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

### 15. Journal 由 Storage typed methods 实现，不配 pub 通知

`Storage.append_model` / `read_models` 已提供 JSONL 追加/读取。
OS 级 `ab` append 跨进程安全。不需要 Zenoh pub 通知——JSONL
原生协议实现者寡，pub 不是默认期待。

**接受**：Journal = Storage typed JSONL 方法即完成。
**拒绝**：额外包装一层 Journal 对象 + Zenoh 通知。

### 16. KVCache 不为 Session 通讯原语

模型推理 memoization 不是跨进程通讯基线能管的事。
`Cache` (KV + TTL + Hash + Lock) 已覆盖 session 级共享缓存需求。

**移除** KVCache。

### 17. ObservableStorage 不为 Session 通讯原语

Storage 写 + Zenoh 通知是应用层组合，不是通讯协议约束本身。
需要此模式的调用方自行组合。

**移除** ObservableStorage。未来可作为功能性高阶封装，不作为 Session 原语。

### 18. Zenoh Queryable 暂不引入为传输原语 (2026-06-07)

Zenoh 的 queryable/get 与 pub/sub 构成传输层 push/pull 对偶：
- pub/sub = push, 生产者 → 消费者
- queryable/get = pull, 消费者 → 生产者 → 消费者

三种 key 路由模式均合法：精确 key (1:1)、wildcard (1:n)、UUID 参数化 (逻辑 1:1)。
理论上是传输层缺失的拉取原语，但 **没有具体需求场景**。

仅有的设想场景是 cell 可控 API (UI 观测/调用 cell)，需求未明确。RPC 可以基于 queryable
实现，但 RPC 本身也非当前优先事项。

**暂不引入** queryable。Session 传输层保持 push-only（stream），等 cell API 需求明确后再评估。

### 19. FutureManager：单一真相源 + 单 Zenoh 通知路径 (2026-06-07)

FutureManager 是跨进程异步结果追踪。核心设计：

**SQLite 是唯一真相源**（对齐 ParameterStore 模式）。Zenoh 只传状态变更通知
（`{future_id, status, version}`），不传 value。观察者收到通知后从 SQLite 读完整结果。

**单 Zenoh 路径 `futures/notifications`**。不按 future_id 分路径——session 级并发量
是几十量级，单路径开销可忽略。和 ParameterStore 的 `parameters/invalidations` 一致。

**Issuer/Receiver 双视图**：
- Issuer: `as_issuer(id)` → 声明身份 → 拿到 create 能力 + 只读自己 future 的查询。
  声明时不做数据加载——issuer 的起点是空的。
- Receiver: `as_receiver()` → 实例化后从 SQLite 加载全量快照 + 订阅 Zenoh 通知 +
  暴露 `refresh()` 手动重载（通知丢失兜底）。不声明不存在。

**FutureHandle** 同时暴露读写。resolve/reject/cancel 对所有人开放，
Future 一旦创建就是公共协调点。`result()` 等 Zenoh 通知唤醒 + SQLite 读。

**持久化在 tmp_storage** (`tmp_storage/futures.db`)。tmp 生命周期 = session 生命周期，
保留 future 历史无问题。

一张 `futures` 表：

```sql
future_id   TEXT PRIMARY KEY,
created_by  TEXT NOT NULL,
status      TEXT NOT NULL,    -- pending|resolved|rejected|cancelled|timed_out
task_json   TEXT,
result_json TEXT,
reason      TEXT,
timeout_at  REAL,             -- 0 = 不过期
created_at  REAL NOT NULL,
updated_at  REAL NOT NULL
```

Issuer 和 Receiver 共享同一张表，查询范围不同（Issuer `WHERE created_by = ?`，
Receiver 全表）。

### 20. ActorQueue 简化为 cell 级 declare (2026-06-07)

ActorQueue 的互斥任务队列语义收敛为：**cell 级声明式能力发现，完全学习 ROS2**。

不再作为独立的 Session 通讯原语设计。运行时发现逻辑、key-based 注册、单消费者路由
都在 cell 层面定义。Session ABC 的 actor protocol todo 对齐到 cell 声明体系。

**接受**：ActorQueue 退出 session-communication-bus 范围，归入 cell 体系。
**拒绝**：在 Session 上再做一个独立的 ActorQueue 原语。

## Primitives Summary

已有（ephemeral，不变）：

| 原语 | 底层 | 用途 |
|------|------|------|
| pub_stream / sub_stream | zenoh | 字节流 pub/sub |
| output / output_buffer | 内存 + 回调 | 结构化消息输出 |
| topics | zenoh | 强类型广播 |
| signal | zenoh | Mindflow 感知信号 |

新增（stateful）：

| 原语 | 持久化? | 底层 | 关键 API | 状态 |
|------|---------|------|----------|------|
| Journal | 是 | JSONL (Storage) | `append_model` / `read_models` | ✅ Storage typed methods 已覆盖 |
| ParameterStore | 混合 | sqlite3 + zenoh | `store.declare(Model)` → `Parameter.get/set/version` | ✅ |
| ActorQueue | 是 | journal + lock | `enqueue` / `dequeue` / `ack` | 移出范围 — 归入 cell 声明体系 |
| FutureManager | 否 | sqlite3 + zenoh | `as_issuer(id).create()` / `as_receiver().list()` / `handle.result()` | pending — 设计收敛 (单路径 + issuer/receiver) |

移除项：

| 原语 | 移除原因 |
|------|---------|
| ObservableStorage | Storage + zenoh 通知是应用层组合，非通讯协议原语。需要时做功能性高阶封装即可。 |
| KVCache | 模型推理 memoization 不属于 Session 通讯总线范围。`Cache` (KV + TTL + Hash + Lock) 已覆盖 session 级共享缓存，不需要第二个缓存概念。 |

## Exploration Paths

讨论过但否定的方向：

- **共享内存替代文件**：mmap 零依赖可行，但 resize 限制 + 需额外同步。当前场景不需要共享内存性能，留到具身智能体 sensor 高频通道时再评估。
- **watchdog 替代 zenoh 通知**：watchdog 基于 OS 事件性能低开销，但 zenoh 已有 pub/sub，引入 watchdog 冗余。文件存储 + zenoh 通知 = 最优组合。
- **Actor 和 Future 合并 API**：并发模型相反（互斥 vs 广播），合并导致一个 API 承担两种语义。底层共用，API 分开。

## Planned Fewshots

每个原语的目标验证场景：

| 原语 | Fewshot 1 | Fewshot 2 |
|------|-----------|-----------|
| Journal | Ghost 关键行为日志 | 跨进程事件溯源 |
| ParameterStore | Ghost 人格参数 | 运行时配置共享 |
| ActorQueue | 模型调用队列（token 预算） | 已移出 scope → cell 声明体系 |
| FutureManager | 审批模块（跨进程等审批） | 模型发起的异步任务追踪 (G1 机器人动作结果等) |

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

- FutureManager — 设计已收敛 (Decision #19), 暂停等待优先级调整 (2026-06-07)
- ActorQueue — 移出 session-communication-bus 范围, 归入 cell 声明体系 (Decision #20)

### 讨论记录

- [discuss/parameter-design-collision.md](discuss/parameter-design-collision.md) — Parameter 设计碰撞记录
- [discuss/2026-06-07-actor-future-design.md](discuss/2026-06-07-actor-future-design.md) — Actor/Future/Queryable 阶段性设计收敛 (待写)

## Original Implementation Notes

- 此 feature 是**锚点文档**——后续 feature 验证、更新、对比，直到目标阶段性完成或舍弃
- Cabinet 的实现可参考 GhostPlayground 的模式（树形约定 + 薄封装）