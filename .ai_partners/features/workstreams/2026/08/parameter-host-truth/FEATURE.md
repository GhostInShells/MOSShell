---
title: Parameter Host Truth
status: draft
priority: P1
created: 2026-08-12
updated: 2026-08-12
depends:
  - session-communication-bus
milestone:
description: >-
  Parameter 存储从"SQLite 文件真相 + Zenoh 失效信号"重定向为"host 为唯一真相 +
  广播(query)". 修跨网络前值丢失, 补 on_change 变更回调, cache 走拉逻辑.
---

# Parameter Host Truth

> Use `moss features set-status parameter-host-truth <status> -m "note"` to update state.
> 本 feature 把 parameter 设计从 session-communication-bus (2026-05, completed) 中捞出独立成篇——
> 它原属该 feature 的 D10/D12。trigger 来源: voice-input-state-machine 2026-08-12 会话的
> VoiceNodeRuntimeTopic 生命周期判定 (依赖前值 → parameter)。

## Motivation

SessionParameterStore 的当前设计 (session-communication-bus D10): **SQLite 是唯一真相源,
Zenoh 只传 (key, version) 失效信号, 不传 value**. 读路径本地 dict 缓存, 写路径 SQLite CAS +
Zenoh invalidation.

这个设计隐含一个前提, 现在破了:

> `session.py:382` — "所有 cell 指向 tmp_storage 下同一个 sqlite db 文件".

SQLite ground truth **只在所有 cell 共享同一文件系统时成立** (同机多进程)。跨网络 (云端
matrix hub / 不在同一 project) 时, 各 cell 有各自的 tmp_storage → 收到 Zenoh 失效信号后,
本地 SQLite 里没有那个 key → 回退 param_default → **前值丢失**。

而状态机当前状态恰恰是需要"前值"的——消费者必须读到"自己启动之前的值"。触发场景:

- ghost 世界模型 (KD9): 中途接入必须立刻知道"我正在收音", 不是"等下一事件"。
- 半双工门控: listener 在 ghost 已开始说话后接入, 必须立刻知道"在说", 否则收音喂 ASR 就是
  TTS 回声 (无 AEC 场景)。
- TUI: 启动即渲染当前状态。

另一个缺口: **parameter 接口没有变更回调**。docstring 声称对齐 ROS2 "declare → get/set →
on-change", 但 ABC (`core/blueprint/parameter.py`) 和实现 (`SessionParameterStore`) 只有
get/set/version/remove。推模型需要 on_change。

session 级别的 cache 和 parameter 都是 matrix cell 改造前设计的, 现在都暴露问题。

## Key Decisions

### D1. 广播 + query, host 为唯一真相

| host 状态 | 真相源 | 机制 |
|---|---|---|
| host 存在 | host | host 广播真值 (推), 本地被真值覆盖 |
| host 不存在 | 本地 | 无广播, 本地即真相 |

- 服务启动时监听 liveness (`CellNetwork.on_updated` / `wait_present`) 判断 host 是否在线。
- host 节点监听全量数据, 内存管理 (它持有全部 parameter 真值)。
- parameter 具体类型启动时做一次 query, 拿到前值——"自己启动之前的值" 的满足。
- 前值的本质 = 广播(推) + 启动 query(拉) 的结合, 不依赖共享文件系统。

### D2. cache = 拉, parameter = 推

| | 方向 | 底层 |
|---|---|---|
| cache | 拉 | 本地按需拉, 底层可基于对 host 的 query |
| parameter | 推 | host 广播真值, 本地值被真值覆盖 |

cache 最好的归宿是 Redis——本轮不引入, 搁置。

### D3. parameter 必须有变更回调 on_change

依赖前值的状态 (如 VoiceNodeRuntime) 走 parameter, 消费者需要 push 通知。当前接口只有
get/set/version/remove, **on_change 是待补缺口**。补上后消费者不再需要轮询 version()。

### D4. host 晚上线竞态的简单解法

host 晚上线时, 服务可能已写入本地真值。简单解法: **host 自己 query 一次, 比较时间戳,
最新胜**。不引入复杂的分布式协调。

### D5. 复用现有 cell/network 基建, 不是 greenfield

改造是 re-point 现有层:
- `has_host()` / `is_host` — host 存在性判断
- `CellPresence.__aenter__` — 声明 liveness / queryable / event 通讯资源, 广播上线
- `CellNetwork.on_updated(callback)` — (Cell, online) 结构变化回调 = liveness 监听
- `CellNetwork.refresh()` — 拉取最新 presence = 拉 (query)
- `CellNetwork.on_event()` — CellEvent 到达回调

## 与旧设计的关系

- **session-communication-bus D10** (SQLite 真值 + Zenoh 轻量失效): 只在同机共享文件系统
  成立, 跨网络前值丢失 → 本 feature 修正它。
- **session-communication-bus D12** (跨网协议未来独立定义, 底层可换 etcd/consul/nats KV):
  方向被具体化为 host-as-truth + broadcast/query。
- **session-communication-bus D18** (Zenoh Queryable 暂不引入): 本 feature 的"拉" (query)
  恰恰是当初缺的 pull 原语场景, 现在是需求明确的时机。

## Implementation Notes

- 已核实的现状: `session_parameter.py` declare() 懒加载 SQLite; get() 纯 dict 零 IO;
  写路径 SQLite CAS + Zenoh invalidation (key, version)。接口无 on_change。
- 前值需求判据 (voice-input-state-machine 2026-08-12): **依赖前值 → parameter,
  不依赖 → topic**。这是状态/事件分家的核心判据。
