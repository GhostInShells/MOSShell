---
created: 2026-08-12
depends:
- session-communication-bus
description: 'Parameter 重做为 matrix 层点对点能力: declare (成为写者) + subscribe (成为读者), 单声明者无仲裁,
  推优先于拉, 读零 IO. Memory/Zenoh 双实现, 老 SessionParameterStore 删除, session 上的 parameters
  接口废弃.'
milestone: null
priority: P1
status: completed
status_note: '单声明者+点对点收敛完成: declare/subscribe 分离, Memory/Zenoh 双实现, 装线到 matrix, session
  接口废弃'
title: Parameter Host Truth
updated: '2026-09-01'
---

# Parameter Host Truth

> Use `moss features set-status parameter-host-truth <status> -m "note"` to update state.

## 历史

parameter 源自 session-communication-bus (2026-05, completed) 的 D10/D12, 2026-08-12
因 voice-input-state-machine 的前值需求独立成篇。

老 SessionParameterStore (SQLite + zenoh 失效信号) 依赖方向反了 — parameter 倒过来吃上层
Session — 2026-08-29 彻底删除。重做过程经历多轮偏航: host-as-arbiter、index 水位线、
双真相、version 连续性, 每一版都因"逻辑难懂"被推翻。最终收敛到最简形态:
**单声明者 + 点对点, 无 host, 无仲裁**。

## Motivation

1. **依赖方向反了**: parameter 是底层原语, 不能反向依赖应用层 Session (sub_stream /
   pub_stream_delta / tmp_storage)。
2. **真相源假设不成立**: SQLite 靠"同机读同一 tmp_storage 文件"跨进程同步, 跨网络不成立。
3. **无推模型**: 老 ABC 只有 get/set/version, 消费者只能轮询; 前值需求 (状态机接入立即
   读到"当前正在收音") 依赖前值 + 推变更, 老接口都不提供。

## Key Decisions

### D1. 单声明者, 无仲裁

`declare(model)` 让"单写者"成为构造性事实 — 谁 declare 谁就是该 parameter 的唯一源,
不存在第二个写者需要仲裁。host / version 连续性 / 双真相全部不需要。

### D2. declare/subscribe 分离 (写者/读者)

- `Parameters.declare(model) -> ParameterDeclaration` — 成为写者。
- `Parameters.subscribe(model, address) -> ParameterSubscriber` — 成为读者。

### D3. 点对点 (matrix 面)

subscribe 耦合 address (cell 地址), 定向到某 cell 的声明。matrix 面 = 耦合 address 的
点对点; session 面 = 全网广播。parameter 属于 matrix 面。

### D4. 推优先于拉 + retention

读零 IO 本地缓存; 写 fire-and-forget (本地立即生效 + push)。retention = 订阅时向声明者
query 一次当前值 + 之后持续收推。

### D5. 双实现

- `MemoryParameters` (core/parameter) — 单进程参考实现。
- `ZenohParameters` (matrix/parameters) — matrix 点对点。
- `AbsParameters` (core/parameter/_base) — 收敛队列 + task + 生命周期, transport 抽象。

### D6. 协议化 + 装线到 matrix

- `Parameters` / `ParameterDeclaration` / `ParameterSubscriber` 是 ABC (Facade 消费面)。
- `matrix.parameters` 惰性门 (lazy-gate), 作为 matrix 默认能力; 底层 zenoh.Session 仍走 IoC。
- session 上的 `parameters` 接口彻底移除。

## Implementation

- [x] 协议化 ABC (Parameters / ParameterDeclaration / ParameterSubscriber / ParameterModel / ParameterSchema)
- [x] AbsParameters (队列 + task + 生命周期, transport 抽象)
- [x] MemoryParameters (单进程参考实现)
- [x] ZenohParameters (matrix 点对点)
- [x] 装线到 Matrix (lazy-gate 默认能力)
- [x] 从 session 移除 parameters
- [x] 老 SessionParameterStore + 测试删除
- [x] 测试 (memory 5 + zenoh 2)

## 备注

- 前值需求判据 (voice-input-state-machine): 消费者依赖前值 → parameter; 不依赖 → topic。
- "耦合 address 的放 matrix 面, 全网广播的走 session 面" — parameter 是前者。
- 设计反复偏航的教训: 逻辑难懂 = 设计本身有问题。host-as-arbiter 的复杂性 (version 连续性 /
  双真相 / host-revive) 是过度设计, 单声明者 + 点对点才是正解。

## 复盘

### 根因

parameter 问题是由 **matrix 升级到 network 级别** 而引爆的: 原本"简单版本"的 parameters
实现在单机/单进程下可靠, 一旦 matrix 投影到 network (跨机器、跨进程), 它的真相源假设
(同机读同一文件) 就失效了。简单实现扛不住 network 级别, 才被迫往复杂方向 (host 仲裁 /
version 连续性) 堆, 越堆越难懂。

### 人机协作复盘

迭代了很多轮, 人类与模型始终无法对齐、无法有效碰撞。事后发现的三个点:

1. **模型不敢动"现有代码"和"interface"设计** — 即便是第一版实现里的最老错误 (如
   `handle → store._private()` 的反向私有调用、`declare(model_type)` 传类而非实例), 也会
   被模型当作"既定事实"继承下去, 而不是质疑重写。
2. **复杂时序 / 并发 / Python 线程-协程卸载场景, 模型几乎总是用简单粗糙的方式做** —
   (如闭包注入、裸 `threading.Thread`、阻塞 `session.get`) 交付优先, 不主动解决拓扑问题。
3. **设计方案层面的问题, 必须由人类主动提出, 才会产生质疑性质的碰撞** — 模型倾向于收敛到
   用户给的方案上, 而不是反向质疑方案本身。

### 最终判断 (倒过来)

最终认定 **parameter 的问题是设计过于复杂, 而不是模型执行能力的问题**。核心是:
**network 级别的 parameters 本身可能就是伪命题** — 为了一个跨网络共享状态, 引入 host 仲裁、
版本连续性、双真相, 复杂度远超收益。

最终选择了 **cell address 可指定 + parameter 读写分离 (declare/subscribe) + 单声明者**,
问题极大简化, 且满足当前需求。

现在回看之前 session level 的竞态 parameters (SQLite + 失效信号), 应该承认: **是设计上
就有问题**, 不是实现的锅。