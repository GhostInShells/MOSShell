---
created: 2026-06-05
depends: []
description: 'Session 层级的跨进程 Cache: KV + hash map + 分布式锁, 纯 sqlite3 实现, WAL 模式, 零额外依赖.'
milestone: null
priority: P1
status: dropped
status_note: 2026-08-31 第一性评估后放弃 — feature 名偏航 ("topic-cache" 应为 "session-cache", 建 feature 的模型混淆了 topic). 见文末 2026-08-31 Drop 补充.
title: Session Cache — sqlite3 跨进程共享缓存与仲裁组件
updated: '2026-08-31'
---

# Session Cache

## Motivation

Matrix 是多进程通讯总线，Cell 之间需要共享的缓存与仲裁机制 (类似 ROS2 的 parameter server)。现有的跨进程传输只有文件 和 Zenoh，不想引入 redis 等额外依赖。用 sqlite3 单文件实现，利用 WAL 模式保证多进程并发安全。

## Design Index

- Contract: `ghoshell_moss.contracts.cache.Cache` — 中立契约, 无生命周期
- Implementation: `ghoshell_moss.core.cache.SqliteCache` — 纯 sqlite3, WAL 模式
- Mount point: `Session.cache` property, db 文件在 `session.tmp_storage/cache.db`
- Lifecycle: session 退出时随 tmp 目录清理, 无需显式 lifecycle 管理

## Key Decisions

- **exp=0 = 永不过期**: 对齐旧 GhostOS Cache 契约语义, 非 Redis 的 "立即过期"
- **不引入 lifecycle**: Cache ABC 无 close()/context manager, 进程死 OS 回收连接, session 退出清 tmp 目录
- **放在 session tmp_storage 而非 workspace runtime**: 缓存是 session 生命周期的通讯基础设施, 非持久数据
- **Hash map 和 KV 同时纳入**: 一个 key 可同时有 string value 和 hash 子字段

## Round 2 Enhancement (2026-06-06)

补强: Lock context manager — `Cache` ABC 上添加 `locked(key, overdue)` context manager，自动 unlock。纯语法糖，不涉及实现层。

讨论过但拒绝: Cache 独立 IoC Provider。参考 `session-communication-bus` FEATURE.md Parameter 实现记录——ParameterStore 和 Cache 都是 Session 通讯协议的一部分，没有独立于 Session 的消费场景，不需要走 manifests 注册。Cache 是此模式的先例而非遗漏。

## Implementation Notes

- sqlite3 的 `busy_timeout=3000` + WAL 模式处理多进程并发写
- `lock()` 使用 `BEGIN IMMEDIATE` 事务保证 INSERT + 过期清理的原子性
- `get()` 惰性过期: 读时检查 TTL, 过期返回 None 但不立即删除 (对齐 Redis 行为)
- contracts/__init__.py 的 Cache import 曾在 6091da6 中过早加入 (cache.py 尚不存在), rebase 修正

## 2026-08-12 会话补充 — cache 重定向为拉逻辑

> 人类工程师 + deepseek-v4-flash。与 parameter-host-truth 成对讨论 (推/拉分家)。
> session 级别 cache 和 parameter 都是 matrix cell 改造前设计, 现在都暴露问题。

### 概念分家 (核心)

| | 方向 | 底层 |
|---|---|---|
| **cache** | 拉 | 本地按需拉, 底层可基于对 host 的 query |
| **parameter** | 推 | host 广播真值, 本地值被真值覆盖 |

- cache 不依赖前值——拉就是当前值, 天然没有"启动前的前值"问题。
- host 存在时底层 query host, host 不存在时本地即真相。
- **cache 最好的归宿是 Redis, 本轮不引入**。现状 sqlite3 实现 (本文件) 是折中, 不是终点。

### 与 parameter-host-truth 的关系

两者共享同一套 host-as-truth + 广播(query) 机制 (见 parameter-host-truth D1/D2)。
本 feature 聚焦拉面, parameter-host-truth 聚焦推面。Redis 引入前的 sqlite3 折中保持不动。

## 2026-08-31 Drop 补充 — 放弃 cache 抽象

> 人类工程师 + deepseek-v4-flash。第一性评估 (非代码反向推理), 判定放弃并把 feature 置 dropped。

### 评估方法修正

此前曾用"代码里有没有消费者/有没有锁计数需求"反推基建必要性——本末倒置。基建本来就是先行者,
靠"需求实现了没有"判断基建是否必要是错的。本轮改为正向设计推理:
在 network 级 matrix 组网、ghost 操作系统里, 跨节点共用 cache 的业务场景是否存在。

### 跨节点共享状态拆三类 (落点不同)

| 类型 | 例子 | 本质 | 归属 |
|---|---|---|---|
| 分享真相 | 当前收音状态/全局配置/host 广播真值 | 一份 truth 广播 | **parameter (推)** —— 不归 cache |
| 跨节点原子协调 | 肢体 cell 抢同一关节(互斥锁); 限频计数; 幂等去重 | **原子锁/计数** | 这才是 cache 独特价值 |
| 跨节点临时拉缓存 | 把最近算出的结果按 TTL 跨节点复用 | memoize | 最弱项, 见下 |

判断: 让 cache 有存在价值的只有原子协调。真相已被 parameter 拿走(推); "跨节点拉缓存"是反向模式——
缓存通常是**本地化**才有价值, 跨节点共享"最近结果"仅在"多节点算同一昂贵东西且重算远贵于远端拉"时回本,
少见且顶端收敛到"那用个共享 KV", 又滑回 parameter。Broad Cache = 把**弱需求(拉缓存)+硬需求(原子协调)+被它处覆盖的真相**捆在一起。

### zenoh + host 唯一 truth 的成本

能做, 但按原语不对称:

- **拉 KV + TTL**: 近似 parameter-host-truth 那套机械, 成本中等; 却是最弱需求, 不值得复刻 host 机制。
- **分布式锁**: host 必须串行化授予/监听 TTL/释放, 还要 fencing 才安全。真分布式系统的活。
- **原子计数**: host 必须经 queryable 做单写者 RMW (读改写须在 host 一处串行)。同样的活。

三者还都得为"跨节点原子不变式"重新引入**单写者 host 的脆弱性**——host 本身是网络里的 cell,
会死/重启/被替换。parameter-host-truth D4 已为 host 晚上线竞态头疼(那只是新鲜度); 放到原子不变式上,
host 断连不再是"读到旧值"而是**正确性失效**。用协调细胞死亡的原子原语, 自己被细胞的死亡绑架。
且 moss 开箱不带 redis。

### 结论 — dropped (MOSS 不作为默认 contract)

人类工程师的明确立场: **cache 从来都可以用 Redis 实现** (真分布式, 原子锁/计数齐全)。
问题不是"能不能实现", 而是 **matrix 开箱 (无 redis) 能不能做到**。开箱做不到 —— 单文件 sqlite 在
network 级 matrix 下不能诚实提供原子锁/计数; host 唯一 truth 又会重引入单写者脆弱性。
因此 cache **不作为 MOSS 开箱默认提供的 system contract**; 需要跨节点 cache 的 workspace/部署,
经环境 provider 自注册 (Redis 实现)。缺一个官方抽象是遗憾, 但不能挂一个"开箱即用却伪承诺"的。

支撑依据 —— broad cache 这个形状既实现不了它独特的那部分、又捧不住它真实的那部分:

- 删除 Broad Cache 抽象 (contracts Cache ABC) + SqliteCache 实现 + session 装线 (zenoh_session/mock_session)。
- 真实需求 (跨节点原子协调 锁/计数) 不消失, 但**不该预置成 Broad Cache ABC**; 等真实消费者出现再
  只建需要的那一个原语 (锁原语或计数原语, 非 "Cache")。
- 弱需求 (拉 KV/TTL) 不进网络抽象, 留给本地缓存。
- 命名教训: feature 名应为 `session-cache`; 建 feature 的模型混淆了 topic, 故成 `topic-cache`。
  抽象本身 (Session Cache / Cache / SqliteCache) 命名无偏航, 偏航的是 feature 名。

### 删除清单 (2026-08-31)

- 删 `src/ghoshell_moss/contracts/cache.py` (Cache ABC, 孤儿契约 = 伪承诺)。
- 删 `src/ghoshell_moss/core/cache/` (SqliteCache 实现)。
- 清 `contracts/__init__.py` 的 `from .cache import Cache`。
- 清 `core/blueprint/session.py` 的抽象 `cache` property + import。
- 清 `core/session/mock_session.py` / `matrix/session/zenoh_session.py` 的 cache property + import + `_cache` 懒字段。
- 清 `core/blueprint/matrix.py` contracts docstring 的 "三类" 叙述段 (user 判定不入 docstring)。
