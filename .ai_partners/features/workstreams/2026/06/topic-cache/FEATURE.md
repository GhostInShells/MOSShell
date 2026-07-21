---
created: 2026-06-05
depends: []
description: 'Session 层级的跨进程 Cache: KV + hash map + 分布式锁, 纯 sqlite3 实现, WAL 模式, 零额外依赖.'
milestone: null
priority: P1
status: completed
title: Session Cache — sqlite3 跨进程共享缓存与仲裁组件
updated: '2026-06-05'
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