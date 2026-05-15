---
title: Module Channel
status: in-progress
priority: P2
created: 2026-05-14
updated: 2026-05-14
depends: []
milestone:
description: >-
  Zero-dependency module reflection channel — expose any Python module's public functions as Channel commands.
---

# Module Channel

## Motivation

Channel 构建目前有三种范式：继承 ABC（复杂运行时）、Builder 装饰器（手工注册）、Typer 包装（CLI 桥接）。每种都需要开发者写胶水代码。

`module_channel` 是第四种——零手工反射。一行 `new_module_channel(os)` 就能让 AI 操作文件系统。它补全了"从零手工到全手工"的梯度，同时是 Code as Prompt 原则最薄的表达层——模块即能力。

作为教程/文档的入门示例也有价值：最简单的 Channel，最少的认知负担。

## Design Index

- `src/ghoshell_moss/channels/module_channel.py` — 实现
- `tests/ghoshell_moss/channels/test_module_channel.py` — 单测

## Key Decisions

1. **`respect_all=True` (默认)**: 有 `__all__` 则尊重, 没有则 fallback 到 `dir()`. `respect_all=False` 强制忽略 `__all__`.
2. **零外部依赖** — 只用 `new_channel` + `Builder`，不引入任何第三方库。
3. **include/exclude 过滤** — 白名单/黑名单都是可选的，给调用方精细控制权。
4. **sync 和 async 函数都支持** — `Builder.command()` 已处理两者的差异。
