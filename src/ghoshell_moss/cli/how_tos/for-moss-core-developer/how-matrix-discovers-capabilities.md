---
title: How Matrix discovers capabilities
description: 理解 MOSS 的 manifests 约定如何发现、声明和隔离环境中的能力与资源
---

# How Matrix Discovers Capabilities

MOSS 的能力发现基于一个简单约定：在约定路径下声明，自动扫描发现。

## 约定路径

```bash
# 环境级 — 全局能力
MOSS.manifests.providers      # IoC 依赖声明
MOSS.manifests.channels       # 一级 Channel
MOSS.manifests.primitives     # Shell 原语
MOSS.manifests.configs        # 配置项
MOSS.manifests.topics         # 事件协议
MOSS.manifests.resources      # ResourceStorageFactory 声明

# 模式级 — 按 mode 隔离，覆盖环境级
MOSS.modes.{mode_name}.providers
MOSS.modes.{mode_name}.resources
# ...
```

每种能力类型对应一个子模块，里面放实例声明。扫描逻辑统一用 `scan_package` + 类型过滤：

```bash
moss codex get-source ghoshell_moss.host.manifests.providers
moss codex get-source ghoshell_moss.host.manifests.resource_storages
```

## 合并：env + mode

```bash
moss codex get-source ghoshell_moss.host.manifests:MergedManifests
```

`MergedManifests` 合并多个 `PackageManifests`，右边优先。典型场景是 env 声明全局能力，mode 覆盖/增加特定场景的能力。合并在同一类别的粒度上进行——providers 只覆盖 providers，不会跨类别影响。

## 谁消费这些声明

```bash
# 看 Host 如何加载 manifests
moss codex get-interface ghoshell_moss.host:Host

# 列出当前环境的全部能力
moss manifests providers
moss manifests channels
moss manifests resources
```

`Host` 创建时指定 mode，内部构造 `MergedManifests`，Matrix 启动时遍历所有声明，注入 IoC 容器，触发 Bootstrapper。

## 隐藏价值

这套约定的核心优势不在于"自动化"——几乎所有框架都有自动扫描。真正重要的：

**声明是声明，实现是实现。** Provider 只是一个 Python 实例，它的 factory 里写什么、从哪个包导入，是自由的。这意味着一个第三方包可以在自己的路径下定义 `MOSS.manifests.providers`，只要被 Python path 可见，就能被 MOSS 自动发现。

**AI 可以无副作用地注册。** AI 在 `MOSS.manifests.resources` 里写一行 `MyFactory()`，不需要碰任何启动代码、不需要理解 Matrix 生命周期、不需要修改容器。下次启动自动生效。不对就改，改了就对。

**隔离靠文件名。** mode 机制让同一套能力在不同场景下有不同表现。测试 mode 可以 mock 外部依赖，生产 mode 用真实实现。隔离边界是文件系统，不需要框架级别的沙箱。

## 延伸

```bash
# ResourceStorageFactory 的发现——用 resource 范式发现 resource 本身
moss manifests resources

# REPL 中内省
moss repl
> /manifests.resource_storages()
```
