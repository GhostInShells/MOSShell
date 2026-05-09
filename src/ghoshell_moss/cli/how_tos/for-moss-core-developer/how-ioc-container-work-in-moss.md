---
title: How IoC container works in MOSS
description: 理解 MOSS 的 IoC 容器、Provider 声明、Bootstrapper 生命周期、manifests 自动发现与 Matrix 启动注入流程
---

# How IoC Container Works in MOSS

MOSS 用 IoC 容器管理所有环境依赖。任何能力——Logger、Zenoh Session、ResourceRegistry——都通过 Provider 声明，容器负责实例化和注入。

## 先看接口，再看实现

```bash
# 容器的核心接口
moss codex get-interface ghoshell_container:IoCContainer

# Provider —— 依赖声明的抽象
moss codex get-interface ghoshell_container:Provider

# Bootstrapper —— 生命周期钩子
moss codex get-interface ghoshell_container:Bootstrapper
```

`Provider` 的核心契约：
- `contract()` — 我提供什么类型
- `singleton()` — 单例还是每次新建
- `factory(container)` — 怎么创建实例

`Bootstrapper` 只有一个方法 `bootstrap(container)`，在容器初始化完成后调用，用于后置注册。

## 发现：manifests 的作用

Provider 实例不是手动注册的，而是在 workspace 的 `MOSS.manifests.providers` 包里声明，由 `PackageManifests` 自动扫描。

```bash
# 看扫描逻辑
moss codex get-source ghoshell_moss.host.manifests.providers

# 看一个实际的 providers 声明
cat host/stubs/workspace/src/MOSS/manifests/providers.py

# 在命令行里列出环境中所有被发现和注入的 Provider
moss manifests providers
```

扫出来的 `ProviderInfo` 包含：contract 类型、文件位置、是否单例。Matrix 启动时遍历这些 Provider，调用 `factory()` 创建实例并注册进容器。

## 启动：Matrix 如何串起来

```bash
moss codex get-interface ghoshell_moss.core.blueprint.matrix:Matrix
```

Matrix 是环境的总线。启动流程：
1. 创建容器
2. 加载 `PackageManifests`（按 mode 合并 env + mode 两层）
3. 遍历 `manifests.providers()`，逐个 `register(provider.factory(container))`
4. 遍历 `manifests.bootstrappers()`，逐个 `bootstrap(container)`

## 隐藏价值

这套体系意味着：

- **声明与实现分离**：Provider 只是声明"我能提供 X"，factory 里怎么写是自由的。测试时可以换 mock。
- **AI 无副作用注册**：AI 只需在 `MOSS.manifests.providers` 下写一个 Provider 实例，不需要理解容器启动流程。Matrix 启动时自动发现并注入。
- **可随时调整**：Provider 是纯 Python 对象，可以随时增删改，下次启动生效。

## 延伸

```bash
# 看 ResourceStorageFactoryBootstrapper —— 专门为 Resource 设计的 Bootstrapper
moss codex get-source ghoshell_moss.contracts.resource:ResourceStorageFactoryBootstrapper
```
