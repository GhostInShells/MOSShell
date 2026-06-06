---
title: Discover the MOSS Environment
description: 进入一个 MOSS 项目后，用最少命令了解环境中已集成的能力：有哪些依赖、channel、配置、模式、app。面向首次进入环境的开发者和 AI 协作者。
---

# How to Discover the MOSS Environment

## 背景

你进入一个 MOSS 项目或 workspace。你不了解之前的开发者做了什么，不知道哪些能力已集成、哪些模式可用、哪些 app 在运行。

MOSS 提供了自解释的环境发现体系——不需要读代码、不需要问人。三组命令可以在几秒内建立对环境能力的完整认知。

## 三步探索

### 第一步：定位

```bash
moss workspace where               # 当前在哪种 workspace？在哪？
moss manifests explain             # 环境能力声明的自解释全景图
```

`workspace where` 告诉你当前是项目 workspace 还是用户 workspace。`manifests explain` 是唯一的真相入口——输出所有声明类型、发现路径、检测方式、键语义。它的输出随代码自动保持准确。

### 第二步：扫描能力

```bash
moss manifests contracts           # IoC 中已绑定了哪些接口？（可以拿到什么）
moss manifests providers           # 这些接口由哪些工厂生产？（由谁生产）
moss manifests channels            # 主 Channel 是什么？有哪些命令？
moss manifests configs             # 有哪些配置？当前值是什么？
moss manifests resources           # 有哪些资源存储？scheme:host 是什么？
moss manifests nuclei              # 有哪些感知核？接收什么信号？
moss manifests topics              # 有哪些事件协议声明？
```

**providers 和 contracts 的关系**：providers 声明"这个接口由这个工厂生产"，contracts 列出容器中已绑定的接口。想加新能力 → 看 providers 怎么写的。想用已有能力 → 看 contracts 有什么可拿。

### 第三步：了解模式和 App

```bash
moss modes list                    # 有哪些模式？
moss modes show <name>             # 某个模式的详细配置、manifest 文件清单

moss apps list                     # 有哪些 app？运行状态？
moss apps show <fullname>          # 某个 app 的详细信息
```

Mode 决定了"当前环境看到什么能力"。不同 mode 的 app 白名单和 bringup 策略不同。`moss --mode <name> manifests explain` 可以看某个 mode 下的完整能力视图（mode 文件通过 `from MOSS.manifests.xxx import *` 显式继承全局能力）。

## 需要深入时

上面三步覆盖了"有什么"。需要理解"为什么"和"怎么用"时：

```bash
# 理解任何模块的接口契约（5 秒内）
moss codex get-interface ghoshell_moss.core.concepts.channel
moss codex get-interface ghoshell_moss.contracts.configs

# 理解核心概念
moss codex concepts                # 架构核心概念一览

# 看源码
moss codex get-source <modulepath>

# 知识库
moss howtos list                  # 有哪些 how-to 文档？
moss howtos recall "<问题>"       # AI 语义召回相关文档
```

原则：**先工具，后源码**。工具确认"有什么"，源码补充"怎么用"。

## 常见场景

### 场景：我要开发一个新 App，怎么知道有哪些能力可以调用？

```bash
moss manifests contracts           # 所有可以从 IoC 拿到的服务
moss manifests topics              # 可以 pub/sub 的 topic 类型
moss modes show <当前mode>         # 当前模式的能力边界
```

在 App 中通过 `container.force_fetch(ContractType)` 获取服务，通过 `Matrix.topic()` 做 pub/sub。

### 场景：我要给项目加新能力，放在哪？

参考已有的声明了解模式：

```bash
moss codex get-source MOSS.manifests.providers   # 看已有的 Provider 怎么写
moss codex get-source MOSS.manifests.channels    # 看主 channel 怎么构建
```

然后对照 `moss howtos read host-dev/register-manifests.md` 在正确位置写代码。

### 场景：当前环境有什么问题？怎么排查？

```bash
moss manifests explain             # 看看声明体系是否完整
moss manifests providers           # 检查 Provider 是否被正确发现
moss manifests channels            # 检查主 channel 是否定义
```

如果某个 manifest 没有出现，检查：文件是否在约定路径？实例是否是正确的类型？变量名是否以 `_` 开头（会被跳过）？

## 文档目标

读者按照本文档操作，应该能够：
1. 用 `moss workspace where` + `moss manifests explain` 在 10 秒内建立对环境的基本认知
2. 通过 manifests 子命令列出所有已集成的能力（providers/channels/configs/nuclei/resources/topics）
3. 通过 modes 和 apps 命令了解模式配置和 app 状态
4. 知道用 `moss --mode <name> manifests explain` 看不同 mode 下的能力视图
