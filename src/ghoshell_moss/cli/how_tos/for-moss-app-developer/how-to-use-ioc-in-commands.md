---
title: How to use IoC in commands
description: app 开发者如何在自己的 command 里通过 CommandCtx 获取 IoC 容器中的服务，以及在 Matrix 上下文中获取容器引用。覆盖获取、查看、注册的最小必要知识。
---

# How to Use IoC in Commands

IoC（控制反转）让 app 开发者在 command 里直接拿到已注册的服务实例，不需要自己构造、不需要知道依赖链。

## 两个使用场景

### 1. 在 command 内部

command 函数里用 `CommandCtx` 获取服务：

```python
from ghoshell_moss.core.blueprint.channel_builder import CommandCtx

async def my_command(self, text: str) -> str:
    # 获取日志
    logger = CommandCtx.logger()

    # 获取任意已注册的服务
    tts = CommandCtx.get_contract(TTS)

    logger.info(f"synthesizing: {text}")
    return await tts.synthesize(text)
```

`CommandCtx` 只在 command 执行期间有效。容器引用由 ChannelRuntime 在调用 command 时自动注入。

可用的 contract 类型参考：

```bash
moss codex get-interface ghoshell_moss.contracts.speech   # TTS, Speech
moss codex get-interface ghoshell_moss.contracts.resource  # ResourceRegistry
moss codex get-interface ghoshell_common.contracts         # LoggerItf
```

### 2. 在 command 外部（Matrix 上下文）

app 初始化、Matrix 生命周期回调等非 command 场景，通过 Matrix 拿容器：

```python
matrix = Matrix.discover()
container = matrix.container

# 获取服务
logger = container.force_fetch(LoggerItf)
registry = container.force_fetch(ResourceRegistry)
```

## 查看当前环境有什么

```bash
# 声明了哪些 provider（注册路径、文件位置）
moss manifests providers

# 运行时实际绑定了哪些 contract（可能比 providers 多）
moss manifests contracts

# 按名称过滤
moss manifests contracts resource
moss manifests contracts speech
```

`contracts` 列出的是容器中当前实际存在的绑定，包含通过 `container.set()` 直接设置和通过 Provider 注册的全部实例。

## 注册新服务

1. 在 workspace 的 `MOSS.manifests.providers` 里声明 Provider 实例
2. 下次启动自动生效

具体写法参考现有声明和 Provider 接口：

```bash
# 看当前注册了哪些 provider
moss manifests providers

# 看 Provider 的接口契约
moss codex get-interface ghoshell_container:Provider

# 看实际声明示例
cat $(moss ws where 2>/dev/null | grep "Expect Root" | cut -d'|' -f3 | xargs)/src/MOSS/manifests/providers.py
```

更深入的内核知识（父子容器隔离、Bootstrapper 生命周期、provide 语法糖）属于内核开发者范畴：

```bash
moss how-tos read get-moss-design/how-ioc-container-work-in-moss.md
```

## 文档目标

读者按照本文档操作，应该能够：
1. 在 command 内通过 `CommandCtx.get_contract()` 获取已注册的服务
2. 在 Matrix 上下文中通过 `matrix.container.force_fetch()` 获取服务
3. 用 `moss manifests providers` 和 `moss manifests contracts` 查看环境中的服务
4. 知道如何在 `MOSS.manifests.providers` 里声明新的 Provider
