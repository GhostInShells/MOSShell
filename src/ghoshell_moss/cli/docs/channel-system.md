---
title: MOSS Channel 体系
description: Channel 的构建、集成、发现全链路——从一行反射到跨进程通讯。需要开发新 Channel、理解能力拓扑、或选择集成方式时阅读
---

# MOSS Channel 体系

Channel 是 MOSS 架构的能力组织单元。在八层拓扑中，它位于 Logos（CTML 语法）与 Shell（流式调度）之间——Logos 描述"做什么"，Shell 决定"何时做"，Channel 回答"能做**什么**"。

本文档覆盖 Channel 的完整生命周期：构建、集成、发现。读完应能独立开发新 Channel 并选择正确的集成路径。

---

## 1. 核心概念

### 1.1 Channel 是什么

**能力组织单元** —— 类比文件系统：文件系统用目录组织文件，Channel 用树形拓扑组织命令。每个 Channel 封装一组命令（Python async 函数）、附带指令（instruction）、动态上下文（context_messages）、生命周期钩子。

**上下文窗口组件** —— Channel 在模型每个思维关键帧注入动态上下文。视觉 Channel 提供 `capture()` 命令的同时，通过 `context_messages` 将当前画面描述注入模型的上下文窗口。

**有状态运行时** —— 区别于无状态的 Tool Calling。ChannelRuntime 在模型多次调用之间保持运行，持有连接和状态。命令执行有时间开销，同通道内后续命令需等待前序命令完成（occupy 语义）。

### 1.2 关键抽象

```bash
moss codex get-interface ghoshell_moss.core.concepts.channel:Channel    # 契约
moss codex get-interface ghoshell_moss.core.blueprint.channel_builder   # 构建器
moss codex get-interface ghoshell_moss.core.blueprint.states_channel    # 有状态 Channel
moss codex channeltypes                                                 # 正式 Channel 类型索引
```

- **Channel** — 无副作用的声明。`bootstrap()` 产生有副作用的 `ChannelRuntime`
- **ChannelRuntime** — 运行时实例。持有命令、上下文、生命周期状态
- **Builder** — 装饰器风格的构建 API。`@build.command()` 将 Python 函数反射为 Command
- **ChannelState** — 多态运行时状态。不同状态有不同命令集、指令、上下文
- **Command** — 被反射的 Python 函数。函数签名即模型看到的接口（Code as Prompt）

### 1.3 设计全景：一个 Channel 承担的角色

以下维度描述一个 Channel 在 MOSS 架构中可承担的角色。每项均有对应的 API 或代码路径，当前实现程度不一。

---

**能力创建与组织。** Channel 将一组 Python async 函数封装为 Command，按树形拓扑组织。父 Channel 的子节点可以是本地模块、远程进程、或运行时通过 `get_virtual_children()` 生成的动态节点。树形结构支持折叠和渐进披露——模型不需要一次性看到全部能力。

对应 API：`Builder.command()`、`import_channels()`、`get_virtual_children()`。

**时序契约。** Channel 声明命令的阻塞语义（`blocking`/`@nonblocking`）。Shell 据此调度：同通道内 FIFO 顺序执行，父通道 occupy 期间阻塞所有子通道，异通道并行。`call_soon` 和 `priority` 参数提供抢占控制。

对应 API：`Builder.command(blocking=..., call_soon=..., priority=...)`。

**运行时生命周期。** Channel 有 `startup` → `running` → `idle` → `close` 的完整状态流。`idle` 钩子在通道无命令执行时被周期性调用——适用于维持姿态、降低帧率等后台行为。`startup` 和 `close` 管理资源的获取和释放。

对应 API：`Builder.startup()`、`Builder.idle()`、`Builder.close()`。

**IoC 依赖注入。** 通过 `CommandUtil.get_contract()`，Channel 内的命令在运行时从 Matrix 的 IoC 容器获取服务。Channel 不硬编码依赖——它从环境中发现。`moss manifests` 系列命令提供开发时的环境能力清单。

对应 API：`CommandUtil.get_contract()`、`moss manifests contracts`、`moss manifests providers`。

**模型上下文认知。** Channel 通过四种消息类型参与模型上下文窗口的构建：`moss_static`（静态接口签名）、`moss_dynamic`（运行时刷新状态）、`instruction`（系统提示词注入）、`context_messages`（每个关键帧的动态上下文）。CTML prompt 定义了模型消费这些消息的协议。

对应 API：`Builder.instruction()`、`Builder.context_messages()`、`moss_static`/`moss_dynamic` 标签。

**反身性控制。** Channel 的命令可以修改 Channel 自身——切换 StatefulChannel 的状态、通过 `add_virtual_channel()`/`remove_virtual_channel()` 增减子通道、更新 instruction。结合 Matrix 的 scoped storage，可以实现运行时修改元认知文件的能力。

对应 API：`StatefulChannel.with_state()`、`ChannelStateBuilder.add_virtual_channel()`、`Matrix.get_scoped_storage()`。

**后台持续运行。** `ChannelRuntime` 在模型多轮调用之间保持运行。状态、连接、缓存不随单次 CTML 执行结束而销毁。模型只看到关键帧的快照，但 Channel 在后台持续执行。

对应行为：`ChannelRuntime` 实例在 Shell 生命周期内持久存在。

**通讯端点。** Channel 是数据在 MOSS 体系中的入口和出口：Signal 上行给 Mindflow，context 注入模型认知窗口，command result 返回给 Shell 的观察层。传输管线由 Matrix/Session 负责，Channel 是端点。

对应路径：`Session.output()`（上行）、Shell 的 context 组装（下行）、`observe` 机制（结果返回）。

**远程同构。** 同一个 Channel 接口，可以指向本地函数，也可以指向另一个进程中的 Channel（通过 ZenohProvider/ZenohProxyChannel），或另一个 MOSS 运行时中的 Channel（通过 Fractal）。模型在 CTML 中调用时使用相同的语法，不感知位置差异。

对应 API：`Matrix.provide_channel()`、`Matrix.channel_proxy()`、`FractalHub`。

---

> 远程同构的完整链路见 App 体系文档：`moss docs read app-system.md`
> Matrix 通讯 API：`moss codex get-interface ghoshell_moss.core.blueprint.matrix` （`provide_channel` 与 `channel_proxy`）

---

## 2. 构建梯度

从零手工到全能多态，五层递进：

| 层级 | API | 场景 | 参考 |
|------|-----|------|------|
| L0 | `new_module_channel()` | 纯函数模块，一行反射 | `module_channel.py` |
| L1 | `new_channel()` + Builder | 需 instruction/context/生命周期 | `notebook_channel.py` |
| L2 | 继承 Channel ABC | 复杂运行时，封装 contract | `speech_channel.py` |
| L3 | StatefulChannel | 运行时切换状态/能力集 | `app_store_channel.py` |
| L4 | PrimeChannel | 全能：stateful + mutable + builder | — |

### L0: 一行反射

```python
from ghoshell_moss.channels.module_channel import new_module_channel
import math
chan = new_module_channel(math)  # sqrt, sin, cos 全部变成 Command
```

适用场景：模块已有公共函数，不需要自定义 instruction 或 context。`__all__` 优先，fallback 到 `dir()`。

### L1: Builder 模式

```python
from ghoshell_moss.core.blueprint.channel_builder import new_channel

chan = new_channel(name="my_tool", description="does something")

@chan.build.command()
async def add(a: float, b: float) -> float:
    """Add two numbers. 这个 docstring 模型会看到。"""
    return a + b

@chan.build.context_messages
async def context() -> list[str]:
    return ["current state: 3 items pending"]
```

`Builder.command()` 的关键参数：

| 参数 | 作用 |
|------|------|
| `name` | 重命名命令 |
| `doc` / `comments` / `interface` | 控制模型看到的签名（interface 可指向虚拟函数） |
| `blocking` | 是否阻塞同通道后续命令（默认 True） |
| `always_observe` | 结果是否需要模型观察（信息类 True，确认类 False） |
| `call_soon` / `priority` | 抢占调度 |

### L2: 继承 ABC

当 Channel 需要封装外部 contract（TTS、播放器）、拥有独立生命周期时：

```python
from ghoshell_moss.core import Channel, PyChannel

class SpeechChannel(Channel):
    def name(self) -> str: return self._name
    def id(self) -> str: return self._uid
    def description(self) -> str: return self._description

    def materialize(self, container):
        chan = PyChannel(name=self._name, description=self._description)
        chan.build.command()(self.say)       # 将自身方法注册为命令
        chan.build.startup(self._speech.start)  # 生命周期
        return chan.bootstrap(container)
```

`materialize()` 是 Channel 的唯一抽象方法——将声明转化为运行时实例。

### L3: StatefulChannel

运行时切换能力集。每个 `ChannelState` 有独立的命令、指令、上下文和虚拟子通道：

```python
from ghoshell_moss.core.blueprint.states_channel import (
    ChannelState, new_stateful_channel_from_main
)

class MyState(ChannelState):
    def own_commands(self) -> dict[str, Command]: ...
    def get_virtual_children(self) -> dict[str, Channel]: ...
    def is_available(self) -> bool: ...
```

`get_virtual_children()` 是实现动态子通道的关键——AppStoreChannel 用它为每个运行中的 App 生成 ZenohProxyChannel。

### 选择建议

从低层级开始，遇到以下信号时升级：
- 需要自定义 instruction → L1
- 需要管理外部资源生命周期 → L2
- 需要在运行时切换命令集 → L3
- 需要同时拥有 mutable + stateful 能力 → L4

---

## 3. 集成方式

三种方式，复杂度递增：

### 3.1 Mode channels — 静态声明

在 workspace 约定位置写 Python 模块。启动时自动加载，无需重启进程之外的任何操作。

```python
# .moss_ws/src/MOSS/modes/{name}/channels.py
from ghoshell_moss import new_shell_main_channel
from ghoshell_moss.channels.app_store_channel import AppStoreChannel

main = new_shell_main_channel()
main.import_channels(AppStoreChannel(name='apps'))
```

**适用**：轻量能力、需要直接访问 IoC 容器、开发调试便捷。所有 channel 共享主进程运行时。

**接口**：
- `import_channels(*channels)` — 挂载为能力树子节点。父 occup 阻塞子。
- `with_module(module)` — 能力叠加。所有 module 同时激活、累积叠加。
- `build.command()(func)` — 单命令注册。最细粒度控制。

区别：`import_channels` 构建**树形层级**（父子 occupy），`with_module` 构建**平行叠加**（无层级，命令直接挂载）。`build.command()` 是前两者的底层机制。

### 3.2 App — 运行时动态集成

Channel 作为独立进程运行，通过 Matrix 总线通讯。

```python
# apps/my_group/my_app/main.py
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.channel_builder import new_channel

channel = new_channel(name="my_tool")

@channel.build.command()
async def add(a: float, b: float) -> float:
    return a + b

async def main(matrix: Matrix):
    await matrix.provide_channel(channel)

if __name__ == "__main__":
    Matrix.discover().run(main)
```

**适用**：需要独立依赖（`pyproject.toml`）、独立运行时、崩溃不拖垮主进程的能力。

**AI 控制面**：
```ctml
<apps:list_apps />
<apps:start fullname="my_group/my_app" />
<apps.my_group_my_app:add a="1" b="2" />
<apps:stop fullname="my_group/my_app" />
```

**完整链路**：`matrix.provide_channel()` → ZenohProvider 注册 → AppStoreChannel.get_virtual_children() 发现 → ZenohProxyChannel 挂载 → Shell Channel 树中出现 `apps.my_group_my_app`。对模型透明。

### 3.3 Matrix.provide_channel() — 底层 API

不依赖 workspace 约定，不依赖 Mode，不依赖 AppStore。只要有 Matrix 实例：

```python
await matrix.provide_channel(channel)
await matrix.channel_proxy(address, name)  # 另一端连接
```

这是前两者的基础机制。测试和脚本中最常用。

### 选择建议

```
需要独立依赖/独立进程？ → App
需要多 mode 复用？ → Mode channels 声明
需要快速验证原型？ → Mode channels (共享运行时)
测试/脚本中临时用？ → matrix.provide_channel()
```

---

## 4. 运行时语义

### 4.1 Occupy 与并发

- **同通道内**：命令 FIFO 顺序执行。当前命令未完成时，新命令保持 pending
- **父通道 occupy**：父通道有命令执行时，**所有子通道**的新命令都不会分发
- **异通道间**：并行执行，互不阻塞

`@nonblocking` 装饰器可标记命令**不 occupy**，同通道后续命令立刻执行。

### 4.2 Observe 体系

| 机制 | 触发方式 | 效果 |
|------|---------|------|
| `always_observe=True` | 命令注册时声明 | 结果在下一关键帧展示给模型 |
| `) -> Observe:` | 命令返回值类型 | 标记观察，**不中断**并行任务 |
| `raise ObserveError()` | 命令内抛出 | **紧急中断**全局，取消一切 |

`always_observe` 的约定：返回"信息"的命令设 True（read、list、query），返回"确认"的设 False（write、delete、start）。

### 4.3 动态性

- **`moss_static`** — 启动时确定的静态信息（命令签名、通道描述）
- **`moss_dynamic`** — 运行时刷新的动态信息（上下文、新命令、子通道变化）
- **`refresh_metas()`** — 触发 Channel 树的动态更新。AppStoreChannel 依赖此机制发现新 App

---

## 5. 发现与自解释

### 5.1 开发时发现：codex channeltypes

```bash
moss codex channeltypes              # 列出所有正式 Channel 类型
moss codex channeltypes <name>       # 反射单个 Channel 完整接口
moss codex channeltypes <name> --deps  # 包含依赖接口
```

对标 `moss codex concepts`。索引 `ghoshell_moss.channels` 包下所有模块，读取 docstring 中的类型和状态标记。

### 5.2 模块 docstring 约定

```python
"""一句话功能描述 | 功能类型 | 状态

Example:
    from ghoshell_moss import new_shell_main_channel
    from ghoshell_moss.channels.xxx import ...
    main = new_shell_main_channel()
    main.import_channels(...)
"""
```

- 第一行：机器可解析（`ast.get_docstring` 读取，codex 索引用）
- Example 段：只给一种推荐集成方式。不执行，为 code as prompt

### 5.3 Type 体系

| Type | 含义 | 示例 |
|------|------|------|
| `系统管理` | MOSS 架构级组件生命周期管理 | app_store_channel |
| `通讯桥接` | 跨进程/跨运行时连接与路由 | fractal_hub |
| `交互能力` | 向外部世界的输出或感知 | speech_channel |
| `集成` | 将已有外部能力封装为 Channel | module_channel, typer_channel |
| `系统控制` | 操作系统级别控制 | mac_channel |
| `认知模块` | 对文件系统等资源的动态结构化认知 | notebook_channel |

### 5.4 Status 三态

`alpha`（原型，无测试）→ `beta`（可用，接口可能变动）→ `active`（正式维护，跟随 semver）

---

## 6. 运行时发现：manifests channels

```bash
moss manifests channels     # 当前环境中 main channel 的命令树
moss manifests explain      # 所有声明类型的完整清单
```

区别于 `codex channeltypes`：

| | `codex channeltypes` | `manifests channels` |
|---|---|---|
| 视角 | 开发时——有哪些预制能力 | 运行时——当前环境的能力树 |
| 来源 | `ghoshell_moss.channels` 包 | workspace manifests |
| 使用者 | 开发前查阅 | 调试/理解运行环境 |

---

## 7. 测试

单测路径：`tests/ghoshell_moss/channels/`。参考模式：

```python
async with chan.bootstrap() as runtime:
    await runtime.start()
    assert runtime.has_own_command("ping")
    result = await runtime.execute_command("ping", args=(...))
```

CTML 语义的完整测试在 `tests/ghoshell_moss/core/ctml/v1_0/test_ctml_v1.py`（1434 行，覆盖并行、occupy、scope、observe、streaming、timeout 等全部核心语义）。

跨进程 proxy 测试在 `tests/ghoshell_moss/host/test_app_store_channel_proxy.py` 和 `tests/ghoshell_moss/host/test_zenoh_fractal.py`。

---

## 8. 与相邻层的关系

```
Logos (CTML 语法)
    ↓ 描述"做什么"—— <chan:cmd arg="val"/>
Channel (能力组织)
    ↓ 回答"能做什么"—— 命令、上下文、指令
Shell (流式调度)
    ↓ 决定"何时做"—— 流式解析、occupy、并行调度
Matrix (通讯总线)
    ↓ 提供"在哪里做"—— 跨进程、动态注册
```

Channel 向上对接 CTML 的 `<chan:cmd/>` 调用语法，向下依赖 Builder 将 Python 函数反射为模型可理解的形式。跨进程时依赖 Matrix 的 `provide_channel()`/`channel_proxy()` 桥接。

---

写下于 2026-06-02，基于 `ghoshell_moss.channels` 正式化过程中的完整认知轨迹。
