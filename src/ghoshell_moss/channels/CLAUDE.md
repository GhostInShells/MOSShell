# Channels — 正式能力模块

本目录是 MOSS 随包分发的正式 Channel 类型。通过 `moss codex channeltypes` 索引。

## 1. 模块约定

### 1.1 Docstring 范式

每个模块第一行 docstring 采用机器可解析格式：

```python
"""一句话功能描述 | 功能类型 | 状态
"""
```

- `功能类型` 从下文的类型体系取值
- `状态` 从下文的 status 取值
- 由 `ast.get_docstring` 读取，对接 `moss codex channeltypes` 的索引表

### 1.2 Example 段

docstring 后续段落可追加 Example 段，只给**一种**推荐集成方式：

```python
"""反射 Python 模块为 Channel 命令集 | 集成 | beta

Example:
    from ghoshell_moss import new_shell_main_channel
    from ghoshell_moss.channels.module_channel import new_module_channel
    import math
    main = new_shell_main_channel()
    main.import_channels(new_module_channel(math))
"""
```

Example 不执行，只为 code as prompt —— 让读代码的模型一眼知道如何接入 main channel。

### 1.3 observe 约定

每个命令必须显式标注 `always_observe`。规则：

| always_observe | 适用场景 | 示例 |
|---|---|---|
| True | 结果是"信息"，模型需基于内容做下一步推理 | read、list、query、exec |
| False | 结果是"确认"，只需知成败 | write、delete、start、stop、say |

不依赖 Builder 的默认值。

## 2. Type 体系

功能导向分类，在此文件维护（单一事实源）。随着 channel 增多演进。

| Type | 含义 | 示例 |
|------|------|------|
| `系统管理` | MOSS 架构级组件生命周期管理 | app_store_channel |
| `通讯桥接` | 跨进程/跨运行时通讯连接与路由 | fractal_hub |
| `交互能力` | 向外部世界的输出或感知 | speech_channel |
| `集成` | 将已有外部能力封装为 Channel | module_channel, typer_channel |
| `系统控制` | 操作系统级别控制 | mac_channel |
| `认知模块` | 对文件系统等资源的动态结构化认知 | — |

*此分类为草拟（2026-06-02），随 channel 增多自然演进。新 channel 的类型可以追加，无需修改已有。*

## 3. Status

三态，线性推进：

```
alpha → beta → active
```

| Status | 含义 |
|--------|------|
| `alpha` | 原型/草图，无测试，接口随意改 |
| `beta` | 功能可用，接口可能变动，需要更多实际使用验证 |
| `active` | 正式维护，有测试覆盖，接口兼容承诺，跟随项目 semver |

- 当前 `active` 的模块：app_store_channel、fractal_hub
- 进入 `active` 后，接口变更需跟随项目的语义化版本号

## 4. 构建梯度

现有 channel 覆盖不同的构建层级，可作为新 channel 开发的参考起点：

| 层级 | API | 场景 | 参考 |
|------|-----|------|------|
| L0 | `new_module_channel()` | 纯函数模块，零手工反射 | module_channel |
| L1 | `new_channel()` + Builder | 需 instruction/context/生命周期 | typer_channel |
| L2 | 继承 Channel ABC | 复杂运行时，封装 contract | speech_channel |
| L3 | StatefulChannel | 运行时切换状态/能力集 | app_store_channel |
| L4 | PrimeChannel | 全能：stateful + mutable + builder | — |

## 5. 封装策略

两种互不排斥的策略，可在同一模块内混用：

**Channel Interface** — 先定义接口，再注册。类自身声明命令方法，在 `bootstrap()`/`materialize()` 中注册。`Builder.command(interface=...)` 可以给模型看虚拟函数签名，实际执行另一实现体。参考：speech_channel。

**As Channel** — 外部已有的事物包装为 Channel。把 module、CLI、API、设备等外部能力反射/封装，不要求被包装者感知 Channel 的存在。参考：module_channel、typer_channel。

## 6. 发现与使用

```bash
# 列出所有正式 channel 类型
moss codex channeltypes

# 反射单个 channel 的完整接口
moss codex channeltypes app_store_channel

# 带依赖反射
moss codex channeltypes app_store_channel --deps
```

运行时环境的能力视图用 `moss manifests channels`，不是 codex。二者的区别：

| | `codex channeltypes` | `manifests channels` |
|---|---|---|
| 视角 | 开发时——有哪些预制能力可用 | 运行时——当前环境的 Channel 树 |
| 来源 | `ghoshell_moss.channels` 包 | workspace manifests |
| 使用者 | 开发新功能/新 app 前查阅 | 调试/理解当前运行环境 |

## 7. 测试

单测路径：`tests/ghoshell_moss/channels/`

参考模式：
- `chan.bootstrap()` 上下文管理器获取 runtime
- `runtime.get_command("name")` 验证命令存在
- `runtime.execute_command("name", args=(...))` 验证执行正确
- `runtime.self_meta()` / `runtime.metas()` 验证元信息

只测本模块职责。CTML 解析、调度时序等问题由各自模块的测试覆盖。

## 8. 深入调研

```bash
# 查看本目录的历史演进
git log -- src/ghoshell_moss/channels/

# 结合 feature 记录理解设计决策
moss --ai features specification

# 核心抽象
moss codex get-interface ghoshell_moss.core.concepts.channel:Channel
moss codex get-interface ghoshell_moss.core.blueprint.channel_builder:Builder
moss codex get-interface ghoshell_moss.core.blueprint.states_channel
```
