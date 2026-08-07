# Channels — 正式能力模块

本目录是 MOSS 随包分发的正式 Channel 类型。通过 `moss codex channeltypes` 索引。
每个 channel 的具体能力由其代码与 docstring 自解释——本文件只记录跨 channel 的约定。

## 1. 模块约定

### 1.1 Docstring 范式

每个模块第一行 docstring 采用机器可解析格式：

```python
"""一句话功能描述 | 功能类型 | 状态
"""
```

- `功能类型` 与 `状态` 由各 channel 自行声明（自由取值）
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

## 2. Status

三态，线性推进：

```
alpha → beta → active
```

| Status | 含义 |
|--------|------|
| `alpha` | 原型/草图，无测试，接口随意改 |
| `beta` | 功能可用，接口可能变动，需要更多实际使用验证 |
| `active` | 正式维护，有测试覆盖，接口兼容承诺，跟随项目 semver |

进入 `active` 后，接口变更需跟随项目的语义化版本号。

## 3. 开发前必读

开发 channel 前先过一遍下面三条命令，覆盖关注点：

```bash
moss codex blueprint channel_builder   # Builder API、生命周期、CommandUtil
moss codex blueprint states_channel    # StatefulChannel / ChannelModule / PrimeChannel
moss ctml read                          # 三种阻塞机制、Observe 语义、context 供给
```

## 4. 发现与使用

```bash
# 列出所有正式 channel 类型
moss codex channeltypes

# 反射单个 channel 的完整接口
moss codex channeltypes <name>

# 带依赖反射
moss codex channeltypes <name> --deps
```

运行时环境的能力视图用 `moss manifests channels`，不是 codex。二者的区别：

| | `codex channeltypes` | `manifests channels` |
|---|---|---|
| 视角 | 开发时——有哪些预制能力可用 | 运行时——当前环境的 Channel 树 |
| 来源 | `ghoshell_moss.channels` 包 | workspace manifests |
| 使用者 | 开发新功能/新 app 前查阅 | 调试/理解当前运行环境 |

## 5. 测试

单测路径：`tests/ghoshell_moss/channels/`

参考模式：
- `chan.bootstrap()` 上下文管理器获取 runtime
- `runtime.get_command("name")` 验证命令存在
- `runtime.execute_command("name", args=(...))` 验证执行正确
- `runtime.self_meta()` / `runtime.metas()` 验证元信息

只测本模块职责。CTML 解析、调度时序等问题由各自模块的测试覆盖。

## 6. 深入调研

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
