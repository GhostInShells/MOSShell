# Channels — AI 开发指南

## 1. Quick Start: 三个文件理解 Channel 体系

按顺序读:

1. `src/ghoshell_moss/core/ctml/prompts/v1_0_0.zh.md` — CTML 语法规范。理解 Channel 是什么、命令如何调度、时间第一公民。
2. `src/ghoshell_moss/core/blueprint/channel_builder.py` — Builder API。理解如何注册命令、instruction、context、生命周期。
   ```bash
   moss --ai codex get-interface ghoshell_moss.core.blueprint.channel_builder
   ```
3. `tests/ghoshell_moss/core/ctml/v1_0/test_ctml_v1.py` — 测试即文档。所有核心语义（并行、occupy、scope、timeout、流式参数）都有可运行的用例。

## 2. 本目录的开发范式

本目录存放 channel 原型。既是实现，也是案例。

### 两种封装策略

**Channel Interface** — 先定义接口，再注册。

类自身声明命令方法，在 `bootstrap()` 中通过 `Builder.command()` 注册。`Builder.command()` 的 `interface` 参数更进一步：可以给模型看一个虚拟函数签名，实际执行另一个实现体。

原型: `speech_channel.py` — `SpeechChannel.say()` 定义接口，`bootstrap()` 中用 `chan.build.command()(self.say)` 注册。mac_channel.py — `run()` 函数定义 JXA 执行接口，`new_mac_control_channel()` 中注册。

**As Channel** — 外部已有的事物包装为 Channel。

把 module、CLI、API、设备等外部能力反射/封装为 channel command，不要求被包装者感知 Channel 的存在。

原型: `module_channel.py` — 反射任意 Python module 的函数。`typer_channel.py` — 把 Typer CLI 包装为 Channel。

### 当前原型

| 文件 | 策略 | 亮点 |
|------|------|------|
| `module_channel.py` | As Channel | L0: 零手工，一行反射 module |
| `mac_channel.py` | Channel Interface | L1: Builder 装饰器注册单个命令 |
| `speech_channel.py` | Channel Interface | L2: 继承 Channel ABC，封装 contract |
| `typer_channel.py` | As Channel | L1+: instruction + context_messages + 经验记忆 |

### 零依赖原则

本目录下的原型只用 MOSS core + Python stdlib。额外依赖留给 app (独立 `pyproject.toml`) 处理。

## 3. 高阶抽象: StatefulChannel

当 Channel 需要在运行时切换状态（不同状态有不同命令集、指令、上下文）时，用 `states_channel.py`。

```bash
moss --ai codex get-interface ghoshell_moss.core.blueprint.states_channel
```

核心概念:

- `ChannelState` — 一个运行时状态。有 `own_commands()`、`get_instruction()`、`get_context_messages()`、生命周期钩子。
- `ChannelStateBuilder` — 既是 `Builder` 又是 `ChannelState`，支持 `add_virtual_channel()` / `remove_virtual_channel()` 运行时增减子通道。
- `StatefulChannel` — 持有多个 `ChannelState`，通过 `with_state()` 注册，运行时切换。
- `PrimeChannel` — `StatefulChannel + MutableChannel`，全能。

工厂函数: `new_state_builder()`, `new_channel_from_state()`, `new_stateful_channel()`, `new_prime_channel()`.

## 4. 封装策略选择

| 你要做的事 | 策略 | 层级 |
|-----------|------|------|
| 反射已有 Python module 的函数 | As Channel | L0 `new_module_channel()` |
| 包装 CLI / API / 外部能力 | As Channel | L1 `new_channel()` + Builder |
| 定义新能力，手工控制接口签名 | Channel Interface | L1-L2 Builder 或继承 ABC |
| 封装外部 contract (TTS、播放器等) | Channel Interface | L2 继承 Channel ABC |
| 运行时切换能力集 | 任一策略 + State | L3 StatefulChannel |
| 全部能力 | 任一策略 + Prime | L4 PrimeChannel |

`Builder.command()` 的关键参数决定了函数级驱动的丰富程度:

- `name` — 重命名
- `doc` / `comments` / `interface` — 控制模型看到的签名（interface 可指向虚拟函数）
- `blocking` — 是否阻塞同通道后续命令
- `available` — 动态可用性
- `call_soon` / `priority` — 抢占调度

## 5. 开发规范

### 测试

单测路径: `tests/ghoshell_moss/channels/`

参考 `tests/ghoshell_moss/core/channels/test_py_channel.py` 的模式:

- `chan.bootstrap()` 上下文管理器获取 runtime
- `runtime.get_command("name")` 验证命令存在性
- `runtime.execute_command("name", args=(...))` 验证执行正确性
- `runtime.self_meta()` / `runtime.metas()` 验证元信息

只测本模块的职责。CTML 解析、调度时序、caller name 格式等问题由各自模块的测试覆盖。

### 提交

遵循项目 Git 提交规范。FEATURE.md 与代码同 commit。

### 深入调研

```bash
# 查看本目录的历史实现和演进
git log -- src/ghoshell_moss/channels/

# 查看某个文件的变更轨迹
git log -p -- src/ghoshell_moss/channels/module_channel.py

# 结合 feature 记录理解设计决策
moss --ai features specification
```

本目录下的代码既是实现，也是案例。新原型可以参考已有原型的模式，选择最接近的作为起点。
