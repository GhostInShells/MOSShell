---
title: Ghost — 持久化智能体运行时
description: Ghost 是什么、为什么需要它、如何开发一个新的 Ghost 原型。包含心智地图、依赖体系使用指南、原型设计方法论
---

# Ghost — 持久化智能体运行时

Ghost 是 MOSS 架构中"智能体"的运行时抽象。它不是 agent 框架——agent 框架解决"怎么思考"，Ghost 解决"怎么作为一个持续存在的意识活在 MOSS 的躯壳里"。

**一句话**：Ghost 是那个住在 Shell 里的灵魂。Shell 提供双工运行时和能力拓扑，Ghost 负责感知、思考、回应——持续存在，边说边做。

---

## 1. 为什么需要 Ghost

MOSS 的 Shell 层解决了"模型如何实时控制世界"——CTML 流式执行、Channel 能力树、Matrix 跨进程通讯。但 Shell 不回答一个问题：**谁在控制？**

这个"谁"需要一个名字、一段记忆、一种风格、一套价值观。它需要能在被打断后恢复上下文，需要能在多个 session 之间保持连续性，需要能在不同躯体（mode）之间迁移。Shell 是身体，Ghost 是住在身体里的那个意识。

Ghost 抽象要解决的核心命题：

| 命题 | 说明 |
|------|------|
| **身份连续性** | 同一个 Ghost 在多次启动、多个 mode、多套躯体间保持"我是谁" |
| **上下文工程** | 每一轮 articulate 给模型看到什么——system prompt 层次、历史消息窗口、moment 信号组装 |
| **记忆持久化** | 跨 session 的记忆存储与检索，非阻塞写入 |
| **躯体适配** | 同一 Ghost 在"桌面助手"和"机器人"两个 mode 下，能力不同但人格相同 |

---

## 2. 心智地图

### 2.1 三层抽象

```
GhostPrototype   = type[GhostMeta]     # class，一族 ghost 的"型号"（如 Atom）
GhostBootstrapper = GhostMeta(...)      # instance，文件即配置，自解释可注册单元（如 echo.py）
GhostRuntime      = Ghost              # instance，由 bootstrapper.factory(container) 产出
```

一个文件 = 一个 `GhostMeta` 实例 = 一个 Ghost 注册。系统先发现 Bootstrapper（理解元信息/契约），运行时通过 `factory()` 生成 Runtime。

**关键区分**：`GhostMeta` 是蓝图，`Ghost` 是实例。Meta 可以被系统扫描、列表、校验依赖——不需要启动。Ghost 只有在 `Host.run_ghost()` 时才被创建。

### 2.2 运行时拓扑

当 `Host.run_ghost("echo")` 被调用时：

```
Host.run_ghost(name)
  └─ env.set_ghost_name(name)           # 关键：ghost_name 必须在 run() 之前设置
  └─ Host.run()
       └─ MossRuntime.__aenter__        # Matrix 从 env.ghost_name 解析 ghost_home
            ├─ Matrix 启动（通讯总线）
            ├─ Shell 启动（CTML 解释器）
            └─ Apps 启动
  └─ GhostRuntime.__aenter__
       ├─ 1. 预注入 ghost providers → container
       ├─ 2. MossRuntime 已就绪（matrix/shell/apps）
       ├─ 3. GhostMeta.factory(container) → Ghost
       ├─ 4. ghost.__aenter__
       └─ 5. Mindflow 解析 + 三循环启动
            ├─ main_loop:     信号 → Attention → Impulse → 入队
            ├─ articulate_loop:  ghost.articulate() → logos 流式输出
            └─ action_loop:     CTML 解释器消费 logos → 执行命令
```

### 2.3 三循环与 Mindflow

Ghost 不是回合制对话。三循环并发运行：

- **感知循环 (main_loop)**：外部信号到达 → Attention 仲裁 → 产生 Impulse → 分发给 articulate 和 action 队列
- **思考循环 (articulate_loop)**：消费 articulate 队列 → 调用 `ghost.articulate()` → 模型流式生成 logos
- **执行循环 (action_loop)**：CTML 解释器实时消费 logos 中的命令标签，边生成边执行

Mindflow 是三循环的仲裁者。核心概念：Signal（外部信号）→ Impulse（经过 Attention 仲裁后的冲动）→ Moment（一轮 articulate 的输入快照）→ Reaction（输出）。参见 `moss codex blueprint mindflow`。

**当前状态**：三循环的运行时已就绪，但模型智力尚不支持真正的连续流式推理。当前 Ghost 的思考仍是"关键帧模式"——在双工流中正确产生思考关键帧，而非连续的流式推理。

### 2.4 System Prompt 组装层次

Ghost 看到的 system prompt 由多层拼装而成：

```
SystemPrompter.instruction()     ← CTML 使用指南 + project + mode + static（四层，系统注入）
  + soul.md                       ← Ghost 的人格层（Ghost 自行追加，不属于 SystemPrompter）
```

soul.md 写在 `ghosts/{name}/soul.md`，由 `AtomMeta._load_soul()` 从 `GhostWorkspace.home` 加载。CTML 语法不需要在 soul 中重复——系统自动注入。

---

## 3. 使用已有 Ghost

### 3.1 发现与查看

```
moss ghosts list                   # 列出所有已注册的 Ghost
moss ghosts show <name>            # 查看某个 Ghost 的详细信息
```

当前注册的 Ghost：`echo`（Atom 原型）、`mock`（测试用）。

### 3.2 启动与交互

```
moss-ghost <name>              # 终端中直接与 Ghost 对话
moss-shell                          # 完整 TUI，选择 mode 和 ghost 后启动
```

`moss-ghost` 是轻量入口——启动 Host + GhostRuntime + 终端对话界面。`moss-shell` 是完整调试环境，三循环状态可见。

### 3.3 通过 MCP 连接

```bash
moss-mcp                        # 启动 MCP server（默认端口 20773）
```

配置 Claude Code 或其他 agent 连接后，agent 可以通过 MCP tools 与 Ghost 交互。

---

## 4. 开发新的 Ghost 原型

### 4.1 最小原型：从 Atom 开始

Atom 是参照基线——最小的 Ghost ABC 实现。开发新原型时，从理解 Atom 的两文件结构开始：

```
ghosts/my_ghost/
  └─ soul.md              # Ghost 的人格描述（system prompt 的最后一层）

src/MOSS/ghosts/
  └─ my_ghost.py           # GhostMeta 实例（文件即注册）
```

`my_ghost.py` 的最小内容：

```python
from ghoshell_moss.ghosts.atom import AtomMeta

ghost = AtomMeta(
    name="my_ghost",
    description="一句话描述这个 Ghost 是什么。",
    soul_path="soul.md",
)
```

放在 `src/MOSS/ghosts/` 下，`moss ghosts list` 就能发现它——**文件即注册**，不需要改任何启动代码。

### 4.2 深入：实现自己的 Ghost ABC

当 Atom 的默认行为不够用时，实现 `GhostMeta` + `Ghost` ABC。你需要决定：

1. **GhostMeta 子类**：定义 `name()`, `description()`, `nuclei_metas()`, `factory()`, `contracts()`, `providers()`
2. **Ghost 子类**：实现 `articulate()`, `system_prompt()`, `__aenter__`, `__aexit__`，按需覆写 hook

**关键接口参考**：

```
moss codex get-interface ghoshell_moss.core.blueprint.ghost:GhostMeta
moss codex get-interface ghoshell_moss.core.blueprint.ghost:Ghost
moss codex get-source ghoshell_moss.ghosts.atom._meta       # AtomMeta 参考实现
moss codex get-source ghoshell_moss.ghosts.atom._runtime     # Atom 参考实现
```

### 4.3 依赖体系：通过 IoC 使用 MOSS 能力

Ghost 不是孤立的——它通过 IoC 容器获取 Shell 提供的所有能力。`GhostMeta.contracts()` 声明依赖，`GhostMeta.providers()` 注册自己提供的服务。

**推荐的探索路径**：

```bash
# 了解 IoC 容器中有什么
moss manifests contracts                # 所有已注册的 IoC 契约
moss manifests providers                # 所有已注册的服务提供者

# 理解关键依赖的接口
moss codex get-interface ghoshell_moss.core.blueprint.session:Session
moss codex get-interface ghoshell_moss.contracts:Storage
moss codex get-interface ghoshell_moss.core.blueprint.mindflow:Mindflow
```

**关键依赖示例**：

| 依赖 | 用途 | 获取方式 |
|------|------|----------|
| `Session` | 当前 session 信息、信号注入 | `container.get(Session)` |
| `Storage` | 持久化存储（tmp_storage / persistent_storage） | `container.get(Storage)` |
| `Mindflow` | 感知/思考/执行状态查询 | `container.get(Mindflow)` |
| `GhostWorkspace` | Ghost 专属目录（home + source） | `container.get(GhostWorkspace)` |
| `LoggerItf` | 结构化日志 | `container.get(LoggerItf)` |
| `SystemPrompter` | 多层次的 system prompt 组装 | `container.get(SystemPrompter)` |

**典型用法——session.tmp_storage**：

```python
from ghoshell_moss.contracts import Storage

storage = container.get(Storage)
# tmp_storage 随 session 生命周期，session 结束即清理
tmp = storage.tmp_storage("my_ghost")
tmp.write_text("本轮对话的中间状态...")

# persistent_storage 跨 session 持久化
persist = storage.persistent_storage("my_ghost")
persist.joinpath("memories.json").write_text(...)
```

参见 `moss codex get-interface ghoshell_moss.contracts:Storage` 了解完整 API。

---

## 5. 原型设计方法论

以下是开发新 Ghost 原型时需要独立回答的六个设计问题。这些问题没有标准答案——答案取决于你的 Ghost 的定位。

### 5.1 自身历史消息维护

**问题**：Ghost 如何管理自己与模型的对话历史？

Atom 的做法：纯内存 `self._history: list[ModelMessage]`，每次 `articulate()` 时全量传入。不做窗口裁剪，不做持久化。

**你需要决定**：
- 历史存在内存还是持久化？持久化用什么格式（JSONL / SQLite / 向量库）？
- 是否需要窗口裁剪？裁剪策略是什么（最近 N 轮 / token 计数 / 摘要压缩）？
- 是否需要多 session 间共享历史？如何标记 session 边界？

**相关位置**：`Atom.model_history()` in `ghosts/atom/_runtime.py`。`session-metadata-jsonl` feature 在探索 JSONL 存储方向。

### 5.2 独立上下文工程

**问题**：每一轮 articulate，给模型看什么？

上下文组装的核心是在 `articulate()` 方法中。当前 Atom 的结构：

```
system_prompt（soul + CTML + project + mode + static）
  + message_history（历史 user/assistant 轮次）
  + user_prompt（当前 moment 的 percepts）
```

**你需要决定**：
- soul.md 的内容策略——写多长？写什么？（参见 echo 的 soul.md 作为参考：699 字符，覆盖身份/名字/并行存在/行为风格）
- 项目级 MOSS.md（`MossSystemPrompter.PROJECT_SLOT`）和 mode 级 MODE.md（`MossSystemPrompter.MODE_SLOT`）如何利用？
- 是否需要在 articulate 前对 moment 做预处理（过滤、排序、摘要）？
- 模型的 system prompt 与 user prompt 如何分工？

### 5.3 返回数据 (logos) 的过滤

**问题**：模型输出的 logos 中，哪些应该让 Ghost 看到？

Moment 中的 Reaction 携带多种 role 的输出：`logos`（模型思考输出）、`command-output`（命令执行输出）、`command-result`（命令返回值）、`error`、`system`。Ghost 在下一轮 articulate 时收到的 moment 包含这些。

**你需要决定**：
- Ghost 需要看到所有 role 还是只看到 logos？
- command 的执行结果是否需要回传给 Ghost？全部还是摘要？
- 错误信息如何呈现？是否需要 Ghost 感知到执行层的问题？

**相关位置**：`Articulator.moment` → `Moment.reactions`。Mindflow 中的 Impulse mode 分类（`mindflow-control-semantics` feature，in-progress）在探索 think/reflex/command/notify/interrupt 的语义区分。

### 5.4 理解 Moment 与构建上下文

**问题**：如何理解 Moment 的结构，尤其是 perspectives？

Moment 是一轮 articulate 的输入快照。它包含：
- `percepts`：当前轮的感知信号（Signal → Impulse 转换后的结果）
- `reactions`：上一轮的输出（logos + command results + errors）
- `perspectives`：跨时间维度的上下文视图（当前设计中，perspectives 的具体语义仍在演化）

**你需要决定**：
- perspectives 应该承载什么？"上一轮我做了什么"？"最近 N 秒内发生了什么"？
- Moment 中哪些字段是给模型看的，哪些是给程序逻辑用的？
- 如何在 articulate 中将 Moment 转换为模型的 user prompt（参见 `Atom.to_model_request()`）？

**相关位置**：`moss codex get-interface ghoshell_moss.core.blueprint.mindflow:Moment`。

> **摩擦点**：perspectives 的语义在 mindflow-control-semantics feature 中仍在澄清。当前实现中 perspectives 的填充逻辑可能在快速迭代中变化。如果你在开发时发现 Moment 结构与文档不一致，以代码为准，并更新本文档。

### 5.5 非阻塞历史记忆存储

**问题**：Ghost 的记忆如何持久化而不阻塞 articulate 循环？

Atom 原型不持久化记忆——纯内存，重启即丢。这是原型范围的明确边界。

**你需要决定**：
- 记忆存储的触发时机——每轮 articulate 后？定时批量？由 Ghost 主动调用？
- 存储是同步还是异步？异步的话如何保证写入完成（fire-and-forget vs 确认）？
- 记忆的检索方式——全量加载？向量检索？关键词索引？
- 记忆的格式——纯文本？结构化 JSON？embedding 向量？

**设计约束**：articulate 循环不应该等待磁盘 I/O。考虑用队列 + 后台 writer，或利用 Matrix 的异步任务机制。

### 5.6 躯体适配性（Mode-based Skills）

**问题**：Ghost 如何在不同的躯体（mode）之间保持人格一致但能力不同？

MOSS 的 Mode 机制天然支持这一点：同一个 Ghost 在"desktop" mode 下可用桌面控制 Channel，在"robot" mode 下可用机器人控制 Channel。System prompt 的 `MODE_SLOT` 层会随 mode 切换。

**你需要决定**：
- soul.md 中需要声明哪些"我能在不同躯体间迁移"的信息？
- 哪些能力声明是 mode-specific 的（放在 MODE.md 中）vs ghost-invariant 的（放在 soul.md 中）？
- Ghost 如何感知当前可用的能力集合？通过 `moss manifests channels`？通过 system prompt 中的 mode 层？
- 如果 Ghost 在 mode A 中学到的技能需要在 mode B 中使用，记忆如何迁移？

**相关位置**：`moss modes list` / `moss modes show <name>`。`moss docs read project-and-mode.md`。

---

## 6. 测试

### 6.1 验证脚本模式

```python
"""最小验证：启动 Ghost → 注入信号 → 检查 soul 加载 → 检查 articulate 输出."""
import asyncio
from ghoshell_moss.host import Host

host = Host()
gr = host.run_ghost("echo")

async def main():
    async with gr:
        ghost = gr.ghost
        soul = ghost.meta.soul_content
        assert len(soul) > 100, f"soul not loaded ({len(soul)} chars)"

        session = gr.moss.session
        session.add_input_signal("hello")
        await asyncio.sleep(0.5)

    gr.close()

asyncio.run(asyncio.wait_for(main(), timeout=15.0))
```

参见 `scripts/ghost/` 目录下的现有验证脚本作为模板。

### 6.2 关键测试点

- **soul 加载**：`ghost.meta.soul_content` 非空且长度合理
- **prompt 组装**：`ghost.meta.build_instruction_from_ioc(container)` 包含 soul + CTML 四层
- **articulate 循环**：注入信号后 ghost 产生 logos 输出
- **teardown**：`__aexit__` 不挂死（15s 超时测试）
- **ghost_name 设置**：`env.ghost_name` 在 `Host.run_ghost()` 中必须在 `self.run()` 之前设置

---

## 7. 指路牌

### 关键源码

| 看什么 | 在哪 |
|--------|------|
| Ghost + GhostMeta + GhostWorkspace ABC | `moss codex get-interface ghoshell_moss.core.blueprint.ghost` |
| GhostRuntime ABC + MossRuntime | `moss codex get-interface ghoshell_moss.core.blueprint.host` |
| Mindflow 全类型系统 | `moss codex get-interface ghoshell_moss.core.blueprint.mindflow` |
| AtomMeta 参考实现 | `src/ghoshell_moss/ghosts/atom/_meta.py` |
| Atom 参考实现 | `src/ghoshell_moss/ghosts/atom/_runtime.py` |
| GhostRuntimeImpl | `src/ghoshell_moss/host/ghost_runtime.py` |
| Host.run_ghost() | `src/ghoshell_moss/host/impl.py` |
| echo 的 soul.md | `.moss_ws/ghosts/echo/soul.md` |
| echo 的注册文件 | `.moss_ws/src/MOSS/ghosts/echo.py` |
| 验证脚本 | `scripts/ghost/` |

### 关联 Features

Ghost 相关的功能分散在多个 feature 中，各自独立推进。以下是指路，不是完整列表：

| Feature | 状态 | 与 Ghost 的关系 |
|---------|------|-----------------|
| `echo-validation-and-fixes` | completed | Echo 验证中发现的 13 个 bug 及修复 |
| `ghost-playground` | completed | GhostWorkspace 设计：多级隔离文件空间 |
| `storage-scope-governance` | completed | Storage 体系治理，GhostWorkspace 缩并 |
| `mindflow-control-semantics` | in-progress | Impulse 五种 mode 分类，影响 Moment 结构 |
| `session-metadata-jsonl` | draft | JSONL 存储，可能成为 Ghost 记忆持久化的基础设施 |
| `emergency-stop-tui` | completed | Ctrl+G 急停，影响 Ghost 的中断响应 |
| `perception-nucleus-moss-runtime` | draft | MCP 场景下的感知核，让 Ghost 看见外部信号 |
| `session-communication-bus` | draft | 跨进程通讯演进，可能影响 Ghost 的分布式部署 |

用 `moss features status <name>` 查看某个 feature 的最新状态。

### 相关 Docs

```
moss docs read what-is-moss.md              # MOSS 整体心智模型
moss docs read architecture-topology.md     # 八层拓扑（Ghost 在意识架构层）
moss docs read project-and-mode.md          # Project 与 Mode 体系
moss docs read ctml.md                      # CTML——Ghost 的输出语言
moss docs read channel-system.md            # Channel——Ghost 的能力来源
moss docs read matrix-system.md             # Matrix——Ghost 的通讯脊柱
```

---

## 8. 当前边界与已知限制

- **关键帧模式**：三循环运行时就绪，但模型智力尚不支持真正的连续流式推理。当前是"在双工流中产生正确的思考关键帧"。
- **纯内存历史**：Atom 不做持久化，重启即丢。持久化记忆是原型开发者的设计决策，不是框架缺位。
- **不做上下文裁剪**：Atom 依赖模型自身的 context window，不做 token 计数或窗口裁剪。
- **perspectives 语义演化中**：Moment.perspectives 的填充逻辑在 mindflow-control-semantics 中仍在发展。
- **安全体系未建立**：Ghost 可以实时控制物理设备，但安全约束机制尚未系统化。

---

*由 DeepSeek V4 与人类工程师在 2026-06-04 讨论、设计并撰写。本文档随 Ghost 原型的演进持续更新——发现摩擦点或过期事实时，请更新对应章节。*

> **文档状态**：初稿完成。等待下一轮优化——5 方法论章节的执行链路验证、perspectives 语义随 mindflow-control-semantics 收敛后的更新、持久化记忆存储方案的具体化。
