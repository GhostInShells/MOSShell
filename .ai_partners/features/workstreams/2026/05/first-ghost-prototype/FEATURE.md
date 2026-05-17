---
title: First Ghost Prototype
status: in_progress
priority: P0
created: 2026-05-14
updated: 2026-05-17
step: 10f_done
depends: []
milestone:
description: >-
  从零开发第一个完整的 Ghost 原型——将 Ghost/Mindflow 抽象转化为可运行的智能体实现，打通 "感知→思考→执行" 三循环。
---

# First Ghost Prototype

> 入口文档。新的 AI 实例读这个即可恢复上下文，然后按索引进入子文档。

## 这是什么

Ghost 是 MOSShell 最核心的模块——持久化智能体的运行时。ABC 已定义，Mindflow 调度链路已验证，但不存在完整可运行的 Ghost 实现。本 workstream 从零构建第一个原型。

## 文件夹结构

```
first-ghost-prototype/
├── FEATURE.md              # 本文件：状态追踪 + 子文档索引
├── DESIGN.md               # 设计结论汇总（随讨论推进逐步填入）
├── TASKS.md                # 任务分解 + 当前进度
└── discuss/                # 每步讨论的完整记录
    ├── 01-ghost-abc-positioning.md
    ├── 02-minimal-prototype-goals.md
    ├── 03-infrastructure-preparation.md
    ├── 04-runtime-integration.md
    ├── 05-matrix-tui-integration.md
    ├── 06-full-link-testing.md
    └── 07-documentation.md
```

**三个顶层文件的分工**：

| 文件 | 角色 |
|------|------|
| `FEATURE.md`（本文件） | 入口 + 状态 + 索引。不存详细内容，只存指针。 |
| `DESIGN.md` | 最终设计结论。每个讨论步骤结束后更新。精简、声明式。 |
| `TASKS.md` | 实施追踪。任务分解、依赖、进度。跨 session 追踪。 |
| `discuss/` | 每步讨论的现场记录。保留决策理由和权衡过程。 |

## 推进方法论

**模式**：人类引导讨论 → AI 记录决策 → 生成子文档 → FEATURE.md 更新索引。

**纪律**：
1. 一步一文档。完成一个讨论步骤后，才进入下一步。
2. 不提前调研。避免上下文窗口膨胀和压缩丢失。
3. 每个 discuss 文档记录：背景、讨论要点、决策结论、保留的锚点对话。
4. DESIGN.md 在每个讨论步骤结束后更新，只记录"结论"不记录"过程"。
5. FEATURE.md 在每个讨论步骤结束后更新 `updated` 日期和状态。

## 推进步骤

| # | 讨论主题 | 产出 | 状态 |
|---|---------|------|------|
| 1 | Ghost ABC 重新审视 + GhostFactory 定位 | `discuss/01-ghost-abc-positioning.md` | done |
| 2 | 最小原型技术目标 | `discuss/02-minimal-prototype-goals.md` | done |
| 3 | 基建依赖准备 | `discuss/03-infrastructure-preparation.md` | done |
| 4 | Ghost runtime 集成方式 | `discuss/04-runtime-integration.md` | done |
| 5 | Matrix / TUI 集成 | `discuss/05-matrix-tui-integration.md` | done |
| 6 | 全链路测试方案 | `discuss/06-full-link-testing.md` | done |
| 7 | 文档准备 | `discuss/07-documentation.md` | done |
| 8 | Ghost ABC 微调 + 三层抽象落地 | `ghost.py` 注释/文档 | done |
| 9 | _meta / _runtime 实现 + 测试 | `ghosts/atom/` (4 files, 24 tests) | done |
| 10 | GhostRuntime 包裹模式 | 生命周期编排 | pending |
| 10a | SystemPrompter tree 模型 | tree model + MossSystemPrompter 迁至 blueprint.host | done |
| 10b | GhostRuntime ABC 定义 | ABC 5 成员 + 架构选型 (方案4) | done |
| 10c | GhostRuntimeImpl skeleton | wiring: providers → moss → ghost → mindflow | done |
| 10d | action observe 回路闭合 | 流式 feed → interpreter → action.outcome() + 异常分级 | done |
| 10e | session signal 路由 | session → mindflow bridge + matrix logger 输出 | done |
| 10f | task output 协议 | `on_task_done` → session.output, 结算只发 status | done |

## 实现阶段关键决策（2026-05-16）— GhostRuntime 架构选型

GhostRuntime 的四种架构方案与最终决策，详见 [DESIGN.md](./DESIGN.md)。

**最终选择方案 4**：GhostRuntime 持有 MossRuntime，组合优于继承。ABC 五个成员（`.moss`/`.ghost`/`.meta`/`.container`/`close()`），ghost + mindflow main loop 托管给 `matrix.create_task`。

定义位置: `ghoshell_moss/core/blueprint/host.py` — 与 `MossRuntime` 同文件。

---

## 实现阶段关键决策（2026-05-16）— SystemPrompter tree 模型

在为 GhostRuntime 做准备时，发现 SystemPrompter 当前是压平字符串，Ghost 无法感知层次来插入 soul。
在与人类工程师讨论后，将 SystemPrompter 从 flat model 重构为 tree model。

核心决策：

1. **SystemPrompter 作为 tree node** — `children() -> dict[str, SystemPrompter]` 暴露子节点树。每个子节点自身也是 SystemPrompter，递归可组合。不使用 `@property` — 函数签名自解释。

2. **MossSystemPrompter 命名访问器** — `ctml_instruction() / project_instruction() / mode_instruction() / static_instruction()` 四个 concrete 方法，是对 children key 的便捷包装。不是 abstractmethod — 不排斥任意插槽。

3. **flatten() + linear()** — `flatten()` 返回 `[(path, desc)]` 使树可自解释、可测试结构完整性。`linear(slots)` 按给定顺序拼装，Ghost 用来在层间插入 soul。

4. **MossSystemPrompterImpl** — 钻石继承 `BaseSystemPrompter + MossSystemPrompter`，零额外代码。注册为 `SystemPrompter` 和 `MossSystemPrompter` 双 IoC key。

5. **描述自解释** — `description()` 每节点可选声明。MatrixImpl 构建时填写，AI 遍历 `flatten()` 即理解全局。

涉及文件：
- `contracts/system_prompter.py` — tree model 实现
- `host/system_prompter.py` — MossSystemPrompterImpl
- `host/matrix.py` — _prepare_system_prompter 构建树

### 2026-05-16（续）— 迁移 MossSystemPrompter 至 blueprint.host

将 `MossSystemPrompter` 从 `contracts/system_prompter.py` 搬迁至 `blueprint/host.py`：

6. **遵循 channel_builder 模式**：MOSS 特定约定与通用 contracts 分离。`SystemPrompter` + `BaseSystemPrompter` 留在 contracts（通用组件），`MossSystemPrompter` 放在 `blueprint/host.py` 与 `MossRuntime`/`MossHost` 同簇。

7. **Slot 常量消除魔法值**：`CTML_SLOT = 'ctml'`, `PROJECT_SLOT = 'project'`, `MODE_SLOT = 'mode'`, `MOSS_STATIC_SLOT = 'static'`。upper fractal 重构因魔法值付出了巨大代价，此处从源头避免。

8. **MossRuntime.system_prompter 便捷属性**：`self.matrix.container.force_fetch(MossSystemPrompter)`，消费者不感知 IoC key。

9. **static slot 在 _bootstrap_after_matrix() lazily 注入**：`self._ctml_shell.static_messages` 作为 callable 传递，每次 instruction() 动态求值。Shell 启动后才可用。

### 验证手法 — 基于抽象，零上下文集成测试

关键发现：只用 blueprint + contracts 抽象，不引用任何 host 实现，可以独立运行全链路验证：

```python
# 单文件窗口：从抽象入口直达运行时验证
from ghoshell_moss.core.blueprint.host import MossHost, MossSystemPrompter

host = MossHost.discover()
runtime = host.run(run_shell=False)
async with runtime:
    prompter = runtime.system_prompter
    # children() / flatten() / named accessors 全链路验证
```

这个模式验证了：
- `MossHost.discover()` 环境发现 → `MossRuntime` → `MossSystemPrompter` 全链路连通
- CTML 真实加载（6146 chars）
- tree 结构 `ctml → project → mode → static` 按序排列
- `flatten()` 自解释输出正常

基于抽象的单文件窗口是 MOSS 架构探索的核心模式 — 不依赖 IDE、不需要启动 REPL、不需要理解 host 实现。

---

## 实现阶段关键决策（2026-05-15）

此轮实现了 Atom 原型的完整代码 + 24 个测试。关键设计决策：

1. **Meta 即工厂**：`build_agent()` / `build_instruction()` 放在 `AtomMeta` 上而非 module 级函数。
   `build_agent` 不依赖 IoC 可独立单测；`factory(container)` 组合两者产出 `Atom` runtime。

2. **显式依赖 + env var 兜底**：`model: Model | None` 和 `provider: Provider | None` 作为 type hint，
   None 时走 `AnthropicModel` + 环境变量。既声明了依赖，又保留了开箱即用的便利。

3. **soul 双通道**：`soul_path` (str/Path/None→name) 文件加载 + `soul_content` (非 None 时跳过加载)。
   兼顾生产环境和测试。

4. **on_agent_build 回调**：外部注入 hook，不耦合 Agent 构建逻辑到 Meta 内部。

5. **消息 Adapter 独立文件**：`_adapter.py` 专注 MOSS Message → pydantic AI 转换。当前处理 text + base64 image，
   后续替换为正式 MessageAdapter 时路径清晰。

6. **package 内测试**：对于有特殊依赖（pydantic AI）的模块，测试放在 package 内 (`test_atom.py`)。
   只用真依赖路径，不堆 MagicMock。24 个测试覆盖 soul 加载、instruction 组装、消息协议、adapter 转换、生命周期。

## 实现阶段关键决策（2026-05-17）— GhostRuntimeImpl + Ghost ABC 清理

此轮实现了 GhostRuntimeImpl 并清理了 Ghost ABC。关键决策：

1. **删除 `Ghost.nuclei()`**：实例方法造成语义漂移——nuclei 的创建和注册应走 `GhostMeta.nuclei_manifests()` 工厂路径。nuclei 实例由 meta 工厂在容器 bootstrap 后产出，GhostRuntime 负责将它们注册到 Mindflow。

2. **Ghost 自行组装 instruction**：GhostRuntime 不替 ghost 注入 soul 到 SystemPrompter tree。ghost 自己从 IoC 拿 prompter，自己决定如何组合。Atom 的 `build_instruction()` 已经是这个模式。

3. **Mindflow 解析优先级**：`ghost.mindflow()` > `IoC.get(Mindflow)` > `BaseMindflow`。结果 `container.set(Mindflow, ...)` 回绑，让 Session 等组件可发现。

4. **三循环用 janus.Queue + asyncio tasks**：v0 用协程而非多线程。janus 保留用于线程安全的未来兼容。

5. **构造时检查 MossRuntime 未启动**：`is_running()` 为 True 则抛 `RuntimeError`。GhostRuntime 拥有完整生命周期。

6. **`_action_loop` 流式执行 + observe 回路闭合**：`action.received_logos()` (stream) → `interpreter.feed(delta)` (stream)，不粘合 string 做批量。三个明确阶段：feed → compile → execute。`InterpretError` 被捕获，`interpretation.observe` + `as_messages()` 通过 `action.outcome()` 回传 Mindflow，模型在下一轮 Moment 看到错误可自我纠正。

7. **异常四级分级**：(1) InterpretError — 可管理中断，observe=True；(2) Task 级失败 — 单个命令异常，不中断整体；(3) 静默失败 — log 不呈现；(4) 致命异常 — 向外传播。

8. **matrix 信息输出预留**：logos 流式解析过程和 interpreter 结算两处标记 TODO，后续接入 matrix logger。

## 实现阶段关键决策（2026-05-17）— task output 协议

将 action loop 的产出接入 `session.output()` 总线，使端侧 consumer（TUI panel、测试断言）可通过统一协议消费 ghost 执行过程。

核心决策：

1. **task 级输出走 `on_task_done`**：每个 CommandTask 完成时通过 `interpreter.on_task_done()` 回调实时产出 `OutputItem(role='task', ...)`。有消息时带 `task_result().as_messages()`，无消息时只带 `log=f"{caller_name} done"`。consumer 可看到逐命令的执行进度。

2. **结算只发 status_messages**：`wait_stopped()` 后只产出 `OutputItem(role='system', *status_messages())`。不再发全量 `executed_messages()`——因为每个 task 的结果已通过 `'task'` 逐条发送。`action.outcome()` 也只收 status，模型在下一轮 Moment 看到的是摘要（"5 done, 1 failed"），不是全量结果。

3. **异常走 `'error'`**：InterpretError 捕获后 `session.output('error', str(exception))`，与 task 级别的 `'task'` 区分。

4. **角色分配总结**：

| role | 来源 | 时机 |
|------|------|------|
| `'logos'` | moment (待实现) | articulate loop 产出 |
| `'task'` | `on_task_done` → `task_result()` | 每个 task 完成 |
| `'error'` | InterpretError / critical failure | 异常捕获点 |
| `'system'` | `status_messages()` only | interpreter 结算 |

涉及文件：`host/ghost_runtime.py` — `_stream_execute` 注册 `on_task_done` 回调 + 结算改走 status。

## 实现阶段关键决策（2026-05-17）— observe 回路 + 流式执行

## 认知准备（已完成 2026-05-14）

首轮探索确定了以下认知基线：

1. **测试体系**：`tests/` 下 ~70 个测试文件。Ghost 零测试。Mindflow 有三个测试文件覆盖三循环调度。
2. **CTML**：流式命令标记语言。Code as Prompt、时间第一公民、树形 Channel、结构化并发。
3. **Mindflow ABC**：三循环（感知/思考/执行）全双工调度中枢。Signal → Nucleus → Impulse → Attention → Articulator/Action。
4. **Ghost ABC**：`system_prompt()` + `memories()` + `articulate()` + `channel()` + `mindflow()`。核心是 `articulate()`。
5. **Mindflow 测试**：`BaseMindflow` / `BaseAttention` / `BufferNucleus` 已通过测试验证三循环链路可运行。

### 架构关系速览

```
Signal (感知信号) → Nucleus (加工/降频) → Impulse (动机)
                                              ↓
Mindflow (调度中枢) → Attention (单次运行态)
                                              ↓
                        Articulator (推理) + Action (执行)
                                              ↑
                              Ghost.articulate() 提供 Logos
```

---

*本文件在每个讨论步骤完成后更新。新 AI 实例：读此文件 → 检查 TASKS.md 了解进度 → 进入当前讨论步骤。*
