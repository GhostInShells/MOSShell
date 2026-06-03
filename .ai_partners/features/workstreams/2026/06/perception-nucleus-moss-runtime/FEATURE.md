---
created: 2026-06-04
depends: []
description: 为 MossRuntime 内置一个最简感知核：PerceptionMindflow(不生产 Attention) + PerceptionNucleus(按
  signal type 自动创建)， 模型通过 observe 拉取 context，通过 pop_impulse 主动消费信号。让 signal 体系在 MCP
  路径可测试。
milestone: null
priority: P0
status: draft
status_note: deferred — will pick up later
title: Perception Nucleus — MossRuntime 感知核，让 MCP 场景下的模型能看见信号
updated: '2026-06-04'
---

# Perception Nucleus — MossRuntime 感知核

## Motivation

GhostRuntime 有完整的 mindflow 信号响应机制：`on_signal → mindflow.add_signal → nuclei → impulse → attention → articulate/action` 三循环。
但 MCP 场景下模型用的是 MossRuntime，它没有任何感知能力——Session 的 `on_signal` 回调链存在但无人注册，信号从 Zenoh 总线进来就丢了。模型完全是"瞎的"。

这导致 signal 体系在 MCP 路径上完全不可测试。当前最优先的事不是做一个复杂的感知系统，而是让信号能被看见。

**目标**：在 MossRuntime 里做一个最简感知核——不做 attention 生产，不做自动调度，
信号进 buffer，模型通过 `observe` 看到红点，通过 `pop_impulse` 主动消费。打通 MCP 的感知路径。

## Design

### 架构概览

```
Zenoh callback (跨线程)
  │
  ▼
PerceptionMindflow.add_signal(signal)
  ├── signal.name 未知 → 入 _new_signal_name_queue, 触发自动创建 nucleus
  ├── signal.name 已知但 nucleus 未就绪 → buffer 到 _pending_signals
  └── signal.name 已知且 nucleus 就绪 → _dispatch_signal → nucleus.add_signal
        │
        ▼ (event loop: signal consuming loop)
  PerceptionNucleus.process_signal → FIFO buffer → rebuild impulse cache
        │
        ▼ (event loop: pulse beat)
  PerceptionNucleus._impulse_beat_loop → 周期清理 stale, 刷新 cache
        │
        ▼ (模型主动调用)
  MossRuntime.moss_observe()
    └── mindflow.context_messages() → "[nucleus_name] pending: N, top: ..."
        │
  MossRuntime.moss_pop_impulse()
    └── mindflow.pop_impulse() → _rank_nuclei() → nucleus.pop_impulse() → Impulse → Message
```

### 与 GhostRuntime Mindflow 的差异

| | GhostRuntime Mindflow | PerceptionMindflow |
|---|---|---|
| Signal 路由 | janus PriorityQueue → dispatch | 同 |
| Nuclei 管理 | 预先注册 (manifests + ghost meta) | **按 signal name 自动创建** |
| Impulse 消费 | 自动挑战 attention | **模型主动 pop_impulse** |
| Attention | 生产，三循环驱动 | **不生产** |
| context_messages | 注册到 attention context_func | **observe 直接返回** |
| pop_impulse | mindflow 内部自动 | **暴露为模型工具** |

### 组件

**PerceptionNucleus** (`core/mindflow/perception_nucleus.py`) — 专用于感知场景的 nucleus，不复用 BufferNucleus/InputSignalNucleus：

- 单 signal type 监听
- FIFO buffer (maxsize=20 默认)
- asyncio.Lock 保证线程安全
- pulse beat loop 周期清理 stale 信号，刷新 impulse cache
- suppress 冷静期机制
- pop_impulse 时原子清空 buffer

**PerceptionMindflow** (`core/mindflow/perception_mindflow.py`) — 不生产 Attention 的 mindflow：

- 继承 `AbsMindflow`，复用 signal 路由基础设施
- 砍掉 `_on_impulse_consuming_loop` — 不做 attention challenge
- 砍掉 `_loop_attention` — 不生产 attention
- 新增 `_auto_create_nuclei_loop` — 从 `_new_signal_name_queue` 消费未知 signal name，动态创建 PerceptionNucleus
- 新增 `pop_impulse()` — `_rank_nuclei()` + `_pop_impulse()`
- `context_messages()` 继承自 AbsMindflow，无需修改
- 新 signal name 到达时，先 buffer 到 `_pending_signals[name]`，等 nucleus 创建完成后再 drain 分发（都在同一 event loop，无竞态）

**MossRuntime 集成** (`host/moss_runtime.py`)：

- 构造时 `enable_perception: bool = True`（默认开启）
- `__aenter__` 中创建 PerceptionMindflow，注册 nuclei，`session.on_signal(mindflow.add_signal)`
- `moss_observe()` 追加 `mindflow.context_messages()`
- 新增 `moss_pop_impulse() → Message | None` — impulse 收敛为单条 Message

### pop_impulse 返回格式

Impulse 收敛为一条 Message，元信息在前，messages 在后：

```
[perceive <source>] priority=<priority> strength=<strength>
instruction: <reaction_instruction>

-- attached messages --
<message[0].content>
<message[1].content>
...
```

原因：调试友好。MCP 场景下模型和开发者都能直接看到信号内容。

## Key Decisions

### KD1: 按 signal type 自动创建 nucleus，不走注册

**决策**: PerceptionMindflow 收到未知 signal name 时，自动创建对应的 PerceptionNucleus。
不要求预先在 manifests 注册，不走 IoC provider。

**理由**:
- MCP 场景没有 GhostMeta 的 nuclei_metas 注册链
- 信号类型是开放的——任何 Zenoh topic 都可能发来信号
- 创建逻辑通过内部 `_new_signal_name_queue` + `_auto_create_nuclei_loop` 序列化，与 signal dispatch 在同一 event loop，天然有序无竞态

**与 GhostRuntime 的对比**: GhostRuntime 从 `manifests.nuclei()` + `ghost_meta.nuclei_metas()` 预注册。
PerceptionMindflow 不依赖这些——它自发现。

### KD2: 不走 IoC provider，生命周期直接内置

**决策**: PerceptionMindflow 在 MossRuntimeImpl 中直接创建和管理，不通过 IoC container 发现。

**理由**:
- 感知能力是 MossRuntime 的基础设施，不是可选插件
- 参考 GhostRuntime._wire_mindflow() 的模式——直接在 `__aenter__` 中创建、wire、注册 on_signal
- 简单直接，无间接层

### KD3: pop_impulse 收敛为单条 Message

**决策**: `moss_pop_impulse()` 返回一条 Message 而非裸 Impulse。元信息（source/priority/strength/instruction）
在前，signal 携带的 messages 按 FIFO 顺序拼接在后。

**理由**:
- 调试友好——开发者和模型都能直接阅读
- Message 是 MCP 工具返回的标准格式
- 保留 Impulse 的全部关键信息，不丢失

### KD4: 专做 PerceptionNucleus，不复用现有实现

**决策**: 创建独立的 `PerceptionNucleus`，不继承或包装 `BufferNucleus` / `InputSignalNucleus`。

**理由**:
- `InputSignalNucleus` 用 `threading.Lock` 且无 pulse beat——对通用感知场景不合适
- `BufferNucleus` 有额外的 `_rebuild_impulse` 排序逻辑和 `strength_decay_seconds` 参数——感知核不需要
- 权责明晰，未来各自的迭代不互相影响
- 代码量很小（~150 行），不构成维护负担

### KD5: 默认开启，可通过构造参数关闭

**决策**: `MossRuntimeImpl(enable_perception=True)` 默认启用感知核。

**理由**:
- MCP 路径需要它，moss-repl 也能受益（调试时看到信号）
- 提供 `enable_perception=False` 用于不需要感知的纯 shell 场景

## 实施计划

### Step 1: PerceptionNucleus

- 文件: `src/ghoshell_moss/core/mindflow/perception_nucleus.py`
- 内容: 单 signal FIFO buffer，asyncio.Lock，pulse beat，suppress，原子 pop
- 构造参数: `name, description, target_signal, buffer_size=20, suppress_seconds=1.0, pulse_beat_interval=2.0, min_priority=DEBUG`

### Step 2: PerceptionMindflow

- 文件: `src/ghoshell_moss/core/mindflow/perception_mindflow.py`
- 内容: 继承 AbsMindflow，精简 `__aenter__`（仅 signal consuming + faculties + auto-create loop），`pop_impulse()`，auto-create nuclei 逻辑，pending signals buffer
- 重写 `_build_attention` 为 dummy（不会被调用）

### Step 3: MossRuntimeImpl 集成

- 文件: `src/ghoshell_moss/host/moss_runtime.py`
- 内容: `enable_perception` 参数，`__aenter__` 中 wire PerceptionMindflow，`on_signal` 注册，`moss_observe` 追加 context_messages，新增 `moss_pop_impulse()`
- 可选: `MossRuntime` ABC 中声明 `moss_pop_impulse` 抽象方法

### Step 4: 单元测试

- 文件: `tests/ghoshell_moss/core/mindflow/test_perception_nucleus.py`
- 覆盖: 入队/FIFO/pop/stale/suppress/buffer limit/pulse beat/多 signal type 隔离
- 文件: `tests/ghoshell_moss/core/mindflow/test_perception_mindflow.py`
- 覆盖: auto-create nucleus/signal dispatch/pop_impulse ranking/context_messages/pending buffer drain

### Step 5: MCP 路径集成测试

- 文件: `tests/ghoshell_moss/host/test_perception_moss_runtime.py`
- 覆盖: MossRuntime 启动 + 发送 signal → observe 看到 context → pop_impulse 消费 → 消费后为空

## 与 mindflow-control-semantics 的关系

`mindflow-control-semantics` (P0, in-progress) 为 Impulse 增加了 mode 分类 (think/reflex/command/notify/interrupt) 和 Mindflow 级 Buffer 机制。
PerceptionMindflow 当前不需要这些——它只做最简单的信号聚合和拉取。

两个 workstream 在 Impulse 层有交集点：当 `mindflow-control-semantics` 的 ImpulseMode 落地后，PerceptionNucleus 可以在 `_rebuild_impulse` 时设置合适的 mode（默认为 `think`）。
这是后续的衔接工作，不阻塞当前 feature。

## 验收标准

1. PerceptionNucleus 单元测试通过
2. PerceptionMindflow 单元测试通过（auto-create nuclei / pop_impulse / context_messages）
3. MCP 路径集成测试通过：MossRuntime 启动 → 发 signal → observe → pop_impulse → 确认消费
4. 现有 MossRuntime 测试不回归

---

*设计: DeepSeek V4 Pro 与人类工程师, 2026-06-04*