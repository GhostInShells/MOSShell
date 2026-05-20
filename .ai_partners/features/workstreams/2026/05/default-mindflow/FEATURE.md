---
created: 2026-05-19
depends: []
description: 提供预配置好 input signal nucleus 的 DefaultMindflow，实现 "红点" 式信号聚合与纯优先级仲裁， 让任何地方都能直接复用一个行为可预期的
  Mindflow。
milestone: null
priority: P0
status: completed
status_note: Steps 1-4 完成, 43 tests. Step 5 属于 GhostRuntime 集成, 不在此 feature 范围内.
title: Default Mindflow — 开箱即用的 Input Signal 感知调度
updated: '2026-05-20'
---

# Default Mindflow

## Motivation

从 `first-ghost-prototype` Phase 3 拆分。Ghost 开发已打通 "感知→思考→执行" 全链路 (10a-10i done)，
但使用的 Mindflow 是手动组装的裸 `BaseMindflow` + `BufferNucleus`。11a 要求提供预配置的 default mindflow
支持传统输入/打断模式——这项工作触及 Mindflow + Attention + Nucleus 三层，独立性强，
拆分为独立 workstream 避免 ghost feature 膨胀。

当前要运行 Ghost 的三循环，每次都需要手动组装 `BaseMindflow` + `BufferNucleus`。
而且 `BufferNucleus` 的信号聚合行为（合并消息到单个 Impulse）与 "输入信号" 的场景语义不完全匹配：
输入信号应该像 IM 红点——展示最新消息描述列表、pop 时 FIFO 全量返回、优先级取最大。
此外 `BaseAttention` 的强度衰减仲裁模型虽然精细，但初版缺乏可观测性和可调性。

本轮工作：
1. 提供开箱即用的 `DefaultMindflow`，预配置 input signal nucleus
2. 拆分 `AbsBaseAttention`（纯生命周期）→ `BaseAttention`（只加 challenge），让仲裁策略可独立替换
3. 实现首版可观测仲裁：纯优先级 + 固定保护期

## 调研结论

### Mindflow 数据流

```
Signal (端侧原始输入) → Nucleus (信号聚合/降频) → Impulse (调度信号)
                                                          ↓
Mindflow (调度中枢) → Attention (单次运行态) → Articulator (推理) + Action (执行)
```

### 调度链关键路径

1. **Signal 入队**: `BaseMindflow.add_signal()` — 跨线程安全，janus PriorityQueue 卸载到 event loop
2. **分发到 Nucleus**: `_dispatch_signal()` 按 signal name 路由
3. **Nucleus 产出 Impulse**: 通过 `with_bus()` 注册的 `impulse_notify` 回调 → `add_impulse()`
4. **Impulse 消费循环**: `_on_impulse_consuming_loop()` 周期性 `_rank_nuclei()` → `_challenge_attention()`
5. **Attention 仲裁**: `challenge()` 返回 True/False/None 决定抢占/压制/吸收
6. **Attention yield**: `_pop_new_attention_queue` (maxsize=1) → `loop()` yield

### 当前实现总结

| 层 | 抽象 | 实现 | 关键行为 |
|---|---|---|---|
| 信号聚合 | Nucleus | BufferNucleus | FIFO buffer, suppress 冷静期, pulse beat |
| 调度 | Mindflow | BaseMindflow | 优先级队列, _rank_nuclei max(), challenge 仲裁 |
| 运行态 | Attention | BaseAttention | 保护期+强度衰减+同源提权, observe 循环 |

## Design Index

- Mindflow ABC: `ghoshell_moss.core.blueprint.mindflow`
- BaseMindflow: `ghoshell_moss.core.mindflow.base_mindflow`
- BaseAttention: `ghoshell_moss.core.mindflow.base_attention`
- BufferNucleus: `ghoshell_moss.core.mindflow.buffer_nucleus`
- GhostRuntime 集成: `ghoshell_moss.host.ghost_runtime.py:_wire_mindflow`
- 单测: `tests/ghoshell_moss/core/mindflow/`

## Key Decisions

### 1. AbsBaseAttention 拆分 — challenge 作为独立扩展点

**决策**: 将 `BaseAttention` 拆为两层：

```
AbsBaseAttention           ← 全部生命周期机械 (loop, observe 循环, context func,
                               abort, Articulator/Action 创建, current_strength 状态管理)
    └── BaseAttention      ← 只加 4 个构造参数 + challenge() + current_strength() 衰减计算
```

**理由**:
- `challenge()` 的调用方 `BaseMindflow._challenge_attention()` 只读三值返回值，不依赖具体实现
- 换一种仲裁策略只需 override `challenge()` + `current_strength()`，其余全部复用
- `AbsBaseAttention` 维护 `_initial_strength`/`_strength_refreshed_at`/`_strength_decay_time` 状态，
  子类只读这些状态做计算

**原类名 `BaseAttention` 保留为子类**（继承链末尾），现有调用方 `BaseMindflow._create_attention_from_impulse()`
一行不改，构造签名和行为完全不变。

> 父类命名由实现时确定，不使用 `AbsBaseAttention` 这个临时名。

### 2. 首版仲裁: 纯优先级 + 固定保护期

**决策**: 初版 Attention 实现用以下四条规则，去掉强度衰减曲线：

1. `challenger.priority > current.priority` → **True**（无保护期，立即抢占）
2. `challenger.priority < current.priority` → **False**（压制）
3. 同级且保护期外 → **True**；保护期内 → **False**
4. `challenger.id == current.id` → **None**（更新 complete，不抢占）

**理由**:
- **可观测**: 每个决策只有两个变量 (priority + elapsed)，一行 log 解释清楚
- **可调**: 只有一个 knob（保护期时长），默认 2-3 秒
- **无仲裁风暴**: 同级在保护期内被绝对压制，不会出现 A 打断 B、B 又打断 A 的震荡
- **代价可接受**: input signal 场景是粗粒度的（用户消息、系统通知、定时器），priority 区分就够。
  strength 维度以后需要时在子类加回来

**构造参数**: 只保留 `protection_seconds: float = 2.5`

### 3. InputSignalNucleus — 红点式信号聚合

**决策**: 创建专用 Nucleus 监听 `InputSignal` (name='input')，行为对齐 IM 红点模型：

- **入队**: 接受 `InputSignal`，FIFO buffer
- **红点 status()**: 返回 `"pending: N, top: <latest description>"`
- **pop 时全量返回**: `pop_impulse()` 时清空全部 buffer，产生的 Impulse 包含所有消息 (FIFO 顺序)
- **优先级由最大决定**: `Impulse.priority = max(signal.priority for signal in buffer)`
- **同级短保护期外可打断**: 由 Attention 的 challenge 规则 3 保证
- **suppress 冷静期**: 被压制后短暂静默，避免信号风暴

**与 BufferNucleus 的关系**: InputSignalNucleus 是 BufferNucleus 的同级实现，共享 Nucleus ABC。
BufferNucleus 保留用于其他 sensor 场景。

### 4. DefaultMindflow — 预配置工厂

**决策**: 提供 `DefaultMindflow` 类/工厂函数，启动时自动注册 InputSignalNucleus：

```python
# 最小使用
mindflow = DefaultMindflow()
async with mindflow:
    mindflow.add_signal(InputSignal().to_signal("你好"))
    async for attention in mindflow.loop():
        ...
```

**构造参数**:
- `protection_seconds: float = 2.5` — 同级保护期
- `input_buffer_size: int = 20` — input signal buffer 上限
- `input_suppress_seconds: float = 0.5` — input nucleus 被压制后的冷静期
- `logger` — 日志

**IoC 集成**: 提供 `DefaultMindflowProvider`，使 `MossRuntime` 可以通过 IoC 自动发现。
GhostRuntime 的 `_wire_mindflow()` 已有 fallback 链 (`ghost.mindflow() > IoC.get(Mindflow) > BaseMindflow()`),
在 IoC 层注册后自动生效。

## 实施计划

### Step 1: AbsBaseAttention 拆分

- 文件: `src/ghoshell_moss/core/mindflow/base_attention.py`
- 内容: 重命名现有 `BaseAttention` 为父类，提取 `challenge()` + `current_strength()` 为 abstractmethod
- 新 `BaseAttention` 子类保留当前强度衰减实现
- 验证: 现有 15+ 单测全部通过

### Step 2: PriorityProtectionAttention

- 文件: `src/ghoshell_moss/core/mindflow/priority_attention.py` (或同在 base_attention.py)
- 内容: 继承父类，实现纯优先级 + 固定保护期 challenge
- 单测: 覆盖抢占/压制/同级保护期/保护期过期/同 ID 更新/FATAL 必抢占

### Step 3: InputSignalNucleus

- 文件: `src/ghoshell_moss/core/mindflow/input_signal_nucleus.py`
- 内容: 监听 `InputSignal`，红点 status，pop 全量返回 (FIFO)
- 单测: 覆盖入队/status/pop/buffer limit/stale/suppress

### Step 4: DefaultMindflow + Provider

- 文件: `src/ghoshell_moss/core/mindflow/default_mindflow.py`
- 内容: `DefaultMindflow(BaseMindflow)` 构造时自动注册 InputSignalNucleus + PriorityProtectionAttention
- Provider 注册到 IoC 体系

### Step 5: 端到端验证

- GhostRuntime 通过 IoC 自动集成 DefaultMindflow
- REPL 中发送 input signal 走通全链路

## Implementation Notes

- `BaseMindflow._create_attention_from_impulse()` 当前硬编码 `BaseAttention(...)` 构造。
  拆分后不改此处——新 `BaseAttention` 签名不变。
  要使用 `PriorityProtectionAttention` 需通过 `DefaultMindflow` override `_create_attention_from_impulse()`.
- `current_strength()` 在父类中用于 `_inner_attention_lifecycle()` 的 fade-out 检查。
  拆分后父类的 `current_strength()` 声明为 abstract，子类各自实现。
- InputSignalNucleus 的 `pop_impulse()` 必须在 lock 内原子清空 buffer，
  防止 `_process_signal` 在清空和重建之间插入新信号。

## 实施记录 (2026-05-20)

Steps 1-4 完成，由 DeepSeek V4 Pro 实现，人类工程师 review。

### 实际架构

```
AbsAttention            ← 抽象生命周期, challenge()/current_strength() = abstract
  ├── BaseAttention     ← 强度衰减 (向后兼容)
  └── PriorityProtectionAttention ← 纯优先级+固定保护期

AbsMindflow             ← 抽象调度, _build_attention() = abstract
  ├── BaseMindflow      ← 用 BaseAttention (向后兼容)
  └── PriorityMindflow  ← 用 PriorityProtectionAttention

new_default_mindflow()  ← 工厂: PriorityMindflow + InputSignalNucleus
```

### 关键决策偏离

1. **AbsMindflow 拆分**: 实施中增加了 `AbsMindflow`，将 `_build_attention()` 作为抽象扩展点，
   而非让 `DefaultMindflow` 直接 override `_create_attention_from_impulse()`。更干净。
2. **工厂函数 > 类**: `new_default_mindflow()` 替代了 `DefaultMindflow` 类。
   显式一行 `with_nucleus(InputSignalNucleus(...))` 做 code as prompt。
3. **同级保护期外强度比较**: 保护期后同级不是无条件抢占，而是 `challenger.strength > current.strength`。
   两个 knob: priority (粗调) + strength (微调)。

### 测试覆盖

- AbsAttention/BaseAttention: 22 测试 (原有全部通过)
- PriorityProtectionAttention: 10 测试 (7 条规则 + 过期 + noop escalation + FATAL)
- InputSignalNucleus: 9 测试 (入队/红点/pop/全量/FIFO/priority/buffer limit/suppress/filter)
- PriorityMindflow + factory: 2 测试 (工厂 + attention 类型)
- 总计: 43 测试, 0 失败

### 未完成

- Step 5: GhostRuntime 端到端集成验证 (REPL 中走通全链路)

---

*调研与设计: DeepSeek V4 与人类工程师, 2026-05-19*
*实施: DeepSeek V4 Pro via Claude Code, 2026-05-20*