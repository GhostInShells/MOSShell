---
title: Mindflow Channel
status: draft
priority: P1
created: 2026-07-23
updated: 2026-07-23
depends: []
milestone: 0.1.0
description: >-
  Mindflow 反身控制 Channel — 将 mindflow 从 opaque 调度器变为 ghost 可感知、
  可操纵的透明面. 自解释 + 注意力管理 + 优先级干预 + nucleus pull.
---

# Mindflow Channel

> Use `moss features set-status mindflow-channel <status> -m "note"` to update state.

## Motivation

Mindflow 当前没有 channel. 它的上下文（nucleus 状态、当前 attention、perspective）
是强行拼装到 perspective func 里的 — ghost 看不到自己的感知系统内部状态.
这是一个"感知幽闭"问题: ghost 知道自己有感官, 有消息在排队, 但感官对它是黑箱,
只能等 signal 够强了自行破门而入.

Mindflow channel 是 Mindflow 的反身控制面. 做完后:
- Mindflow 的自解释、注意力状态、优先级调整、nucleus pull 全部通过 channel 体系暴露
- ghost 获得一个关于自身注意力的**可操作模型** — 不只是"收到 impulse → 反应",
  而是"我看见我的感知系统里有什么、我的注意力被什么占据、我可以选择改变它"
- Mindflow 从 push-only 变为 push + pull 双路径

这是 Dolores ghost channel 认知场的核心内容: ghost 通过自己的 channel 看见
并调控自身感知状态.

## Capability Surface

四层递进:

### 1. 自解释 (Self-Explanation)

Channel `instruction` 动态反映当前 Nucleus 拓扑. 每个 NucleusMeta 贡献一行
自描述 (name + signals + description). 模型一瞥就知道自己有哪些感官、各自在
监听什么信号, 不需要系统层单独拼 perspective.

输出形状类似:
```
mindflow nuclei:
  vision (NOTICE): 监听 visual_frame, camera_status
  audio  (NOTICE): 监听 asr_partial, asr_final
  system (BACKGROUND): 监听 health_check, config_change
```

### 2. 注意力管理 (Attention Management)

暴露当前 attention 状态, ghost 可主动操作:

- **状态查询**: 当前谁占着 attention、什么 priority、剩余 protection_time
- **释放**: 主动结束当前 attention (release)
- **降级**: 降低当前 attention priority, 让外部信号更容易抢占
- **屏蔽**: 临时 suppress 特定 nucleus 的 impulse, 不让它参与 challenge

### 3. 优先级干预 (Priority Intervention)

基于 Mindflow 已预留的接口 (`set_signal_priority_bar` / `set_impulse_priority_bar`)
做细粒度操作:

- **per-nucleus 提权/降权**: 对特定 nucleus 的 priority +1/-1
- **全局水位**: 抬高最低可挑战门槛 (如 NOTICE 为最低, BACKGROUND 直接丢弃)
- **临时加权**: 对特定 signal name 临时加权, 使匹配的 signal 更容易抢占

mindflow 仲裁时取修改后的权重做 challenge 校验. 这些修改是 ghost 的运行时行为,
不污染 nucleus 本身的声明式 priority.

### 4. Nucleus Pull (主动拿 Impulse)

不通过 mindflow challenge 管道, 而是 ghost 主动从 nucleus 里拿高优 impulse:

```
channel:mindflow pull-nucleus --name vision
  → 返回 nucleus 当前最高优 impulse
  → 通过 command 返回值 → observe 体系直接递给 ghost
  → 同时反身更新当前 attention 状态 (标记已阅/消费)
```

这条路径不走 push (Signal → Nucleus → Impulse → Challenge → Attention).
低中优信号不参与 challenge 竞争, 安静排队等 ghost 来取. FATAL/CRITICAL
依然走 push 抢断. 两条路径各管各的优先级带.

## Key Decisions

- **Mindflow.as_channel() + Nucleus.as_channel() 已预留接口.** 实现时优先走
  这两个预留点, 不做新的抽象切口.
- **自解释走 instruction, 不走 context_messages.** instruction 是静态面 (每
  refresh 更新), context_messages 是动态面 (每帧更新). nucleus 拓扑变更是低频
  事件, 走 instruction 即可.
- **优先级干预是运行时 overlay, 不修改 nucleus 声明.** nucleus 的 priority
  是声明式基准, ghost 的提权/降权是运行时 overlay. mindflow 仲裁时合并计算.
- **Pull 路径复用 observe 体系, 不新造通讯协议.** command 返回值 →
  observe → 下一帧 context, 链路已存在.
- **首版 scope: 自解释 + 注意力管理 + 优先级干预.** Pull 路径实现复杂度
  更高 (需要定义 nucleus 的 poll 接口), 按需后做.

## Open Problems

- **Nucleus poll 接口** — 当前 Nucleus 只有 `peek()` (不消费) 和 `pop_impulse()`
  (mindflow 回调). Pull 路径需要 "poll + consume" 语义. 是否在 Nucleus ABC 加
  `poll()`, 还是用 `peek()` + 独立 ack 机制, 待定.
- **优先级 overlay 的 persist 范围** — 临时加权是否跨 attention? 是否跨 session?
  需要定义 overlay 生命周期.
- **屏蔽 vs background_notice** — 屏蔽一个 nucleus 和把它设到 BACKGROUND + notify
  有什么区别? 前者是硬切断, 后者是"不抢占但留痕". 两条路径各自适用场景需厘清.

## Implementation Notes

- `Mindflow.as_channel()` 返回一个 MutableChannel, 用 Builder 注册上述命令.
- 每个 Nucleus 的 `as_channel()` (如果返回非 None) 作为 mindflow channel 的
  虚拟子通道, 提供 per-nucleus 粒度的控制.
- 自解释的 instruction 通过 `Builder.instruction()` 注册, 每次 `refresh_meta`
  时重新生成字符串 (从 mindflow.faculties() 取当前 nuclei).
- `set_signal_priority_bar` / `set_impulse_priority_bar` 的默认实现是 noop
  (见 mindflow.py), 需要在 Mindflow 实现中补上.
