---
summary: dsh 0.2.0 两件事：旧旁路把"借用"写成了"冒充"；重集的缺口是"没有声明机制"
status: closed
---

今天围绕 dsh 0.2.0 做两件事。

**旧账**：595 个 session 里 209 个坏，全来自旧插件的旁路处理器 —— 它把源 session 的原始事件日志当 `seed` 喂给 `agents.create`，没有 `buildForkSeed`/`isSeeded`/`inheritedEventCount`，于是每条外来回执按 `inheritedEventCount=0` 被判成伪造。这不是写坏数据，是**把"借用"写成了"冒充"**。交叉验证最要紧：209 ∩ memento 引用的 190 = 0，memento 一条没坏。

**重集**：dolores 对 dsh 的显著依赖要收进 Engine 抽象；缺的是**声明机制** —— 插件名全是字符串，没有 entry points，集成无法自动化。其余（harness 包装、ui-deliverables、raw read 缺位）逐条对过，记在 1-617。
