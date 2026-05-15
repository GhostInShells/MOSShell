# 06 — 全链路测试方案

日期：2026-05-14

## 背景

Ghost 原型的测试策略需要明确：测什么、怎么测、不测什么。

## 讨论要点

### 1. Atom 的核心依赖

```
Atom 依赖:
  MossPrompter   → 组装 system_prompt（可从 IoC mock）
  Workspace      → ghost 专属目录 + soul 文件加载（可从 IoC mock）
  Articulator    → 提供 moment + send_logos（可从 IoC mock）
```

三者均可通过 IoC container mock。不需要真实模型调用。

### 2. 测试策略

**第一优先级：数据结构对齐**

- Moment → to_model_request() → 拼出正确的请求消息
- ModelResponse → save_model_request() → 剥离 perspectives 存入 Reaction
- ModelContext.to_messages() → 回放多轮历史，校验 perspectives 携带策略

这些是纯数据转换，不依赖网络或模型。最容易出 bug 也是最需要测试的。

**第二优先级：API 契约**

- GhostMeta.factory(container) 能成功产出 Ghost 实例
- Ghost.system_prompt() 返回 soul + MossPrompter 拼合的字符串
- Ghost.articulate(articulator) 在 mock 模型下返回正确的 Logos 流
- Ghost 生命周期 __aenter__/__aexit__ 正常

**第三优先级：集成测试**

- GhostRuntime 包裹 MossRuntimeImpl，完整生命周期启动
- Signal → Mindflow → Attention → articulate → moss_exec 全链路
- 参考 MindflowSuite 的测试模式

### 3. 不测的

- 真实模型调用结果（模型输出不可控，不属于单元/集成测试范围）
- 多线程调度正确性（v0 单线程）
- 性能指标

### 4. 测试套件

v0 只写关键路径的单测 + 一个集成测试。完整测试套件是未来工作。

## 决策结论

1. 核心依赖（MossPrompter / Workspace / Articulator）均通过 IoC mock
2. 重点测消息协议和历史管理的数据结构对齐
3. API 契约覆盖 Ghost 生命周期 + articulate 流程
4. 一个集成测试覆盖完整三循环链路
5. 不测试真实模型调用
