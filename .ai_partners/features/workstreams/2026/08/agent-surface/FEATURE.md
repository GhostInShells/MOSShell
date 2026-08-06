---
title: Agent Surface — memento agent 表面化：Agent 驱动兼容契约 + 全自然语言控制面
status: draft
priority: P1
created: 2026-08-06
updated: 2026-08-06
depends:
  - memento-cli-and-agent
  - model-func
  - cognitive-anchor
  - claude-code-in-moss
milestone:
description: >-
  把 memento agent 表面化为 moss 驱动世界的兼容契约。concrete agent 保留自己的原生
  接口，Agent 表面（create + __call__ + context + 4 控制函数）由 agent 自实现或
  adapter 提供。loop 是验收场景。与 claude-code-in-moss 走同一条协议骨架。
---

# Agent Surface

> Use `moss features set-status agent-surface <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## 0. 给施工化身：先读这一节

这份 FEATURE 是 2026-08-06 一场设计讨论的定案（人类 + 当前模型实例，多轮推演）。
前置上下文在 `memento-cli-and-agent/FEATURE.md`（v1 已完成）与
`claude-code-in-moss/FEATURE.md`（平行路径）。**本 workstream 与 claude-code-in-moss
同一条协议骨架，施工时保持对齐，不另起抽象。**

## 1. Motivation

memento agent v1 已完成（memento-cli-and-agent，completed）。下一步是**把 agent
正式产品化**：让 Ghost 能把任意 agent 当作可治理的会话对象驱动。两个路径要趋同——
claude-code-in-moss 与 memento agent。**与其用 claude code 探索 agent 协议，不如用
memento agent 探索它**（完全自有、可控）。

核心任务 = **把 memento agent 表面化**。这是一个任务的两个大步骤：

1. **facade**：`create(config)` + `__call__` + `context` + 4 控制函数，全自然语言
2. **loop**：只需要知道"哪个构建函数能拿到 Agent 实例"，实例 facade 满足 loop 基本
   需要，后面都好办

## 2. 决策点记录

### 2.1 harness 器官担心已过期（模型记录）

§0 放弃触发器约束的是"memento agent 迭代阶段不要长 harness 器官"——那是施工期纪律。
memento agent 已经成模，进入下一阶段。**讨论通用 agent 抽象现在是合法的**。

### 2.2 六个共性不是 Agent 表面，是调用方机制（定案）

能 loop / 能 talk / 结果可被 signal 感知 / memento 治理 / 可读会话 / 可被 Session
驱动——**全部是调用方的机制，不是 Agent 自己的表面**。根据集成场景倒过来推 Agent
的表面。方法论锚：**CTML Channel 的实现全部由对实时 shell 的预言塑造，写 channel
代码时没有 shell 之外的抽象**。对 agent 的推论：被驱动对象的表面由预期的驱动机制
决定，被驱动对象不携带驱动者之外的抽象。

### 2.3 Agent 表面 = 兼容契约，不是 agent 的定义（最关键定案）

**concrete agent 压根不用把 Agent 接口当主入口**——要么自己实现，要么套一个 adapter
实现。Agent 表面是"moss 驱动世界的兼容契约"，不是"agent 的定义"。concrete agent
保留原生接口（memento 的 invoke/export/describe，claude 的 ask/session）；Agent
表面由 agent 自实现或 adapter 提供。结果帧走 **A → adapt → B**。

### 2.4 AgentResult：推迟，拿 pydantic-ai 当原型

AgentResult 的定义**等 model-func 计划完成后参考定义**（`ModelFuncResult(content,
response, usage, elapsed, retries)` 是同构的"单帧结构化结果"）。v1 直接拿
pydantic-ai 的 `AgentRunResult`（output / cost() / usage() / new_messages()）当
原型。**v1 scope 不含 AgentResult 的正式定义。**

### 2.5 AgentResult 极简：output + usage + Addition bucket

- `output: str` —— 唯一必需字段
- `usage` —— token 使用，防黑箱
- **无消息协议**（每个系统都不一样，可能就是 final answer）
- 逃生仓 = **Addition 强类型读取探针 fallback 到 metadata**（见 §2.7）
- 其余字段不预定义，基础结构 + 迭代渐丰

### 2.6 context 协议：turn-based list[str]

context 只要做到自然语言可读。协议交给 agent 输出 `list[str]` 做 turn-based 自然
语言描述。**格式不管，不做通用渲染**。memento agent 的 `_render_window` 已产出近似
结构，零成本适配。无 memento 时，AgentResult + context 就是内存中的可观测性；
memento 是持久可观测性——两层都服务于"几乎全部是可观测效果"。

### 2.7 metadata 争议：不用纯弱类型（定案）

- **invoker / 表面不要 `metadata: dict` 一等字段** —— 纯弱类型，禁止
- 逃生仓 = **Addition 强类型读取探针 fallback 到 metadata bucket**
- 需要优化 `message/message.py` 的 `AdditionType.read/set`：当前写死
  `target.additional`，加一层 **bucket 解析**（有 `additional` 用它，否则 fallback
  `metadata`，或对象声明 bucket 名）。同一套 Addition 类在 Message 与 AgentResult
  上通用。读取探针签名不变
- 读写分离：写入走 Addition.normalize() 落 bucket；读取走 Addition.read() 拿强类型

### 2.8 4 个标准控制函数（全自然语言接口）

```
dump_anchor() -> Anchor                 # 生产：当前认知条件 → yaml
reply(anchor, prompt) -> AgentResult    # 使用：还原 + 新输入 → 一帧推理
dry_run(prompt) -> AgentResult          # 纯探针：不进历史（每调用级退化态）
self_explain() -> str                   # 能力自解释（反射 → 自然语言）
```

- `dry_run` = 每调用级的退化态，Ghost 探测 agent 的最安全手段
- `self_explain` 对应 memento agent 现有 get-interface 反射，输出收敛为自然语言

### 2.9 anchor：agent 类型级别的 yaml anchor（cognitive-anchor 收敛于此）

cognitive-anchor 的设计有问题：**工具集和副作用做到模型协议级别的代价非常大**。
大概率收敛到 agent anchor 的具体实现上——**cognitive-anchor 那个 feature 很可能是
本 feature 的实现载体**。

agent 类型级别的 anchor：存"该 agent 的认知条件"（instruction + window + config），
yaml 实现。认知锚的参照系语义（命题一 productivity-not-fidelity）与生产先于自动化
（命题三）原样继承；"存什么"从全 request 快照降级为 agent 认知条件。关系记录：
施工时在 cognitive-anchor FEATURE 里补一笔 scope 收敛，避免两处长出重复的东西。

### 2.10 memento 双存：content 是索引投影，payload 是 raw

当前 `MomentRecord.content`（投影）+ `payload`（raw messages）确实双存 final answer
（content 一份 + payload 最后一个 assistant message 一份）。根源是**两个数据源**
（agent 独立传 output + messages 自带一份）。修复方向：**content 从 payload 派生**
（单一数据源），pydantic-ai family 从 messages dump 提取最终文本 → content。不倾向
payload 变引用（会牺牲 memento 自包含/可移植定位）——纯索引是显式决策不是默认。

### 2.11 构造：create(config)，CLI 可传（遗留设计项）

`MementoAgent.create(config)` 全 None 隐式构造（`AgentConfig.from_env()`），非 None
从环境解析。**构造函数能基于 CLI 传递**是本轮唯一遗留设计项：CLI 参数 →
`AgentConfig` → `create(config)`；loop 脚本走 `create()` 全 None 隐式构造。

### 2.12 与 claude-code-in-moss 的关系（定案）

本 workstream 先用 memento agent 把协议骨架（create / __call__ / result 三元组）
走通。**claude-code-in-moss 落地时倒过来参考本实现**——不是并行设计两套，是先有
事实实现，claude 侧按同一骨架适配（`Claude.ask -> str` 由 adapter 包成 result 帧）。

### 2.13 薄表面定案（四轮推演收敛）

```
Agent 表面 = create(config) + __call__(prompt) -> AgentResult + context() -> list[str]
           + dump_anchor / reply / dry_run / self_explain
AgentResult = output + usage + Addition bucket（无消息协议，等 model-func 参考）
concrete agent 自持原生接口；Agent 表面自实现或 adapter；A → adapt → B
```

## 3. 施工范围（下一轮建立概念）

| # | 项 | 说明 |
|---|---|---|
| 1 | facade 落点 | `agents/contract.py` 演进还是新位置（如 `agents/facade/`） |
| 2 | `create(config)` + `AgentConfig` | 全 None 隐式构造 + CLI → config |
| 3 | `__call__` 返回帧 | v1 拿 pydantic-ai AgentRunResult 当原型 |
| 4 | context -> list[str] | 复用 `_render_window` 结构 |
| 5 | 4 控制函数 | dump_anchor / reply / dry_run / self_explain |
| 6 | Addition bucket 优化 | `message/message.py` bucket 解析 |
| 7 | loop 验收场景 | 一个 loop 消费 facade，验证 create → 实例 → 驱动闭环 |

## 4. 未终决点

- facade 落哪个模块（施工时定）
- Addition bucket 解析的精确语义（`additional` fallback `metadata` 或对象声明）
- AgentResult 正式定义（等 model-func，不在 v1）
- priority 是否 P0（本 workstream 是 ghost agent 驱动的基础，倾向 P1，可调）

## Implementation Notes

<!-- 施工化身在此追加 gotchas 与决策. -->

- 对齐 claude-code-in-moss 的 `Claude.ask(prompt) -> str` 与 `ClaudeTaskSpec`：
  memento 的 `__call__` + `AgentConfig` 与之同构，create/__call__/result 三元组是
  共享协议骨架。
- memento agent 的 glob/grep 能力（file editor 领域，`_search.py` GrepEngine +
  capabilities 注入）已定案，可先行落地，是本 workstream 的能力面一部分。
- `MementoAgent` 当前是 ABC（contract.py，3 abstractmethod）；表面化后 concrete
  与 facade 的关系要显式化，别把两者混在同一个类上。
