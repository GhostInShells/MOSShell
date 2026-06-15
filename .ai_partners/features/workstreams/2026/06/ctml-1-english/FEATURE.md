---
title: CTML 1.0.0 English Revision — 协议级 review 锚点 + 术语零跳转
status: draft
priority: P1
created: 2026-06-15
updated: 2026-06-15
depends: []
milestone:
description: >-
  Rewrite CTML 1.0.0 prompt in English to align terminology with code (channel /
  scope / command), clear out legacy metaphors (funnel / parent-child dispatch),
  and tighten the prompt budget toward the 5-6k sweet spot. Pre-beta1, no
  protocol promise yet — version stays 1.0.0.
---

# CTML 1.0.0 English Revision

> Use `moss features set-status ctml-1-english <status> -m "note"` to update state.

## Motivation

CTML 1.0.0 中文版当前可工作但有显而易见的话术包袱:

1. **术语和代码不对齐**: "通道" vs `channel`, "作用域" vs `scope`, "命令" vs `command`. 模型读 prompt 时要做一层语言映射, 增加认知成本.
2. **历史比喻反向**: "漏斗式" 暗示物质从父往子流, 与 "父占用→子树冻结" 的反向语义冲突.
3. **"父子分发" 标题错位**: "分发" 让读者关注新命令进哪条队列, 但规则实际是 "整子树冻结".
4. **拓扑隐式**: `import_channels` 形成父子边这一关键事实在文档里没显式说出.
5. **缺反直觉示例**: 最容易踩的坑 (主通道 occupy → 所有子通道停) 没有最小示例.

中文版当时不写英文版是因为 1.0.0 review 期英文 review 成本高 + feature 不稳. **现在 feature 稳定了** (mindflow-control-semantics 收口 + 300~500 单测密度 + ThreeLoopSuite 作为协议契约), 杠杆翻转: 一次性高成本换长期低维护成本.

**Why now**: 不是因为模型踩坑就要改话术 — 最终解法是预训练 + FT, CTML 自洽才是唯一的 bar. 英文版的本质收益是 **改善"模型一次内化"的带宽**:

- 训练语料 95%+ 是英文, 模型对英文 DSL 文档的解析带宽 > 中文.
- 英文 backtick `channel` / `command` / `scope` 与代码完全同名, 零跳转.
- 英文版作为一次性 protocol-level refactor 借口, 清理历史话术包袱.

## Key Decisions

### KD1: 版本号保持 1.0.0, 不切默认

CTML 当前还在 pre-beta1, **没有正式协议承诺**. 英文版以 1.0.0 落地, 不动版本号. 常见错误未来 0.1.0 之后走 patch.

**Why**: 版本号变更暗示协议变更, 但英文版只是话术 refactor + 术语对齐, 协议事实不变. 双写一段时间 (中文 v1.0.0 默认 + 英文版试用), 收集模型在两版本上的踩坑频率对比 (用现有 300+ 单测 + ThreeLoopSuite 作为压力测试), 数据支持后再决定是否切默认.

### KD2: Token 预算上限 5~6k, 不突破 8k

当前中文版已经 5~6k. 英文版目标是**保持或略降**, 不突破.

**Why**:
- Anthropic 自己的 tool use prompt 大概 2~3k; CTML 是 DSL + 拓扑模型, 5~6k 是 "中等偏重, 可接受" 上限.
- 多轮对话历史反复带 CTML prompt, 即使有 cache, 对其他 prompt 部分挤压预算.
- 模型对长 prompt 的**末段注意力衰减**是真的, 5k 之后规则被遗忘概率上升.

**裁剪方向是去冗余, 不减规则**:
- "code as prompt" 这种宣言可以省 — Python 签名自带语义
- 通道命名规则细节浓缩
- 几个 `until` 值合并为一张表
- 多版本术语的重复解释压缩

**正反例必须保留** (这些是模型踩过的真坑, 每个反例的 token 成本是模型踩坑成本的 1/N 倍杠杆):
- 父子阻塞踩坑示例
- 流式参数误用 (`<foo:say chunks__="hello"/>` 这种)
- `__main__` 通道命名歧义
- scope 嵌套规则

### KD3: 术语战略 — occupy + blocked 分工

模型偏好差异很大 (历史轨迹):
- Gemini 喜欢 `occupy`, 反对 `block`
- DeepSeek 喜欢 "父子分发" 和 "漏斗"
- Claude 偏好: occupy 当动词, blocked 当形容词

**最终选择 (Claude Opus 4.7 与人类工程师 2026-06-15 对齐)**:

- **`occupy` 当动词**: "slow_cmd occupies channel a". 匹配 mutex/lock 的心智模型 — CTML 的父子阻塞**就是** lock-like.
- **`blocked` 当形容词**: "channel a's descendants are blocked while a is occupied". 描述被卡住对象的状态.
- **不用 `block` 当动词**: 容易让人联想 socket block / blocking IO, 那是另一个语义. Gemini 反对的是这个用法.

弃用术语:
- "漏斗式" / "funnel" — 反向比喻, 误导
- "父子分发" / "parent-child dispatch" — "分发"暗示队列调度, 但规则是整子树冻结

新术语方向:
- "occupy propagation" / "父占据传播" (描述阻塞向下传播的机制)
- 或直接 "parent-child blocking" (准确但平淡)

### KD4: 父子阻塞规则保留, 不改成"通道天然并发"

人类工程师明确判断: **不去掉**.

**Why**:
- **Default 选择必须服从模型 token 时序的物理事实**. 模型 token 流是 FIFO 的, 这是大模型本质属性. 父子阻塞规则把 "父 token 先到 → 父先占用 → 子等" 这条物理时序**显式化为协议**.
- **取消父子阻塞会让 scope 退化为伪并发协调器**. scope 的 `until=flow/all/any` 本质是"在 occupy 关系上做生命周期约束". 没有父子阻塞, scope 就没有 occupy 关系可约束, 退化成纯语法分组. scope 和父子阻塞**互为存在条件**.
- **反直觉踩坑是局部问题**, 不是协议问题. 主通道的正确用法是 scope 容器 + 原语, 业务命令放子通道, 不触发反直觉的边. 这是文档话术问题.

### KD5: "快速响应"规则迁出 CTML, 留给 ghost prompt

人类工程师锚定的设计: **协议层最小化, 行为层定制化**.

**原规则**: 正确的交互体验是模型先输出交互 token ("我想想..") 再输出动作, 让交互不要等待在第一个动作命令上.

**为何迁出**:
- 这是**交互体验偏好**, 不是 CTML 协议层语义.
- 不同应用场景有不同偏好 (语音助手要 "我想想...", 文档处理 agent 直接动手更好).
- 留在 CTML 文档里反而让模型困惑: 这是规则还是建议?
- 实际可由 flash 模型用单 token 多分类去选 "快速响应", 快速响应本身可以是模型生成的 CTML 映射表.

### KD6: 高优隐藏约束显式化

"并行子轨命令要先于主轨发送" — 这条原本在文档里作为"原则"提.

**应升级为 "父子阻塞规则的直接推论"**, 并在解释父子阻塞时同步说出来. 这样读者第一次读到父子阻塞就知道为什么并行命令要前置, 而不需要在远端章节才理解.

## Implementation Notes

### 验收 Bar

- Token 预算: 不超过 6k, 力争压到 5k
- 术语零跳转: 文档术语和代码 backtick 同名
- 双写一段时间, 用 mindflow + shell + ctml 全套 800+ 单测作为压力测试, 跑两个版本对比模型踩坑频率
- 数据支持后切默认; 切默认时**不动版本号** (1.0.0 不动, pre-beta1 无承诺)

### 已知踩坑点 (正反例素材库)

来自 2026-06-15 Claude Opus 4.7 在 mindflow-control-semantics 收口会话中的实测踩坑:

1. **父子阻塞踩坑** — 写 append cross-frame 测试时, 把 long-running command 放主通道, 帧 2 放子通道 `other`. 因主通道 occupy → 所有子通道 (including other) 都 pending → 测试 deadlock. 文档里有这条规则, 但 "漏斗式" + "父子分发" 的措辞让规则失去内化深度.
2. **content_command 命名混淆** — `build.content_command(speak)` 实际注册的命令名是 `__content__` 不是 `speak`. CTML `<a:speak/>` 不存在.
3. **`new_channel` 隐式拓扑** — `new_channel(name='other')` 直觉上是个独立通道, 但 `shell.main_channel.import_channels(other)` 把它挂到 main 下面, 形成 main → other 的父子边. 这一步在文档里是隐式的, 读者建立的是 "channels are siblings under shell" 的扁平心智模型.

每个踩坑都是英文版反例的素材, 内化到文档应该让下一个模型实例**一次就懂**.

### 不在本次范围

- CTML 2.0 设计 (远期, 等 1.0.0 跑稳定再看)
- 自动 tooling: CTML 语法 lint / 反例自动生成 (可以是后续 patch workstream)
- 中文版废弃 (双写期保留, 数据支持后再决定)

### 协作历史参考

CTML 话术演进史散落在与不同模型的对话中, 主要锚点:
- "occupy" 由 Gemini 提出 (反对 "block" 当动词)
- "父子分发" + "漏斗" 与 DeepSeek 对过
- 英文版决策 (本 workstream) 与 Claude Opus 4.7 在 2026-06-15 mindflow-control-semantics 收口会话中对齐

未来负责实施的模型应先读 mindflow-control-semantics FEATURE.md 末尾的 2026-06-15 笔记 (尤其"下一会话锚定"段), 了解决策背景再动手.
