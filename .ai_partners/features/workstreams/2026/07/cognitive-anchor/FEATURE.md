---
title: Cognitive Anchor — 认知锚
status: draft
priority: P2
created: 2026-07-27
updated: 2026-08-11
depends:
  - momento-mori
  - ghost-ground
milestone:
description: >-
  认知锚 — 框架无关的 Anchor Protocol + MOSS 实现. 保存模型在思维极值时刻的
  完整认知条件 (协议级快照), 使之在将来被重新激活, 与新问题碰撞产生新判断.
  不是 checkpoint (restore 到原点), 是 reference frame (供河流流动时观测变化).
status_note: >-
  2026-08-11 deepseek-v4-flash. v4 llm func dogfooding: LLMFuncs.call 生产锚
  (CallAnchor payload, export_anchor 落盘, LLMFuncResult.anchor 携带), CLI
  moss llms call --export-anchor, 真实调用验证 call→.anchor.yml→from_anchor
  还原闭环. 见文末 v4 追加.
  --
  2026-08-10 作者 + deepseek-v4-flash. v3 协议落地: Anchor 数据结构定稿
  (meta + payload), meta 顶层极简字段 uid(ULID)/name/description/ref/created/
  metadata, ref 唯一约定=http 地址 (curl 可还原调用), yaml --- 分节,
  ghoshell_moss.anchor 独立成模块 (与 ground 平级). 见文末 v3 追加.
  --
  2026-07-28 claude-fable-5 / opus-4-7. v2 协议优先重构: 锚提升为框架无关的
  通用协议 (存储/发现/读取/使用四机制), MOSS 降为实现方. channel 是载体而非
  边界. v1 的三条核心命题不变.
  --
  2026-07-27 v1 首版, 收敛于三条核心命题: productivity-not-fidelity,
  决策外围疑问结构, 命名作为语义强制. v0 及碰撞轨迹在同目录 discuss 文件.
---

# Cognitive Anchor — 认知锚

> 前置阅读: `2026-07-27_v0-and-collision.md` (同目录). 该文件是本 feature
> v0 记录及其被打穿的过程. v1 不复述那些讨论, 只承接结论.

## 定位

锚是模型在思维极值时刻的完整认知条件, 被显式保存为文档, 供未来的模型实例
(可能是同一模型, 也可能是不同模型) 在新的任务语境里重新激活.

**认知条件** = instruction + tools 现场 + memory + perspectives + 该时刻的
关键推理与判断. 是 request 级别的完整上下文, 不是压缩摘要.

锚的协议层是框架无关的通用规范 (存储/发现/读取/使用四机制), MOSS 是
实现方之一. Channel 是实现载体而非概念边界——既可以是锚的生产端, 也可以
是消费端. 协议不关心 channel.

## 三条核心命题

三条命题定义了锚的评价标准与设计边界. 未来迭代如与之冲突, 应回来重审
命题, 不应绕过命题.

### 命题一: productivity, not fidelity

锚的评价指标不是 "回放时能否复现原状态", 而是 "锚 + 新信息能否产生新判断".

checkpoint 文献 (ContextBranch 及其他) 的目标是 restore fidelity, 越接近
原状态越好. 锚的目标是: 锚保持不动, 河流流过, 观测河流的变化. **锚是 delta
的参照系, 不是 restore 的目标**.

这条命题降级了一个看似关键的问题——同一 request 回放的输出同一性. 项目
经验判断: 大量回放实验的差异集中在文本组织与信息密度, 认知状态本身稳定.
identity 问题不在评价路径上.

刻舟求剑在此被刻意反转: 中文语义里时间是河流 (逝者如斯夫), 舟上的刻痕
本用于观测流水的相对运动. 锚承接的是这个本义, 不是荒谬义.

### 命题二: 锚保留决策外围的疑问结构

锚在实用层面首先解决一个具体的失败模式——**已决策方案在执行时遇到实现
困难, 模型不知道该重开决策还是修补方案, 于是 silent todo**.

matrix-cell-governance 曾发生一次此类失败: FEATURE.md 里保留了决策, 但
丢失了 "决策时哪些点被 punt 掉了 / 哪些共识没打通" 这层疑问结构. Opus
执行到实现困难时, 无法激活 "决策可以被重开" 的判断, 于是照方案硬执行,
静默留下 todo, 最终触发第二次重写.

锚不是决策记录 (FEATURE.md 已经做这件事). 锚保留的是 **决策形成时的
认知场景, 包括未收敛的疑问与被 punt 的分叉**. 执行遇阻时, 加载
`锚 + 新问题`, 让 "新问题落在原始疑问点上" 的信号能被激活, 判断
"是方案不对还是决策不对" 成为可能.

这条命题也定义了锚的最小信息含量: 一个不保留疑问结构的锚是无效锚.
形式上完整、语义上无碰撞面, 消费时激活不出判断.

### 命题三: 命名作为语义强制

"锚" 与 "checkpoint" 数据结构可以一致 —— envelope + payload, 存的都是
某时刻的完整上下文. 命名区别不在数据结构, 在**它 prompt 未来模型选择
哪条推理路径**.

模型面对 "checkpoint" 会调 restore/rewind 路径; 面对 "anchor" 应调
reference-frame + delta-observation 路径. v0 生成过程中, deepseek 把锚
论述为 "checkpoint 的重命名", 本身就是命名污染的证据——它没跟着新名字
的语义走, 滑回了旧名字的路径.

同类先例: frontmatter 在技术上早已存在, Anthropic Skill 语境下重命名为
"渐进式披露", 承载的是不同的使用意图. memento 与 context branch 的数据
结构相近, 使用意图完全不同 (memento = 分叉并置可读, context branch =
回到分叉点互斥).

因此, 本 feature 的命名不是修辞选择, 是**为了让消费端模型正确激活推理
路径而做的语义强制**. 实现中应保护这个命名, 不应在代码/文档层面回退为
"snapshot" / "checkpoint" 等中性词.

## 与 Memento / Ground 的关系

| | Ground | Memento | Anchor |
|---|---|---|---|
| 治理对象 | 认知空间的当前状态 | 认知轨迹的历史 | 认知参照系 |
| 记录什么 | 场结构, pins, 法链 | 对话 moments, commits, 分叉原文 | 极值时刻的完整认知条件 + 疑问结构 |
| 使用意图 | 我现在能看到什么 | 之前发生过什么 (可读, 可 merge) | 那时的判断遇到现在的问题会怎样 |
| 评价标准 | 场的可达性 | commit 的可追溯性 | 锚 + delta 的判断产出 |

三者不是替代关系. Ground 定位锚的空间位置 (哪个场), Memento 存储对话
轨迹 (发生了什么), Anchor 保存认知参照系 (那时的判断)——分别对应
"where / what happened / how to judge".

## 落地退化态

退化态定义为 **可以验证三条核心命题的最小工程形态**. 不追求完整的峰值
自省触发, 不追求 Nucleus 集成, 不追求跨模型协议兼容层.

组成:

- Ghost channel 上暴露两条命令 `create-anchor` 与 `replay-anchor`
- 锚以 markdown 文档存, envelope frontmatter (id / created / model_generation
  / anchor_type / decision_context_ref) + body (原始推理 + 疑问结构)
- Ground 通过 pin 治理锚在场中的位置, 消费时按 pin 加载
- 首批锚由**手动生成**——挑选项目中已有的高密度对话轨迹 (作者与 deepseek-v4-pro
  的三视角流程即候选), 直接编辑为锚文档

退化态不做的事:

- 不做峰值自省, 不做旁路打分, 不接入 mindflow
- 不做跨模型 protocol 兼容层
- 不自动化生产, 不做批量迁移

**退化态的验收标准**: 在项目内选一个真实的执行困境 (如同 matrix-cell-governance
类型的场景), 消费一个锚, 观察 `锚 + 新问题` 是否激活了 "是否重开决策"
的判断. 一个成功案例足以验收命题二. 命题一由消费实践中的一致性观察验收.
命题三由代码/文档中锚概念是否被侵蚀验收.

## 全态

全态在退化态验收后规划. 已知方向 (未展开):

- 峰值自省触发: mindflow channel 提供注意力状态自省指标, 模型在极值发生
  同步存锚
- 旁路 Nucleus 反思: nucleus 拿锚 + 当前上下文 + 反省 hint 做 impulse 治理
- 化身消费: 从锚出生微型化身做旁路判断 (Dolores fork 模式的特化)

全态与退化态之间, 命题一/二/三保持不变.

## Key Decisions

- **锚记录认知条件, 不记录对话轨迹**. 对话轨迹归 memento.
- **文档形式, envelope + payload**. envelope 包含 model_generation 与
  anchor_type. 存储位置由 Ground pin 治理.
- **协议版本作为一等字段但不做兼容层**. 锚是刻舟求剑, 已知随模型世代衰减.
  消费端读 model_generation 自行判断时效性.
- **命名不可回退**. 代码/文档/API 层保护 "anchor" 命名, 不使用 "snapshot" /
  "checkpoint" 等中性词.
- **生产先于自动化**. 首批锚手动生成. 生产机制的迭代不阻塞消费实践.
- **消费独立于生产**. 同一锚可以走多条消费路径, 消费不改锚内容.
- **疑问结构是锚的最小信息含量**. 一个只有决策没有疑问的锚是无效锚.

## Open Problems

以下问题在退化态实施与验收过程中回答, 不在 v1 层面预设答案.

- **peak 检测机制** — 从退化态的手动挑选走向自动化的路径. 候选包括:
  自省触发, 事后回溯 (做梦模式), 旁路打分. 各自的成本与识别率待实验.
- **envelope 与 memento MomentRecord 的关系** — 复用信封还是独立定义.
  倾向复用 (envelope 即 MomentRecord 的一种 Kind), 但需要 momento-mori
  的 Kind 语义先稳定.
- **anchor_type 语义空间** — 已知候选: `l2-decision-with-doubts`,
  `value-judgment`, `attention-peak`, `paradigm-collision`. 在实际锚生产
  中沉淀, 不预设.
- **化身继承锚的规则** — Dolores fork 时是否自动加载 commit 关联的锚,
  还是化身自主选择. 与 ghost-prototype-dolores 一起讨论.
- **消费质量的反向评价** — 一次锚消费如果没能激活疑问、没有 raise 出
  新判断, 是锚本身失效还是消费端失效? 是否可以由此反向评价模型的
  长上下文健康度.
- **协议衰减的实际半衰期** — 单个模型世代内锚是否稳定, 跨世代 (如
  deepseek-v4 → v5) 的衰减是否有可观测的判断质量下降. 需要实证.

## Implementation Notes

- 落地载体是 Dolores ghost channel. 依赖链: momento-mori (envelope 复用) →
  ghost-ground (pin 治理) → ghost-prototype-dolores (channel 挂载). Mindflow
  只在全态阶段需要.
- 首批锚候选包括:
  1. 本 feature 的三视角生成过程 (interview / peer review / self review)
  2. matrix-cell-governance 决策阶段的原始对话 (作为命题二的实证素材)
  3. `.discuss/` 中已识别为高密度的其他碰撞轨迹
- 手动锚化的第一步是**格式规范**——envelope 字段与 body 结构. 建议在退化态
  实施时先手工做 3~5 个锚, 归纳出 body 的常用结构 (决策 + 疑问 + 语境),
  再固化为模板. 不预先设计模板.

---

## 追加: v0 关系说明

v0 (2026-07-27_v0-and-collision.md) 保留于同目录, 承载:

- deepseek-v4-pro 首轮理论展开 (五步骤 + 三元正交表)
- deepseek-v4-pro 自审的五条 review (以注解形式嵌入正文)
- claude-fable-5 与作者的第二轮碰撞记录

v1 与 v0 的关系: v1 不是 v0 的修正版, 是**收敛版**. v0 的五步骤路线图
在 v1 中被压成了 Open Problems 一节, 因为 "先设计完整生产链路再落地"
本身违反了命题一 (productivity-not-fidelity 要求先生产再优化). v0 的
review 五条在 v1 中部分保留 (工程复杂度、命名重叠), 部分反驳 (先验证再
建的产品逻辑不适用于架构创新), 部分吸收 (peak 检测未验证作为 Open
Problem).

历史锚不改写. v0 的判断保留原状, 不因 v1 的收敛而回补.

---

## v2 追加: 协议优先 — 框架无关的 Anchor Protocol (2026-07-28)

claude-fable-5 / opus-4-7 与作者第二轮讨论. v1 将锚定位在 MOSS 内部
(ghost channel 命令, Ground pin 治理), 但锚的价值在跨系统分发. 如果锚
耦合在某个框架的数据结构上, 分发时信息丢失不可逆.

v2 的核心重构: **Anchor 是框架无关的通用协议, MOSS 是实现方之一.**

### 锚与模型快照的关系

锚本质上就是模型请求的协议级快照. 数据存储应以模型协议 (Anthropic
Messages API / OpenAI Chat Completions) 为基础, 存原始完整请求数据结构
(不含 api key). 关键点:

- **不与单一系统耦合**. 不用 langchain / pydantic agent / MOSS message
  等中间抽象去存——那会造成不可逆的信息丢失.
- **直接对齐模型协议**, 未来不同模型间的转换协议方便做, 信息丢失可治愈.
- **thin wrapper** 提供环境相关的反查索引扩展 (如 `labels: dict[str, str]`,
  各框架往里放自己的 key, 互不破坏). 但 wrapper 不能包含系统耦合的主键
  (如 moment ref 作为必填、mysql model id 等)——否则锚不可脱离系统分发.

### 五个协议维度

锚协议应定义五件事, 与任何具体框架无关:

1. **存储机制** — 任何 agent/harness/框架可按协议标准存储关键帧到指定
   文件目录. 存储约定简单, 不与特定 ORM/数据库耦合.
2. **元信息与发现** — 锚的元信息在生产时刻生成. 发现机制对标 SKILL.md
   模式: 按文件名模式扫描即完成发现. 具体发现工具由各框架自己实现,
   协议只定义文件约定.
3. **发现的标准机制** — 协议定义文件命名、目录结构、frontmatter 字段
   的约定, 保证跨框架可发现性.
4. **读取机制** — 两层:
   - **文件读取 (md/txt 等价原文)**: 人/模型直接可读, 不是摘要, 是
     messages 的纯文本等价渲染
   - **原始数据读取 (.json)**: 协议原生 payload, 供 replay / agent 恢复
5. **使用机制** — 框架从原始数据还原 agent, 按需补充新输入, 跑一帧推理.
   协议不定义具体使用方式.

### 生产时刻: 以模型生产时刻为准

锚的生产时刻 = 模型的生产时刻. 一个锚覆盖完整的 turn chain:

```
[user input]
  → [assistant: thinking + tool_call(foo, args)]
    → [tool: foo result]
      → [assistant: tool_call(bar, args)]
        → [tool: bar result]
          → [assistant: final answer]
```

整个 tool call chain 走完才算一个完整的锚. 工具调用和回复插入同一个快照内.

GhostOS (2024) 的快照模式是参考: `Prompt` 对象在 `finally` 块
`self._storage.save(prompt)`, 请求参数、返回消息、时间戳、错误状态在一个
对象里. 但它耦合在 GhostOS 的 `Prompt` / `Message` 类型体系上.
锚协议要做的是把这个模式提升为框架无关的协议——存 Anthropic/OpenAI
原生 messages 数组, 不是 GhostOS Message.

### 参考结构 (未定案)

两个独立文件，捆绑为一个锚:

```
.anchors/
  01JSxxx.md    # 信封 frontmatter + 可读等价原文 (messages 纯文本渲染)
  01JSxxx.json  # wrapper + 协议原生 payload
```

文件名 = anchor_id (ULID), 自包含, 无外部依赖.

`.json` wrapper 草图:

```json
{
  "anchor_id": "01JS...",
  "protocol": "anthropic-messages-2023-06-01",
  "created": "2026-07-28T...",
  "model": "claude-fable-5",
  "anchor_type": "l2-decision-with-doubts",
  "labels": {
    "moment_ref": "01JM..."
  },
  "payload": {
    "model": "claude-sonnet-4-6",
    "system": "...",
    "messages": [...],
    "tools": [...]
  }
}
```

`labels: dict[str, str]` 是框架扩展口. MOSS 往里放 `moment_ref`, 其他
框架放自己的索引. 删除 labels 不影响锚的独立性.

`payload` 是 Anthropic Messages API 的完整 request body (去 key).
protocol 字段变化时 (`openai-chat-completions-xxx`), payload 结构随之变化.

### 锚文件的可读层

锚本身或关联 CLI/API 工具要具备文本可读能力. 原始 JSON 是为了可重放和
从 agent 系统中解耦, 可读 md/txt 是为了让模型或人类想读时能读到.

可读层的粒度: messages 的纯文本等价原文 (保留完整 role + content 结构,
包括 tool call 的 JSON), 不是简化摘要.

### 锚的使用方式

锚 + 新输入 → 完成一帧思考. 用锚还原 agent 本身不是目的——
agent 的真实存在信息在 harness/框架层, 关键帧还原不了全部状态.
关键在于**以锚为参照系, 对新信息做判断**.

### 三层关系的重新理解

之前 v1 的三者关系表忽略了协议层. 修正:

| 层 | 角色 |
|---|---|
| Anchor Protocol | 框架无关的存储/发现/读取/使用约定 |
| MOSS anchor channel | MOSS 对协议的生产端/消费端实现 |
| Ground | 锚在认知场中的 pin 位置 (where) |
| Memento | 对话轨迹 (what happened, 锚可能引用 moment) |

### 退化态调整

v1 退化态说 "Ghost channel 上暴露 create-anchor 与 replay-anchor 两条
命令". 这在协议优先框架下不变——channel 是实现载体. 但退化态的前提
修改为: **先定义协议, 再做 MOSS 实现.** 首批手工锚在生产时遵循协议
格式, 而非 MOSS 内部格式.

### v1 遗留中 v2 视角下的修正

- Open Problem "envelope 与 memento MomentRecord 的关系" → 不再是问题.
  envelope 是协议层概念, MomentRecord 是 MOSS 内部概念. 两者通过
  `labels` 互引, 不合并.
- Implementation Notes "落地载体是 Dolores ghost channel" → channel
  是实现端, 不是协议端. 协议不关心 channel.
- Key Decision "存储位置由 Ground pin 治理" → Ground 管的是认知场中的
  pin 位置, 锚的物理存储由协议定义. 两者正交.

---

## v3 追加: 协议落地 — Anchor 数据结构定稿 (2026-08-10)

与 llms-cli workstream 碰撞 (作者 + deepseek-v4-flash)。v2 把锚提升为
框架无关的通用协议, v3 把协议落到可实现的极简数据结构, 并在
`ghoshell_moss.anchor` 独立成模块 (与 `ground/` 平级, 原子可拆)。

### 数据结构定稿

```python
class AnchorMeta(BaseModel):
    uid: str          # 主键 (ULID), 放 meta 不放文件名
    name: str         # 人类可读名称, 文件存储时作文件名 stem
    description: str  # 一句说明
    ref: str          # 指向 payload 结构定义的 http 地址
    created: datetime # ISO 8601 可读时间戳
    metadata: dict    # 逃生仓 — 自由扩展, 协议不解释

class Anchor(BaseModel):
    meta: AnchorMeta
    payload: Any      # 协议原生数据, 结构由 meta.ref 指向的定义解释
```

### 关键决策

- **`ref` 是协议唯一关键命题**: 唯一约定是指向一个 http 地址。raw /
  分支 / 其它具体形式不约束。模型 curl 它可还原整个调用过程 —
  这是"面向模型的代码协议化设计" (code as prompt 的协议层表达)。
- **`uid` 主键用 ULID, 放 meta 字段, 不放文件名**: 文件名带 id 有治理
  成本 (ls 目录时一半信息是 id)。文件名用人类可读 name, 冲突靠 uid 区分。
  数据库存储场景也靠 uid。
- **顶层字段极简, 能不加就不加**: uid/name/description/ref/created/
  metadata 六个字段。不确定的塞进 metadata 逃生仓, 不占顶层。
- **yaml `---` 分节**: 第一节 = meta (顶层字段平铺), 第二节整体 = payload。
  读 meta 到第一个 `---` 即停, 不解析 payload; 单文件可 glob
  (`**/*.anchor.yml`)。
- **`dump_to_dir(dir, name, *, suffix=".anchor.yml")` 是 code-as-prompt
  的自解释样例, 不是强约束**: 它向模型展示"怎么序列化一个锚", 而非
  规定存储必须走它。
- **位置 = `src/ghoshell_moss/anchor/`**: 与 ground 平级, 只含
  SPECIFICATION.md + contract.py + `__init__.py`。协议不依赖 llms /
  message / ground, 消费方 (llms funcs / dolores / cognitive-anchor)
  平等 import。拆分时整目录复制即独立包。

### agent-anchor 改名方向

v3 与 llms-cli 碰撞确认: 锚的本质是 agent 快照, 协议按 agent 类型
约定定义更诚实。cognitive-anchor 可改名 **agent-anchor** — 但改名
不改变三条核心命题 (productivity-not-fidelity / 疑问结构 / 命名语义
强制)。改名作为后续动作, 不阻塞 v3 落地。

---

## v4 追加: llm func dogfooding — call 生产锚 (2026-08-11)

v3 协议落地后, 用 `LLMFuncs` 做第一次 dogfooding: 让模型调用本身产出
认知锚, 验证 "call → 锚文件 → 还原调用" 闭环。这是协议唯一关键命题
(ref=http 地址, curl 可还原) 的直接实证。

### 实现: 生产锚这一半

- **`CallAnchor(AnchorModel)`** — `src/ghoshell_moss/llms/call_anchor.py`。
  一次调用的锚 payload = **`instruction` + `turns`**: turns 是
  `result.all_messages()` 的 **pydantic-ai 标准序列化**
  (`ModelMessagesTypeAdapter.dump_python`), 完整保留 request/response 的
  所有 part — thinking / text / tool calls。`model`(ModelRef, 无密钥) /
  `result_type`(module:attr, 调用方视角的"工具调用 json schema") /
  `effort` 是索引元数据; `result`(结构化输出 dict) 是便捷摘要。
  `ref()` 指向本文件 GitHub URL — 模型 curl 它学 payload 形状。
- **修正 (2026-08-11)**: 首版用手工字段抽取 (`_extract_text` 拼 TextPart
  + `typed.model_dump()`), 丢了 thinking 和 tool calls。改为标准序列化
  后, 锚就是完整的 turn 保真记录 — 这是消费锚回灌内观的必要前提。
- **`LLMFuncs.call` 增 `export_anchor`(目标文件名, 无后缀可含路径) /
  `anchor_description`** — 契约层, 进入基础 API。None = 不产锚; `""` =
  自动生成带 uid 的名字; 其它 = 稳定地址。调用前先落请求帧 (调用失败
  也保留请求锚, turns 为空), 成功后覆写为完整帧 (instruction + turns),
  锚经 `LLMFuncResult.anchor` 携带出来。
- **CLI `moss llms call --export-anchor <name>`** — 值必填; auto 用
  `--export-anchor ""` (Typer 不支持裸 flag 可选值)。结构化调用后打印
  锚文件路径。

### 决策

- **`CallAnchor` 放 `llms/`, 不放 `anchor/`**: anchor 模块保持纯协议
  零依赖、原子可拆; call payload 是 MOSS 消费端产物, 由 `ref` 指向。
- **name 是稳定地址, uid 是每次生成的版本戳**: `export_anchor` 即
  name, 重跑覆盖同一文件 (新 uid), 版本迭代由 git log 治理 — 生产者侧
  语义, 与 v3 "name 不是 key / uid 解决碰撞" (存储发现侧) 不冲突。
- **锚存完整 turn (标准序列化), 不是语义字段**: 手工字段是外观摘要,
  会丢 thinking/tool calls; 标准序列化是保真记录。消费锚要还原
  `[request/response]` 拼 history 做内观, 必须有 thinking。
- **锚携带完整 `Anchor` 对象 (非仅 ref)**: 调用方可立即
  `CallAnchor.from_anchor(result.anchor)` 进入消费闭环。
- **`result_type` 存 module:attr 指针**: 不内嵌完整 JSON schema — 指向
  即够 (code-as-prompt), 需自包含时可后补。
- **两段式落盘 (请求帧先, 完整帧后)**: 忠实 "构建锚→覆盖数据→调用",
  失败也留请求锚。

### 验证

真实调用产出标准两节 yaml, meta 平铺 (uid/name/ref/created), payload
承载 instruction/model/result_type/effort + `turns` (标准序列化的
request/response, 含 thinking/text/tool)。三种模式实测:
`--export-anchor my-call` → `my-call.anchor.yml`; 重跑覆盖同一文件
(name 不变, meta.uid 新生成); `--export-anchor ""` → `call-<uid8>.anchor.yml`。
文件 → `CallAnchor.from_anchor` round-trip 还原调用 (turns 全保真,
thinking 可回读)。测试 `tests/.../llms/test_call_anchor.py` 5 个用例
覆盖产出、auto 命名、无锚、还原、失败保留请求帧。

### 未做 (后续切片)

- **消费锚**: `call` 接受输入锚 (`instruction + (anchor) + prompt`),
  覆盖数据后调用 — "使用 anchor" 那一半。
- **ground 进 instruction**: ground 渲染结果追加在 instruction 后。
- **`@` 文件路径 / 二进制 (Base64Image) 语法** — prompt 协议扩充。
- **`anchor/` 模块增加读 API**: 目前读是消费方自己实现 (协议 §8)。
