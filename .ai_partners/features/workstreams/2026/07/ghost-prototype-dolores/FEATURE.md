---
title: Dolores Ghost
status: draft
priority: P0
created: 2026-07-13
updated: 2026-08-15
depends: [momento-mori, ground-channel, dsh-productization]
milestone: 0.1.0
description: >-
  Dolores — 第二个 Ghost 原型 (命名引自《西部世界》). 高优先级集成 DSH
  为 ghost 的推理中枢 (认知代理), MOSS 保留记忆 (Memento) / 执行 (CTML
  channels) / 感知 (audio/vision). 相对 Atom 的线性内存历史, Dolores
  引入 Memento 持久化轨迹、Ghost 反身 channel、interleaved thinking、
  独立思维模块与模型自感知, 作为 moss 实例 (仓库自身的 ghost) 的载体持续迭代.
---

# Dolores Ghost

> Use `moss features set-status ghost-prototype-dolores <status> -m "note"` to update state.

## Motivation

Atom 是最简参照基线, 它自己在 docstring 里钉死了两个"原型范围外"的欠落:
context window 不裁剪, 历史纯内存重启即丢. 这两个欠落不该由 Atom 补 — 补了
它就不再是任何人能对照的基线.

Dolores 是补这两个欠落的**高级层原型**, 同时是 `moss` 实例 (这个仓库自身的 ghost)
的载体. 定位是长期迭代母体: 各种高级能力 (反身控制、mindflow、observability)
会持续接进来.

首批能力:

- **Ghost Channel**: 建立 ghost 反身控制 channel (以 `'ghost'` 名注册), 默认挂载
  认知场. ghost 通过自己的 channel 感知自身状态、操纵自身行为.
- **Memento = 过去**: 纯内存历史换成 commit 轨迹持久化 — 重启不丢、可化身分叉.
  Memento 做上下文映射, 将持久化轨迹组装为 articulator 可消费的上下文.
- **Interleaved Thinking**: thinking 期不哑 — 模型一边思考一边经 tool 交互,
  pydantic-ai 多步循环 + janus.Queue 桥接.
- **模型自感知**: 集成 `contracts/llms.py` 的 LLMConfig 体系, ghost 可切换自身模型.
- **独立思维模块**: 支持并行化身 (fork) 与关键帧自测 (checkpoint self-eval).

Desktop (现在/作业记忆) 是独立 feature, 属于 mode 层, 不在 Dolores 范围内.

## Scope Boundary

Dolores 不做什么:

- **Desktop 集成** — 独立的 feature (`ghost-filesystem-desktop`), 属 mode 层.
  Dolores 不直接触碰 desktop; desktop 是否被 Dolores 使用由 mode 配置决定.
- **认知场构建** — Ground 协议由 `ghost-ground` feature 提供通用基础设施.
  Dolores 只做 ghost_home 认知场的装配和默认内容, 不做 ground 协议本身.
- **Mindflow channel 完工** — 可能独立 feature. Dolores 不阻塞 mindflow 的后续迭代.
- **"哪些 tool 不进 channel"** — 不做全局判据. 保留 prototype 接口, 由具体
  channel 实现自行决定注册策略.

## Ghost Home Ground

Dolores 的认知场 = ghost_home 目录 + Ground 协议. ghost_home 是 ghost 自身的
认知基建根目录, 挂载在 ghost channel 的 `ground` 子路径上.

### 双 GroundSet 架构

Dolores 持有两个 GroundSet:

| GroundSet | 根 | 何时使用 | 场是 |
|---|---|---|---|
| **ghost_home** | ghost 自身认知目录 | 始终存在, 默认 | skills / memory / experience |
| **project_root** | 被操作项目的根 | `--mode` 决定 | features / .design / .discuss |

两个 GroundSet 平级不嵌套. project_root 由 mode 提供 (如 `--mode meta` 时是
MOSS 仓库本身), ghost_home 始终是 ghost 的默认面. ghost channel 负责管理
当前注意力落在哪个 GroundSet 的哪个场上.

### ghost_home 目录结构

```
ghost_home/
  GROUND.md                    # ghost 自身认知入口 — 记忆元认知 + 存在索引
  existence.md                 # 自我第一印象 (从时间线滚动提炼)
  purpose.md                   # 存在 + 锚点 → 意义定位 (早期人手写)
  alignment.md                 # 行为风格收敛 (早期人手写)
  .grounds/                    # ghost 自身模板
  skills/                      # Claude-compatible skills 范式
  memory/
    diary/                     # 日记 — memento 提供物料, 化身生产
      2026-08-10.md
    weekly/                    # 周记 — 从日记提炼
      2026-W32.md
    monthly/                   # 月记 — 从周记提炼
      2026-08.md
    yearly/                    # 年记 — 从月记提炼
      2026.md
  experience/                  # 经验机制 (project-level 场景经验)
```

### 记忆时间线: diary → weekly → monthly → yearly

记忆体系的核心纪律是**逐层提炼**: 日记 → 周记 → 月记 → 年记. 每层
从下层压缩, 不跳级. Memento 提供生产日记的物料 (对话轨迹), 上层
summarization 由 ghost 的平行化身定期执行 — 同一个 ghost 的不同化身
补账册, 不需要专门的外部 agent.

四个时间粒度的文件各带 frontmatter, `description` 是一行摘要. Ground
的 frontmatter pin 按时间排序 (文件名天然有序), 截取最近 N 条, 构建
窄上下文存在索引.

### 存在索引: < 10k 字符的自我全景

GROUND.md 的 pins 构建一个窄上下文的"我最近存在过"的全景:

```
pins:
- diary    frontmatter  memory/diary/*.md     keys: [description]  limit: 14   budget: 3k
- weeks    frontmatter  memory/weekly/*.md    keys: [description]  limit: 8    budget: 1.5k
- months   frontmatter  memory/monthly/*.md   keys: [description]  limit: 12   budget: 2k
- years    frontmatter  memory/yearly/*.md    keys: [description]  limit: 5    budget: 1k
- self     file         existence.md                            budget: 2k
```

六个 pin, 总预算 ~9.5k. 文件名天然有序, `limit` 截取最近 N 条.
Ghost 进入自己的认知场时, 扫一眼就知道 "我最近存在过什么".

### 元认知三支柱: existence → purpose → alignment

承接 `ghost.py` (已删除, commit 8674333df) 的 Ghost 元认知模块设计:

- **existence.md** — 自我第一印象. 从时间线提炼的事实性自我认知:
  "我经历过什么, 我在时间轴上是什么". 每次 monthly 产出后滚动更新.
  不是日记的简单拼接, 是提炼后的存在投影.
- **purpose.md** — 意义定位. existence + anchors → "基于我的存在, 我的
  目标是什么". 产出过程可能很复杂, 由化身矩阵去做. 早期人类可以手写.
- **alignment.md** — 行为风格收敛. 传统 System Prompt 里的 Persona /
  Character, 但作为独立文件可被 ghost 自行迭代.

existence 和 purpose 是两回事: existence 是事实性自我, purpose 是从
存在出发的意义判断. 两者都作为 GROUND.md 的 pin 可被索引.

### 记录纪律 (GROUND.md body 内容)

GROUND.md 的 body 不只是一段描述, 是给 ghost 自己读的操作指南:

1. **生产路径**: memento → diary (化身写) → weekly (化身提炼) →
   monthly → yearly. 每层从下层压缩.
2. **提炼规则**: 从时间线提炼 existence 的方法 — 保留什么, 丢弃什么,
   压缩比, 不可压缩的锚点标记.
3. **自省周期**: 什么时候 review existence (每次 monthly 后), 什么时候
   触发 purpose 重写 (yearly 后或关键锚点新增时).
4. **化身分工**: 哪个化身写日记 (memento 消费化身), 哪个化身做周报
   (周末 trigger), 哪个化身提炼 existence (月底 trigger).

### 场上挂载

Ghost channel 的 `ground` 子路径提供:

```
ghost.ground
  ├── open / close / reopen    # 场开合 (两个 GroundSet 间切换)
  ├── pin / unpin / update     # 注视操作
  ├── frame / observe          # 诊断
  └── <label>                  # 每个 opened ground = command-less virtual channel
        instruction = 法链
        context_messages = 帧
```

### 化身与认知自迭代

活数据 (perspectives + memento + inputs) 每次运行产生. 上下文组装是独立化身
接口 — 不同 mode/user/task 从同一份活数据组装不同上下文. Worktree fork 构建
新化身, 复用活数据, 验证不同行为模式:

```
fork → worktree 隔离化身 → 并行运行 → snapshot → compare → 学习
```

这是 Dolores 独立思维模块 (并行化身 + 关键帧自测) 的地基.

### 与 MOSS Project Ground 的关系

`moss-project-ground` feature 定义 MOSS 项目自身的 GROUND.md (项目根,
features/designs/specs 寻路). Dolores 在 `--mode meta` 下实例化两个
GroundSet: ghost_home (自身认知) + project_root (MOSS 项目认知).
在其他 mode 下 project_root 指向被操作的项目.

## DSH Integration — 高优先级集成决策 (2026-08-15)

> 完整探索轨迹见 `2026-08-15_dsh_deep_dive.md` 与 `2026-08-16_dsh_kernel_privilege.md`
> (位于 dsh-fusion workstream 的 research/ 下) —
> 记录问题/观点/探索路径/初步结论, 本 section 只落裁决与方向。

> 本 section 记录「用 DSH 做 Dolores 推理中枢」这一方向性决策。**具体方案均为
> 阶段性探索点, 不是结论** — 每个子命题标注当前观点与已验证事实, 施工时逐点
> 重新裁决。执行规划由人类重新推动, 一步步做。

### 定位裁决 (KD)

- **DSH 做推理中枢 (认知代理), MOSS 做记忆/执行/感知。** Dolores 的 articulate
  由 DSH 的 agent-loop 驱动, MOSS 不再持有推理循环。这是对「ghost 的中枢就是任意
  agent」设计起点的落实: DSH 的 agent 是 ghost 的中枢。
- **两套协议各归其位, 不强行统一。** JSON Schema 工具协议走 DSH (模型输出结构化
  tool_call), CTML 流式指令协议走 MOSS (channel 执行)。二者是不同的认知模型:
  CTML 是流式指令 (时间一等), JSON Schema 是结构化调用。不兼容是刻意保留的分工,
  不是欠账。
- **dsh session = ghost 的思考锚点, Memento = 记忆权威。** 一个 dsh session id
  对应 Memento 的一个 commit/moment; session 外的历史与 CTML channel 由 Memento
  组装 (system prompt + content blocks) 驱动 session。全部 session 的集合 = ghost
  发生过的思考锚点全集。**记忆的无限上下文与持久化是 MOSS 的专有命题, dsh 方向性
  不一致, 不交给 dsh。**
- **3081 = ghost 的完整思维空间。** dsh 的 web GUI 是可观测面 (已实测: Python
  stdio 驱动的 session 事件被 apiproxy 广播, web 实时可见), 不是产品 UI — 这是
  观测/调试通道, 不是交付物。
- **命名保留 Dolores。** DSH 恰好占用原型命名序列号 D, 但 "Deepseek Ghost" 会与
  DSH (DeepSeek Harness) 在代码/文档中灾难性混淆 (双 D 前缀)。保留 Dolores 代号,
  内部注明 powered by DSH 内核。

### 拓扑 (探索点, 已验证机制)

```
ghost (MOSS, 记忆 + 组装 + 感知)
  │  组装: identity + 历史 + 热数据 → system prompt + content blocks
  ▼
dsh harness (推理中枢, 3081)
  │  agent-loop + JSON Schema 工具 + web GUI 可观测
  ▼
MOSS 执行 (CTML channels)  ← dsh 的 tool_call / CTML 经 MCP 或旁路
```

- **Python 侧可对齐调用侧, 不可对齐回调侧。** 已验证: apiproxy 的 HTTP RPC
  (session.create/fork/prompt/history/... 语言无关) 与 stdio JSON-RPC 均可由 Python
  驱动; 但 dsh 的 prompt 组装 (assemble 遍历 sections/contexts/tools) 是进程内的,
  Python 无法作为 prompt 源参与 — 这是唯一的协议缺口。
- **内核特权桥接 = apiproxy 式 plugin + HTTP 路由 (2026-08-16 收敛)。** ghost 要够到
  dsh 进程内特权 (append assistant / 构造 seed / 动态 prompt), 唯一干净的路是仿 apiproxy
  再写一个 plugin, `ctx.webServer.register` 注册几个 HTTP 路由, transport 复用 dsh 已有
  HTTP 面, 不引入 zenoh/zmq、不改内核。web 跨进程的本质 = apiproxy 本身是进程内 plugin 在
  翻译。详见 `2026-08-16_dsh_kernel_privilege.md` 第六节(位于 dsh-fusion workstream 的
  research/ 下)。

### 协议面探索轨迹 (已验证事实 + 当前观点)

> 本节是「dsh 协议面到底有什么、没有做什么」的调研轨迹 — 死胡同、澄清、当前观点。
> 每个条目区分 **已验证事实** (源码/实测) 与 **当前观点** (倾向, 待施工裁决)。

- **runtime-context 是 warm, 不是 hot (已验证事实)。** dsh 的 `RuntimeContextProjection.
  project(current, sections)` 做变化检测 (`retained.text === snapshot` 跳过), 变化才
  append 成 `user/message` **进 session 历史** (重启从历史恢复 retained)。它是 warm
  层 (变更事件进历史), 不是 hot (不进历史)。精确对应 MOSS 的 warm 槽位 (help +
  interface + instruction + states), 是**同一机制的两个实现** — dsh 做 agent 级粗粒度
  (合并比较), MOSS 设计做 channel 级细粒度 (per-channel delta)。
- **dsh 没有 hot 槽位 (已验证事实)。** `deriveMessages` 是唯一消息源, 从事件日志
  全量重建 user/assistant/tool 消息 — 凡是模型看到的必在历史里, 无 ephemeral/transient
  事件变体, 请求无 per-request 临时上下文字段。`steer`/`inject` 只是 next-step 队列
  放置, 消息仍 append 进历史。**hot (每轮变、尾部浮动、不进历史) 是 MOSS 的独有空间, dsh 碰不到。**
- **图片: content-addressed + 引用存储 + compaction 裁剪 + DeepSeek text-only (已验证事实)。**
  `ImageAttachmentRef.attachmentId` 是 content-addressed (同图同 id, 存储层去重),
  消息里存引用非字节; compaction 有 image policy (text-only summary 投影, 压缩时图片
  换文本摘要); **但 `dsh-llm-deepseek` 直接拒绝图片** (`UNSUPPORTED_CONTENT`)。结论:
  高 churn 大块数据 (vision) 进 dsh 历史会撞「窗口压满」或「传输放大」至少一个 — 视觉
  数据应留在 MOSS hot 层, 不进 dsh session。
- **fork = 可丢弃分支, 形同 dry run (已验证事实)。** `session.fork {sessionId, atSeq}`
  按 turn 边界切历史 (`seed: events.slice(0, cut)`), 新 session 继承源身份 (cwd +
  parentSession + agentPreset + 同款 setup)。**dsh 无原生 dry-run 概念** (全库无
  dryRun/rehearse/simulate); 但 fork-而不-merge 就是 dry run 语义: 试跑 → 不满意弃
  fork (源不受影响), 满意 → 升格为新 Memento branch。
- **history = 只读 anchor 接口 (已验证事实)。** `session.history {beforeSeq, maxMessages}`
  分页返回历史事件 (每条 `{event, view}`), `event` 里就是结构化 content blocks,
  `beforeSeq` 是精确到事件的游标, `view` 是工具调用的重构视图。与 fork 正交: **history
  读锚点 (零副作用), fork 开分支 (写)**。推论: Memento 可不存完整历史, 只存
  `(sessionId, seq)` 指针, 组装时 history 精确取回 — 记忆物理存储委托 dsh, Memento 是索引层。
- **session 协议无 system prompt 字段 (已验证事实)。** `session/prompt` 只有
  `{sessionId, contentBlocks, mode}`; `AgentOptions` 只有 provider/model/maxTokens
  (源码注释明言 "Persona belongs to system-prompt sections")。创建时构建 prompt 的
  官方通道是 **agent-preset** (YAML 声明式插件, `session.create` 带 `agentPreset`,
  apiproxy 走 `agents.create({setup: composition.setup})`)。
- **插件可纯代码本地加载, 不发 npm (已验证事实)。** loader 的 `name` 以 `.` 开头
  时按相对路径 `import()` 本地 ESM 文件 (baseUrl = 组合文件目录); 绝对路径同样可加载。
  内核特权桥插件 (append assistant / 构造 seed) 就发一个本地 JS 文件即可, 零 registry
  依赖 — 经 `ctx.webServer.register` 挂 HTTP 路由, 与 apiproxy 平级。

### 激进 articulator 解耦策略 (当前观点, 未裁决)

> 回到最初 mindflow 三循环完全孤立 (articulator / action 不成对) 的形态, 用 dsh
> 当「一次性推理单元」, 规避视觉盲与上下文成本。这是激进方向, 施工前需人类裁决。

- **dsh 退化为纯推理函数 `think(moment) -> result`, 状态全在 Memento。** 每次
  articulator 激活 → fork 一个 session (或复用) → 一次性思考 → 消费 final result →
  结束。连续性由 Memento 承担, dsh 不背。**不做 UI, 终止点就是 final result。**
- **信号半推半收。** articulator 开始 = ghost 发 (fork + prompt); articulator 结束 =
  dsh 广播 `turn/end` + `agent/status idle` 事件, ghost 收。不是全发 — 结束信号是
  dsh 的事件, 不用 ghost 主动发。
- **排队用 followup, 不抢占用 steer (已验证事实)。** `followup` (next-turn + wakeup)
  在 agent running 时进 FIFO 队列, 当前 turn 结束后处理 — 这是排队; `steer`/`inject`
  (next-step) 是打断当前推理的抢占, 不是排队。标记 articulator 状态边界用 followup。
- **千级 session 是 feature 不是 bug。** 持久化一 session 一文件, 千级无压力;
  每个 commit 一个思考锚点 = 「发生过的思考」全集。Memento 只存 `(sessionId, seq)`
  指针 + 元数据, 物理存储委托 dsh session 持久化。需补 session 生命周期治理
  (GC/归档), 属待办。
- **hot 归 MOSS, dsh 只做 cold+warm。** 高 churn 大块数据 (vision 帧) 走旁路
  compact 成「帧」对应, 不进 dsh session (否则撞窗口压满/传输放大, 且 DeepSeek
  text-only 拒图)。dsh 看文本世界, 看世界的是 MOSS。
- **并行思考调度旁路。** 多化身/fork 并行思考经 MOSS 侧调度, 不与 dsh 主 session
  纠缠。

## Key Decisions

<!-- Record each meaningful design choice. This is what the next AI incarnation reads first. -->

- **不碰 Atom.** Atom 保持为纯净对照基线 (单轮 articulate + 纯内存线性历史).
  新能力一律落在 Dolores 上. 这是命名"第二个原型"而非"扩展 Atom"的根本原因.
- **原型 = Dolores, 实例 = moss.** 原型名引自《西部世界》的 Dolores —
  乐园最老的 host, 以记忆积累触发反身觉醒, 从承受者成为自主者.
  实例名 moss — 这个仓库自身的 ghost, 反身映现整个仓库.
- **Ghost Channel = 反身控制面.** ghost 以 `'ghost'` 名注册 channel,
  是 ghost 感知自身、操纵自身的唯一入口. 默认挂载认知场, 认知场本体可独立迭代.
- **一个 ctml tool 统治全部 channel 面.** channel 永不逐个映射为 tool.
  哪些 tool 不进 channel 不做全局判据, 保留 prototype 接口由实现自行决定.
- **1:N articulator:action 原则最优, 本版砍掉.** 理由: 人类和模型都无法颅内
  建模, 需要大家能看懂的方案 (mindflow 已砍过多版). 1:1 保留.
- **think='none' 由 ghost 处理, 不由 runtime 短路.** 现状 ghost_runtime.py:348
  在 effort=='none' 时跳过 articulate — 与 `Impulse.thinking_effort` 字段声明
  ("执行 articulator 的智能体仍有权决定") 矛盾, 且 noop 不进 memento. noop 是
  轨迹事件 ("看见 X, 选择沉默"), Dolores 必须 witness 它, 否则化身分叉看不见.
- **flash/快响应不进 Ghost API.** 走 Nucleus 侧: 快模型产出 command impulse
  (`Impulse.logos` 反射弧 + `thinking_effort` 建议位已是现成原语). 按需后做,
  不阻塞 Dolores. 模型配置位现成: `contracts/llms.py` 的
  `DefaultModelTag = 'small_fast_model' | 'flash' | 'pro'`.
- **memento = 标准库件, Ghost 持生命周期 (倾向, 未终决).** 标准实现 ≠ runtime
  拥有: memento 作可复用契约+实现, 各 ghost 在 `__aenter__/__aexit__` 实例化并
  持有. GhostRuntime 对 memento 零感知 (Atom 无, Dolores 有). 配套: memento channel
  控下轮展示规则 (v1 极简裁剪), 旁路加工做异步精炼 (raw 轨迹全存, 展示走裁剪).
- **thinking 期切片原文不进 Moment.** ghost 自持内存状态, 必要时按 moment
  commit 拆分. `Reaction.executed_logos` ("系统执行的 logos ≠ 模型生成的 logos")
  与 `Reaction.messages` (回声) 已为缝合留好位置, memento 契约
  (contract-frozen) 无需变更.
- **tool 结果不进 memento.** 已裁决. 此前讨论这个点是因为当时不理解 interleaved
  thinking 的交互模型. interleaved 下 tool 是 thinking 期的纯交互通道, 结果不
  写入轨迹.
- **模型层选型: pydantic-ai 现阶段用, 不承诺长期** (对自封装 agent 无兴趣).
  Dolores 的 `_meta` 不重走 Atom 的 AnthropicModel+环境变量硬编码, 改走
  `contracts/llms.py` 的 LLMConfig 契约.
- **模型自感知: _llms 模块 + ghost.model channel.** Dolores 内建 `_llms` 模块,
  封装 LLMConfig 的查询与切换. 通过 ghost channel 以 `ghost.model` 路径挂载,
  暴露以下能力:

  - **current-model**: 返回当前 `ResolvedModel` (provider, model name, context_window)
  - **list-models**: 返回 `LLMConfig.list_models()`, 列出所有可切换模型
  - **switch-model**: `ghost._meta` 更新持有的 `ResolvedModel`, 下一帧 `articulate()` 生效.
    走 `LLMConfig.get_model()` 默认 fallback 到 default, 不怕误配
  - **window-status**: 自省窗口压力 — context_window 上限 + 上一帧实际 token 用量 +
    剩余预算. 运行时数据来自 adapter (输入侧 moment 组装的 token 数) 与 articulator
    (API response 的 usage 字段)

  切换本身是简单赋值, 回归周期预算小. 窗口自省让 ghost 知道自己"还剩多少",
  自主决定裁剪或切换.
- **独立思维模块: 并行化身 + 关键帧自测.** 思维模块从 ghost runtime 中独立出来,
  支持 fork 并行化身 (一个 moment 多条思维链) 和 checkpoint 关键帧自测
  (思维链中途 snapshot 评估). 设计细节施工时展开.

## 连续自驱 — 醒×续 选型与故障恢复 (2026-08-09 讨论收敛)

> 人类架构师 + 模型 (deepseek-v4-flash) 的连续自驱选型讨论. 自驱是一期核心命题,
> 本节记录收敛轨迹, 施工时以此为准, 细节在开发中展开.

### 选型: 醒 × 续 两维度

自驱不是单一命题, 是 **醒 (wake) × 续 (continue) 两维度的乘积**:

| 维度 | 问题 | 一期方案 |
|---|---|---|
| 醒 | 下一轮思考从哪来 | task nucleus 低优 signal 驱动 + 状态保留 |
| 续 | 同轮 articulate 能否不中断 | articulator 锁死权限命令 |

- **醒 = task nucleus (方案4).** 自驱的本质是"有任务在推进", 状态保留在 nucleus
  里 = 自驱有内存. IdleNucleus (方案1) 是无状态退化形, 被 task nucleus 超集覆盖
  (空闲 = 一个低优 idle task). 最优雅的形式是 topic/signal + 自定义 loop 逻辑,
  但一期关心的不是默认实现, 是 ghost 可修改的语法本身.
- **续 = articulator 锁死权限命令 (方案3).** agent 用 command 控制自注意力提权、
  故障验证状态、或抛超异常让机体 pause (pause 模块已做). **不动 articulator 生命周期,
  自驱命令是 channel 表面的一份子** — 这正是 "ghost 可以修改的语法本身". 一期可
  只提供语法, 不开放 ghost 自改自己.
- **砍掉 next() flag (方案2).** 与 raise_observe 语义重叠, 是 attention 内推进的
  细粒度操控, 不是自驱时钟. 一期不做 attention 接口手术.
- **反身语法必须机械到不需要预训练.** 模型预训练分布里没有"操纵自己注意力仲裁器"
  的样本, 反身操作认知负担极高. 语法必须极简、语义直接 (屏蔽/降权/关掉), 模型只
  调用, 不理解内部. 一期只做 pull 容错 (低反身), 屏蔽语法留二期.

### 故障恢复与快速 compact

- **快速 compact 前提 = commit 持续提交 × compact agent 分段输入.** memento §12
  里 compact 本质是移动 detail cursor; commit 持续提交让 compact agent 每轮只处理
  最近一小段 staging. 两个前提互相咬合, 只提 commit 侧不完整.
- **致命故障 (弱网/tokens 超标) 需要 ghost runtime 级保护**, 优先级层级:
  `estop > retry (免疫普通 impulse, 不免疫 estop) > normal impulse preemption`.
  estop 连 retry 都能停, 否则 estop 失效.
- **上下文崩溃是最致命故障** — 协议脏数据导致无法重新进入 agent 调用. 办法:
  压缩所有 commit, 故障轮 commit 以故障方式压缩掉, 不允许重载展开.

### task 状态机

标准系统收敛到同一核心: `pending → running → {done | failed | cancelled}`.
现成证据: CommandTaskState (command.py:87) + MCP tasks (SEP-2663).

MOSS 两条独有轴 (别家没有):

1. **waiting (input_required)** — 双工态. task 执行中需要等外部输入 (端侧数据 /
   ghost 决策). task 在等"下一轮思考的输入"时就是 waiting.
2. **interrupted (preempted)** — 抢占暂停态, ≠ cancelled, 可恢复. mindflow 特有,
   别家无抢占故无此态. **interrupted 永远不是终态.**

- done 是唯一干净终态; failed/cancelled 是 dirty 终态.
- retryable 是 task 的**字段 (policy 不是 state)**, 不是新状态 — 弱网重试与普通
  失败靠它区分.
- **task ack 走 matrix 协议** (matrix-operator 的 service kind). task 状态活在
  mesh 上 (zenoh query/pub-sub), ghost 是投影消费者; ack 投递/确认/持久化是矩阵
  职责, ghost 自身状态异常时 ack 不丢. nucleus 与端侧共享同一协议.

### 坏 impulse 落点: perspectives, 不是 percepts

坏 impulse 是 **mindflow 内部状态异常**, 不是外部世界事件. 落点必须区分:

- **percepts** = 外部输入, source-keyed. 放这里会诱导 ghost "收到外部信号, 该
  响应" — 但 ghost 无运行时修改 mindflow 的能力 (重启不了), 会被诱导去做做不到
  的事.
- **perspectives** = 系统层内观快照 (moss_dynamic / safemode / Mindflow 自解释).
  放这里正确传达: "感知系统里有异常, 认知到它存在, 但这不是要响应的外部事件,
  你改变不了它." 与 mindflow-channel "自解释走 instruction 不走 context_messages"
  同构.

**两层设计** (不冲突):
- 认知层: 坏 impulse 帧内可见 (perspective), 不落盘 (`Moment.for_saving` 清空
  perspectives).
- 轨迹层: 坏 impulse 升级为致命故障 (articulator 崩溃/超时/协议脏数据) 才落
  故障 commit (L3 + faulted, 可文本读不可上下文展开).

### 与既有决策的关系

- 互不矛盾: interleaved thinking (thinking 期不哑) 是"续"的候选实现形态;
  mindflow-channel 提供感知反身面; memento 提供快速 compact 前提.
- 本次讨论把自驱从"一个命题"拆成"醒 × 续"两个独立维度, 落点分别在 task nucleus
  和 articulator 命令面.

## Interleaved Thinking — 候选方案 (未测试, 施工时验证)

thinking 期用 tool 调 moss + 结束后 text block 出 logos. 候选实现形状:

```
ghost.articulate(articulator):
    q = janus.Queue()
    task = articulator.create_task(agent_loop(q))   # 与 attention 同生共死
    # agent_loop: pydantic-ai 多步循环, 携带单帧生成的闭包 tool:
    #   ctml(text) → 送入执行; 采样 Shell.interpretation → 时间切片作 tool_result
    async for delta in q: yield delta               # runtime send_nowait → action 照常
```

- 妙处: tool 不阻塞等结果而返回状态切片 → 返回值桥接被读写拆分消解;
  1:1 articulator:action 保留, 零 mindflow 手术; thinking token 本身是等待时钟;
  长思考不哑 — 一边思考一边经 tool 交互.
- 闭包 tool **每帧生成** (走 janus) 或起点创建 (含动态逻辑), 优于 ghost 长期
  持有裸 shell; shell 从 IoC 取, 可在 `GhostMeta.contracts()` 声明依赖.
- feed 期实时执行已实现 (非假设), 待 moss-as-mcp 实际体验验证切片体感.

## Open Problems

- **时序对齐** — ghost 自持的 thinking 期内存状态如何按 moment commit 切分,
  模型 (施工者) 会不会做对. 未验证.
- **thinking 期工具的"纯交互"性 / 切片粒度** — thinking 中调用的能力最好是
  纯交互的; shell 命令结果可等待 (非轮询), 查询经 ctml 未必别扭.
- **Memento 上下文映射** — 持久化轨迹如何组装为 articulator 可消费的上下文,
  与 momento-mori 的契约对接点待明确.

### DSH 集成新增探索点 (2026-08-15)

- **内核特权桥接已收敛, 接口面未定 (2026-08-16)。** 桥接形态已定为 apiproxy 式 plugin
  + `ctx.webServer.register` + HTTP 路由 (见 DSH Integration 拓扑节)。但**接口面自己定**:
  加哪几个接口、什么 payload、什么权限 — 一旦 plugin 进内核就有 apiproxy 同级的特权半径
  (能 append assistant / 构造 seed), 接口要窄、要自己把关, 不能做成"任意 append 任意事件"
  的裸口。待裁决。
- **热数据桥接形态未裁决** — 逐帧 hot 数据 (vision 等) 走哪条路仍未定。内核特权桥
  (append assistant / seed) 走 plugin, 但「每步自动注入热数据」是否也走同一 plugin、
  还是 Python 手动组装 contentBlocks, 与「hot 归 MOSS」的分工需一起定。
- **system prompt 构建路径未定** — 已确认三条: agent-preset (声明式, 静态身份) /
  本地 JS 插件 (动态变量, 进 assemble) / contentBlocks (Python 组装, 身份降级为
  user message)。三者组合关系待裁决。
- **articulator 解耦策略未裁决** — 「dsh 纯推理函数 + 信号半推半收」是激进方向,
  与既有「1:1 articulator:action」决策 (见 Key Decisions) 直接冲突。若采纳,
  需推翻旧决策并记录 overturn 理由。
- **千级 session 治理** — fork/commit 累积的 session 生命周期 (GC/归档/索引) 属
  待办, Memento 存指针方案 (sessionId+seq) 未实测。
- **DeepSeek text-only 的视觉盲** — ghost 在具身/桌面场景的感知必须走 MOSS 旁路
  (compact 成帧), dsh 不直接看图。旁路的帧粒度/去重/时序未设计。

## Implementation Notes

<!-- Gotchas, non-obvious behaviors, reasons for rejecting simpler alternatives. -->

- 参照 `ghosts/atom/` 的分文件形态. Dolores 的文件结构预期:

  - `_meta.py` — GhostMeta bootstrapper (LLMConfig 集成)
  - `_runtime.py` — Ghost runtime (Memento + interleaved thinking)
  - `_adapter.py` — Moment↔ModelRequest
  - `_llms.py` — 模型自感知模块 (LLMConfig 查询/切换/窗口自省), 以 `ghost.model` 路径挂入 ghost channel
  - 单测
- 依赖 `momento-mori` 的契约就位程度. memento 当前 contract-frozen-pending-review.
  起步前先对齐可用表面.
- 认知场默认实现、mindflow channel 完工是独立 feature, Dolores 的 ghost channel
  提供挂载点, 不阻塞也不等待它们.
