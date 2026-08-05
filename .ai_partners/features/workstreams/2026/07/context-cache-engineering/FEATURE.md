---
title: Context Cache Engineering
status: in-progress
status_note: ContextMonitor 设计收敛, 独立模块先行 (host/context_monitor.py), 与 interleaved thinking 同款交付节奏
priority: P1
created: 2026-07-30
updated: 2026-08-05
depends:
  - momento-mori
milestone:
description: >-
  Ghost runtime 上下文构建的缓存工程化：append-only 历史 + cold/warm/hot 分层 +
  A/B 双投影 + memento 重绘对齐。前缀 KV 缓存经济学作为一等约束进入 context 组装。
---

# Context Cache Engineering

> Use `moss features set-status context-cache-engineering <status> -m "note"` to update state.

## Motivation

主流 LLM API（Anthropic/OpenAI/Gemini/DeepSeek）的缓存机制同构：**前缀 KV 缓存，
字节级精确最长前缀匹配，从 token 0 起算，读价约为写价的 10%（Anthropic/DeepSeek）**。
铁律：前缀中任何 token 变更，其后缓存全部失效。

MOSS 当前的 context 组装没有把这个约束当一等公民：

1. **Atom runtime 存储 bug**：`save_model_request`（`ghosts/atom/_runtime.py`）把
   全量请求（含 perspectives + hint）原样存进 `_history`——存储路径复用请求构建
   路径，没做 `as_history_messages` 的历史视图转换。每轮 perspectives 快照全部
   沉入历史（token 膨胀 + 陈旧快照污染），且与 memento 层设计意图脱节。
2. **消息顺序反了**：`as_request_messages`（`core/blueprint/memento.py`）里
   perspectives 排在 percepts 前。ephemeral 内容插在 durable 内容前面，导致
   即使修了存储 bug，percepts 也会因位移而每轮 miss。
3. **所有动态数据被当 hot 处理**：`make_dynamic_messages`（`core/ctml/v1_0/prompts.py`）
   每轮全量重绘 channel 的 interface/instruction/description——这些是低频变更的
   warm 数据，却在为"位置移动"付每轮全价。
4. **tools 方案不可取的缓存学论证**：JSON Schema tools 序列化在 prompt 最前
   （tools → system → messages），变更 = 从 token 0 全量 miss。MOSS channel 热插拔
   的特性决定它不能做成 tools——这在 Code as Prompt 的表达力理由之外补上了
   缓存经济学理由。

## Key Decisions

### D1. cold / warm / hot 三层，按变更周期 k 归位

| 层 | 定义 | 位置 | 成本 |
|----|------|------|------|
| cold | 会话内不变（CTML meta、ghost 人格） | system prompt | 付一次，恒 0.1 |
| warm | 偶变（channel 的 interface/instruction/description/states/failure） | **变更事件追加进历史**（event sourcing） | 变更时付 ΔS，其余恒 0.1 |
| hot | 每轮变（`context` 传感快照、hint、perspectives） | 尾部浮动，用后即弃 | 每轮全价，有界不累积 |

一句话判据：**k=1 走尾部，k>2 走变更事件，k=∞ 走前缀**（0.1 读写价比推出
盈亏点 k≈1.1）。

**warm 不是预先声明的类别，是运行时观测的行为**——变更检测消解了"变更频率
不可知"的问题。唯一需要人为声明的是 hot（channel `context` 字段）。

**static 收编为 k=∞ 的 warm**：会话重建时 static+warm 全量物化为历史顶部一次
提交（`<moss_static>` 包装保留），之后一切变更（含热插拔导致的 static 变更）
走 warm-delta。system prompt 瘦身为纯 cold，token 0 前缀永不失效。
`make_static_messages` 拼进 instruction 的现行做法废弃。

CTML 规范无需修改：`moss_dynamic` 的 "append + same-name override + only the
most recent is authoritative" 语义就是 event sourcing，规范层面是现成的。

### D2. 变更检测：hash 渲染文本，不 hash 数据结构

对照面是 `sha256(rendered_warm_block_text)`，以 channel path 为 key。
**不要**对 ChannelMeta 做规范化序列化对照——`created` 等 default_factory 时间戳
字段会假阳性，exclude 清单会腐烂。进上下文的是渲染文本，文本没变即语义没变。

确定性前提（现有代码的三颗地雷）：
- `metas: dict` 遍历序（prompts.py `make_dynamic_messages`）→ 按 path 排序；
- `states: dict` 迭代序 → 排序输出；
- `refreshed` 时间戳 → 移出参与 hash 的文本（delta 消息自身带时间戳合法，
  写一次即成稳定历史）。

### D3. 防抖三态机（时序敏感检查）

warm 数据每轮都变时，delta 入史会退化为方案 4（历史累积 + 污染）。按 channel
block 为单位：

```
WARM ──连续 2 轮变更──→ DEMOTED(hot)
DEMOTED ──连续 M 轮未变──→ 提交一次当前版本入史 ──→ WARM
```

- 首次变更永远走 delta（不会首轮 hot）。
- DEMOTED 期间当前版本改在刷新尾部渲染；历史里的旧版本由 CTML
  "latest authoritative" 语义覆盖，规范免改。
- 回归时等 M 轮（建议 3，需实验）稳定再提交一次最终版。

### D4. A/B 双投影与前缀不变式（承重墙）

请求周期：moment → **单次渲染**产出 `(durable, ephemeral)` 两段：

```
request_A = durable + ephemeral    # 发送（全量）
request_B = durable                # 存储（历史投影）
```

**B 必须是 A 的字面前缀**——缓存收益成立的唯一条件。禁止 A/B 走两条独立渲染
路径（`as_request_messages` / `as_history_messages` 各渲染一次）：任何格式差异
都导致每轮缓存静默全丢，且无报错，只能靠账单发现。前缀性质由构造保证。

配套约束：
- moment 在周期开始时冻结，发送后异步转化 B 期间不允许写入；
- pydantic-ai 一次 run 的多个 request/response 对中，**只有第一帧携带 ephemeral
  内容**（后续是 tool returns，天然 durable）——入史时仅替换第一帧 A 为 B；
- durable 段内序：`[reaction][warm-delta][percepts]`；ephemeral 段：
  `[perspectives][executing][hint]`。核心规则：**每轮内部 durable 必须排在
  ephemeral 之前**（下一轮 ephemeral 消失，若在前则其后 durable 全部位移 miss）。
  现行 `as_request_messages` 的 perspectives-before-percepts 顺序必须翻转。

warm-delta 的通道候选：作为系统 source 走 `percepts`（source-keyed、durable、
append-only，`to_history_turns` 自动正确处理），moment 层几乎零改动。

### D5. 压缩重绘：usage 触发 + memento 三层折叠

- 触发：`response.usage` 的 input_tokens ≈ 当前上下文权威长度，超阈值（80%）
  触发重绘。与 momento-mori §7 的 cache 遥测（Open Problem #3）互补：规则触发
  是主力，模型自宣 `<memento:commit/>` 是加分项。
- **重绘前先把 staging 机械 commit**（§17.3 #2 "ref 移动前活边先落锚"同款），
  重绘天然从 commit 锚点出发——"commit 是重绘的起点"从哲学变成运行时时序。
- 重绘结果分层（memento §3.5 窗口渲染 O(K+m) 的具体化）：

```
[instruction]                          cold
[久远 commits: 深度摘要]                compact-memento
[最近 N 个 commits: CommitNote 索引]    一行释义, 可后补
[最后 M 个 commits: moments 全量展开]   通常 M=1
[staging: 活边全量展开]                 含 warm-delta percepts
[本轮刷新尾部]                          ephemeral
```

  上下文仍超长才折叠最后 m 个 commit。折叠无损：commit 永远可寻址，
  `show <commit_id>` 缺页中断（memento §5 分页调度）负责按需展开。
- **hash map 重置基准 = 重绘物化版本**，不是清空——清空会让重绘后首轮把全部
  channel 误判为变更，提交冗余 delta。
- 物化必须是 commit 锚点的**确定性纯函数**：memento §7 的化身扇出共享前缀
  （N × 10x 节省）依赖这一点。

### D6. commit 粒度（纠错记录）

讨论中曾提议 "commit = 一个 articulate 回合"——**错误，已撤回**。回合对应
Moment（单帧），commit 是 staging 时间前缀的冻结批（memento §16/§17）。
折叠/索引/展开的单位都是 memento commit。

### D7. 缓存注入与计量由 pydantic_ai AnthropicModelSettings 承载

pydantic_ai 2.5.0 已提供完整的 Anthropic prompt caching 支持，MOSS 无需直接
操作 Anthropic SDK 的 `cache_control` 参数。

**注入侧**（`AnthropicModelSettings` 三个字段）：

| 字段 | 作用 | MOSS 用途 |
|---|---|---|
| `anthropic_cache_instructions: bool\|'5m'\|'1h'` | system prompt 末尾放 cache_control | Phase 0：一行启用 system instruction 全量缓存 |
| `anthropic_cache: bool\|'5m'\|'1h'` | 服务端 automatic caching，断点随对话前移 | Phase 3+：history 前缀自动缓存 |
| `anthropic_cache_messages: bool\|'5m'\|'1h'` | 最后一条 user message 末尾放 cache_control | 备选（与 `anthropic_cache` 互斥，给不兼容网关用） |

`anthropic_cache` 与 `anthropic_cache_messages` 互斥，但均可与 `anthropic_cache_instructions` 组合。
三者共享 Anthropic 的 4 个 cache point 预算（automatic 占 1 个，剩 3 个给 explicit）；
pydantic_ai 内部 `_limit_cache_points` 做溢出裁剪。

**计量侧**（`UsageBase` → `RequestUsage` / `RunUsage`）：

```
Anthropic API usage                      pydantic_ai UsageBase
────────────────────────────────────────────────────────────────
cache_creation_input_tokens          →   cache_write_tokens
cache_read_input_tokens              →   cache_read_tokens
input_tokens                         →   input_tokens
```

`_map_usage()` 通过 genai-prices 自动提取并映射，流式/非流式路径均覆盖。
每个 `ModelResponse.usage` 携带当次请求的缓存计量，`RunUsage` 跨请求累加。

**MOSS 接入点**：`Atom.articulate()`（`_runtime.py:86-92`）中
`stream.response.usage` 在 stream 退出后可用，包含 `cache_write_tokens`、
`cache_read_tokens`、`input_tokens`。当前未读取——需在 `on_articulate_exit`
或 usage 监控点接入。

缓存命中率 = `cache_read_tokens / input_tokens`。
有效计费 = `(input_tokens - cache_read_tokens) + cache_read_tokens * 0.1 + cache_write_tokens * 1.25`。

## V2 — Context Monitor Architecture (2026-08-04 收敛)

本轮讨论（人类 + deepseek-v4-flash via claude code）把 interleaved thinking 的游标
机制泛化为运行时上下文监控模块，取代 GhostRuntime 的上下文构造。收敛结论作为可
重放任务讯息。**以下 V2 决策部分替代 V1 的 D1/D3/D4 的分类与交付**，差异在文内标注。

### V2.0 概念与边界
- ContextMonitor 复用 InterleavedThinkingToolset 的三件套：生命周期（new_from_shell +
  async with）/ 线程安全（双锁 + ThreadSafeEvent）/ 游标语义（drain/status）。
- 从 MOSShell 稳定拿到 static / dynamic / warm-changed / 执行游标，替代
  `_run_articulator` 的 `refresh_metas` + `with_perspective('moss_dynamic')`。
- **单一路径约束**（"两个事实"）：monitor 激活时 warm 渲染完全归 monitor，否则完全
  走旧 `dynamic_messages()`。两条路径渲染同一内容而字节不同 → A/B 前缀静默破裂。
- 交付节奏 = interleaved thinking 同款：独立模块 + 单测验证 → 再决定装线。
  装线保留 `moss_static` / `moss_dynamic` 向前兼容接口，两个非关键帧判断（是否激活）
  即可无痛。

### V2.1 warm 获取 = 帧级 compare-and-emit（非事件订阅）
- 基准 B = {path → P(path, meta)}，来自**最后一次 ACK** 的帧投影，非最后一次所见。
- 每轮帧级 diff `P(last_metas, new_metas)`：
  - L0 存在性：ADD / REMOVE（集合差）
  - L1/L2 存活 channel：逐 warm 单元投影文本对比
- P = 确定性投影（dynamic 块去掉 context），metas/states/commands 全排序、无时间戳。
  `created` 字段逻辑无关（不参与投影）。
- 为何非事件订阅：变更源是宽的（refresh_metas / hot-swap / states_channel /
  add_virtual_channel / node 进程），事件订阅须枚举全部变更源，漏一路 = 静默 warm
  miss（模型拿过期 interface → 发错 CTML，正确性 bug 且无报错）。漏斗出口对比构造上
  不可能漏。
- 不丢保证：基线只在 ACK 后推进，ACK 前一切可重放。

### V2.2 分层修正：定义 vs 状态（替代 D1 的 failure-in-warm）
| 层 | 内容 | 本质 |
|----|------|------|
| warm | interface、states 目录、description、instruction | 能力定义，低频，进历史 |
| hot  | failure、connected、available、context | 能力状态，每轮重绘，尾部 |
- `failure` = "unavailable reason" 字符串（字段命名是历史包袱，语义是"此刻为何不可用"）。
  connected 现合并进 failure（`tree.py:158` new_empty "not connected"），保持合并，不拆。
- hot 两个来源：**声明式（状态层）** + **检测式（定义层降级）**。状态层天然无需降级机。

### V2.3 warm 单元与发射粒度
- 单元 = 字段组（desc-instruction / states / interface），独立 hash / delta / 降级。
  不拆 channel 级：状态机 channel 的 current_state 跳变不该带动大 interface 重发
  （speech 是典型）；行为树 channel 的 interface churn 也不该拖累其余单元。
- 发射 = 单元内整块重发，interface 不拆 command 级。整块 override 自洽、无 merge
  风险；command 级发射不解决病理 churn（照样污染历史）。command 级只用于检测
  （round 2：变了几个 command 作 churn 信号，驱动降级决策）。

### V2.4 降级（防过敏）与 hot→warm 路径
- 每单元三态机：WARM --连续2轮变更--> DEMOTED（尾部渲染，不入史）；
  DEMOTED --连续M=3轮未变--> 当前版入史一次 --> WARM。
  一个降级周期历史增量上界 = 3（2 条降级前 delta + 1 条复位 commit），与 churn 时长
  无关。阈值（2, M）留 dogfood 校准。
- **hot→warm 只有一条真状态机（P1: 降级单元复位）**。P2 状态恢复（failure 清除）不是
  热→温——定义层从未离开 warm，只需停止尾部渲染 + 写恢复锚点。P3 卡热 = 安全终态
  （不重入历史，无 override 语义，无正确性风险）。
- 卡热安全是设计关键：降级不要求复位保证正确性，只关乎经济学。故脱敏机制可后置。

### V2.5 历史锚点 = 统一状态转换事件
- REMOVE tombstone / failure 转换 / connected 转换 = 同一种 durable 小锚点，
  **转换时写一次**；持久态由尾部 hot 每轮承载。模型读历史 = 转换时间线，读尾部 =
  当前权威状态。
- 状态快速翻转时锚点会 churn，是否加防抖（稳定 N 轮才落锚）留 dogfood 校准。

### V2.6 ACK 纪律
- 基线只在"warm 确认进入历史"后推进，必须得到调用方 ack——模型请求失败轨迹即丢失，
  没有 ack 无法保证 warm 最终可见。
- 接线归 GhostRuntime（moss 框架承诺义务，屏蔽复杂度）。ACK 引用帧快照，非当前 metas
  （防 mid-round 漂移）。

### V2.7 Moment 插入与既有改动
- durable: [reaction][warm-delta][percepts]; ephemeral: [perspectives][executing][hint]。
- 需要 `as_request_messages`（`memento.py:240-256`）翻转 perspectives/percepts 顺序
  （Phase 1，独立改动，影响所有 Moment 消费者）。
- 渲染确定性排序在 monitor 内部做，**不动 prompts.py**。monitor 独立实现 warm 渲染，
  可复用 ChannelMetaPrompter 字段方法（description/instruction/states/interface/failure）
  但不含 context。interleaved thinking 已验证：interface 在历史中不在最新位置，可以跑。

### V2.8 交付顺序（修订 V1 落地顺序）
1. **独立模块 `host/context_monitor.py` + 单测**（纯函数注入 metas，零 shell 依赖，
   并行无冲突）。单测清单：投影确定性 / 无变更无delta / 字段变更delta / ADD/REMOVE /
   ACK 纪律（算过未ACK下轮重算）/ 降级三态机。全部可构造，不依赖 shell。
2. Phase 1: memento `as_request_messages` 顺序翻转 + Atom 存储 A/B 修复（成对）。
3. 装线决定: monitor 激活于 GhostRuntime，moss_static/moss_dynamic 干净切换。
4. dogfood 度量: 状态转换频率 / interface churn 率（按 channel 类型分桶）/ 缓存命中率
   （`cache_read/input_tokens`）/ 模型行为（更快停止调用已挂 channel）。

**悬而未决（全部留 dogfood）**: available 的处理（状态浮现 vs 现行 filter 消失）、
锚点防抖、降级阈值（2/M）、REMOVE tombstone 形式。

### V2.9 ContextSnapshot API 与三类原语（2026-08-05 收敛）
- **`snapshot(metas) -> ContextSnapshot`**（原 V2.8 的 `update()`）。返回冻帧快照,
  承载 `warm_deltas`（进 durable）+ `hot_messages`（进 ephemeral 尾部）。
- **`ack(snapshot)`** 替代无参 `ack()`。快照是显式 ack 令牌, 携带单调帧号;
  双重 ack 幂等、旧快照被新快照 supersede 后 ack no-op。消除游标幻觉。
- **三类原语统一于 ContextMonitor**（InterleavedThinkingToolset 标准化超集）:
  1. 静态只读透传: `meta_instruction` / `static_messages` / `dynamic_messages`
     （向前兼容）/ `channel_full_block(path)`。
  2. warm + hot 治理: `snapshot` / `ack`。**待补 gap**: demoted 单元的当前版本
     应出现在 `hot_messages` 中（模型在尾部可见）, 目前 hot 只渲染 failure+context。
  3. interpreter 游标 + 双工 task 历史（装线后激活, 继承 InterleavedThinkingToolset
     的 Tracer 订阅 + 线程安全 buffer）: `drain` / `status` / `wait_interpreter_done`。
  三类统一在同一个 monitor, 不同时钟的数据路径各自独立（K5 二维分解）。
- **已落码**: `host/context_monitor.py` + 33 tests (0.65s), 纯函数零 shell 依赖,
  覆盖投影确定性 / 无变更无delta / 字段变更delta / ADD/REMOVE / ACK 纪律 /
  降级三态机 / hot 层渲染。并行无冲突, 与 interleaved thinking 同款交付节奏。
- 下一步: 填补 demoted-in-hot gap; 装线三路径（moss-mcp / ghost runtime /
  ghost articulate 内部双上下文困境待讨论）; InterleavedThinkingToolset
  游标逻辑并入 monitor。

## Implementation Notes

### 落地顺序（每步独立可验收）

0. **`anthropic_cache_instructions=True`**（零依赖，一行改动）：`_meta.py:131-137`
   的 `AnthropicModelSettings` 加 `anthropic_cache_instructions=True`。
   system instruction（CTML + project + mode + static + soul）首次请求后
   缓存 5 分钟，后续每轮静态部分 90% off。不依赖 A/B 修复、warm/hot 拆分、
   变更检测——当前 system instruction 已是纯静态（`moss_dynamic` 走 Moment
   perspective 通道，不污染 system prompt）。独立可验收：开前后各跑一轮，
   对比 `stream.response.usage.cache_write_tokens` / `cache_read_tokens`。
   后续 Phase 1-4 不会使此缓存失效——cold 内容不变，token 0 前缀不变。
1. **Atom A/B 修复**（最小可落地，不依赖其余）：单次渲染两段返回；
   `save_model_request` 存 B；`as_request_messages` 翻转为 durable-first。
   注意：**单修存储不改顺序会更糟**（抽中段导致 percepts 位移 + 存发不一致），
   两处必须成对落地。
2. **prompts.py warm/hot 拆分**：`ChannelMetaPrompter` 拆 `make_warm_block()`
   （description/instruction/interface/states/failure）与 `make_hot_block()`
   （context）；渲染确定性三颗地雷排除（D2）。
3. **变更检测 + 防抖机**：会话级 `{channel_path: hash}`，三态机（D3）。
   hash map 是纯内存态，崩溃丢失退化为一次全量重绘，无需持久化。
4. **usage 监控 + 重绘**：依赖 memento FORMAT v2 冻结与 ghost runtime 集成
   （memento §19.4：真正的集成位置是 ghost runtime + API）。
   **已具备**：每轮 `ModelResponse.usage` 提供 `cache_write_tokens` /
   `cache_read_tokens` / `input_tokens`，可直接计算缓存命中率与上下文
   饱和度。监控逻辑建议挂在 `on_articulate_exit` 或独立的 usage hook。
5. **`anthropic_cache=True` 叠加**（Phase 1-4 完成后）：在 Phase 0 的
   instruction 缓存基础上叠加 automatic caching，让 history 前缀也享受
   90% 折扣。前提是 A/B 投影已落地（否则 B 投影与前缀不一致导致静默 miss）。
   与 `anthropic_cache_instructions` 组合时共享 4 个 cache point 预算。

### 与相邻 workstream 的边界

- **momento-mori**：本 workstream 是其 ghost runtime 侧第一个消费者。
  消费面：commit 锚点、窗口渲染、CommitNote 释义、机械 commit。
  不动 memento 契约层。
- **memento-cli-and-agent**：无重叠；其 agent 若跑长会话，是本设计的验证场。

### 待实验参数

- 防抖 M（回归稳定轮数）、demote 阈值（连续变更轮数）。
- 重绘阈值 80% 与三层 N/M/m 的取值。
- warm-delta 走 percepts source 的具体 source 命名与消息包装。

### 讨论轨迹

2026-07-30 一场长讨论（人类 + claude-fable-5 via claude code）推完整条链：
缓存机制事实核对 → 四方案算账 → cold/warm/hot 分层 → Atom 存储 bug 定位 →
A/B 前缀不变式 → memento 对齐与 commit 粒度纠错。市场事实基准：Anthropic/
DeepSeek 读价 ~10%，OpenAI 25–50%，写价加成 Anthropic 1.25×/2×（5min/1h TTL）。

2026-07-31 补充（人类 + claude-sonnet-4.6 via claude code）：
pydantic_ai 2.5.0 缓存能力全量调研。发现 AnthropicModelSettings 已提供
`anthropic_cache_instructions` / `anthropic_cache` / `anthropic_cache_messages`
三个注入字段，且 `UsageBase.cache_write_tokens` / `cache_read_tokens` 已通过
`_map_usage()` → genai-prices 从 Anthropic API response 自动提取。MOSS 当前
未读取这些字段。新增 D7（缓存注入与计量承载）+ Phase 0（一行启用 instruction
缓存）+ Phase 5（automatic caching 叠加，需 A/B 投影前置）。与 channel-meta-dyn-static
的 LSM 框架对对齐：Phase 0 先缓存 cold，LSM warm-delta 落地后 cold 进一步
瘦身，token 0 前缀永不失效。

2026-08-04 收敛（人类 + deepseek-v4-flash via claude code）：将 interleaved thinking
的游标机制泛化为 ContextMonitor（详见 V2 节）。要点：warm 用帧级 compare-and-emit
（漏斗出口对比，非事件订阅）；分层基准修正为"定义 vs 状态"（failure 从 warm 移 hot，
hot 两来源 = 声明式状态 + 检测式降级）；hot→warm 只有降级复位一条真状态机，卡热是
安全终态；ACK 纪律（基线只在确认入史后推进，接线归 GhostRuntime）；历史锚点统一为
状态转换事件；发射整块、单元字段组、command 级只做检测；独立模块先验证再装线。
interleaved thinking 验证讯息：interface 在历史中不在最新位置，可以跑。唯一悬而未决：
available 处理、锚点防抖、降级阈值——均留 dogfood 校准。

2026-08-05 重构（人类 + deepseek-v4-flash via claude code）: 引入 `ContextSnapshot`
冻帧作为 ack 显式令牌, `snapshot(metas) -> ContextSnapshot` 替代 `update()`,
`ack(snapshot)` 替代无参 `ack()`。hot 层渲染并入快照。明确三类原语统一于
ContextMonitor（InterleavedThinkingToolset 标准化超集）——静态透传 / warm+hot
治理 / interpreter 游标。已落码 33 tests 全绿。待补 gap: demoted 单元当前版
应进 hot_messages。InterleavedThinkingToolset 游标逻辑并入路径已对齐。
