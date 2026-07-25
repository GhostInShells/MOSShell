---
title: Channel Meta Dyn/Static — 动静分离、主动推送与 Channel Memory
status: design-locked
priority: P2
created: 2026-07-26
updated: 2026-07-26
depends: []
milestone:
description: >-
  ChannelMeta 动/静信息分离 + 主动推送协议 + builder memory_messages 挂载点。
  设计已锁定（本文件即合同），实现待排期。核心动机是 interleaved thinking
  时代的上下文缓存经济学。
---

# Channel Meta Dyn/Static

> Use `moss features set-status channel-meta-dyn-static <status> -m "note"` to update state.

## Motivation

三个长期悬置的命题在 2026-07 的 channel_builder 设计审计讨论中收敛为同一个 workstream：

1. **动/静 meta 信息分离** — ChannelMeta 当前每次 refresh 全量重建、全量传输。静态部分
   （interface 签名、description、instruction）变更低频，动态部分（context、failure、
   available）每关键帧变。不分离意味着传输浪费 + 上下文缓存无法利用。
2. **主动推送模式** — 当前 proxy 只能拉。provider 侧 channel 变更（尤其 virtual channel
   上下线）无法主动上行，导致 proxy 侧要么高频轮询要么信息陈旧。
3. **Channel Memory** — ChannelMeta.memory 字段已存在，但 Builder 没有挂载点
   （MessageFunction docstring 承诺了 "Memory Messages"，是有意保留的自解释入口）。

**为什么是 P2 而非 P3**：interleaved thinking 落地后（见 interleaved-ctml-thinking
workstream），模型关键帧频率上升一个数量级，prompt cache 命中率直接决定成本与延迟。
virtual channel 全量挂 context messages 的现状是每轮全价重付。

**为什么现在才敢锁定设计**：此前"没有最佳实践"的堵点（static 重绘时机）在讨论中
找到了确定性答案（见 Key Decisions #7 LSM 框架）。

## 哲学定位：Channel Memory 是躯体的记忆

Ghost 有 ghost-owned memory。Channel memory 是另一个东西：**躯体的记忆**。
Ghost in Shells 架构里模型可以换 shell——要装上某个 shell 才会回忆起一些东西，
人也是如此。典型案例：channel 连接机体时，memory 里直接携带自拍照。这类内容
必须直接在上下文里（图片无法通过指针让模型自取）、又不能每轮重发进对话历史。

Channel 是极大的授权，memory 语义扩大了授权面，因此此前一直慎重。轻量替代
（instruction 里提示资源位置，走渐进披露）覆盖了大部分场景；memory 挂载点只
服务"必须在上下文内 + 不可指针化"的窄需求。

## Key Decisions

以下六点为协议级答案（人类架构师 2026-07-25 锁定），加第 7 点上下文经济学框架
（讨论中共同推导）：

### 1. 动静分离是 dump 逻辑，不是字段标记

ChannelMeta 保持单模型。static 不标到字段上，用 dump 函数做 code as prompt——
ChannelMeta 是序列化传输协议，method 可解释约定。全字段容错已就位
（`py_channel.py` 有完整示范）。

### 2. 拉机制：refresh 链路穿 `full` flag

静态协议用拉机制，拉时传参。MOSShell 抽象层所有 refresh 增加 `full: bool`。
不做 version/hash 对比。proxy 侧 `cached_meta.merge(new_meta)`，merge 函数
自解释逐字段规则。核心实现位置：`ghoshell_moss/core/runtime/tree.py`
（`ChannelRuntimeNode._refresh` / `refresh_own_metas` 是穿 flag 的自然位置）。

### 3. 推拉均走 Duplex 协议

传输层无关。协议事件已预铺（**不存在迁移，设计之初已铺好所有路径**）：

- `ChannelMetaUpdateEvent` 已带 `all: bool` 字段（partial 更新预留）
- `SyncChannelMetasEvent` 已是 proxy→provider 拉取事件，只差加 `full` 参数
- 见 `ghoshell_moss/core/duplex/protocol.py`

### 4. 注册面确定为 `memory_messages`

Builder 增加 `memory_messages(func: MessageFunction)` 钩子，填充
`ChannelMeta.memory`。与 `context_messages` 对称：memory 归静态侧
（对话之前、持久召回），context 归动态侧（inputs 之前、新鲜感知）。

### 5. Runtime 暴露 `on_proactive_refreshed` 回调协议

`ChannelRuntime` 暴露 `on_proactive_refreshed(Callable[[ChannelMeta], None])`，
tree 注册、更新本地缓存。Duplex 实现：channel provider tree 暴露全局 `on_...`
接口，provider 监听，构建推送协议上行。`dynamic=False`（flag 语义翻新）表示
拒绝被主动刷新——该 channel 的 meta 永远只住 base layer，不产生 delta。

### 6. 无迁移

现存 proxy/provider 不需要兼容分支。`full` 默认值取现状全量语义。

### 7. 上下文经济学：LSM 分层框架（本 workstream 的设计依据）

**缓存锚点结构**（interleaved thinking 交互模式下）：

1. instruction 后
2. memory 区
3. 会话历史折叠位置
4. 会话最新点（thinking/tool-use 的 1+n 轮从此开始，纯后缀 append，天然全命中）

**上下文变更节奏四分**：S(tatic) / C(onversation) / I(nput 回合) / T(ool Use)。
T 节奏全命中与布局无关；布局选择只影响 I 节奏的税。

**四策略对比**（S=static 体积, C=历史体积, δ=单次变更增量, 前缀失效模型）：

| 策略 | 稳态成本/轮 | S 变更成本 | 备注 |
|---|---|---|---|
| 前置 + 立刻重绘 | ~0 | S + C 全部重入 | K* ≈ C/S + 1 轮 break-even |
| 尾置（每轮重定位） | S 全价/轮, 无条件 | 0 | C 每轮增长 ⇒ C 之后内容每轮失效 |
| 全量挂 context messages（现状） | 0 | S 全量 append, 膨胀 C | virtual channel 现状 |
| **LSM: 前置 + delta append + fold 合并** | ~0 | δ（append 一次即成缓存历史） | **锁定方案** |

**LSM 框架**：moss_static = base layer（前置，锚点 3 之前，跨 fold 缓存）；
变更增量以 override 语义 append 进对话流 = delta layer（CTML 已声明
"dynamic overrides static, only latest authoritative"，模型侧零新协议）；
历史折叠 = compaction，累积 δ 合并回 static 块，边际缓存成本为零
（fold 时锚点 3 之后本来就全部重入，S 块自身重写 ~1.25×S 是搭便车）。
补充触发：δ 累积超过阈值（S 的 20%~50%）时主动触发 fold。

**S 前置块的重绘频率被钳制为恰好等于 fold 频率**——"没有最佳实践"的
开放问题由此消解。

**节奏三分（渲染层治理）**：按变更节奏而非容器分层：

- 冷：static interface / instruction / memory → base layer
- 温：dynamic interface（状态迁移节奏）→ delta layer，fold 时晋升
- 热：perception context（关键帧节奏）→ 永远尾部，不参与缓存

数据模型无需改动：`CommandMeta.dynamic` 已是逐命令粒度，
`make_interfaces(dynamic=, sustain=)` 渲染已参数化
（`ghoshell_moss/core/ctml/v1_0/prompts.py`）。治理全部发生在渲染策略层。

**预算参考**：I(nstruction) 5% / S 10% / C 70% / T 15%。S 若膨胀到 30%+，
手段是渐进披露（折叠子树），不是换布局。

**边界**：fold 时机归 Ghost/Agent 工程（上下文构建者）。MOSShell 侧只保证
merge 语义正确 + 暴露"当前累积 delta 体积"信号。MOSShell 工作面因此封闭可测。

## 待 review 的微决策

实现前需人类确认（均非阻塞，模型倾向已标注）：

1. **`merge()` 逐字段规则** — 倾向：`commands` 按 name 覆盖；`context`/`memory`
   整体替换；`states`/`current_state` 整体替换。最易出隐性 bug 处，需逐字段单测。
2. **`dynamic` flag 语义翻新** — 原地重定义 docstring（"拒绝被主动刷新"）
   vs 换名（如 `proactive_refresh: bool`）。倾向：原地重定义，避免 wire 协议变更。
3. **memory 刷新节奏** — 倾向：默认归静态侧、按需刷新（自拍照案例支持）。
4. **`SyncChannelMetasEvent.full` 默认值** — 倾向：默认 True = 现状全量，
   新行为显式 opt-in。

## Implementation Notes

- 实现面全部落在已有骨架关节处，无新抽象：`ChannelMeta.merge()` +
  refresh 链路穿 `full` + `on_proactive_refreshed` 回调 + provider 监听上行 +
  `memory_messages` 钩子。
- **`ChannelMeta.merge()` 一函数两用**：proxy 侧动静合并缓存 = compaction 时
  delta 合并回 static。传输层与上下文层在此汇合，这是两个命题同属一个
  workstream 的原因。
- `py_channel.py` builder state 已持有 `_on_refresh_meta_funcs`，注册面模式现成。
- 实现纪律：此工作"不难但不能错"，channel tree 级改造 + 跨进程语义，
  单测必须覆盖 proxy/provider round-trip 与 merge 逐字段规则。
  适合 worktree 隔离实现 + 人类一致性 review（对照本文件）。
- 关联小修：channel_builder.py 的模型面文档优化与本设计同 commit 落地
  （instruction 红线、perspective_messages 幽灵引用、MessageFunction 注记）。
