---
name: anchor-call
description: 无副作用的单次调用 — 从一个 session 已发生的位置发起一次请求，继承它看到的一切，不写回任何东西
---

# Anchor Call

> 这是 `dolores-concurrent-mind` 的一个子命题：**旁路请求完全继承 session，但不留存储文件**。
> 严格说它是 dsh 的能力缺口，本该进 dsh tracker；那边正在大改造，先不插入，记在这里。
>
> 本文件按 `Question → Context → Found → Answer → Verification` 写：问题是索引，`Found` 是
> 可重放的到达路径，`Verification` 是可跑的确认。**别人不需要重查。**

## Question — 索引

| # | 问题 | 答案位置 |
|---|---|---|
| Q1 | dsh 有没有"不入库"的单次请求？ | §Found F1/F2：LLM 层有，Agent 层没有 |
| Q2 | 怎么从一次已有对话里正好切到某一点？ | F3：`snapshotEvents` + `buildForkSeed`；活会话可直接 `deriveMessages()` |
| Q3 | 怎么能"继承主路看到的全部"而不自己拼前缀？ | F4：`Session.deriveMessages()`，与主路同一段代码 |
| Q4 | 传 `system` 会怎样？ | F5：适配器把它和历史 system 拼接 → 前缀缓存作废 |
| Q5 | 能改 provider / model / tools 吗？ | F6 |
| Q6 | 有没有非 tool 的结构化输出协议？ | F7：没有 |
| Q7 | usage 怎么记？不写日志会被记到哪？ | F8：哪里都不记；账本唯一入口是日志事件 |
| Q8 | 我们的现状有多糟？ | F9 |

## Context

- 版本：**dsh 0.2.0-rc.2**（装着的）。证据都取自 `~/.dsh/profiles/node_modules/@deepseek-ai/*/lib/*.js`
  ——编译后的源码，注释完整。`.vendors/dsh-research/v0.2.2-alpha.2+ce7073d/source` 是更新的 TS 树，
  只用来对照"dsh 自己要往哪走"。
- 动机：Dolores 每次 commit 要写一条 note，这条 note 需要一个"站在主路同一位置往外看一眼"的请求。
  现在的做法是**新开一个 dsh session**，于是每次 note 留一个目录：实测 `571` 个 bypass session /
  `323MB`，主路只有 `3` 个。
- 目标不是"更省 token"，是**一次这样的请求不应该在世界上留下痕迹**。

## Found

**F1 — LLM 层完全无 session。**
`GenerateOptions`（`dsh-llm/lib/types/types.d.ts:489`）里 `sessionId` 是**可选**的；注释写明
*System prompt text for one-shot callers*、*a hand-built one-shot may include identity-free
user inputs*，后者指 `RequestUserInput`（同文件 `:471`）：`id?: never`、`source?: never`，
只能活在一次请求里，进不了历史。

**F2 — Agent 层没有"不入库"这一档。**
`CreateAgentOptions`（`dsh-agent/lib/types/index.d.ts:48`）没有 persist / ephemeral 任何字段；
`AgentFactory.createAgent` 的文档写死：*inserts both session and agent, announces session
creation*。持久化插件 `dsh-session-persistence-jsonl` 由 base bundle **无条件**挂载
（`dsh-base/cordis.patch.yml:130`），且没有跳过条件。
**但** `SessionStore.prepare(options)`（`dsh-session/lib/types/index.d.ts:390`）明确是
*Build a session WITHOUT entering it into the store*，发布钩子是在 `enter()` 里装的
（同文件 `:413`）。→ 不 `enter` 就不写盘，这是结构，不是配置。

**F3 — 切点。**
`buildForkSeed(events, boundary)` 从 `@deepseek-ai/dsh-session/fork` 导出（`exports` 里有
`./fork`）；`events` 由 `Session.snapshotEvents(fromSeq?, toSeqExclusive?)` 给。种子会追加一个
`session/end-seed` 标记，并在尾部还开着时补合成闭合器（`repair.js` 的 `openTurnClosers`）。
切在 `turn/end` 的 seq 上则不需要闭合器。

**F4 — 继承不需要我们拼前缀。**
每轮请求的组装点是 `dsh-agent-loop/lib/index.js:1262-1277`：

```js
const boundaryMessages = session.deriveMessages();
return markAgentLoopRequest(Object.freeze({
  ...header.config,
  messages: boundaryMessages,
  toolHistory: session.toolHistory(),
  ...header.tools !== void 0 ? { tools: header.tools } : {},
  sessionId: this.session.id,
  signal
}));
```

`Session.deriveMessages()`（`dsh-session/lib/types/index.d.ts:303`）是纯读、有缓存、返回
provider 看到的那份历史；规则是"按 `surfaceOp` 标记走"，所以没有标记的事件（chunk、
turn 边界、`session/end-seed`、合成闭合器）**不进历史**。system prompt 是历史里的一条
`system/message`，所以它也在里面。

**F5 — 传 `system` 会拼在历史 system 前面。**
`dsh-llm-deepseek/lib/index.js:1701`：`const system = [options.system, historySystem].filter(Boolean).join("\n\n")`。
→ 继承前缀的路径必须**不传** `system`。`EpochHeader.system` 的类型就是 `never`，注释写
*Retired request text; system prompts belong to system/message events*。

**F6 — provider / model / tools 都能改。**
`...header.config` 后覆盖再 `ctx.llm.prepareCall(newConfig)` 即可。改 provider/model 前缀缓存
必然不命中（换模型本来也不该复用）。`tools` 是我们传的数组，但**历史里已有 tool/call +
tool/result 对**，声明里没有对应名字时部分 provider 会拒 → 默认原样透传或只增不减。

**F7 — 没有结构化输出协议。**
`GenerateOptions` 无 `toolChoice` / `responseFormat` / `json_schema`。`purpose` 在 0.2.0 是写死的
`'compaction' | 'session-title'`；在 0.2.2-alpha.2 里变成 `interface LlmRequestPurposeMap {}`
——空接口，留给声明合并，即 dsh 正在把它变成公开扩展点。结构化只能：**prompt 要求 + 本地校验
+ 重试**。

**F8 — token 账本只有一个入口。**
`assistant/message` 事件上的 `usage` 字段（`dsh-agent-loop/lib/index.js:1148`）。`ctx.tokenMeter`
只有 `measure(session, requestHeader?)`，折叠的是**这个 session 的事件**；
`dsh-session-stats` 同样读日志。`llm/stream` 是 waterfall 事件但在 0.2.0 全树**零订阅者**
——即没有任何"provider 调用级"的全局记账。
→ 我们这种不写日志的请求，**一分都不会被记**；要记账只能调用方自己订阅 `llm/stream` 截
`usage`。`TokenUsage` 字段：`inputTokens` / `outputTokens` / `totalTokens?` /
`cacheReadTokens?`（= DeepSeek 的 `prompt_cache_hit_tokens`）/ `cacheWriteTokens?` /
`reasoningTokens?`。**没有地方持久化它**，只在流里出现一次。

**F9 — 现状账。**
`DSH_HOME/sessions/` 下 578 个目录 / 323MB：`session-*` 前缀 3 个（真主路），纯 UUID 571 个
（bypass 留下的）；散在 3 个按 cwd 分键的目录里。另有 `storages/session_projcache/`、
`storages/workspace.json`、`session-query-sqlite` 索引会跟着变脏。

## Answer — 协议

**Anchor 是坐标，不是快照，也不可读。** 任何"预览"都会退化成上下文搬运，而它的全部价值就在于
它不是内容。

```
Anchor      { session_id: str, seq: int }          # 切点含端：events[0..seq]
AnchorCall  { anchor, message, strategy? }         # strategy: provider/model/reasoningEffort/
                                                   #           maxTokens/temperature/stop
CallResult  { text? | value?, usage, finish, retries }
```

四条纪律：

1. **输入可以给 turn，出口一律 seq。** 长期存的坐标（`deepseek_harness/types/refs.py` 的
   `DshSessionRef`）以 turn 为主坐标、seq 为派生加速器，那是"跨天引用"的需求；anchor call 是
   "立刻用"，精确优先。两者并存，不合并成一个类型。
2. **不传 `system`**，system 已在历史里（F5）。
3. **默认不写任何东西**：不 append、不 `enter`、不建 session、不落记录。记录是**调用方的事**，
   所以协议必须把 `usage` 完整交出去（F8）。
4. **结构化调用**（可选）：声明一个 tool schema，要求模型"只通过调用该工具作答"，解析时剥离
   其 arguments 作为 `value`；没拿到 tool 调用就重试。**不存在强制**，所以"必须"只能是要求 +
   校验 + 重试（F7）。

实现的两种 anchor：

- **活会话**：`ctx.sessions.get(id)` → `deriveMessages()` → `[...hist, tail]`。零对象构造。
- **历史位置**：`snapshotEvents()` → `buildForkSeed(events, seq)` → `sessions.prepare({seed,
  inheritedEventCount, meta})`（不 `enter`）→ 同上。

## 不属于我们 — 上游缺口（现在不提 issue）

dsh 缺的是"对外暴露的内核语义"，四项，按对我们的重要性：

1. 一条"在未发布 session 上跑一轮"的接缝（或 `CreateAgentOptions.publish: false`）。有它，
   anchor call 可以复用一个完整 agent（含 tool gate / prompt 组装），而不只是裸请求。
2. `purpose` 的扩展点（0.2.2 已有 `LlmRequestPurposeMap`）——我们的调用需要自己的分类，
   而不是冒充 `session-title`。
3. 一个 session-less 的 usage 上报口（现在只能调用方自己订阅 `llm/stream`）。
4. 可选：`toolChoice` / `responseFormat`。

## Verification

`.playground/anchor-call/`：一个独立 dsh 插件，只暴露一条 HTTP route；在默认 0.2.0 环境起 dsh，
打两次请求，看三个数：

1. 返回了文本；
2. `usage.cacheReadTokens` 接近主路前缀长度（证明前缀真的被继承、并且命中缓存）；
3. `DSH_HOME/sessions/` 目录数**不增**。

失败模式要显式报错，不静默降级（F1-F8 依赖的都不是"官方 API"，只有"公开导出"）。
