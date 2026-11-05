# 01 — 一次请求，而不是一个 session

> **更正（同日）**：本篇结尾那段"易变信息"的担心是错的。dsh 把渲染结果写成事件
> （system prompt → `system/message`，tools/config → `request/header`），前缀稳定是构造出来的，
> 不是靠运气。另，"复用轨迹上某个节点"本篇没有回答，见 `02-reuse-a-node.md`。

## 动机

并行判断如果每次都开 session，产出会按判断数量线性增长（文件、日志、注册表）。
想要的是：**一段稳定的上下文 + 每次判断一条尾巴**，前缀逐字节不变，命中 provider 的前缀缓存。

## 路径（可重放）

```sh
cd .vendors/dsh-research/v0.2.2-alpha.2+ce7073d/source

grep -n "hand-built one-shot" packages/llm/llm/src/types.ts
grep -n -A8 "export interface RequestUserInput" packages/llm/llm/src/types.ts
grep -rn "RequestUserInput" --include=*.ts packages | grep -v tests | grep -v types.ts
grep -n "in-memory store" packages/core/session/src/index.ts
grep -rln "session-persistence-jsonl" --include=*.ts --include=*.yaml apps packages
```

## 结论

- `[读码]` `GenerateOptions`（`packages/llm/llm/src/types.ts:564`）在类型注释里直接承认
  hand-built one-shot：`system` 字段写着 *System prompt text for one-shot callers*，
  `messages` 写着 *a hand-built one-shot may include identity-free user inputs*。
- `[读码]` `RequestMessage = Message | RequestUserInput`（`types.ts:539`）。`RequestUserInput`
  （`types.ts:530-536`）是**没有 session 身份**的 user 输入：`id?: never`、`source?: never`。
  它只能活在一次请求里，进不了历史。
- `[读码]` 树里已有三处把"会话当上下文 + 一次性提问"当正规用法，各自用
  `LlmRequestPurposeMap` 声明自己的 purpose：
  - `packages/session/session-title-llm/src/index.ts:10`（起标题）
  - `packages/compaction/compaction-basic/src/types.ts:8`（压缩）
  - `packages/experimental/translator/src/deepseek.ts:176`（翻译）
  这条路不是旁门，是 dsh 自己预留的扩展点。
- `[读码]` session store 本身就是内存的（`packages/core/session/src/index.ts:1058`，
  *In-memory session store*）。落盘是**独立插件** `session-persistence-jsonl`；
  不挂它，一个文件都不会产生。
- `[读码]` 有"未发布 session"这一档：`SessionStore.prepare()` 接受 `RestoredSessionOptions`
  （`seed` + `meta` + `inheritedEventCount`，`types.ts:177-186`），
  `SessionPreparation`（`packages/core/session/src/preparation.ts`）管它的生命周期，用完 release。
  **`SessionSeed` 属于恢复/分叉，不属于单次请求**——两者的消费者是
  `session-persistence/handle.ts` 与 `session-query/cold-read.ts`。
- `[推断]` 因此目标是可达的：`一次判断 = 稳定前缀（durable history） + 一条 RequestUserInput 尾巴`。
  尾巴不进历史 ⇒ 下一次前缀不变 ⇒ 可以持续命中。
- `[推断]` 未验证的部分，是下一篇的题目：从 session events 派生 history 的算法是否确定、
  system prompt 里有没有每轮重渲染的易变段、工具声明顺序是否稳定。

## 观点

它把"n 个 session"换成"**1 个 session + n 个请求**"：省的不是上下文，是前缀。

代价全在"逐字节相同"这一条上。**唯一可信的验法是 `prompt_cache_hit_tokens`**
（DeepSeek usage 字段；dsh 有 `packages/llm/token-meter` 在测它），读码读不出结论。

失效条件：派生 history 含时间戳 / cwd / session id / git 状态，或工具声明顺序不稳定。
任一条成立，"无代价"就退化成"每次都全价"。
