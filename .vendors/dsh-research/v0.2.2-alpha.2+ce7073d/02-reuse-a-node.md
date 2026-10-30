# 02 — 从轨迹上任意一个节点并行发请求

## 动机

要 N 个并行判断，共用 session 轨迹上某个节点的前缀，且都不进入历史。
01 只回答了"一次请求可以没有 session 身份"，没回答"怎么复用某个节点"。

## 路径（可重放）

```sh
cd .vendors/dsh-research/v0.2.2-alpha.2+ce7073d/source

sed -n '1,45p'  packages/core/session/src/fork.ts
sed -n '1,50p'  packages/core/session/src/request-header.ts
sed -n '1,80p'  packages/core/agent-loop/src/runtime-context.ts
grep -n "system/message" packages/core/agent-loop/src/agent.ts
grep -n -A10 "export interface RestoredSessionOptions" packages/core/session/src/types.ts
```

## 结论

- `[读码]` **节点 = fork seed。** `buildForkSeed(events, boundary)` 取含 `boundary` 的前缀，
  追加 `session/end-seed{inherited:true}`，再用 `openTurnClosers(prefix, {kind:'forked'})`
  把开着的 step / turn 闭合。返回的数组**保留源事件对象**，只是多出合成 closer。
  （`packages/core/session/src/fork.ts:19-40`）
- `[读码]` **未发布 session。** `RestoredSessionOptions = {seed, meta, inheritedEventCount, eventState}`
  （`packages/core/session/src/types.ts:177-186`）；`SessionStore.prepare(id, options)` 用它造
  session，`SessionPreparation` 管生命周期，用完 release，不进注册表、不写盘。
  （`packages/core/session/src/preparation.ts`）
- `[读码]` **请求的前置不是每轮重渲染，而是已落事件。**
  system prompt → `system/message` 事件（`packages/core/agent-loop/src/agent.ts:460`），
  只在文本真的变了才替换（`runtime-context.ts:58-71`，*surface node 0*）；
  tools / config → `request/header` 事件，用 `headerEquals` 去重（`request-header.ts:26-45`）。
  任何拿到日志的人都能重建"某个请求是在哪个 header 下建的"。
- `[读码]` **尾巴没有身份。** `RequestUserInput`（`types.ts:530-536`）：`id?: never`、`source?: never`，
  只能活在一次请求里，进不了历史。
- `[推断]` 三件拼起来即形状：`fork(log, k)` → 未发布 session → 加一条 `RequestUserInput` →
  `prepareCall().stream()`。N 个并行 = N 个这样的批次；前缀是同一段不可变事件，尾巴各自不同，
  **源轨迹一个字节都不动**。
- `[推断]` 并发安全的前提是 fork seed 的快照语义（源事件对象被保留、session 构造时快照）。

## 观点

**节点 = fork seed；尾巴 = request-only；并发 = 未发布 session。** 三件都是现成的，不用新造。

唯一要实测的，不是"前缀会不会变"，而是**fork 的合成 closer 会不会改变消息序列**：
`session/end-seed` 与 forked 的 turn/step 结束如果被投影成 provider 看得见的消息
（比如补一条 assistant 或 tool 结果），那"节点 k"的前缀就不再等于原轨迹在 k 处的请求。

这一点读码定不了，只有一次实测能定：用同一前缀发两条不同尾巴，看第二条的
`prompt_cache_hit_tokens` 是否 ≈ 前缀长度。命中 → 前缀成立；不命中 → 换一条不补 closer 的截取路径。
