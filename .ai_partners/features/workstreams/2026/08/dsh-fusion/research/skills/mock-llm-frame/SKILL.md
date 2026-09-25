---
name: mock-llm-frame
description: >-
  验证「mock LlmAdapter 确定性产出 tool-use」的机制: 一个 ego 专属 mock LLM provider,
  agent/request 路由切到它, 模型"产出" frame tool-call, agent-loop 原生暂停 dispatch,
  plugin 的 frame tool Consumer resolve 帧, 下一 step 切回真模型。Dolores thinking
  transaction 的候选形态之一 (vs perStep enter-with-messages)。
---

# Mock LLM 产出 Tool-Use 机制验证

本 skill 验证 Dolores thinking transaction 设计讨论 (2026-08-27) 的一个候选形态:
**用 mock LlmAdapter 让"模型产出" frame tool-call**, 替代合成 session 事件 / perStep 注入。

## 命题 (要断言的机制)

| # | 断言 | 源码锚点 |
|---|------|---------|
| ① | mock `stream()` 产 tool-call-delta → BlockAssembler delta-only 宽容 → `executeToolCalls` 可调度 | `llm/llm/src/assembler.ts:131-133` (open block 从累计 delta 组装); `agent-loop/src/tool-calls.ts:263` (appendToolCall) |
| ② | frame tool resolve → `tool/result` 落 session (surfaceOp append + sourceEventSeqs=[callSeq]) | `agent-loop/src/tool-calls.ts:276-289` (appendToolResult) |
| ③ | `agent/request` 路由切换: 同 turn step1 provider=moss-frame, 之后 moss-real | `agent-loop/src/agent.ts:438-445` (agent/request waterfall → prepareCall) |

## 拓扑

```
plugin.ts (挂在 dsh web 进程, :3085)
  ├─ FrameMockAdapter(LlmAdapter 子类): moss-frame 产 moss_frame tool-call / moss-real 产文本
  ├─ ctx.llm.registerAdapter(['moss-frame','moss-real'], adapter)
  ├─ ctx.on('agent/request'): step===1 → moss-frame, 否则 moss-real (短路 next, 全 mock 驱动)
  ├─ ctx.tools.register(defineTool moss_frame): execute 返回占位帧
  └─ ctx.agents.create: 惰性建活 agent (agentOptions.provider=moss-real)
       RPC: GET /plugin-api/frame-log | POST /plugin-api/frame-trigger (steer 唤醒 turn)
verify.py (纯标准库):
  ├─ 建 home/workspace (agent cwd)
  ├─ 启 dsh → 轮询 frame-log 可用 → POST frame-trigger
  ├─ 轮询 frame-log 直到 turn/end → 断言 ①~⑤ → 退出码即判定
```

## 运行方式

```sh
cd .ai_partners/features/workstreams/2026/08/dsh-fusion/research/skills/mock-llm-frame
python3 verify.py
```

全量 PASS → 退出码 0; 任一 FAIL → 非零退出码并打印失败项。

## 判定标准 (verify.py 断言)

1. **tool/call moss_frame 出现** — mock 产出可被调度 (①)。
2. **tool/result 成对出现** — frame tool resolve 落 session (②)。
3. **request/header 里 moss-frame 在 moss-real 之前** — 路由切换真实发生 (③)。
4. **turn/end completed** — turn 正常收线, 无死循环/无 abort。
5. **切回后产文本 assistant/message** — moss-real step 真正跑了 (续走成立)。

## 文件

| 文件 | 作用 |
| --- | --- |
| `verify.py` | 建 cwd → 启 dsh → 触发 → 轮询 → 断言, 退出码即判定 |
| `home/profiles/web/plugin.ts` | mock adapter + agent/request 路由 + frame tool + agent 创建 + RPC |
| `home/profiles/web/cordis.patch.yml` | 挂 `./plugin.ts` |
| `home/profiles/web/{cordis.yml,package.json,pnpm-workspace.yaml}` | web profile 骨架 |

## 踩坑记录 (预期)

- **零真实 LLM** 是全 mock 的关键: `agent/request` listener **短路 `next()`** 直接返回
  mock provider, 不依赖默认 provider/model, 也不触达任何真实 adapter。
- BlockAssembler **delta-only 宽容**: 只发 `tool-call-delta` + `finish` 即可组装出
  tool-call block, 不需要 `block-start`/`block-end` (open block 从累计 delta 组装)。
- `tool/result` 的 `sourceEventSeqs` 必须引用对应的 `tool/call` seq — 这是 agent-loop
  `appendToolResult` 内部强制的 (成对规则), 不是 plugin 自己拼的。
- agent cwd (`home/workspace`) 必须先存在 — session 边界会验证绝对 cwd。verify.py 启动前创建。
- 若 agent/request 短路导致其它 bundle listener 被跳过, 只影响本 skill 自己的 turn, 无碍断言。
- plugin.ts 是 ESM, `package.json` 已有 `type: module`。
