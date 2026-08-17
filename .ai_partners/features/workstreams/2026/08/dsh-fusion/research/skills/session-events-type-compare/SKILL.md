---
name: session-events-type-compare
description: >-
  长期可验证哨兵: TS 侧构造 session event 各类型 mock 实例 (ground truth),
  Python 侧喂给 ghoshell_moss.agents.deepseek_harness.types.session_events 的强类型模型,
  验证类型转换正确且与 TS 类型保持同步。dsh 类型漂移时能捕获差异。
---

# TS ↔ Python Session Event 类型数据比较

本 skill 验证 dsh 融合基建的**类型映射正确性**: dsh 的 13 种 session event 已封装成
Python 强类型模型(`src/ghoshell_moss/agents/deepseek_harness/types/session_events.py`),
本 skill 在 TS 侧构造各类型的 mock 实例作为 ground truth, 让 Python 模型逐条转换,
保证两边类型保持同步。

dsh 是开发者预览(协议漂移, 破坏性变更), 本 skill 是长期哨兵: 类型变更时 `verify.py`
应能捕获差异。

## 拓扑

```
plugin.ts (挂在 dsh web 进程, :3084)
  └─ 注册 HTTP rpc GET /plugin-api/session-events-mock
       → 返回 TS 构造的 19 条 mock session events (13 种类型 + 嵌套变体)
verify.py (MOSS repo venv)
  ├─ 启动 dsh web
  ├─ 轮询 rpc 直到可用
  ├─ 拉取 mock → SessionEvent.from_dict → 按 type 分发到具体 SessionEventModel
  └─ 断言: 分发正确类 / round-trip to_dict()==原 mock / seq·time·type 借道 / 不错配
```

## 运行方式

```sh
# 需要 MOSS 仓库 venv (因为 import ghoshell_moss)
cd .ai_partners/features/workstreams/2026/08/dsh-fusion/research/skills/session-events-type-compare
<repo>/.venv/bin/python verify.py
```

全量 PASS → 退出码 0; 任一 FAIL → 非零退出码并打印差异。

## 判定标准 (每条 mock 事件)

1. **分发正确**: `from_session_event` 返回正确的具体类实例。
2. **round-trip**: `to_dict()` 与原始 mock 逐字段相等。
3. **信封借道**: `seq` / `time` / `type` 从 `meta` 读出, 不复制。
4. **不错配**: 其它类型的 `from_session_event` 对该事件返回 `None`。

## Mock 覆盖

全部 13 种事件类型 + 嵌套判别联合的多样变体:
- turn/start, turn/end(completed + aborted/user), step/start, step/end
- user/message: source=user / plugin(form=notice) / image 块 / tool-result 块
- assistant/chunk: text-delta / finish / block-end(tool-call 块)
- assistant/message(message=model source + usage)、tool/call、tool/result(message + error + meta)
- todo/write、request/header(完整 EpochHeader)、request/context、session/end-seed
- 3 个 surface 事件带 `surfaceOp:'append'` + `sourceEventSeqs`

## 文件

| 文件 | 作用 |
| --- | --- |
| `verify.py` | 启 dsh → 拉 mock → 全量类型转换校验, 退出码即判定 |
| `home/profiles/web/plugin.ts` | 注册 rpc, 构造 TS mock ground truth |
| `home/profiles/web/cordis.patch.yml` | 挂 `./plugin.ts` |
| `home/profiles/web/{cordis.yml,package.json,pnpm-workspace.yaml}` | web profile 骨架 |

## 踩坑记录

- plugin 只用 `ctx.webServer`(注入), 零第三方依赖。`import type` 被 Node 24
  strip-types 擦除, 无运行时依赖。
- dsh 信封是**扁平** dict `{type,seq,time,data,ignorable?,surfaceOp?,sourceEventSeqs?}`,
  没有顶层 `meta` 字段; Python 侧 `SessionEventMeta` 是重组。`tool/result` 的 `data.meta`
  (工具私有载荷)与信封 `meta` 是两回事, Python 模型里改名 `tool_meta` 处理。
- verify.py 依赖 ghoshell_moss, 必须用 MOSS 仓库 venv 跑, 不是裸 `python3`。
