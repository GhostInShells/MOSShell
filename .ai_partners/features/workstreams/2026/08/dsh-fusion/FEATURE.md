---
created: 2026-08-14
depends:
- agent-surface
description: 'dsh 融合 — DeepSeek Harness (dsh) 作为 MOSS 核心推理组件的一体化集成。dsh 从「候选外部 agent
  面」升格为 MOSS 的推理中枢: MOSS 保留记忆 (Memento) / 执行 (CTML) / 感知 (audio/vision), dsh 承载 agent-loop
  推理。本 workstream 决策融合本身, 落点分两条路径: gui 管理的 agent 与 dolores ghost。'
milestone: 0.1.0
priority: P0
status: in-progress
status_note: 'agent loop 驱动与治理边界调研完成 (wake/perStep/interrupt); 融合基建落成于 deepseek_harness/ (launcher/client/session/message_mapper/types + pyproject dsh optional); Dolores Path B 的 articulate/enter 收敛为「参数面自设计的组合入口」(2026-08-27) — 具体方案在源码 docstring (dolores/_ego.py「transaction / RPC 旁路」+ plugin.ts articulate/enter), 读源码优先'
title: DSH Fusion
updated: '2026-08-27'
---

# DSH Fusion

> Use `moss features set-status dsh-fusion <status> -m "note"` to update state.
> 从 `dsh-productization`(候选取代 claude-code-in-moss / 开箱不实现)翻篇改型而来,
> 2026-08-16。旧定位与预判方向不在此保留, 见 [Legacy](#legacy)。

## Motivation

DeepSeek Harness(`deepseek-ai/deepseek-harness`, MIT, 2026-08-13 开发者预览)是一个
「Model + Harness = Agent」的 coding harness。其「更工程化的协议面」(Python SDK + 三套
进程外协议 + session 持久化 + 一切皆插件)使它能成为 MOSS 的**核心推理组件**, 而非
仅仅一个外部 agent 面。

融合的分工裁决(2026-08-15 收敛, 详见 research/ 轨迹):

- **DSH 做推理中枢 (认知代理), MOSS 做记忆/执行/感知。** Dolores 的 articulate 由 DSH
  的 agent-loop 驱动, MOSS 不再持有推理循环。
- **两套协议各归其位, 不强行统一。** JSON Schema 工具协议走 DSH, CTML 流式指令协议走
  MOSS。不兼容是刻意保留的分工。
- **dsh session = ghost 的思考锚点, Memento = 记忆权威。** 无限上下文与持久化是 MOSS 的
  专有命题, 不交给 dsh。

## 两条落点路径

融合本体已定, 落地走两条路径, 分别由独立 workstream 承载:

### 路径 A: gui 管理的 agent (独立 feature)

把 dsh 作为一个 **gui 可管理的 agent 实例** 暴露给 MOSS。基于已验证的 vendor 可行性:
dsh web 前端是近乎通用的 agent session 表面(只认 `ISession` + `SessionEvent`), 任何满足
contract 的 backend 都能挂它当界面。收敛形态 = matrix node 持有 session(owner 归 MOSS),
vendor 的 chat 界面作 GUI 子进程, 父进程 proxy 转发 event 流 + 8 verb。
源码级调研见 research/source/deepseek-harness/。

### 路径 B: dolores ghost (推理中枢)

dsh 作为 Dolores 的推理中枢, 决策落在 `ghost-prototype-dolores` FEATURE 的
DSH Integration 节。本 workstream 只提供融合基建, 不重复路径决策。
见 `ghost-prototype-dolores` workstream 的 DSH Integration 节与本 workstream 的
research/ 调研轨迹。

## 融合基建 (Scope)

本 workstream 交付 dsh 融合的**可复用基建**, 供两条路径消费。**基建已落成于
`src/ghoshell_moss/deepseek_harness/`**(2026-08-20/22, 见 git log)——落点从原计划
`agents/dsh/` 改到 `deepseek_harness/`, 自成一族协议客户端包, 官方 SDK 不用。

1. **optional 依赖** — pyproject `[project.optional-dependencies]` 的 `dsh = [...]`:
   `deepseek-harness-sdk` 仅作参考锚点(自研 client 不用)、`httpx`(outbound HTTP)、
   `websockets`(mux WS 下行)。
2. **Python 基建**(`deepseek_harness/`, 自定义协议客户端为主):
   - `launcher.py` — `DshLauncher`/`DshLauncherConfig`/`DshExit`: spawn dsh
     web-profile 子进程(经 MOSS Subprocesses 契约 DI), 连 web 表面。传输走
     dsh web profile + 内置 `/api/events.mux` WS 下行 + plugin 注册 HTTP 路由上行
     (零依赖伪双工), **不用 stdio JSON-RPC**。push 式就绪(ws 连上→started→aenter 返回)。
     子进程/RPC 异常收成 `DshExit` + `exception()`。
   - `client.py` — `DshClient`(全局管理面 facade)+ `DshRpcException`: 每方法 =
     一个 apiproxy 动词(session/workspace/host/agent-preset/settings/credentials/
     llm/skill/goal 只读+CRUD); `plugin_call(path)` 走 plugin webServer 路由。
   - `session.py` — `DshSession`(会话级 facade): 绑 sessionId, 屏蔽 rpc 入参对象,
     挂驱动动词(prompt/cancel/update-queue/select-model/history/fork/attachment)。
     `accept_frame` 喂帧(反转依赖, owner 注册), 按事件名分派到 `on_session_event*`;
     token 记账; `instruction()`/`surface_messages()` 经 plugin 路由 pull;
     `when_{running,idle}` 等运行态镜像。
   - `message_mapper.py` — MOSS Message → dsh UserMessage 单向映射(role=user;
     image 抛 NotImplementedError, 需 attachment ref 或走 session.prompt 提升)。
   - `types/` — 强类型 pydantic 数据面(rpc/nouns/events/sessions/domains/sdk,
     按依赖序拆防环 import; 信封热路径用 TypedDict, 载荷用 pydantic validate)。
3. **测试** — `tests/ghoshell_moss/deepseek_harness/`(test_launcher / test_session /
   test_message_mapper)。旧的 `dsh_web_probe` 测试 node 已不在 system_test_nodes。
4. **装线** — 路径 B 的 dolores 装线进行中: `_runtime.py` 用 `DshLauncher`(经
   `matrix.processes`), `_run.py` 用 `DshSession`, `topics.py` 消费 `SessionEvent`;
   `_ego.py`(`DoloresEgo`)目前只是表面草稿(方法体 `...`)。

> **强烈提示: 具体方案在源码 docstring, 不在本 FEATURE。** articulate/enter 收敛为
> 「参数面自设计的组合入口」(2026-08-27): enter 是 plugin 侧 `/articulate/enter`
> 自定义入口, 参数面由 MOSS 设计, handler 内部组合多个 dsh 接口调用(非 1:1 代理)。
> 读 `ghosts/dolores/_ego.py` 模块 docstring「transaction / RPC 旁路」与
> `dsh_plugin/moss-dolores-ghost-plugin.ts` 的 articulate/enter 注释。

## Key Decisions

- **dsh = 推理中枢, MOSS = 记忆/执行/感知。** 这是融合的第一裁决, 两路径共享。
  详见 Motivation。
- **官方 SDK 不是基建底座。** 源码级调研(Python 类型面极薄, 只懂 3 个 event type)与
  实验(早期 `DshChannel` 走 stdio JSON-RPC)一致——基建以自定义协议客户端为主, 官方
  SDK 仅作参考锚点。传输终版选**dsh web profile + `/api/events.mux` WS 下行 + plugin
  HTTP 上行**(零依赖伪双工), 弃 stdio JSON-RPC(见 `launcher.py` 注释)。
- **`deepseek_harness/` 落点, 自成一族(已改)。** 原计划 `agents/dsh/`(与 memento
  同族)未沿用——代码实际在 `src/ghoshell_moss/deepseek_harness/`。独立包 `ghoshell_dsh`
  保持被否: 会切断与 agent-surface 骨架、memento 参照的关系。
- **apiproxy 式 plugin 桥接内核特权 (2026-08-16 收敛)。** ghost 要够到 dsh 进程内
  特权(append assistant / 构造 seed / 动态 prompt), 唯一干净的路是仿 apiproxy 再写
  一个 plugin, `ctx.webServer.register` 注册 HTTP 路由, transport 复用 dsh 已有 HTTP
  面, 不引入 zenoh/zmq、不改内核。接口面待裁决, 要窄。
- **hot 归 MOSS, dsh 只做 cold+warm。** 高 churn 大块数据 (vision) 走 MOSS 旁路,
  不进 dsh session(DeepSeek text-only 拒图, 撞窗口压满/传输放大)。
- **激进 articulator 解耦策略未裁决。** 「dsh 退化为纯推理函数 `think(moment)->result`,
  状态全在 Memento」与既有「1:1 articulator:action」决策冲突, 采纳需显式 overturn。

## Shared Resources

> 融合相关的调研轨迹与验证物索引, 避免盲找。完整历史见 git log。

- **调研轨迹**(早期 2026-08-14/15/16 三篇结论被后续实验推翻, 已删, 见 git
  `git log -- research/`; 下列为当前存留的 research/ 文件):
  - `research_2026-08-17_dsh_agent_status_and_context_modeling.md` — agent status /
    context 建模
  - `research_2026-08-19_dsh_agent_loop_drive_and_governance.md` — agent loop 驱动与
    治理边界(wake/perStep/interrupt)
  - `research_2026-08-20_dsh_agent_api_surface_and_timing.md` — session 级 agent 两个
    调用面(http rpc 外侧 vs plugin 内侧)与各自时序语义
  - `research_2026-08-20_dsh_session_surface_and_message_protocol.md` — **纠错**: dsh
    session 不是 append-only 上下文, 是 **log(append-only 真相源, 永不删) + surface
    (可 replace 的模型可见投影) 两层**。compact 用 surface `replace` 影藏旧节点, log
    只增不变小——"compact 只能追加"不成立。**"hot 归 MOSS / dsh 只做 cold+warm"须基于
    此 surface 语义校准。**
- **当前可信 skill**(`research/skills/` 下, 自包含可复跑):
  - `plugin-api-session-event/` — 已验证:「dsh web 内置 `/api/events.mux` WS 下行 +
    plugin 注册 HTTP 回调」构成零依赖伪双工, ghost runtime 不开对外接口
- **基建源码锚点**: `src/ghoshell_moss/deepseek_harness/`(launcher/client/session/
  message_mapper/types)+ `tests/ghoshell_moss/deepseek_harness/`。plugin 面:
  `ghosts/dolores/dsh_plugin/moss-dolores-ghost-plugin.ts`。dsh 官方源码在
  `research/source/deepseek-harness/`(`python/sdk` 仅参考锚点、`packages/acp` 权限仲裁、
  `packages/client/runtime` vendor 面、`dsh-host-apiproxy` 桥接范本)。

## Legacy

旧 `dsh-productization`(2026-08-14)定位: 候选取代 `claude-code-in-moss`、开箱不实现、
把 dsh 的 mode/工具/session/权限面原样搬到 MOSS。该定位已翻篇(2026-08-16): dsh 从
外部 agent 面升格为核心推理组件。预判方向(集成路径/融合点/定位边界)与源码级调研细节
不再在此保留——`git log -- .ai_partners/features/workstreams/2026/08/dsh-fusion/`
可见完整演进历史, 源码级调研沉淀于 Shared Resources。

## Open Problems

- **内核特权桥接口面未定** — 加哪几个接口、什么 payload、什么权限(plugin 进内核后
  有 apiproxy 同级特权半径, 不能做成"任意 append 任意事件"的裸口)。
- **热数据桥接形态未定** — 逐帧 hot 数据走哪条路, 与「hot 归 MOSS」分工一起定。
- **system prompt 构建路径未定** — agent-preset(声明式) / 本地 JS 插件(动态变量) /
  contentBlocks(Python 组装)三者组合关系。
- **激进解耦策略与旧决策的冲突** — 见 Key Decisions。
- **千级 session 治理** — fork/commit 累积的 session 生命周期(GC/归档/索引)待办。
- **协议漂移** — dsh 开发者预览, 破坏性变更; 集成层需容忍(`serverInfo.version` 恒 0.0.1)。