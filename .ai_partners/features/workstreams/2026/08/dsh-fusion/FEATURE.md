---
title: DSH Fusion
status: draft
# priority: importance within the current stage (iteration cycle) — not development urgency
priority: P0
created: 2026-08-14
updated: 2026-08-16
depends:
  - agent-surface
milestone: 0.1.0
description: >-
  dsh 融合 — DeepSeek Harness (dsh) 作为 MOSS 核心推理组件的一体化集成。dsh
  从「候选外部 agent 面」升格为 MOSS 的推理中枢: MOSS 保留记忆 (Memento) /
  执行 (CTML) / 感知 (audio/vision), dsh 承载 agent-loop 推理。本 workstream
  决策融合本身, 落点分两条路径: gui 管理的 agent 与 dolores ghost。
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
源码级调研见 research/2026-08-14_dsh_source_survey.md。

### 路径 B: dolores ghost (推理中枢)

dsh 作为 Dolores 的推理中枢, 决策落在 `ghost-prototype-dolores` FEATURE 的
DSH Integration 节。本 workstream 只提供融合基建, 不重复路径决策。
见 `ghost-prototype-dolores` workstream 的 DSH Integration 节与
research/2026-08-15_dsh_deep_dive.md。

## 融合基建 (Scope)

本 workstream 交付 dsh 融合的**可复用基建**, 供两条路径消费:

1. **optional 依赖** — dsh 作为 `ghoshell-moss` 的 optional dependency(pyproject
   `[project.optional-dependencies]`), 不污染核心安装。
2. **Python 基建** — 落在 `src/ghoshell_moss/agents/dsh/`, 与 `memento_pydantic_agent`
   同族。现状(2026-08-15 实验): `DshChannel` 直接 speak dsh stdio JSON-RPC, 已验证
   全链路(3080 → MCP → moss → CTML → mesh → DshChannel → 3081)。官方 SDK 太简陋,
   基建以自定义协议客户端为主。
3. **测试 node** — `.moss/system_test_nodes/dsh_web_probe/` 已验证, 系统化测试承接。
4. **装线** — 路径 B 的 dolores 装线同时进行。

## Key Decisions

- **dsh = 推理中枢, MOSS = 记忆/执行/感知。** 这是融合的第一裁决, 两路径共享。
  详见 Motivation。
- **官方 SDK 不是基建底座。** 源码级调研(Python 类型面极薄, 只懂 3 个 event type)
  与实验(DshChannel 直接 speak stdio JSON-RPC 跑通全链路)一致——基建以自定义协议
  客户端为主, 官方 SDK 仅作参考。
- **`agents/dsh/` 落点, 与 memento 同族。** 触发 `agents/` 包"第二个家族提级"的
  约定(`agents/__init__.py`)。独立包 `ghoshell_dsh` 被否: 会切断与 agent-surface
  骨架、memento 参照的关系。
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

- **调研轨迹**(workstream research/ 下):
  - `2026-08-14_dsh_source_survey.md` — 源码级调研(三套协议 / SDK / provider 注册 /
    web vendor 可行性)
  - `2026-08-15_dsh_deep_dive.md` — dsh 深入调研(沙盒 / Python SDK / JSON-RPC /
    全链路实验 / 架构收敛)
  - `2026-08-16_dsh_kernel_privilege.md` — 内核特权与三方桥(fork/compact 追加式 /
    协议面不对称 / apiproxy 式桥接收敛)
- **验证物**:
  - `.moss/system_test_nodes/dsh_web_probe/` — DshChannel node, 全链路已跑通
  - scripts/ — 后续系统化测试脚本落点
- **官方源码锚点**: `python/sdk`(Python SDK)、`packages/acp`(权限仲裁)、
  `packages/client/runtime`(web store, vendor 面)、`dsh-host-apiproxy`(桥接范本)

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
