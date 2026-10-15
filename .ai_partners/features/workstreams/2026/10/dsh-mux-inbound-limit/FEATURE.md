---
title: Dsh Mux Inbound Limit
status: completed
status_note: >-
  最小交付: 直接写死 max_size=None 修根因 + 一条 WS 级回归; 配置项/退避/计数/1009 分类/
  观测面全部裁掉, 见 Key Decisions。未真机复验。
# priority: importance within the current stage (iteration cycle) — not development urgency
priority: P0
created: 2026-10-15
updated: 2026-10-15
depends: [dsh-fusion]
milestone: 0.1.0
description: >-
  mux 下行入站上限修复：websockets 默认 1 MiB 让长会话的整份重连下发物越线，宿主侧工具
  (CTML) 静默全废。一行 kwarg 修根因；重连可见性欠账记在案，另行决策。
---

# Dsh Mux Inbound Limit

> 修复的**是什么**（code fact）以源码与 `git log` 为准；本文件只留决策与未决项。
> 上游 workstream: [dsh-fusion](../../08/dsh-fusion/FEATURE.md)（已完成；本组是它传输面的新决策集）。

## Motivation

2026-10-15 现场：`ghoshell_moss 0.1.0b2` + `websockets 17.2`，会话长过 1 MiB 后，每一次
mux 重连下发的整份物都越过 websockets 客户端默认入站上限（`max_size = 2**20`）→ 本端
`fail_connection(1009)` → 每秒连上又断，持续 23 分钟（`message too big` 1346 次、
稳定 59 次/分钟）。

后果的**形状**才是重点：宿主侧 dolores 工具（CTML / `moss_interpret`、`moss_react`）是这条
mux 的唯一消费者，mux 一死它们静默全废；而同会话的 `bash` / `read` / `write` 跑在 dsh
进程内，完全正常；MOSS 本体也活着。**没有任何一个面会说“CTML 废了”**——一次没有报警的失效。

这不是拼写错误，而是一条**没有主人的默认值**：库的 1 MiB 被继承，没人问过“dsh 会往这条
链路上发多大一坨东西”。

## Key Decisions

- **mux 入站不设上限（`max_size=None`）。** dsh 侧 node `ws` 的上限
  在 100 MiB 级，卡点只在宿主客户端；且超限的是**随会话增长的整份下发物**（现场尺寸
  1182866 → 1068567 → 1070726，缓慢漂移），不是某个固定大文件。被否：保留 1 MiB 靠
  “少聊点”绕过（长会话过 1 MiB 是常态）；把下发物切碎（那是 dsh 服务端的事，宿主改不了）；
  给一个“够大”的固定值（会话体积无上界，任何固定值都会在某个更长的会话上重演）。
- **直接写死 `max_size=None`，不做配置项（人类裁定）。** 曾做成 `DshConnectionConfig.mux_max_size`
  默认 None，被要求撤回：没人会去设它，而**每多一个没人设的旋钮就是一处将来会骗人的默认值**。
  需要收紧内存时再加不迟——那会是拿真实约束换来的决定，不是预留。
- **只止血，不顺手重做重连策略（人类裁定）。** 下列三块都写出并跑绿过，最后全部裁掉，
  理由是本次要的是“别再静默失效”，而不是借机改重连行为——改动面越小越可审：
  1. 重连退避 + 连续失败计数 + 到阈值升一次 ERROR 明说 CTML 不可用；
  2. 1009 单独分类，日志点名**入站**方向（原文 `sent 1009 …` 易被读成“发出去太大”）并给出
     最后一条成功解析的入站帧的 `streamId` / 帧类型作归因线索；
  3. `MuxHealth` 值模型 + `DshConnection.mux_health()` + `DshConnectionSurface` 投影
     （当时无任何生产消费者，只有测试在读）。
- **不变的口径：不得用 bash/read/write 是否可用反推 mux。** 它们跑在 dsh 进程内，不走这条
  链路——这正是本缺陷之所以静默的原因。判定必须落在 mux 日志或一次真实 CTML 回执上。

## Honest gaps（被裁方案留下的欠账）

- **同类故障仍会每秒刷屏。** 退避被裁，重连仍是固定 `sleep(1.0)`；下一次任何原因导致
  链路持续坏掉，日志还是 59 次/分钟的量级，且除一行行 WARNING 外没有“CTML 已不可用”的信号。
- **断连日志仍是库原文。** 1009 分类被裁，现场读日志的人仍可能把 `sent 1009 …` 读反方向。
- **“CTML 已不可用”没进到 dsh 会话内部（模型看不见）。** 要让它进 dsh 侧模型，需要在 TS
  插件上加一条注入路由（`session/instruction` 只读），那是另一个决策集。
- **那帧超限数据属于哪条逻辑流，没有钉到行级。** 倾向 `session/follow` 的 snapshot 重下发
  （随会话增长，与现场尺寸漂移吻合），不排除 `$events` 的断线期积压回放；两者同源
  （都由会话内容派生），本次修法对两者都成立。
- **未做真机复验。** 本组只有 WS 级回归（本地假 mux 发 >1 MiB 帧）。真机尺子（报告原文）：
  ① `grep -c "message too big" .moss/runtime/logs/moss.log` 新增停止增长；
  ② `mux connected` 之后不再紧跟 1009；③ 一条最短 CTML 能拿到回执。

## Exploration paths

- 起初想给“入站上限”找一个“合理的大值”（如 64 MiB）。放弃了：会话体积无上界，任何有限值
  都只是把同一个 bug 推到更长的会话上；`None` 才是与“整份下发物”语义一致的选择。
- 失败计数曾经想“连上就清零”。否掉：本缺陷里每轮都连上，那样永远清零、永远健康——
  **若将来重拾退避，计数必须按“连接活了多久”清**，这是本组留下的一条硬结论。

## Methods

- **WS 级回归用真实 `websockets.serve` 假 mux**（`tests/.../test_launcher.py`），而不是 mock
  `websockets.connect`：`max_size` 的语义（逐片累加、越线 1009）只有真跑协议栈才测得可信。
  这条测试发一条 >1 MiB 的帧，断言它被收下并分派——换回库默认即必挂。
- 用公开生命周期 `async with DshConnection(...)` 起 `_ws_loop`，不手掰私有状态。

## Implementation Notes

- **无新增配置项**：`max_size=None` 直接写在 `_ws_loop` 的 `websockets.connect(...)` 上，
  `DshConnectionConfig` 一行未动。
- `max_size` 只作用于**入站**消息；出站不受影响，宿主发出的开流帧始终很小。
