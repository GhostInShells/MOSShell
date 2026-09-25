---
title: Calendar Node — 可见 web 日程表 + 区间/到点/过期三级提醒
status: in-progress
priority: P2
created: 2026-09-22
updated: 2026-09-22
depends: []
milestone:
description: >-
  nodes/webview_apps/calendar/：node 起来就是一个视觉可见的 FullCalendar web，
  sqlite3 是人类与 ghost 共享的真相，提醒经 node 发 signal 进 mindflow
  (区间→低优 aside, 到点→分级 input, 过期→notify-next 且须过等级门槛)。
---

# Calendar Node

> Use `moss features set-status calendar-node <status> -m "note"` to update state.

## Motivation

要一个日程表：人类看得见、能改，ghost 也读得懂、到点会来提醒。

技术核心（人类 2026-09-22 明确）：**node 要能启动一个视觉可见的 web**。其余一切
（sqlite、提醒）都建立在这个可见面上——没有可见面，人类无法审计 ghost 看到的日程，
也无法与 ghost 共享同一份真相。

## Key Decisions

### KD1. 可见 web 是技术核心，先立它

`nodes/webview_apps/calendar/`，`singleton: true`。单端口同时服务 `index.html` 与 `/ws`
（照搬 `nodes/webview_apps/artifacts/main.py` 的 `SurfaceServer`，含
`del response.headers["Content-Type"]` 那个坑的修复）。端口默认 ephemeral，
真实地址从 channel 的 `url` named notice 读——**不许假设固定端口**。

前端 = FullCalendar 6 standalone，jsdelivr CDN（`artifacts/index.html` 已确立 CDN 惯例）：
`https://cdn.jsdelivr.net/npm/fullcalendar@6.1.15/index.global.min.js`。零新依赖
（`sqlite3`/`asyncio` stdlib + `websockets` 已在 `[host]`）→ 不建 INSTALL.md。

### KD2. 感知面不用 `context_messages`（人类否决）

`context_messages` 是 hot 面，**每思考帧刷新，污染行为**。日历的"现在几点、下一件事是什么"
是**慢变量**，不该占 hot 面。改用两条：

| 面 | 承载 |
|---|---|
| `named_notices`（warm, delta） | **当前所在区间**（天/时）的状态 |
| node 发 signal（异步推送） | **提醒本身** |

`named_notices` 的四值语义天然适配区间（`channel_builder.py:472`）：区间没变 → `""`
（零 token，模型保持上一帧读数）；区间切换 → 新文本重发一次；离开区间 → `None`
（`<name removed/>` 一次）。**这就是"区间内提示"**——delta 只在跨越区间时才花 token。

### KD3. 三级提醒：区间 / 到点 / 过期（人类定，非 notify 一元方案）

| 级 | 时机 | Signal | 优先级 | 频率 |
|---|---|---|---|---|
| 区间总量 | 天 / 时 桶切换 | `aside` | **低优** | **每区间一次**，聚合总量 |
| 到点 | `start_ts` 到达 | `input` | **按事件等级** | 每事件一次 |
| 过期 | `start_ts` 已过仍未处理 | `notify(next=True)` | 须 **> 门槛 X** 才够格升级 | 每事件一次 |

- **到点不是 notify 而是分级 input**：`input` 走 `InputSignalNucleus`（聚合 buffer，
  优先级取 max，赢了才转向）。提醒的"该多想一次"强度由事件等级决定——
  等级高则优先级高、更容易赢；等级低则可能让位于当前思考。这是刻意的**可让位**。
- **过期才用强提示**：`notify(next=True)` 保证"做完手上事就转向它"。但**有门槛**：
  只有等级 > X 的事件才配升到 notify；低等级过期不升级（否则过期信息会把 ghost 淹掉）。
  门槛 X 初值定在 `Priority.NOTICE`（即等级 ≥ WARNING 才升级），后续可调。
- **区间总量低优 aside**：`aside` 永不打断（`AsideNucleus` 只 buffer 不抢占，
  `min_priority=BACKGROUND` 收低优），配合低 priority，正好是"知会一声"。

三条都**由 node 主动发 signal**（`matrix.send_signal_to_ghost`，`matrix.py:123`），
不靠模型轮询、不靠 context 帧。

### KD4. 事件带 `level`，等级驱动提醒强度

`events.level`（0..3，越高越要命）是提醒强度的唯一旋钮，同时决定：
到点 input 的优先级（`NOTICE + level`）、以及过期是否够格升 notify（`level >= X` 的门槛）。
不设 per-reminder 覆盖——一个事件一个等级，简单可解释。

### KD5. sqlite3 是共享真相，人类与 ghost 读写同一张表

db 落 `matrix.home / "calendar.db"`（可被 `MOSS_CALENDAR_DB` 覆盖），
`PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000`——`sqlite-channel` workstream 的结论：
**WAL + busy_timeout 是跨进程成败关键**。

单表 `events`，不建第二张表：`remind` 用 `start_ts` + 相对偏移，去重与账本共用
`fired_ts`（NULL=pending）。多次提醒 / snooze / RRULE 留后续。

## Implementation Notes

- 布局走非平凡 node 的惯例（同 `nodes/os/*`、`ghost-in-bilibili`）：`src/ghoshell_calendar/`
  包 + `tests/`，`main.py` 薄入口。
- `ReminderLoop` 的 `send_signal` 与时钟由外部注入 → 循环本身不依赖 Matrix，可单测。
- 唤醒机制：`asyncio.Event`（写库即 set）+ `wait_for` 上限 30s（兜住外部写入与时钟漂移）。
- stub 三面纪律（`node-lifecycle`）：NODE.md（模型面）+ README.md（人类面），
  无 INSTALL.md（零新依赖），英文。
- 时间输入三级回退（`YYYY-MM-DD HH:MM` / `YYYY-MM-DD` / `HH:MM`），本地时区。

## Open Questions

- 门槛 X 与 level 分档的具体取值（先 `NOTICE` + 0..3），dogfood 后再定。
- 提醒的"处理"语义（ghost 说一句算处理？还是人类在页面 ack？）——本版只做"发出即记账"。
- 是否给 ghost 一个 `snooze` / `ack` 命令（先不做，观察真实摩擦）。
