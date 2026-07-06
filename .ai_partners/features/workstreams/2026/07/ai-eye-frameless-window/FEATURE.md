---
created: 2026-07-06
depends:
- pygame-games
description: 把 ai_eye 从常规 pygame 窗口迁移为无边框、透明、置顶的 PySide6 窗口——只有眼睛悬浮在桌面上， 可拖拽移动、边缘缩放。原有化身控制（look_at/dilate/blink/set_expression）随迁移完整保留。
  这是「GUI 就是本质、屏幕是 N 层独立透明图层」这一 AIOS 显示论的第一个实证。
milestone: null
priority: P2
status: completed
status_note: frameless transparent PySide6 window + preserved avatar controls
title: AI Eye Frameless Window — 无边框透明桌面图层
updated: '2026-07-06'
---

# AI Eye Frameless Window — 无边框透明桌面图层

> Use `moss features set-status ai-eye-frameless-window <status> -m "note"` to update state.

## Motivation

`pygame-games` 里的 ai_eye 是「AI 化身」的最小锚点，但它跑在一个**普通带边框的 pygame 窗口**里——
标题栏、灰底、窗口 chrome 全在，眼睛被框在一个"应用窗口"里，破坏了"AI 降临到桌面上"的存在感。

真正想要的是：**只有眼睛本身悬浮在桌面上**，没有窗口的痕迹。这不只是视觉打磨——它是
`.forge/notes/2026-06-30-streaming-gui-aios-display.md` 那条显示论的第一个可运行实证：
每个 app 的窗口本质上是一层**全屏透明图层**，人看到的"窗口"只是它的不透明区域；
现代 OS 一直支持无边框/透明/层叠，只是人被 UI 习惯骗了。ai_eye 是把这个论断落到代码的最小切口。

## Key Decisions

### KD1: 迁移到 PySide6 承载窗口，pygame 只做绘制

窗口壳换成 PySide6（`QWidget`，`FramelessWindowHint` + `WA_TranslucentBackground` + 置顶）。
眼睛绘制**仍留在 pygame**：绘制到一个离屏 `SRCALPHA` surface，每帧 blit 进 Qt 窗口的 `paintEvent`。

**Why 不纯 pygame**：pygame 自身对"无边框 + 逐像素透明 + 桌面合成"的跨平台支持很弱（尤其 macOS）。
Qt 的 translucent frameless widget 是成熟路径。保留 pygame 绘制则**完整复用**了 `pygame-games` 里
已验证的化身逻辑——眼睛渲染数学、以及 `look_at`/`dilate`/`blink`/`set_expression` 全套 Channel 控制与
动画（注视 lerp、瞳孔呼吸、自动眨眼、空闲飘移）——不推倒重来。迁移**没有丢任何一条命令**
（新旧版都是 7 条 command + `context_messages`）。

**Why 不换 Live2D / Qt 原生绘制**：`pygame-games` KD4 的硬约束是"两个椭圆 + 一个圆就能做出有灵魂的眼睛"，
不引入 3D/Live2D。沿用之。

### KD2: 交互靠自绘 grip，而非系统窗口边框

无边框意味着失去系统提供的移动/缩放把手，自己实现：
- **移动**：在窗口体任意处拖拽（`mousePressEvent`/`mouseMoveEvent`）。
- **缩放**：拖右边缘 / 下边缘 / 右下角（8px grip，`_detect_grip`），眼睛随窗口 `relayout` 等比缩放。
- **退出**：`ESC`（或 ghost 会话结束关闭窗口）。

**Why**：这正是那条显示论的注脚——人需要边框来缩放/拖拽/关闭，AI 不需要。这里为"人还要用"
保留了最小交互；面向 AI 的场景（窗口管理器 app 调坐标/大小）是后续方向，不在本切口内。

## Implementation Notes

- 依赖新增 `PySide6>=6.5.0`（`apps/games/ai_eye/pyproject.toml`）。依赖隔离在 app 内，不污染主环境
  （延续 `pygame-games` KD2 的依赖隔离原则）。
- pygame 绘制 → 离屏 SRCALPHA surface → 每帧 blit 进 `QWidget.paintEvent`，是 pygame 与 Qt 事件循环
  共存的关键桥。注意帧节奏由 Qt 侧驱动 repaint。
- app 内 `CLAUDE.md` 已同步更新窗口行为说明（Move/Resize/Quit）。

## Status note

完成。无边框透明窗口 + 拖拽/缩放已落地并实测通过；`pygame-games` 的化身控制
（`look_at` / `dilate` / `blink` / `set_expression` + 注视 lerp / 呼吸 / 自动眨眼 / 空闲飘移）
随迁移完整保留并在新 PySide6 窗口上正常工作（已核代码：main.py 命令定义 154–244、动画在
`game_loop` :646、渲染在 `_draw_frame`/`_draw_eye` :515-517）。面向 AI 的窗口管理器场景
（调坐标/大小/多图层）是后续独立方向，不属于本切口。