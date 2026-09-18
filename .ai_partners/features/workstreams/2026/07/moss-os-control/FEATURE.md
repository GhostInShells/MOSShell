---
title: MOSS OS Control — 操作系统的统一能力域
status: in-progress
priority: P1
created: 2026-09-13
updated: 2026-09-15
milestone: beta-release
description: >-
  把操作系统当作一个整体能力域来校准。nodes/os/ 下每个 node 管一个 OS 原生面，
  各自提供轻量的 web 交互面。概念由 desktop-gui（bash + file_editor 的人类 GUI）
  扩大而来；其 node 是空 UI 原型，已删除。
---

# MOSS OS Control

> 本目录是本 feature 的 workstream。**每个子功能一个子文档**,本 FEATURE.md 只做索引与
> 关键决策。前身 `desktop-gui` 的原文档保留为本目录的 `FEATURE.old.md`,讨论记录在
> `discuss/`。本 feature 的**能力实现**落在 `nodes/os/`(category),文档留在这里。

## Motivation

desktop-gui 最初只是 bash + file_editor 的人类 GUI。现在把它扩大成一个**操作系统
能力域**:把 OS 当作一个整体对象 —— 窗口、屏幕截屏、shell、文档、原生 app 控制
都是这个域里的面。

目标是一条**统一集中的管理机制**,不是第一轮把面摆齐。机制对了,面随成熟度塞进来。

## 关键决策

### KD1. 能力域,不是机器

category = `nodes/os/`(能力优先)。**不**按 `nodes/macos` / `nodes/linux` 顶层切 ——
OS 是实现维度,不是能力单位。node 的 probe 探针 + install 已是现成的判别机制,
平台差异收在各 node 内部按 OS 分派。

这是 ghost 中心化的关键:面是给 ghost 的,谁落在哪台机器由运行时判别。

### KD2. 一个 node = 一个 OS 面 = 一个进程 = 一个 channel

不是一个 node 嵌 N 个子 channel。category 下是 N 个平级 node。

### KD3. 不做聚合,每个 node 自带 web 交互面

**所有 os node 都要有至少 web 级别的交互面,且尽可能轻量。** 采集/数据管线与模型面
**共用一条**(vision 家族契约第 6 条的推广:图形化是人类面,不建第二套管线)。

不建聚合机制:每个 node 就是自己的 web 服务;"打开"交给已存在的 browsers / 躯体
node 去做。node 只专注于**能力 + 交互体验**。

### KD4. 屏幕截屏是控制面,图形界面是躯体面

> **已被取代 (2026-09-15)**: 见 feature `vision-stream` 的
> `design/2026-09-15-stream_vision_unified_input_perception.md` KD3。
> 屏幕截屏的实质是**感知**,归 vision;os 域若保留什么,其身份是
> "**把屏幕推成地址的本地推流模块**"(producer),不是"截屏能力"。原文保留如下。

- **屏幕截屏** = OS 原生能力的**控制面** → 属于本域(node `screen`)。
- **图形界面/合成器**(`nodes/screens/qt_screen`)= **躯体能力** → 不属于本域。

两者语义不同,不混。`qt_screen` 若因命名易混,改它的名。

### KD5. `file_editor` 是文档内存编辑产品,不是文件系统

不喜欢现有 coding agent 的文档编辑交互,要做产品化:

```
read(文档 → 内存)
  → editor(内存里多轮改)
      → [ action(可流式呈现) → effect(可 diff) → dialog(可对话) → result ] 循环
  → export(落回原文件) / 放弃
```

界面是一个**流**,export 是终点;副作用更新循环。命名仍叫 `file_editor`。详细方案
另立子文档展开(本轮只钉循环)。

### KD6. 存量 alpha channel = 实现底座

`terminal_channel` / `file_editor_channel` / `mac_channel` / `desktop_channel` 是 package
里的 alpha channel,是**实现底座代码**。怎么处理(直接复用 / 包成 node / 重写)按各自
**实现成熟度**逐个判,不现在一刀切。

### KD7. desktop-gui node 是空原型,已删除

原 `nodes/skins/desktop-gui` 只有一个 Reflex UI 原型:命令流 / 审批 / 对话全部是 mock
(`inject_mock_command` 手工注入 + `_mock_result` 硬编码),没有任何 Matrix / channel
接线(全文件唯一 "Matrix" 字样是 `app.py` 的一句 docstring)。**能力边界为零**,故删除,
不搬进本域。

留下的价值是**设计**,不在代码:

- 审批即对话(approval-as-dialogue)、diff 渲染、呼吸灯状态 —— 保留在本目录的
  `FEATURE.old.md` + `discuss/`。
- 真正的实现底座是 `terminal_channel` / `file_editor_channel`(见 KD6),不是这个 node。

`terminal` 重做时,上述设计作为人类面的输入之一。

## 命名

| node | 面 | 说明 |
|---|---|---|
| `terminal` | 进程编排 | 待建;全异步 bash + python 调度器,见 `terminal.md`。持久 shell 会话是 `pexpect node`(另立) |
| `window_control` | 窗口 | 枚举/几何/聚焦;下面按 OS 分派实现 |
| ~~`screen`~~ | ~~截屏~~ | **已移出本域 (2026-09-15, 见 KD4)** → feature `vision-stream` |
| `osascript` | mac 原生 | 按控制语言原名,不发明 "app" 中间名词 |
| `file_editor` | 文档 | 见 KD5 |

## 子文档

| node | 状态 | 子文档 |
|---|---|---|
| terminal | 已实现 v1 | [terminal.md](terminal.md) |
| window_control | 待建 | — |
| screen | 已移出本域 (2026-09-15) | → feature `vision-stream` |
| osascript | 待建 | — |
| file_editor | 已实现 v1（卡片机制，同 terminal） | [file_editor.md](file_editor.md) |

## 当前状态

- `nodes/os/terminal`（`bash` 卡片进程编排）与 `nodes/os/file_editor`（人机共享感知的文本
  工作副本，卡片机制与 terminal 同构，见 `file_editor.md`）已落地；
  `window_control` / `osascript` 仍待建。
- 原 `nodes/skins/desktop-gui`(空 UI 原型)已删除。
- 旧 workstream `2026/07/desktop-gui` 改名 `2026/07/moss-os-control`;老 `FEATURE.md`
  改名 `FEATURE.old.md`,本文件为新的 `FEATURE.md`。

## 待办

- `tutorials/L1_create_a_node.md` 以 desktop-gui 为示例 node,该 node 已删 —— 示例需换。
- `window_control` / `osascript` 两个面仍待建（存量 alpha channel `mac_channel` /
  `desktop_channel` 是底座，见 KD6）。
