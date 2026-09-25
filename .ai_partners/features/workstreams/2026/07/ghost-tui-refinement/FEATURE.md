---
created: 2026-07-27
depends:
- ghost-runtime-safemode
description: Ghost TUI 交互治理：logos 尾段丢失、input 计数残留、感知边界、快捷键约定、状态渲染跨线程、 日志可视化。safemode
  dogfooding 时暴露的一批交互问题，集中做一轮 refinement。
milestone: null
priority: P2
status: completed
status_note: '项 #1 logos 尾段渲染丢失已修复: 原地重绘 RESPONSE panel + queue.Queue 毒丸生命周期 + RichCaller.stop()
  让 interrupt/close 立即停渲染. 其余项未做, workstream 关闭.'
title: Ghost TUI Refinement — 交互时序与治理
updated: '2026-08-28'
---

# Ghost TUI Refinement — 交互时序与治理

> Use `moss features set-status ghost-tui-refinement <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

`ghost-runtime-safemode` dogfooding 时集中暴露了 Ghost TUI 的多个交互问题。
安全模式让人类介入 articulate→action 链路，这一介入把原本流式路径中被
掩盖的时序、状态、感知边界问题都翻了出来。本 workstream 集中做一轮 TUI
治理。

## Design Index

- Key design documents: `design/`
- Key discussion records: `discuss/`
- 相关 dogfooding 记录见 `ghost-runtime-safemode/FEATURE.md` Round 2

## Key Decisions

<!-- 下面每一项都是从 dogfooding 中提出的观察，尚未 design-lock，需要各自展开设计。 -->

### 1. Logos 尾段渲染丢失

**现象**：ghost 每轮输出的 logos 最后一段没有在 TUI 中展示，要等到下一次
input 时才会补上。dogfooding 中 ghost 自己也注意到并反馈了这个问题。

**猜测方向**：解析层 (CTML)、渲染层 (pub_logos 到 rich console)、或 flush
时机。需要先定位是哪一层的 bug。

### 2. TUI input 计数不消失

**现象**：mindflow perspective 里 "pending input from ghost tui" 的计数
持续累加，不下降。ghost 感知到这些 pending 但实际上没有信号积压——大概率
是 input nucleus 的状态没有正确清理。

**猜测方向**：input signal 消费后没 pop，或 status snapshot 没实时更新。

### 3. Input 调度：notify + 高优，而非 priority queue

**现状**：目前 TUI input 走某种 priority queue 逻辑。
**目标**：改成 notify + 高优逻辑。TUI 输入应该是一个 "抢占式的通知"，
而不是 "排队等待被处理的信号"。ghost 应该在收到 TUI 输入时立刻感知，而不是
等自己的当前回合完成才从队列里拉。

需要在 mindflow / attention 层看清当前的调度语义再决定改法。

### 4. TUI 感知过强，收敛提醒范围

**现象**：ghost 每一帧的 perspective 都会看到 "来自 ghost tui 的 input
signal"，包括通道配置、语音接口等元信息被重复声明。ghost 明确表达
"我的声音配置被重新声明了一遍，语音接口也完整地再次给出了"。

**目标**：ghost 应该知道 TUI 是一个信号源，但不该每帧都被这些元数据轰炸。
需要区分 "首次接入的静态信息" 和 "本轮变化的动态信号"。

（可能与 `channel-meta-dyn-static` workstream 有交叉，视情合并或对齐。）

### 5. Ctrl+C 清空输入区

按下 Ctrl+C 时，如果输入区有内容，清空当前 buffer。属于常规 REPL 惯例。

### 6. Ctrl+C 空输入区 → interrupt / exit 两级

在输入区已空的前提下：
- 第一次 Ctrl+C 发送 interrupt（打断当前 ghost 生成）
- 第二次 Ctrl+C 直接退出

需要短暂 debounce 窗口（1-2s？）区分"连按"与"两次不同意图"。

**定位（2026-08-03，确认 bug）**：TUI 的 interrupt 根本不触发 `shell.clear()`。现状两个按键都
不走 interrupt 通路：

- `c-c`（tui.py `default_key_bindings`）绑定 `graceful_exit` → `self.close()` 直接关掉整个
  TUI，没有 interrupt 一级。
- `escape` → `current_state().on_interrupt`，但 `REPLState.on_interrupt` 只 cancel 本地
  `_operation_task`（文本输入处理），不产生 interrupt signal → ghost 生成不停、半截命令继续跑。

而正确通路已存在，只是 TUI 没接：

```
new_interrupt_signal()  (core/mindflow/interrupt_nucleus.py:196)
  → InterruptSignalMeta("interrupt") 信号
  → InterruptNucleus 包装成 FATAL + notify + interrupt=True impulse
  → ghost_runtime._main_loop  (host/ghost_runtime.py:298-299)
  → await shell.clear()   ← 关 interpreter + 清 speech 缓冲 + 取消 pending commands
```

**修复设计（已实现，本会话）**：

- `ghost_ui.py _GhostStateBase.on_interrupt` override：先 cancel 本地 REPL operation（复用
  REPLState 逻辑），再向 session 发 `new_interrupt_signal(description="from ghost tui")`，
  接入既有通路触发 `shell.clear()`。
- **interrupt 与 exit 分离**：interrupt 唯一入口是 `escape`（走 `on_interrupt`，真实打断
  生成）。`c-c`（tui.py）不再承担 interrupt——空输入区首按只提示 `press c-c again to exit`，
  debounce 1.5s 内二按才 exit；输入区有内容则清空 buffer（项 #5，不打断不退出）。
  理由（人类工程师定）：有 esc 时 c-c 做 interrupt 冗余，提示"下一轮 c-c 关闭"更合理。
- 顺手修复暴露的老 bug（关闭路径）：`tui_exception_handler` 原用 `self.host.matrix()`
  （Host 不缓存 matrix → AttributeError 让 "Task was destroyed" 变成 exception handler 崩溃）；
  改为 abstract `_log_loop_exception`，三个 concrete TUI 各自经 runtime 取 matrix logger。
  `run()` finally 里 cancel + gather pending tasks，消除 loop.close() 的 pending 噪音。

### 7. 跨线程 rich status 对象的可提交性

需验证：rich 的 `Status` / `Live` 对象是否支持 "A 线程 render, B 线程改
状态"。若可行，很多状态展示可以走这个模式（例如 pending input 数量、
safemode gate 状态）不需要 invalidate 整个 prompt。

**行动**：写一个最小实验脚本验证 threadsafe 行为。

### 8. `/clear` 清屏命令

REPL 常见惯例，直接加入。

### 9. TUI REPL state: tail 系统日志

新增一个 TUI 状态：tail `.moss/runtime/logs/moss.log`（或其他系统日志），
支持在 tail 视图中直接输入命令。

**前置**：project 需要暴露 API 把 "系统日志文件路径" 显性化（不能让 TUI
自己猜路径）。相关命令可能是 `moss project logs-path` 或
`moss project.log_paths()`。

参考 `ghost-runtime-safemode` Round 2 复盘：静默 `except Exception` 让
真错误消失在日志里，用户看不到。TUI 内置日志 tail 能让这类 bug 提前浮现。

## Implementation Notes

- 各点独立性较强，可按优先级拆分实现，不必一次落地全部
- 优先级建议（初判，可讨论）：
  - P0: (1) logos 尾段、(2) input 计数 —— 都是明确的 bug
  - P1: (3) input 调度、(4) 感知收敛 —— 语义层重构，影响面大
  - P2: (5)(6)(8) 快捷键与 /clear —— 用户体验治理
  - P2: (7) rich status 跨线程 —— 需要先做技术验证
  - P2: (9) 日志 tail —— 需要 project API 支持
- (4) 与 `channel-meta-dyn-static` 可能有语义重叠，实现前先看那边有没有
  已 design-locked 的方向

## Session Log

### 2026-08-03 — interrupt → shell.clear 通路修复（项 #5、#6）

修复确认 bug：TUI interrupt 不触发 `shell.clear()`（详见项 #6 定位）。

- `ghost_ui.py`：`_GhostStateBase` 新增 `on_interrupt` override —— cancel 本地 REPL
  operation 后向 session 发 `new_interrupt_signal()`，接入既有
  InterruptNucleus → `shell.clear()` 通路。
- `tui.py`：`c-c` 绑定从 `graceful_exit`（无条件关 TUI）改为提示两级 —— 输入区有内容清空
  buffer；空则首按提示 `press c-c again to exit`、1.5s debounce 内二按 exit。interrupt
  收归 `escape` 单一入口（项 #6 二次修订，人类工程师定）。
- `tui.py` + 三个 concrete TUI：关闭路径修复 —— `tui_exception_handler` 弃用
  `self.host.matrix()`，改 abstract `_log_loop_exception`；`run()` finally cancel +
  gather pending tasks，消除 "Task was destroyed" 噪音与 handler 崩溃。
- 未动：`moss-shell`（REPLState 本地 op cancel 语义保留）、项 #1 logos 尾段、项 #2 计数残留。