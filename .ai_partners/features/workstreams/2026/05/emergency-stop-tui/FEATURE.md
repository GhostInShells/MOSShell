---
created: 2026-05-29
depends:
- first-ghost-prototype
description: TUI 快捷键触发全局急停，级联中断所有有状态双工节点（mindflow → shell → interpreter → speech），
  暂停 ghost 三循环但保持 TUI 存活可恢复。
priority: P0
status: completed
title: Emergency Stop — TUI 快捷键全局急停
updated: '2026-06-04'
---

# Emergency Stop — TUI 快捷键全局急停

## Motivation

当前 TUI 的 Escape 只做 REPL 级 interrupt（取消 `_operation_task`），ghost 三循环继续运行。
需要一个真正的急停——一键暂停所有双工节点，模型停止思考、命令停止执行、音频静音，
但 TUI 保持存活，可以恢复。

## 双工节点急停现状分析（2026-05-29 讨论）

链路拓扑：

```
input signal → mindflow → articulate_loop → action_loop
                  │            │                │
             attention   ghost.articulate   interpreter → command tasks
             impulse     (model API)         shell
             nuclei                          speech (TTS/audio)
```

逐节点 pause 覆盖情况：

| 节点 | pause 方法 | 做了什么 | 缺口 |
|------|-----------|---------|------|
| **Mindflow** | `BaseMindflow.pause()` | abort attention, 清空 signal 队列, nuclei.clear() | ✅ 已完备 |
| **Shell** | `CTMLShell.pause()` → `clear()` | `speech.clear()` + `main_runtime.tree.clear()` | ❌ 不调 `stop_interpretation()`，in-flight interpreter 继续跑 |
| **Interpreter** | 无独立 pause | 只能通过 `Shell.stop_interpretation(cancel_executing=True)` 中断 | Shell.clear() 漏调 |
| **Speech** | `Speech.clear()` | 清空活跃 TTS/音频输出流 | ✅ 被 Shell.clear() 调到 |
| **articulate_loop** | 无 pause | GhostRuntimeImpl 的 janus queue consumer | ❌ 模型 API 调用无法中断 |
| **action_loop** | 无 pause | GhostRuntimeImpl 的 janus queue consumer | ❌ 已在跑的 action 无法中断 |

## Key Decisions

### 1. 补齐 Shell.clear() 的 interpreter 中断

在 `CTMLShell._clear()` 中补 `stop_interpretation()`，使 `shell.pause()` 能中断 in-flight interpreter。

**改动点**: `ctml_shell.py:_clear()` — 在 `asyncio.gather` 中加 `self.stop_interpretation()`

### 2. GhostRuntime.pause() 统一急停入口

新增 `GhostRuntime.pause(toggle)` 方法，级联调用：
1. `mindflow.pause(toggle)` — 停分发、清信号
2. `shell.pause(toggle)` — 清 speech + channel tasks + interpreter（补齐后）
3. articulate_loop 的 in-flight model call 如何中断（待定）

pause(False) 逆序恢复。

### 3. TUI 快捷键绑定

在 `MossHostTUI.default_key_bindings()` 中新增全局急停快捷键（如 `Ctrl+\`），
直接调用 `GhostRuntime.pause(True/False)` 切换。

因为是全局快捷键，无论当前在哪个 state 都能触发。
需要在 TUI 状态栏显示是否处于急停状态。

## Open Questions

1. 快捷键选 `Ctrl+G`（macOS 无冲突，语义匹配 "stop the ghost"）。

2. pause/resume 的 TUI 状态指示——底部 toolbar 显示 `[PAUSED]` 标记。

## Implementation Notes

- mindflow 的 pause 已完备，本次主要补 shell 侧缺口
- `Shell.stop_interpretation()` 已存在且调 `old.close(cancel_executing=True)`，直接复用
- REPL 已有 `/ghost.pause()` 和 `/ghost.resume()`（GhostInspector），快捷键是对它的 TUI 级补充

## Design Revision (2026-06-02)

**GhostRuntime.pause() 只级联 mindflow.pause()，不额外调 shell.pause()。**

理由：`mindflow.pause()` → `attention.abort('paused')` 已通过以下机制自然中断 articulate/action 循环：

- `BaseArticulator._wait_aborted_and_cancel()` 检测 abort → task group close → `async with articulator:` 退出
- `BaseAction._wait_aborted_and_cancel()` + `action.received_logos()` 内部 `is_aborted()` 检查 → 停止 yield

`Shell.pause()` 仍保留独立路径，供 `GhostInspector.pause()` REPL 命令使用。

## Cross-Feature Verification Points

以下点由 **mindflow-control-semantics** (F3: abort 传播到 action loop + shell.clear) 验收：

1. **Articulate loop moment 保全**：`mindflow.pause()` 触发 attention abort 后，若 `async with articulator:` 因 `_wait_aborted_and_cancel` 提前退出，`on_articulate_exit()` 是否仍被调用确保 moment 不丢失。

2. **Action loop 容错**：abort 发生在 `_stream_execute()` 期间时，`received_logos()` 提前结束但 interpreter 收到不完整 CTML，`action.outcome()` 是否总是被调用来闭合 observe 回路。