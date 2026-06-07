---
title: TUI Render Governance — 有状态渲染、全局通知通道与底部状态栏
status: completed
priority: P1
created: 2026-06-03
updated: 2026-06-07
depends: []
milestone:
description: >-
  TUI 渲染基础设施增强：ConsoleOutput 重命名为 TuiRender，所有 state 统一有状态 ring buffer，
  非 current state 输出进 buffer 不丢弃，切换 state 时 clear + replay buffer，
  增加 prompt_toolkit bottom_toolbar 全局通知通道，加 urgent 参数区分普通输出与审批级通知。
---

# TUI Render Governance

## Motivation

当前 TUI 渲染体系（`host/tui.py`）有三个缺口：

1. **非 current state 输出被丢弃**。`ConsoleOutput.rprint()` 检查 `_alive_fn()`，
   不 alive 时直接 return——切走 state 后 Ghost 流式输出、审批通知全部丢失。
2. **state 切换无 buffer**。切回 state 时看不到历史输出，只有一条切换分隔线。
3. **无全局通知通道**。审批请求、系统告警等需要跨 state 可见的信息没有落点。

这三个缺口是所有需要"人在 TUI 里审批 AI 操作"的场景的前置依赖。

## Key Decisions

### 1. ConsoleOutput → TuiRender 重命名

原名 "ConsoleOutput" 语义不准——它不直接操作 console，而是管理渲染输出、buffer、和
底部状态。新名 `TuiRender` 精确描述职责。

### 2. 所有 state 统一有状态 ring buffer

`deque(maxlen=buffer_size)` 存所有输出项。两个开关：
- `buffer_enabled` — 全局开关，默认 True
- `buffer_rprint` — 普通 `rprint()` 是否进 buffer，默认 True

所有输出路径统一进 buffer。所有 state 共享相同 buffer_size 配置。

### 3. 非 alive 输出进 buffer，加 urgent 全局通道

```python
def rprint(self, *items, urgent: bool = False):
    self._buffer.extend(items)
    if urgent:
        self._global_channel.push(items)  # → bottom_toolbar 更新
    if not self._alive_fn():
        return  # buffer 已存，不渲染，但不丢弃
    self._queue.put_nowait(items)
```

`urgent=True` 走全局通道 → 触发 bottom_toolbar 更新（如审批计数）。
`urgent=False` 只进 buffer，切回 state 后 replay。

### 4. 切换 state = clear + replay_buffer

`on_switch(True)` 时调用 `render.replay_buffer()` 重放所有历史输出。
接口统一为 `TUIState` 的默认行为，每个 state 自动继承。

### 5. bottom_toolbar 注册给 prompt_toolkit

`TuiRender` 暴露 `get_bottom_toolbar() -> Callable[[], str]`，
在 `MossHostTUI._input_loop` 的 `prompt_async()` 中挂载。
prompt_toolkit 每帧自动调用，无需 push。

### 6. 不引入新依赖，不改 state 生命周期

所有改动局限在 `host/tui.py`。不改 PromptSession → Application 的重构。
不改 state 的 `__aenter__/__aexit__` 生命周期。纯增量改动。

## Implementation Notes

- `deque` 来自 collections 标准库
- buffer_size 默认 200，可配置
- `replay_buffer()` 把 deque 内容全部重新 push 到 render queue
- bottom_toolbar 文本由 ApprovalModule 等外部模块通过 `TuiRender.set_bottom()` 写入

## Dependencies

无。纯 TUI 基础设施改动，不依赖 Terminal、ApprovalModule、或任何上层模块。

## Relationship to ai-terminal

`ai-terminal` feature 的第三层（GUI App Cell）可以选择由本 feature 支持的
TUI ApprovalState 替代。两个 feature 独立迭代，在 bottom_toolbar 回调点汇合。
顺序：TuiRender 重构 → ApprovalState → Terminal 审批模块。

## Implementation (2026-06-07, deepseek-v4-pro via claude code)

### Changed files

| File | Change |
|------|--------|
| `src/ghoshell_moss/host/tui.py` | ConsoleOutput → TuiRender (backward compat alias), deque ring buffer, urgent= param, replay_buffer(), bottom_toolbar on prompt_async |
| `src/ghoshell_moss/host/repl/inspector_moss_runtime.py` | ConsoleOutput → TuiRender type annotation |
| `src/ghoshell_moss/host/tui_entries/ghost_ui.py` | _prompt_status: ANSI escapes → FormattedText tuples |

### Decisions made during implementation

- **replay_buffer drain**: per-state TuiRender gets `_drain_render_queue` as clear_func (not full console clear) — drains shared queue without clearing rich console, then replays buffer.
- **on_urgent callback**: per-state TuiRender instances wired to `MossHostTUI.set_bottom_toolbar` — urgent rprint from any state updates the global bottom toolbar.
- **_prompt_status contract**: changed from `str` with ANSI escapes → `list[tuple[str, str]]` (FormattedText). ANSI codes in prompt_toolkit message cause render conflicts.
- **backward compat**: `ConsoleOutput = TuiRender` alias at module level. All existing rprint() calls unchanged (urgent defaults to False).

### Remaining for downstream

- ApprovalState (ai-terminal Phase 2): uses `urgent=True` + `set_bottom_toolbar()` for approval queue display.
- Bottom toolbar styling: currently plain text; FormattedText support can be added when needed.
