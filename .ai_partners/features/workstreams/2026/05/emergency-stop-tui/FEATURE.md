---
created: 2026-05-29
depends:
- first-ghost-prototype
description: TUI 快捷键触发全局急停，级联中断所有有状态双工节点（mindflow → shell → interpreter → speech），
  暂停 ghost 三循环但保持 TUI 存活可恢复。
priority: P0
status: completed
title: Emergency Stop — TUI 快捷键全局急停
updated: '2026-06-16'
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

## Acceptance Testing Findings (2026-06-15)

验收发现三个核心问题，状态回滚为 in-progress。

### Problem 1: TUI toggle 导致状态永远不对齐

当前 `GhostTUI._on_emergency_pause()` 使用 `self._paused = not self._paused` 做 toggle，
但 pause 状态分布在多个组件中（TUI._paused、GhostRuntime 无自身状态、Mindflow._paused、
Shell._paused），没有单一真源：

- TUI 按 Ctrl+G → toggle `self._paused` → `runtime.pause(self._paused)`
- 如果 mindflow 已通过 REPL `/ghost.pause()` 被暂停，第二次 Ctrl+G 会 toggle 成 False，
  调 `runtime.pause(False)` 反而 resume 了一个 REPL 暂停的状态
- 如果 `runtime.pause()` 内部任何环节失败（无声），TUI 的 `_paused` 与真实状态永久偏离
- 多个入口（TUI Ctrl+G、REPL `/ghost.pause()`、程序化调用）之间没有仲裁，各自 toggle

**修正方向**: GhostRuntime 应作为 pause 状态的单一真源，提供 `is_paused()` 查询接口。
TUI 不再自己 toggle，而是调 `runtime.pause(True/False)` 并查询 `runtime.is_paused()` 来显示状态。

### Problem 2: Shell.pause() 实现不对

`Shell.pause()` 当前实现：
```python
def pause(self, toggle: bool = True) -> None:
    self._paused = toggle
    if self._paused:
        self.clear()
```

问题：
- `clear()` 是 async（返回 Future），`pause()` 是 sync——如果 clear 内部创建了 `_clearing_task`，
  pause() 返回时 clear 可能还没完成，调用方以为已经停了但实际还在清理中
- `_paused` 只在 `_check_paused()` 里用，而这个检查仅在 `interpreter()` 和 `push_task()` 入口—
  不阻止已经在跑的 interpreter 继续执行
- pause 之后没有阻止 shell 继续接收和处理来自外部的数据

### Problem 3: pause 没有做 loop 卸载

设计修订决定只调 `mindflow.pause()` 而不调 `shell.pause()`，依赖 attention abort 的自然传播。
但 abort 传播是协作式的——只在特定检查点（`_check_abort_and_clear` 在 feed/compile/execute 阶段边界，
`_wait_aborted_and_cancel` 在 `async with articulator:` 的 __aexit__）响应：

- in-flight 模型 API 调用（`ghost.articulate()`）不会被 abort 中断，必须等 API 返回
- 命令执行阶段中（`_stream_execute`）abort 只在三个 phase 之间检查，长耗时命令不会被打断
- 三循环本身（`_main_loop`、`_articulate_loop`、`_action_loop`）没有暂停机制，只在 queue.get()
  处阻塞等待下一个 item

**修正方向**: 三循环需要能响应 pause——最简单的方式是让 `pause()` 影响 queue 的行为
（如 shutdown + 重建，或在 loop 的 queue.get 前加 is_paused 检查）。同时 shell.pause()
 需要同步等待 clear 完成。

## Design Revision (2026-06-15)

验收后完整重设计。核心原则：**幂等、回调通知、副作用与 clear 分离、向前兼容、分层单测**。

### Layer 1: GhostRuntime — 单一真源 + callback 参数

**回调协议选型**: 不用注册式 (`on_pause_changed`)，不用 ThreadSafeEvent（所有权不清），不用 Future（cancel 语义在 pause 场景下是误导——副作用已发生，无法回滚）。采用 `pause(toggle, callback=None)` — 谁调谁收通知，无生命周期负担。

跨组件感知（如 REPL pause 后 TUI 也要显示 `[PAUSED]`）不靠回调，靠 TUI 渲染时读 `runtime.is_paused()` 轮询——prompt 刷新频率本身就高。

```python
from typing import Callable

PauseCallback = Callable[[bool], None]  # bool = 新的 pause 状态


class GhostRuntime(ABC):
    def pause(self, toggle: bool = True, callback: PauseCallback | None = None) -> None:
        """急停. 幂等: pause(True) 多次调用仍是 paused. 向前兼容.
        
        callback 在当前调用栈中同步 fire (pause 已设完状态后).
        调用方必须保证 callback 是线程安全的 (如果跨线程调 pause).
        """
        pass

    def is_paused(self) -> bool:
        """查询当前 pause 状态. TUI/REPL 统一入口."""
        return False


class GhostRuntimeImpl(GhostRuntime):
    def __init__(self, ...):
        self._paused = False
        self._pause_lock = asyncio.Lock()

    def is_paused(self) -> bool:
        return self._paused

    def pause(self, toggle: bool = True, callback: PauseCallback | None = None) -> None:
        if self._paused == toggle:
            # 幂等: 同状态不重复执行. 但仍 fire callback 让调用方对齐.
            if callback:
                callback(self._paused)
            return
        self._paused = toggle  # 先设状态, 乐观更新
        if toggle:
            if self._mindflow is not None:
                self._mindflow.pause(True)
            if self.moss.shell is not None:
                asyncio.create_task(self._async_pause_shell())
        else:
            if self._mindflow is not None:
                self._mindflow.pause(False)
            if self.moss.shell is not None:
                self.moss.shell.pause(False)
        if callback:
            callback(self._paused)

    async def _async_pause_shell(self) -> None:
        """在 event loop 内执行 shell 侧暂停, Lock 防重入."""
        async with self._pause_lock:
            await self.moss.shell._pause()
```

关键设计决策:
- `_paused` **先设值**再做清理 — 乐观更新, callback 立即 fire, 清理在后台跑
- 幂等时仍 fire callback — 调用方不知道当前状态, callback 让它对齐自己的 UI
- `_pause_lock` 保证 shell 侧的 async 清理不会重入
- callback 在 pause() 内部同步调用 — 调用方不需要管理 Future/Event 生命周期

### Layer 2: Shell — async _pause + 副作用与 clear 分离

```python
class CTMLShell:
    async def _pause(self) -> None:
        """异步暂停 — 在 event loop 内执行, 返回时保证副作用完成.
        与 clear() 分开: clear 是通用清理, _pause 是暂停专用路径."""
        self._paused = True
        # 直接 await _clear, 不等 Future
        done = await asyncio.gather(
            self._speech.clear(),
            self._main_runtime.tree.clear(self._main_runtime),
            self.stop_interpretation(),
            return_exceptions=True,
        )
        for t in done:
            if isinstance(t, Exception):
                self._logger.error(...)

    def pause(self, toggle: bool = True) -> None:
        """sync pause — 向前兼容. 不等待清理完成, fire-and-forget.
        GhostRuntime.pause() 内部走 _pause(), 此路径保留给 REPL 直接调用."""
        self._paused = toggle
        if self._paused:
            self.clear()  # fire and forget
```

关键设计决策:
- `_pause()` 是 async, 在 event loop 内执行, 返回时**保证** stop_interpretation + speech.clear + tree.clear 全部完成
- `pause()` 保持 sync 签名 — REPL `/ghost.pause()` 和旧调用方不受影响
- `_pause()` 和 `_clear()` 共享实现体 (`asyncio.gather` 那段), 但 `_pause()` await 它, `clear()` fire-and-forget 它
- 副作用分离: 暂停逻辑 (_paused flag + lock) 和清理逻辑 (clear the runtime) 是两层

### Layer 3: Mindflow — 已有实现够用, 补回调

BaseMindflow.pause() 当前实现基本正确:
- abort 当前 attention → 级联 `_wait_aborted_and_cancel`
- 堵住 impulse 消费循环 (`_unpaused_event`)
- 清空 signal 队列并重建

缺口: pause 期间 discard signal (而不仅仅是 defer)。如果未来需要 "pause 期间缓存 signal, resume 后消费", 需要改 `_on_signal_consuming_loop` 的 discard 逻辑。当前先保持丢弃行为。

### Layer 4: TUI — 去掉 toggle, 通过 callback 同步状态

```python
class MossHostTUI:
    def __init__(self, ...):
        self._paused = False  # 缓存, 真源在 GhostRuntime.is_paused()

    def _on_emergency_pause(self) -> None:
        """Ctrl+G: 调 runtime.pause() 翻转, callback 同步 TUI 缓存."""
        target = not self.runtime.is_paused()
        self.runtime.pause(target, self._on_pause_callback)

    def _on_pause_callback(self, paused: bool) -> None:
        """由 runtime.pause() 同步调用, 更新 TUI 缓存 + 强制刷新 toolbar."""
        self._paused = paused
        self.set_bottom_toolbar("[PAUSED]" if paused else "")
        # 强制 prompt 重绘以刷新 _prompt_status
        if self._prompt_session and self._prompt_session.app:
            self._prompt_session.app.invalidate()

    def _prompt_status(self) -> list[tuple[str, str]]:
        # 每帧渲染时从缓存读, 兜底校准
        if not self._paused and self.runtime and self.runtime.is_paused():
            self._paused = True  # 兜底: callback 丢了也能校准
        if self._paused:
            return [("fg:red bold", "[PAUSED] ")]
        return []
```

关键设计决策:
- TUI 不再 toggle — 调 `runtime.pause(target, callback)` 设值
- `_on_pause_callback` 更新 `self._paused` 缓存 + 强制 invalidate prompt
- 兜底校准: `_prompt_status` 每次渲染时交叉校验 `runtime.is_paused()`, callback 丢了也不漂移
- toolbar 在 callback 中直接更新, 不等下次渲染周期

### Layer 5: GhostInspector REPL — 走统一入口

```python
class GhostInspector:
    def pause(self) -> None:
        """REPL /ghost.pause() — 走 GhostRuntime 统一入口."""
        self._ghost_runtime.pause(True)

    def resume(self) -> None:
        self._ghost_runtime.pause(False)
```

### 三循环暂停机制

三循环入口检查 `is_paused()`, 暂停时阻塞等待:

```python
async def _main_loop(self) -> None:
    while True:
        if self.is_paused():
            await self._unpaused_event.wait()
        ...

async def _articulate_loop(self) -> None:
    while mindflow.is_running():
        if self.is_paused():
            await self._unpaused_event.wait()
            continue
        articulator = await self._articulate_queue.async_q.get()
        ...

async def _action_loop(self) -> None:
    while mindflow.is_running():
        if self.is_paused():
            await self._unpaused_event.wait()
            continue
        action = await self._action_queue.async_q.get()
        ...
```

`_unpaused_event` 在 `pause(False)` 时 set, `pause(True)` 时 clear。

### 实现顺序与单测

| 步骤 | 内容 | 单测 |
|------|------|------|
| 1 | Shell._pause() async + Lock | `test_shell_pause.py`: 验证 _pause() 返回后 interpreter 已停止, speech cleared |
| 2 | GhostRuntime.is_paused() + on_pause_changed() | `test_ghost_runtime_pause.py`: 幂等, 回调 fire, 状态查询 |
| 3 | GhostRuntime.pause() 集成 mindflow + shell | mock mindflow/shell, 验证级联调用顺序 |
| 4 | TUI 去掉 toggle, 订阅回调 | `test_tui_pause.py`: Ctrl+G → runtime.pause() 被调, [PAUSED] 显示正确 |
| 5 | 三循环入口暂停点 | `test_loop_pause.py`: pause 后 loop 阻塞, resume 后继续 |
| 6 | 端到端集成 | 手动验收: Ctrl+G → 暂停 → Ctrl+G → 恢复 → REPL /ghost.pause() 同样生效 |