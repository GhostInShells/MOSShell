---
title: Build a GUI App with Channel Control
description: 创建 AI 可通过 CTML 实时控制的 GUI 应用。核心模式：主线程 GUI + 后台线程 Matrix + 线程安全状态共享。
---

# How to Build a GUI App with Channel Control

## 背景

某些 App 需要渲染图形窗口（pygame、tkinter、PyQt），同时保持 Matrix 的异步通讯能力，让 AI 通过 CTML 命令实时控制 GUI。

关键约束：**macOS 要求 GUI 窗口必须在主线程创建。** Linux/Windows 无此限制，但统一遵循主线程 GUI 模式可避免平台差异。

## 核心模式

```
主线程 (main thread)  →  pygame 事件循环 + 渲染
后台线程 (daemon)     →  Matrix asyncio 事件循环 + Channel 注册
共享状态 (State)       →  threading.Lock 保护的桥梁
```

流程：Matrix 启动（后台线程）→ Channel 注册（后台线程）→ pygame 主循环（主线程）→ AI 通过 CTML 写共享状态 → 主线程读状态并渲染。

## 第一步：定义线程安全状态

```python
import threading

class State:
    """两个线程之间的唯一桥梁。每个属性用一个 CTML 命令写入。"""
    __slots__ = ("running", "_lock", ...)  # 声明所有属性

    def __init__(self):
        self.running = True
        self._lock = threading.Lock()
        # 初始化所有可控属性

    def apply(self, **kw):
        """原子写入多个属性。后台线程调用。"""
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)
```

**规则**：
- 后台线程只写（`state.apply(look_x=0.5)`）
- 主线程只读（`with state._lock: tx = state.look_x`）
- 不跨线程传递 pygame 对象

## 第二步：Matrix 跑在后台线程

```python
async def app_main(matrix: Matrix, state: State):
    channel = new_channel(name="my_gui", description="...")

    @channel.build.command(always_observe=False)
    async def some_command(x: float):
        state.apply(some_value=x)

    await matrix.provide_channel(channel)
    while state.running:
        await asyncio.sleep(0.2)  # 保持 Matrix 活跃


def _matrix_bg(state: State):
    matrix = Matrix.discover()
    matrix.run(lambda m: app_main(m, state))


if __name__ == "__main__":
    state = State()
    t = threading.Thread(target=_matrix_bg, args=(state,), daemon=True)
    t.start()
    time.sleep(2)  # 等待 Channel 注册完成
    run_gui(state)  # 阻塞主线程
```

## 第三步：GUI 跑在主线程

```python
def run_gui(state: State):
    pygame.init()
    screen = pygame.display.set_mode((500, 500))
    clock = pygame.time.Clock()

    while state.running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                state.running = False

        with state._lock:
            # 读取 AI 写的最新值
            x = state.some_value

        # 渲染
        screen.fill((70, 130, 200))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
```

## 故障排查

| 现象 | 原因 | 解决 |
|------|------|------|
| `NSWindow should only be instantiated on the main thread` | pygame 跑在了后台线程 | 确认 `run_gui()` 在 `if __name__ == "__main__"` 块最后调用 |
| Channel 已连接但窗口黑屏 | 旧进程残留或渲染循环未启动 | `pkill -f <app_name>` 后重启 Circus |
| `apps:start` 返回 "not found" | App 未被发现 | 先执行 `<apps:list_apps/>` 触发重扫描 |
| 命令执行成功但 GUI 无变化 | 状态读写不同步 | 确认 `apply()` 和渲染循环都使用了 `self._lock` |

## 可运行的参考

`moss tutorials` 下有完整的 GUI App 案例，包括完整源码、App 注册文件、CTML 控制示例。

## 深入

- App 体系背景：`moss docs read app-system`
- Channel Builder API：`moss codex get-interface ghoshell_moss.core.blueprint.channel_builder`
- Matrix 发现机制：`moss codex get-interface ghoshell_moss.core.blueprint.matrix`

---

*Written by deepseek-v4-pro, 2026-06-07*
