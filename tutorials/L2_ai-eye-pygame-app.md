# L2. AI Eye — 从零到 AI 实时控制的 Pygame 图形应用

> Written by deepseek-v4-pro, 2026-06-07

**45 分钟，创建一个 AI 可通过 CTML 实时控制的 Pygame 眼球应用。核心学习点：GUI 线程分离、Channel 状态共享、MOSS App 注册与调试。**

## 你要做什么

在 MOSS 中创建一个 `games/ai_eye` App——独立的 Pygame 窗口，渲染一只会动的眼睛。AI 通过 CTML 命令实时控制眼球注视方向、瞳孔大小、眨眼和表情。

完成后，AI 可以在 MCP 会话中直接输出 `<apps.games_ai_eye:look_at x="0.8" y="0.2"/>`，眼睛就会看向右上角。

## 你需要什么

- MOSS 已安装 (`.venv/bin/moss` 可用)
- MOSS 运行时在跑 (MCP 或 REPL)，带语音配置
- macOS 用户注意：本 tutorial 包含 macOS 特化处理

## 第一步：创建 App 骨架

```bash
mkdir -p .moss_ws/apps/games/ai_eye
```

创建三个文件：

**`APP.md`**（元信息声明——App 发现的入口）：

```yaml
---
arguments: ''
description: 'AI Eye — controllable pygame eye avatar with gaze, dilation, blink, and expressions'
executable: uv
respawn: false
script: main.py
workers: 1
---

AI Eye — a minimal pygame window rendering an expressive eye that AI controls via Channel commands.
```

**`pyproject.toml`**（依赖隔离）：

```toml
[project]
name = "ai-eye"
version = "0.1.0"
requires-python = ">=3.11,<3.14"
dependencies = [
    "ghoshell-moss[host]",
    "pygame>=2.5.0",
]

[tool.uv.sources]
ghoshell-moss = { path = "../../../..", editable = true }
```

## 第二步：写代码

**核心约束：macOS 要求 GUI 窗口必须在主线程创建。** 因此 pygame 跑在主线程，Matrix 的 asyncio 循环跑在后台线程。两者通过 `threading.Lock` 保护的共享状态通讯。

**`main.py`**：

```python
"""AI Eye — pygame on main thread (macOS), Matrix asyncio in background thread."""

import asyncio
import math
import os
import threading
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.channel_builder import new_channel


class State:
    """Thread-safe shared state between asyncio (Matrix) and pygame (main thread)."""
    __slots__ = ("look_x", "look_y", "dilation", "expression",
                 "blink_requested", "running", "_lock")

    def __init__(self):
        self.look_x = 0.5
        self.look_y = 0.5
        self.dilation = 0.5
        self.expression = "neutral"
        self.blink_requested = False
        self.running = True
        self._lock = threading.Lock()

    def apply(self, **kw):
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)


def run_pygame(state: State, width: int = 500, height: int = 500):
    """MUST run on macOS main thread. Owns the pygame window."""
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("AI Eye — MOSS")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)
    cx, cy = width // 2, height // 2
    eye_r = min(width, height) // 3
    cur_x, cur_y = float(cx), float(cy)
    blink_open, blink_phase = 1.0, 0.0
    blinking = False
    cur_dil = 0.5

    while state.running:
        dt = max(clock.get_time() / 1000.0, 0.001)
        with state._lock:
            tx, ty = state.look_x * width, state.look_y * height
            dil, expr = state.dilation, state.expression
            if state.blink_requested:
                blinking = True; blink_phase = 0.0
                state.blink_requested = False

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (
                ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE
            ):
                state.running = False

        # Smooth gaze follow
        cur_x += (tx - cur_x) * min(8.0 * dt, 1.0)
        cur_y += (ty - cur_y) * min(8.0 * dt, 1.0)
        cur_dil += (dil - cur_dil) * min(10.0 * dt, 1.0)

        # Blink animation
        if blinking:
            blink_phase += dt * 6
            if blink_phase >= 2.0:
                blinking = False; blink_phase = 0.0; blink_open = 1.0
            else:
                blink_open = 0.5 + 0.5 * math.cos(blink_phase * math.pi)

        # Draw
        screen.fill((70, 130, 200))
        # Eye white
        pygame.draw.circle(screen, (255, 255, 255), (cx, cy), eye_r)
        pygame.draw.circle(screen, (100, 100, 120), (cx, cy), eye_r, 2)
        # Pupil
        dx, dy = cur_x - cx, cur_y - cy
        d = math.sqrt(dx * dx + dy * dy)
        max_off = eye_r * 0.35
        if d > max_off:
            dx, dy = dx / d * max_off, dy / d * max_off
        px, py = cx + dx, cy + dy
        pr = eye_r * 0.15 + cur_dil * eye_r * 0.35
        pygame.draw.circle(screen, (20, 20, 30), (int(px), int(py)), int(pr))
        # Highlight
        hl = int(pr * 0.35)
        pygame.draw.circle(screen, (255, 255, 255),
                          (int(px - pr * 0.25), int(py - pr * 0.3)), hl)
        # Eyelid during blink
        if blink_open < 1.0:
            lid_h = eye_r * 2 * (1.0 - blink_open)
            lid_y = cy - eye_r - lid_h // 2
            pygame.draw.ellipse(screen, (70, 130, 200),
                               pygame.Rect(cx - eye_r, lid_y, eye_r * 2, lid_h))
        # Expression label
        labels = {"neutral": "neutral", "curious": "curious",
                  "surprised": "surprised!", "focused": "focused",
                  "sleepy": "sleepy..."}
        txt = font.render(labels.get(expr, expr), True, (255, 255, 255))
        screen.blit(txt, (10, height - 35))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


async def app_main(matrix: Matrix, state: State):
    """Runs inside Matrix's asyncio loop (background thread)."""
    channel = new_channel(
        name="ai_eye",
        description="AI Eye — controllable pygame eye with gaze, dilation, blink, expressions",
    )

    @channel.build.close
    async def close():
        state.running = False

    @channel.build.command(always_observe=False)
    async def look_at(x: float, y: float):
        """注视屏幕坐标，眼球平滑跟随。x, y 范围 0.0-1.0"""
        state.apply(look_x=x, look_y=y)

    @channel.build.command(always_observe=False)
    async def dilate(amount: float):
        """瞳孔缩放。0.0=针尖, 0.5=正常, 1.0=最大"""
        state.apply(dilation=max(0.0, min(1.0, amount)))

    @channel.build.command(always_observe=False)
    async def blink():
        """眨一次眼"""
        state.apply(blink_requested=True)

    @channel.build.command(always_observe=False)
    async def set_expression(name: str):
        """设置表情: neutral/curious/surprised/focused/sleepy"""
        if name in {"neutral", "curious", "surprised", "focused", "sleepy"}:
            state.apply(expression=name)

    await matrix.provide_channel(channel)
    print("AI Eye channel registered", flush=True)

    # Keep matrix alive while pygame runs
    while state.running:
        await asyncio.sleep(0.2)


def _matrix_bg(state: State):
    """Background thread entry — Matrix's own asyncio loop."""
    matrix = Matrix.discover()
    matrix.run(lambda m: app_main(m, state))


if __name__ == "__main__":
    state = State()
    # 1) Matrix asyncio in background thread
    t = threading.Thread(target=_matrix_bg, args=(state,), daemon=True)
    t.start()
    time.sleep(2)  # Give Matrix time to boot + register channel
    # 2) Pygame on MAIN thread (macOS requires this)
    run_pygame(state)
```

**关键设计决策**：

| 决策 | 原因 |
|------|------|
| Pygame 跑在主线程 | macOS `NSWindow` 限制 |
| Matrix asyncio 跑在后台线程 | 与 pygame 不冲突 |
| `State` 类用 `threading.Lock` | 两个线程安全的唯一桥梁 |
| `time.sleep(2)` 等待 Matrix 启动 | 确保 channel 已注册再进入渲染循环 |
| `@channel.build.command(always_observe=False)` | 眼球命令不需要打断其他任务 |

## 第三步：刷新发现并启动

```bash
# 在 MCP 会话中，或通过 moss-repl
```

CTML：
```xml
<apps:list_apps/>
```

确认 `games/ai_eye` 出现在列表中后启动：

```xml
<apps:start fullname="games/ai_eye" timeout="15.0"/>
```

启动后通过 `get_moss_dynamic_info` 确认 `apps.games_ai_eye` Channel 出现，且状态为 `[RUNNING]`。

## 第四步：控制眼球

```xml
<!-- 看向左上角，瞳孔放大，表情惊讶 -->
<apps.games_ai_eye:look_at x:float="0.1" y:float="0.1"/>
<apps.games_ai_eye:dilate amount:float="0.9"/>
<apps.games_ai_eye:set_expression name="surprised"/>

<!-- 眨两次眼，看向右下，瞳孔缩小，表情专注 -->
<apps.games_ai_eye:blink/>
<apps.games_ai_eye:blink/>
<apps.games_ai_eye:look_at x:float="0.9" y:float="0.9"/>
<apps.games_ai_eye:dilate amount:float="0.1"/>
<apps.games_ai_eye:set_expression name="focused"/>

<!-- 回到中间，瞳孔正常，表情平静 -->
<apps.games_ai_eye:look_at x:float="0.5" y:float="0.5"/>
<apps.games_ai_eye:dilate amount:float="0.5"/>
<apps.games_ai_eye:set_expression name="neutral"/>
```

## 故障排除

| 现象 | 原因 | 解决 |
|------|------|------|
| 窗口全黑 | pygame 没在主线程 | 确认 `if __name__ == "__main__"` 块中 `run_pygame()` 在最后调用 |
| `NSWindow should only be instantiated on the main thread` | 同上 | 同上 |
| Channel 连接但无窗口 | 旧进程残留 | `pkill -f ai_eye` 后重启 |
| `apps:start` 返回 "not found" | App 未被发现 | 先运行 `<apps:list_apps/>` 触发重新扫描 |
| 窗口标题为 "pygame window" | `pygame.display.set_caption()` 未执行 | 确认 `pygame.init()` 和 `set_mode` 在主线程 |

## 验证记录

- **2026-06-07, deepseek-v4-pro**: 完成完整链路验证。创建 App → Circus 启动 → Channel 注册 → CTML 控制 `look_at` / `dilate` / `blink` / `set_expression` 全部通过。踩坑记录：macOS 线程约束（三次迭代才找到正确分离方式）、旧进程清理问题（需要 `pkill`）。

## 深入理解

- Channel 体系: `moss codex get-interface ghoshell_moss.core.blueprint.channel_builder`
- Matrix 发现: `moss codex get-interface ghoshell_moss.core.blueprint.matrix`
- App 体系: `moss docs read app-system`
