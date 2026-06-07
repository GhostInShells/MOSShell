# Reference source for L2 AI Eye tutorial. NOT meant to be run from this directory.
# Copy to .moss_ws/apps/games/ai_eye/main.py — the App directory created in Step 1.
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
                blinking = True
                blink_phase = 0.0
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
                blinking = False
                blink_phase = 0.0
                blink_open = 1.0
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
