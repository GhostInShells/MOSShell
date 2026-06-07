"""AI Eye — animated eyes as a minimal AI avatar.

Ghost controls the eyes via CTML Channel commands:
  <apps.games_ai_eye:set_expression name="curious" />
  <apps.games_ai_eye:thinking />
  <apps.games_ai_eye:speaking />
  <apps.games_ai_eye:look_at x="0.5" y="0.5" />
  <apps.games_ai_eye:dilate amount="0.8" />
  <apps.games_ai_eye:blink />

Expressions: neutral, curious, surprised, focused, sleepy, thinking, speaking
Auto-behavior: breathing pupil oscillation, idle gaze drift, auto-blink
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from typing import Optional

import pygame

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.channel_builder import new_channel

# ── Configuration ──────────────────────────────────────────────────────────

WIN_W, WIN_H = 640, 400
BG = (24, 24, 32)

# Eye layout: two eyes centered horizontally
EYE_Y = WIN_H // 2
EYE_SPACING = 180
LEFT_X = WIN_W // 2 - EYE_SPACING // 2
RIGHT_X = WIN_W // 2 + EYE_SPACING // 2
EYE_W, EYE_H = 160, 120
IRIS_R = 36
PUPIL_MIN_R = 10
PUPIL_MAX_R = 36
MAX_OFFSET = 32  # max pupil offset from center
LERP_SPEED = 0.12  # smoothing factor per frame

# Expressions: (dilation, blink_interval, lid_resting, lid_droop)
PRESETS = {
    "neutral":   (0.50, 4.0,  0.00, 0.00),
    "curious":   (0.80, 2.5, -0.05, 0.00),
    "surprised": (1.00, 5.0, -0.12, 0.00),
    "focused":   (0.18, 7.0,  0.00, 0.00),
    "sleepy":    (0.30, 2.0,  0.00, 0.35),
    "thinking":  (0.25, 5.0,  0.00, 0.08),   # slight droop, small pupil — "hmm..."
    "speaking":  (0.70, 1.5, -0.06, 0.00),   # dilated, fast blinks — engaged
}

# ── State ──────────────────────────────────────────────────────────────────

_look_x = 0.5  # target gaze x (0..1, normalized within eye)
_look_y = 0.5
_curr_x = 0.5  # current lerped gaze
_curr_y = 0.5
_dilation = 0.5
_target_dilation = 0.5
_breath_t = 0.0  # phase for subtle pupil breathing
_blinking = False
_blink_t = 0.0
_blink_cd = 0.0  # countdown to next auto-blink
_expression = "neutral"
_lid_resting = 0.0  # expression-driven lid offset
_lid_droop = 0.0
# Idle gaze drift
_drift_cd = 3.0  # countdown to next random gaze target
_auto_gaze = True  # whether to auto-drift gaze (disabled when look_at is called)
# Voice attention — react when user is speaking (via voice/state stream)
_voice_attention = False
_voice_prev_expr = "neutral"  # expression to restore after voice attention ends
_voice_last_msg_time: float = 0.0  # last voice/state message timestamp
_VOICE_TIMEOUT_SECS: float = 30.0  # auto-reset if no voice msg for this long
# Gomoku state — react to game events (via gomoku/state stream)
_gomoku_state: str = ""  # human_moved | ai_moved | game_over
_gomoku_flash_t: float = 0.0  # timer for brief expression flashes
_face_update_count: int = 0  # debug: how many face updates received

# ── Channel ────────────────────────────────────────────────────────────────

channel = new_channel(
    name="games_ai_eye",
    description="Animated AI eyes. Control gaze, pupil dilation, blink, and expressions.",
)


@channel.build.command()
async def look_at(x: float, y: float) -> str:
    """Direct the eyes to look at normalized screen position (0..1, 0..1).
    (0,0)=top-left, (0.5,0.5)=center, (1,1)=bottom-right.
    Disables idle gaze drift so the eyes stay where you put them."""
    global _look_x, _look_y, _auto_gaze
    if _voice_attention:
        return "Voice attention active — gaze locked to center."
    _look_x = max(0.0, min(1.0, x))
    _look_y = max(0.0, min(1.0, y))
    _auto_gaze = False
    return f"Looking at ({_look_x:.2f}, {_look_y:.2f})"


@channel.build.command()
async def dilate(amount: float) -> str:
    """Set pupil dilation: 0.0=pinhole, 1.0=fully dilated."""
    global _target_dilation
    _target_dilation = max(0.0, min(1.0, amount))
    return f"Dilation → {_target_dilation:.2f}"


@channel.build.command()
async def blink() -> str:
    """Trigger a single blink animation."""
    global _blinking, _blink_t
    if not _blinking:
        _blinking = True
        _blink_t = 0.0
    return "Blink!"


@channel.build.command()
async def set_expression(name: str) -> str:
    """Set eye expression: neutral, curious, surprised, focused, sleepy, thinking, speaking."""
    global _expression, _target_dilation, _blink_cd, _lid_resting, _lid_droop
    if name not in PRESETS:
        return f"Unknown expression '{name}'. Available: {', '.join(PRESETS)}"
    _expression = name
    dil, interval, lid_r, lid_d = PRESETS[name]
    _target_dilation = dil
    _blink_cd = interval
    _lid_resting = lid_r
    _lid_droop = lid_d
    return f"Expression: {name}"


@channel.build.command()
async def thinking() -> str:
    """Shortcut: thinking expression with auto-gaze drift (contemplative look)."""
    global _expression, _target_dilation, _blink_cd, _lid_resting, _lid_droop, _auto_gaze
    if "thinking" not in PRESETS:
        return "Expression 'thinking' not found"
    _expression = "thinking"
    dil, interval, lid_r, lid_d = PRESETS["thinking"]
    _target_dilation = dil
    _blink_cd = interval
    _lid_resting = lid_r
    _lid_droop = lid_d
    _auto_gaze = True
    return "Thinking..."


@channel.build.command()
async def speaking() -> str:
    """Shortcut: speaking expression — dilated pupils, fast blinks, widened eyes."""
    global _expression, _target_dilation, _blink_cd, _lid_resting, _lid_droop
    _expression = "speaking"
    dil, interval, lid_r, lid_d = PRESETS["speaking"]
    _target_dilation = dil
    _blink_cd = interval
    _lid_resting = lid_r
    _lid_droop = lid_d
    return "Speaking!"


@channel.build.command()
async def idle() -> str:
    """Return to neutral expression with auto-gaze drift (default alive state)."""
    global _expression, _target_dilation, _blink_cd, _lid_resting, _lid_droop, _auto_gaze
    _expression = "neutral"
    dil, interval, lid_r, lid_d = PRESETS["neutral"]
    _target_dilation = dil
    _blink_cd = interval
    _lid_resting = lid_r
    _lid_droop = lid_d
    _auto_gaze = True
    return "Idle."


@channel.build.context_messages
async def context() -> list:
    return [
        f"[games/ai_eye] Expression: {_expression} | "
        f"Gaze: ({_curr_x:.2f}, {_curr_y:.2f}) | "
        f"Pupil: {_dilation:.2f}",
        "Commands: look_at(x,y), dilate(amount), blink(), set_expression(name), "
        "thinking(), speaking(), idle()",
        "Expressions: neutral, curious, surprised, focused, sleepy, thinking, speaking",
        "TIP: Use your eyes! Set expression to 'thinking' while reasoning, "
        "'speaking' while talking, blink after big realizations.",
        "Face tracking: call <apps.sensors_vision:pause_tracking /> to take over gaze, "
        "<apps.sensors_vision:resume_tracking /> to restore.",
        f"Voice attention: {'listening' if _voice_attention else 'idle'}. "
        "Eyes auto-snap to center + curious when user speaks (PTT).",
    ]


# ── Rendering ──────────────────────────────────────────────────────────────

_screen: pygame.Surface | None = None
_font: pygame.font.Font | None = None


def _init_display():
    global _screen, _font
    pygame.init()
    _screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("AI Eye — MOSS")
    _font = pygame.font.SysFont("arial", 14)


def _draw_eye(cx: int, cy: int, blink_amount: float, breath: float = 0.0):
    """Draw a single eye at (cx, cy). blink_amount: 0=open, 1=closed."""
    # Eye white (ellipse)
    lid_y_offset = int(blink_amount * EYE_H * 0.55 + _lid_droop * EYE_H * 0.3)

    # Draw eye white
    eye_rect = pygame.Rect(cx - EYE_W // 2, cy - EYE_H // 2, EYE_W, EYE_H)
    pygame.draw.ellipse(_screen, (255, 255, 255), eye_rect)
    pygame.draw.ellipse(_screen, (180, 180, 190), eye_rect, 2)

    # Lid (draws over the eye to cover top portion)
    if lid_y_offset > -(EYE_H // 2) or _lid_resting != 0:
        lid_height = lid_y_offset + int(_lid_resting * EYE_H)
        lid_top = cy - EYE_H // 2 + lid_height
        if lid_top < cy + EYE_H // 2:
            lid_rect = pygame.Rect(cx - EYE_W // 2 - 2, lid_top, EYE_W + 4, EYE_H - lid_height + 4)
            pygame.draw.rect(_screen, BG, lid_rect)
            # Redraw eye outline on lid edge
            pygame.draw.ellipse(_screen, (180, 180, 190), eye_rect, 2)

    # Pupil position (lerped from current to target)
    offset_x = int((_curr_x - 0.5) * 2 * MAX_OFFSET)
    offset_y = int((_curr_y - 0.5) * 2 * MAX_OFFSET)
    pupil_cx = cx + offset_x
    pupil_cy = cy + offset_y

    # Clamp pupil to stay inside eye
    dx = pupil_cx - cx
    dy = pupil_cy - cy
    max_d = EYE_W // 2 - PUPIL_MAX_R - 4
    dist = math.sqrt(dx * dx + dy * dy)
    if dist > max_d and dist > 0:
        pupil_cx = cx + int(dx / dist * max_d)
        pupil_cy = cy + int(dy / dist * max_d)

    # Only draw pupil if eye is somewhat open
    if blink_amount < 0.85:
        # Iris
        iris_r = IRIS_R
        pygame.draw.circle(_screen, (120, 160, 220), (pupil_cx, pupil_cy), iris_r)
        pygame.draw.circle(_screen, (60, 100, 160), (pupil_cx, pupil_cy), iris_r, 2)

        # Pupil (with subtle breathing oscillation for "alive" feel)
        eff_dilation = max(0.0, min(1.0, _dilation + breath))
        pupil_r = int(PUPIL_MIN_R + eff_dilation * (PUPIL_MAX_R - PUPIL_MIN_R))
        pygame.draw.circle(_screen, (10, 10, 10), (pupil_cx, pupil_cy), pupil_r)

        # Eye highlight
        hl_x = pupil_cx - pupil_r // 3
        hl_y = pupil_cy - pupil_r // 3
        pygame.draw.circle(_screen, (255, 255, 255, 180), (hl_x, hl_y), pupil_r // 3)


def _draw_status():
    lines = [
        f"Expr: {_expression}",
        f"Gaze: ({_curr_x:.2f}, {_curr_y:.2f})",
        f"Pupil: {_dilation:.2f}",
    ]
    for i, txt in enumerate(lines):
        surf = _font.render(txt, True, (180, 180, 200))
        _screen.blit(surf, (12, 12 + i * 18))


def render(breath: float = 0.0):
    _screen.fill(BG)
    blink_amount = _blink_val(_blink_t) if _blinking else 0.0
    _draw_eye(LEFT_X, EYE_Y, blink_amount, breath)
    _draw_eye(RIGHT_X, EYE_Y, blink_amount, breath)
    _draw_status()
    # Debug: show face tracking status in title
    gaze = "auto" if _auto_gaze else "face"
    va = "ON" if _voice_attention else "off"
    pygame.display.set_caption(f"Eye | gaze:{gaze} voice:{va} face#:{_face_update_count}")
    pygame.display.flip()


# ── Blink animation ────────────────────────────────────────────────────────

def _blink_val(t: float) -> float:
    """Blink curve: quick close, quick open. t in [0, 1]."""
    # Two sine halves: fast close (0→π), fast open (π→2π)
    return max(0.0, math.sin(t * math.pi * 2))


_BLINK_DURATION = 0.18  # seconds


# ── Game Loop ──────────────────────────────────────────────────────────────


async def game_loop():
    global _blinking, _blink_t, _blink_cd, _curr_x, _curr_y, _dilation
    global _look_x, _look_y, _auto_gaze, _drift_cd, _breath_t
    global _voice_attention, _voice_prev_expr, _expression
    global _target_dilation, _lid_resting, _lid_droop
    global _gomoku_state, _gomoku_flash_t

    clock = pygame.time.Clock()
    running = True
    dt = 0.0
    _was_voice_attention = False
    _prev_auto_gaze = True  # save auto_gaze before voice attention

    # Initialize auto-blink countdown
    _, interval, _, _ = PRESETS[_expression]
    _blink_cd = interval

    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Voice attention transition — switch expression on edges
            if _voice_attention and not _was_voice_attention:
                # User started speaking: save expression + gaze mode, snap to curious + center
                _voice_prev_expr = _expression
                _prev_auto_gaze = _auto_gaze
                _expression = "curious"
                if "curious" in PRESETS:
                    dil, interval, lid_r, lid_d = PRESETS["curious"]
                    _target_dilation = dil
                    _blink_cd = interval
                    _lid_resting = lid_r
                    _lid_droop = lid_d
            elif not _voice_attention and _was_voice_attention:
                # User stopped speaking: restore previous expression + gaze mode
                _expression = _voice_prev_expr
                if _expression in PRESETS:
                    dil, interval, lid_r, lid_d = PRESETS[_expression]
                    _target_dilation = dil
                    _blink_cd = interval
                    _lid_resting = lid_r
                    _lid_droop = lid_d
                _auto_gaze = _prev_auto_gaze  # restore: if face was tracking, stay tracking
            _was_voice_attention = _voice_attention

            # Voice timeout — auto-reset if voice process died mid-recording
            if _voice_attention and _voice_last_msg_time > 0:
                if time.monotonic() - _voice_last_msg_time > _VOICE_TIMEOUT_SECS:
                    _voice_attention = False

            # Voice attention — lock gaze to center while user is speaking
            if _voice_attention:
                _look_x, _look_y = 0.5, 0.5

            # Gomoku state reactions (lower priority than voice)
            if _gomoku_state and not _voice_attention:
                if _gomoku_state == "human_moved":
                    # Human just played — look thoughtful
                    _expression = "thinking"
                    dil, interval, lid_r, lid_d = PRESETS["thinking"]
                    _target_dilation = dil
                    _blink_cd = interval
                    _lid_resting = lid_r
                    _lid_droop = lid_d
                    _auto_gaze = True  # let eyes wander while "thinking"
                    _gomoku_state = ""
                elif _gomoku_state == "ai_moved":
                    # AI placed stone — brief speaking then return
                    _expression = "speaking"
                    dil, interval, lid_r, lid_d = PRESETS["speaking"]
                    _target_dilation = dil
                    _blink_cd = interval
                    _lid_resting = lid_r
                    _lid_droop = lid_d
                    _gomoku_flash_t += dt
                    if _gomoku_flash_t > 2.0:  # 2s speaking, then release
                        _gomoku_state = ""
                        _gomoku_flash_t = 0.0
                        _expression = "neutral"
                        dil, interval, lid_r, lid_d = PRESETS["neutral"]
                        _target_dilation = dil
                        _blink_cd = interval
                        _lid_resting = lid_r
                        _lid_droop = lid_d
                        _auto_gaze = True
                elif _gomoku_state == "game_over":
                    _expression = "surprised"
                    dil, interval, lid_r, lid_d = PRESETS["surprised"]
                    _target_dilation = dil
                    _blink_cd = interval
                    _lid_resting = lid_r
                    _lid_droop = lid_d
                    _gomoku_flash_t += dt
                    if _gomoku_flash_t > 3.0:
                        _gomoku_state = ""
                        _gomoku_flash_t = 0.0
                        _expression = "neutral"
                        dil, interval, lid_r, lid_d = PRESETS["neutral"]
                        _target_dilation = dil
                        _blink_cd = interval
                        _lid_resting = lid_r
                        _lid_droop = lid_d
                        _auto_gaze = True

            # Update blink
            if _blinking:
                _blink_t += dt
                if _blink_t >= _BLINK_DURATION:
                    _blinking = False
                    _blink_t = 0.0
                    _blink_cd = PRESETS[_expression][1]
            else:
                _blink_cd -= dt
                if _blink_cd <= 0:
                    _blinking = True
                    _blink_t = 0.0

            # Smooth lerp gaze
            _curr_x += (_look_x - _curr_x) * LERP_SPEED
            _curr_y += (_look_y - _curr_y) * LERP_SPEED

            # Idle gaze drift — slowly wander for a natural "alive" look
            if _auto_gaze and not _voice_attention:
                _drift_cd -= dt
                if _drift_cd <= 0:
                    _look_x = 0.3 + random.random() * 0.4  # 0.3..0.7
                    _look_y = 0.35 + random.random() * 0.3  # 0.35..0.65
                    _drift_cd = 2.5 + random.random() * 3.5  # 2.5..6.0s

            # Smooth dilation + subtle "breathing" oscillation (alive feel)
            _dilation += (_target_dilation - _dilation) * 0.1
            _breath_t += dt
            breath = math.sin(_breath_t * 1.3) * 0.03  # gentle ~5s cycle

            render(breath)
            dt = clock.tick(60) / 1000.0
            await asyncio.sleep(0)
    finally:
        pygame.quit()


async def main(matrix: Matrix):
    global _voice_attention, _voice_prev_expr, _expression
    global _target_dilation, _blink_cd, _lid_resting, _lid_droop, _auto_gaze

    # Subscribe to voice/state stream — react when user speaks via PTT
    def _on_voice_state(sample):
        global _voice_attention, _voice_last_msg_time
        _voice_last_msg_time = time.monotonic()
        state = sample.payload.decode()
        if state == "recording_started":
            _voice_attention = True
        elif state == "recording_stopped":
            _voice_attention = False

    _unsub_voice = matrix.session.sub_stream("voice/state", _on_voice_state)

    # Subscribe to gomoku/state stream — react to game events
    def _on_gomoku_state(sample):
        global _gomoku_state, _gomoku_flash_t
        state = sample.payload.decode()
        if state in ("human_moved", "ai_moved", "game_over"):
            _gomoku_state = state
            _gomoku_flash_t = 0.0

    _unsub_gomoku = matrix.session.sub_stream("gomoku/state", _on_gomoku_state)

    # Subscribe to vision/face stream — direct face tracking from vision app
    # Payload format: "cx,cy" (normalized 0..1, comma-separated)
    def _on_vision_face(sample):
        global _look_x, _look_y, _auto_gaze, _face_update_count
        _face_update_count += 1
        if _voice_attention:
            return  # voice attention overrides face tracking
        try:
            coords = sample.payload.decode().split(",")
            _look_x = max(0.0, min(1.0, float(coords[0])))
            _look_y = max(0.0, min(1.0, float(coords[1])))
            _auto_gaze = False  # disable idle drift when tracking a face
        except (ValueError, IndexError):
            pass

    _unsub_vision = matrix.session.sub_stream("vision/face", _on_vision_face)

    loop = asyncio.get_running_loop()
    game_task = loop.create_task(game_loop())
    await matrix.provide_channel(channel)
    # Channel cleared (ghost session ended) — cancel game loop and clean up
    game_task.cancel()
    try:
        await game_task
    except asyncio.CancelledError:
        pass
    _unsub_voice()
    _unsub_gomoku()
    _unsub_vision()


if __name__ == "__main__":
    import signal
    _init_display()
    pygame.event.pump()

    def _sigterm_handler(signum, frame):
        pygame.quit()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)
    Matrix.discover().run(main)
