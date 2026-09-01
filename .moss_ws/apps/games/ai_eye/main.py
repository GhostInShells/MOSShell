"""AI Eye — animated eyes as a minimal AI avatar.

Floating on the desktop: a frameless, transparent PySide6 window showing
only two eyes. The eye drawing is done with pygame onto an offscreen
SRCALPHA surface, then blitted into the Qt window each frame. Drag the
window body to move it; drag the right/bottom edges or bottom-right corner
to resize (eyes scale with the window). Press ESC to quit.

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
import sys
import time
from typing import Optional

import pygame
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication, QWidget

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.channel_builder import new_channel

# ── Configuration ──────────────────────────────────────────────────────────

# Reference layout — actual layout scales from the live window size in relayout().
# The window tightly hugs the eyes (small margin for the resize grip + soft
# shadow), so the resize handle sits right at the eye's corner — easy to find.
# Wide-and-short default matches two side-by-side eyes (each wider than tall),
# so the bounding box stays tight without forcing round, tall eyes.
BASE_W, BASE_H = 460, 200
MIN_W, MIN_H = 190, 130
LERP_SPEED = 0.18  # smoothing factor per frame (snappy so gaze motion reads)
GRIP = 8  # px from the right/bottom edge that counts as a resize handle
# Transparent padding around the eyes: keeps the grip + soft shadow inside the
# window. Must be >= GRIP so the resize handle never overlaps the eye.
_EYE_MARGIN_X = 14
_EYE_MARGIN_Y = 12
# Supersample factor: render at N× then smoothscale down → anti-aliased curves.
_SS = 2

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

# ── Live layout (recomputed by relayout on resize) ─────────────────────────

_win_w, _win_h = BASE_W, BASE_H
_eye_y = BASE_H // 2
_eye_spacing = 180
_left_x = BASE_W // 2 - _eye_spacing // 2
_right_x = BASE_W // 2 + _eye_spacing // 2
_eye_w, _eye_h = 160, 120
_iris_r = 36
_pupil_min_r = 10
_pupil_max_r = 36
_max_offset = 32  # max pupil offset from center


def relayout(w: int, h: int) -> None:
    """Recompute eye layout so the eyes fill the window (tight bounding).

    The window is meant to be about the size of the two eyes, so eye geometry
    is derived directly from the window size minus a small margin (for the
    resize grip + soft shadow), not from a fixed base resolution."""
    global _win_w, _win_h, _eye_y, _eye_spacing, _left_x, _right_x
    global _eye_w, _eye_h, _iris_r, _pupil_min_r, _pupil_max_r, _max_offset
    _win_w, _win_h = w, h
    avail_w = max(60, w - 2 * _EYE_MARGIN_X)
    gap = max(14, int(avail_w * 0.10))
    _eye_w = max(40, (avail_w - gap) // 2)
    # A natural female eye: a smooth almond, clearly wider than tall (~1.55:1).
    # _eye_h is the eye opening height (upper peak → lower trough).
    _eye_h = max(28, min(int(h * 0.66), int(_eye_w * 0.64)))
    _eye_spacing = _eye_w + gap
    # Vertically place the eye block so the upper lashes have headroom above and
    # the ground shadow clears the bottom. The eye opening is split up_h(0.54)
    # above the corner line and lo_h(0.46) below; lashes add ~0.30·eye_h above.
    block_h = 1.45 * _eye_h
    voff = max(0.0, (h - block_h) / 2.0)
    _eye_y = int(voff + 0.84 * _eye_h)   # the corner line (y=cy in _eye_outline)
    _left_x = _EYE_MARGIN_X + _eye_w // 2
    _right_x = w - _EYE_MARGIN_X - _eye_w // 2
    # Iris (blue circle-lens). It's cropped to the eye shape by a mask, so it can
    # be large and pretty — the lids cover its top/bottom like a real eye — while
    # still sliding visibly as gaze moves.
    iris = int(_eye_h * 0.40)
    _iris_r = max(8, iris)
    # Pupil radius stays a moderate fraction of the iris — not a near-full black
    # pool. Neutral dilation (~0.5) → pupil ≈ 0.43×iris, natural-looking.
    _pupil_max_r = max(6, int(_iris_r * 0.52))
    _pupil_min_r = max(4, int(_iris_r * 0.24))
    _max_offset = max(6, _eye_w // 2 - _iris_r - 4)


# ── State ──────────────────────────────────────────────────────────────────

_look_x = 0.5  # target gaze x (0..1, normalized within eye)
_look_y = 0.5
_curr_x = 0.5  # current lerped gaze
_curr_y = 0.5
_dilation = 0.5
_target_dilation = 0.5
_breath_t = 0.0  # phase for subtle pupil breathing
_breath = 0.0    # current breath delta applied to dilation (set in game_loop)
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


# ── Rendering (pygame onto an offscreen SRCALPHA surface) ──────────────────

def _lerp_color(c1, c2, t):
    """Linear interpolate two RGB(A) colors. t clamped to [0, 1]."""
    t = max(0.0, min(1.0, t))
    out = tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(len(c1)))
    if len(out) == 3:
        out = (*out, 255)
    return out


def _radial_gradient(surface, center, rmax, inner, outer, steps=16):
    """Filled circle fading from `inner` at the center to `outer` at the edge."""
    cx, cy = center
    for i in range(steps, 0, -1):
        t = i / steps  # 1 at the edge → 0 at the center
        r = max(1, int(rmax * t))
        pygame.draw.circle(surface, _lerp_color(inner, outer, t), (cx, cy), r)


def _soft_shadow(surface, cx, cy, w, h, layers=9, max_alpha=44, expand=0.28):
    """Soft dark blob: faint & large on the outside, denser toward the center.
    Reads as a gentle contact shadow that grounds the eye on the desktop."""
    for i in range(layers - 1, -1, -1):
        t = i / max(1, layers - 1)  # 1 outer → 0 inner
        a = int(max_alpha * (1.0 - t) * 0.7 + max_alpha * 0.15)
        ew = int(w * (1 + expand * t))
        eh = int(h * (1 + expand * t))
        rect = pygame.Rect(cx - ew // 2, cy - eh // 2, ew, eh)
        pygame.draw.ellipse(surface, (0, 0, 0, a), rect)


# ── Eye palette (watery blue circle-lens / 美瞳) ────────────────────────────
# Sclera is a cool, faintly blue white — watery, not stark. The iris is a big
# bright blue disk (circle-lens look); a big colored disk sliding in the sclera
# makes gaze motion very obvious.
# Soft sclera (eye-white): a gentle top→bottom gradient — slightly cool and
# shaded under the upper lid, brightening toward the lower lid — plus a faint
# warm blush translucently pooled at the corners. Not flat, not pure white.
SCLERA = (240, 244, 250, 255)         # base (used as a fallback)
SCLERA_TOP = (224, 232, 244, 255)     # softly shaded/cool — under the upper lid
SCLERA_BOT = (250, 252, 255, 255)     # bright, near-white — toward the lower lid
SCLERA_BLUSH = (252, 232, 226, 38)    # faint warm tint at the corners (translucent)
LASH = (28, 24, 34, 255)              # upper lash line + lashes (dark, cool)
LOWER_LID = (150, 168, 184, 210)
CREASE = (190, 206, 222, 120)         # double-eyelid crease above the upper lid
IRIS_LIMBAL = (20, 44, 92, 255)       # dark navy limbal ring
IRIS_OUTER = (46, 104, 176, 255)      # deep blue iris edge
IRIS_INNER = (120, 192, 236, 255)     # bright watery blue center
IRIS_COLLAR = (40, 96, 168, 255)      # darker blue ring around the pupil
PUPIL = (10, 14, 20, 255)


def _eye_outline(cx: float, cy: float, hw: float, up_h: float, lo_h: float,
                 n: int = 36):
    """A natural female eye outline: a smooth lens/almond built from two
    quadratic Bézier arcs. Both corners sit level on the line y=cy; the upper
    lid is a graceful arch (peaks at cy-up_h), the lower lid a shallower curve
    (troughs at cy+lo_h). `up_h`/`lo_h` scale with openness so blinking closes
    the eye to a slit. Returns (upper_pts, lower_pts), left → right."""
    inner = (cx - hw, cy)
    outer = (cx + hw, cy)
    upper, lower = [], []
    for i in range(n + 1):
        t = i / n
        m = 4 * t * (1 - t)             # bulge: 0 → 1 → 0, peaks at 1.0 (t=0.5)
        x = inner[0] * (1 - t) + outer[0] * t
        upper.append((x, cy - up_h * m))
        lower.append((x, cy + lo_h * m))
    return upper, lower


def _draw_lash(surface, p0, ang: float, length: float, base_w: float, curl: float):
    """One tapered eyelash: a triangle from `p0` (on the lid) outward at angle
    `ang`, curling by `curl` radians toward the tip."""
    dx, dy = math.cos(ang), math.sin(ang)
    nx, ny = -dy, dx  # perpendicular (for the tapered base)
    p1 = (p0[0] + dx * length * 0.55, p0[1] + dy * length * 0.55)
    cdx, cdy = math.cos(ang + curl), math.sin(ang + curl)
    p2 = (p1[0] + cdx * length * 0.55, p1[1] + cdy * length * 0.55)
    a = (p0[0] + nx * base_w / 2, p0[1] + ny * base_w / 2)
    b = (p0[0] - nx * base_w / 2, p0[1] - ny * base_w / 2)
    pygame.draw.polygon(surface, LASH, [a, b, p2])


def _draw_eye(surface: pygame.Surface, cx: int, cy: int,
              blink_amount: float, breath: float = 0.0, ss: int = 1):
    """Draw a single feminine cartoon eye at (cx, cy) onto `surface`.

    cx, cy are in the surface's own (supersampled) coordinates; `ss` is the
    supersample factor — all eye geometry (sized via the layout globals) is
    multiplied by `ss` so the drawing matches the supersampled surface.
    blink_amount: 0 = open, 1 = closed."""
    ew = _eye_w * ss
    eh = _eye_h * ss
    hw = ew / 2.0

    # Soft contact shadow tucked under the eye — a thin ground halo so the eye
    # reads as floating just above the desktop. Kept tight (narrower than the
    # eye) so it never hazes the gap between the two eyes or the window edges.
    _soft_shadow(surface, cx, int(cy + eh * 0.30),
                 int(ew * 0.80), int(eh * 0.30),
                 layers=5, max_alpha=34, expand=0.10)

    # openness folds blink + expression droop into one arch-height factor.
    openness = 1.0 - blink_amount * 0.92 - _lid_droop * 0.45 - _lid_resting
    openness = max(0.06, min(1.0, openness))
    # Upper lid arches a bit more than the lower (natural female eye); both
    # scale with openness so blinking closes to a slit.
    up_h = eh * 0.54 * openness
    lo_h = eh * 0.46 * openness

    upper, lower = _eye_outline(cx, cy, hw, up_h, lo_h)
    eye_poly = upper + lower[::-1]

    # Render the eye interior (sclera + iris + pupil + highlights) onto a temp
    # surface, then crop it to the eye shape with a mask. This lets the iris be
    # large and pretty — the eyelids naturally cover its top and bottom, exactly
    # like a real eye — without the iris ever overflowing the lid outline.
    interior = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    # Soft sclera: a gentle vertical gradient (slightly cool/shaded under the
    # upper lid, brightening toward the lower lid). Not flat, not pure white.
    top_y = int(cy - up_h)
    bot_y = int(cy + lo_h)
    for y in range(top_y, bot_y + 1):
        f = (y - top_y) / max(1, bot_y - top_y)      # 0 at top → 1 at bottom
        col = _lerp_color(SCLERA_TOP, SCLERA_BOT, f)
        pygame.draw.line(interior, col, (int(cx - hw), y), (int(cx + hw), y))
    # Faint warm blush at the corners — drawn on its own surface so it blends
    # softly over the gradient instead of stamping a hard patch.
    blush = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    blush_r = int(hw * 0.5)
    for corner_x in (int(cx - hw * 0.7), int(cx + hw * 0.7)):
        pygame.draw.circle(blush, SCLERA_BLUSH, (corner_x, int(cy)), blush_r)
    interior.blit(blush, (0, 0))

    if blink_amount < 0.82:
        iris_r = _iris_r * ss
        icx = cx
        # Iris vertical center: roughly the opening mid, biased up a touch so the
        # upper lid covers a sliver of the iris (the pretty, alert look).
        icy = cy - int(eh * 0.05 * openness)
        # Horizontal gaze stays fully inside the eye; vertical can range wider
        # because the mask crops whatever the lids cover.
        max_ox = hw - iris_r * 0.55
        max_oy = max(lo_h, up_h) * 0.55
        ox = max(-max_ox, min(max_ox, (_curr_x - 0.5) * 2 * max_ox))
        oy = max(-max_oy, min(max_oy, (_curr_y - 0.5) * 2 * max_oy))
        px = int(icx + ox)
        py = int(icy + oy)

        # Limbal ring → blue gradient → collar around the pupil.
        pygame.draw.circle(interior, IRIS_LIMBAL, (px, py), iris_r)
        _radial_gradient(interior, (px, py), int(iris_r * 0.94),
                         inner=IRIS_INNER, outer=IRIS_OUTER, steps=22)
        # Pupil collar (a slightly darker blue ring tightening the pupil edge).
        pygame.draw.circle(interior, IRIS_COLLAR, (px, py),
                           int(iris_r * 0.55), max(1, int(2 * ss)))

        # Pupil (with a subtle breathing oscillation for an "alive" feel).
        eff_dilation = max(0.0, min(1.0, _dilation + breath))
        pupil_r = max(_pupil_min_r, int(_pupil_min_r
                     + eff_dilation * (_pupil_max_r - _pupil_min_r))) * ss
        pygame.draw.circle(interior, PUPIL, (px, py), pupil_r)

        # Glossy catch-light — a crisp white dot inside the upper-left of the
        # pupil, plus a tiny secondary spark. Both stay within the pupil so they
        # read as a wet reflection, not a blob spilling onto the iris.
        pygame.draw.circle(interior, (255, 255, 255, 255),
                           (px - int(pupil_r * 0.38), py - int(pupil_r * 0.40)),
                           max(2, int(pupil_r * 0.34)))
        pygame.draw.circle(interior, (255, 255, 255, 210),
                           (px + int(pupil_r * 0.34), py + int(pupil_r * 0.34)),
                           max(1, int(pupil_r * 0.15)))

    # Crop the interior to the eye shape: build an alpha mask from the polygon
    # and multiply it in, then blit the masked interior onto the main surface.
    mask = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    pygame.draw.polygon(mask, (255, 255, 255, 255), eye_poly)
    interior.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    surface.blit(interior, (0, 0))

    # Lower lid line — very soft and thin (just defines the lower rim).
    pygame.draw.lines(surface, LOWER_LID, False, lower, max(1, int(1.3 * ss)))

    # Upper lash line (the eyeliner) — drawn as overlapping circles whose radius
    # swells toward the outer third, so the liner tapers gracefully instead of
    # being a uniform stroke.
    n_up = len(upper)
    for k in range(n_up):
        t = k / (n_up - 1)
        edge = abs(t - 0.5) * 2.0                     # 0 at center, 1 at corners
        r = (1.0 + 2.2 * edge ** 1.5) * ss            # thin mid, thick toward corners
        pygame.draw.circle(surface, LASH, (int(upper[k][0]), int(upper[k][1])),
                           max(1, int(r)))

    # Double-eyelid crease — a faint soft arc above the upper lid, fading at the
    # inner corner (only the outer half shows, like a real crease).
    crease = [(x, y - (3.5 + 3.5 * (k / (n_up - 1))) * ss)
              for k, (x, y) in enumerate(upper)]
    pygame.draw.lines(surface, CREASE, False, crease[n_up // 3:], max(1, ss))

    # Eyelashes — fine tapered strokes that sweep outward-and-up, clustered on
    # the outer ~65% of the lid, longest at the outer corner. Each side flicks
    # toward its own outer corner so the pair stays symmetric.
    for k in range(2, n_up - 1):
        t = k / (n_up - 1)
        if 0.40 < t < 0.60:           # skip the very middle (sparser there)
            if k % 2 == 0:
                continue
        side = 1.0 if t >= 0.5 else -1.0
        edge = abs(t - 0.5) * 2.0                      # 0 center → 1 corner
        if edge < 0.12:
            continue
        p0 = upper[k]
        # base angle points up; lean toward this side's outer corner, more so
        # toward the corner (a graceful outward flick).
        ang = -math.pi / 2 + side * (0.35 + 0.55 * edge)
        length = (5.0 + 12.0 * edge) * ss              # short inner → long outer
        curl = side * (0.25 + 0.35 * edge)
        _draw_lash(surface, p0, ang, length, 1.5 * ss, curl)

    # A couple of short lower lashes at the outer corners.
    for k in range(2, n_up - 1):
        t = k / (n_up - 1)
        edge = abs(t - 0.5) * 2.0
        if edge < 0.6 or k % 2 == 0:
            continue
        side = 1.0 if t >= 0.5 else -1.0
        ang = math.pi / 2 + side * 0.35
        _draw_lash(surface, lower[k], ang, (2.5 + 4.0 * edge) * ss,
                   1.1 * ss, side * 0.2)


# Supersampled backing surface — recreated when the window size changes.
_hi_surface: Optional[pygame.Surface] = None
_hi_size: tuple = (0, 0)


def _draw_frame(surface: pygame.Surface, breath: float = 0.0):
    """Paint one frame of the two eyes onto the transparent offscreen surface.

    Draws at 2× resolution onto `_hi_surface`, then smoothscales down onto
    `surface` — this anti-aliases the ellipses/arcs/circles cheaply."""
    global _hi_surface, _hi_size
    w, h = surface.get_width(), surface.get_height()
    if _hi_size != (w, h):
        _hi_surface = pygame.Surface((w * _SS, h * _SS), pygame.SRCALPHA)
        _hi_size = (w, h)
    hi = _hi_surface
    assert hi is not None
    hi.fill((0, 0, 0, 0))
    blink_amount = _blink_val(_blink_t) if _blinking else 0.0
    b = _breath if breath == 0.0 else breath
    _draw_eye(hi, _left_x * _SS, _eye_y * _SS, blink_amount, b, ss=_SS)
    _draw_eye(hi, _right_x * _SS, _eye_y * _SS, blink_amount, b, ss=_SS)
    pygame.transform.smoothscale(hi, (w, h), surface)


# ── Blink animation ────────────────────────────────────────────────────────

def _blink_val(t: float) -> float:
    """Blink curve: quick close, quick open. t in [0, 1]."""
    # Two sine halves: fast close (0→π), fast open (π→2π)
    return max(0.0, math.sin(t * math.pi * 2))


_BLINK_DURATION = 0.18  # seconds


# ── Window (PySide6 frameless + transparent) ───────────────────────────────

class EyeWindow(QWidget):
    """A frameless, translucent widget that floats on the desktop showing only
    the eyes. Drag the body to move; drag the right/bottom edges or bottom-right
    corner to resize. ESC quits."""

    def __init__(self, draw_frame_cb, initial_size=(BASE_W, BASE_H)):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint
                            | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self._draw_frame = draw_frame_cb
        self.resize(*initial_size)
        relayout(*initial_size)
        self._surface = pygame.Surface(initial_size, pygame.SRCALPHA)
        self._drag_offset: Optional[tuple] = None   # for move
        self._resize_grip: Optional[str] = None     # 'r' | 'b' | 'rb'
        self._resize_start: Optional[tuple] = None  # (w, h, mx, my)
        self._last_data = None  # keep QImage byte buffer alive during paint
        self._running = True

    def is_running(self) -> bool:
        return self._running

    def stop(self) -> None:
        self._running = False

    def _detect_grip(self, x: float, y: float) -> Optional[str]:
        w, h = self.width(), self.height()
        on_right = x >= w - GRIP
        on_bottom = y >= h - GRIP
        if on_right and on_bottom:
            return "rb"
        if on_right:
            return "r"
        if on_bottom:
            return "b"
        return None

    def mousePressEvent(self, e):
        x, y = e.position().x(), e.position().y()
        grip = self._detect_grip(x, y)
        if grip:
            self._resize_grip = grip
            self._resize_start = (self.width(), self.height(), x, y)
        else:
            gp = e.globalPosition()
            self._drag_offset = (gp.x() - self.x(), gp.y() - self.y())

    def mouseMoveEvent(self, e):
        if self._resize_grip and self._resize_start:
            w0, h0, mx0, my0 = self._resize_start
            mx, my = e.position().x(), e.position().y()
            new_w, new_h = w0, h0
            if "r" in self._resize_grip:
                new_w = max(MIN_W, w0 + int(mx - mx0))
            if "b" in self._resize_grip:
                new_h = max(MIN_H, h0 + int(my - my0))
            if (new_w, new_h) != (self.width(), self.height()):
                self.resize(new_w, new_h)
        elif self._drag_offset is not None:
            gp = e.globalPosition()
            self.move(int(gp.x() - self._drag_offset[0]),
                      int(gp.y() - self._drag_offset[1]))

    def mouseReleaseEvent(self, e):
        self._resize_grip = None
        self._resize_start = None
        self._drag_offset = None

    def resizeEvent(self, e):
        w, h = self.width(), self.height()
        if (w, h) != (self._surface.get_width(), self._surface.get_height()):
            self._surface = pygame.Surface((w, h), pygame.SRCALPHA)
            relayout(w, h)
        super().resizeEvent(e)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self.stop()
            self.close()

    def closeEvent(self, e):
        self.stop()
        super().closeEvent(e)

    def paintEvent(self, e):
        self._draw_frame(self._surface)
        w = self._surface.get_width()
        h = self._surface.get_height()
        data = pygame.image.tobytes(self._surface, "RGBA")
        img = QImage(data, w, h, 4 * w, QImage.Format.Format_RGBA8888)
        self._last_data = data  # hold reference until paint ends
        painter = QPainter(self)
        painter.drawImage(0, 0, img)
        painter.end()


_app: Optional[QApplication] = None
_widget: Optional[EyeWindow] = None


def _init_display() -> None:
    global _app, _widget
    _app = QApplication.instance() or QApplication(sys.argv)
    _widget = EyeWindow(_draw_frame, (BASE_W, BASE_H))
    _widget.show()


# ── Game Loop ──────────────────────────────────────────────────────────────


async def game_loop():
    global _blinking, _blink_t, _blink_cd, _curr_x, _curr_y, _dilation
    global _look_x, _look_y, _auto_gaze, _drift_cd, _breath_t, _breath
    global _voice_attention, _voice_prev_expr, _expression
    global _target_dilation, _lid_resting, _lid_droop
    global _gomoku_state, _gomoku_flash_t

    clock = pygame.time.Clock()
    dt = 0.0
    _was_voice_attention = False
    _prev_auto_gaze = True  # save auto_gaze before voice attention

    # Initialize auto-blink countdown
    _, interval, _, _ = PRESETS[_expression]
    _blink_cd = interval

    try:
        while _widget is not None and _widget.is_running():
            # Pump Qt events (mouse move/resize/close, paint). Drives the GUI
            # from the asyncio main thread without a separate Qt event loop.
            QApplication.processEvents()

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
            _breath = math.sin(_breath_t * 1.3) * 0.03  # gentle ~5s cycle

            # Trigger a repaint; processEvents() above will flush it next frame.
            _widget.update()
            dt = clock.tick(60) / 1000.0
            await asyncio.sleep(0)
    finally:
        if _widget is not None:
            _widget.stop()
            _widget.close()


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
    # provide_channel already returns an asyncio.Task (created on the matrix's
    # event loop), so we must not wrap it in create_task again.
    channel_task = matrix.provide_channel(channel)

    # Exit when EITHER the window closes (user hit ESC / closed it) OR the
    # ghost clears the channel — whichever happens first.
    done, pending = await asyncio.wait(
        {game_task, channel_task}, return_when=asyncio.FIRST_COMPLETED
    )
    for t in pending:
        t.cancel()
    for t in pending:
        try:
            await t
        except (asyncio.CancelledError, Exception):
            pass

    _unsub_voice()
    _unsub_gomoku()
    _unsub_vision()
    if _widget is not None:
        _widget.close()


if __name__ == "__main__":
    import signal

    _init_display()

    def _sigterm_handler(signum, frame):
        if _widget is not None:
            _widget.stop()
            _widget.close()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)
    Matrix.discover().run(main)
