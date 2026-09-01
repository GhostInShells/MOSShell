# AI Eye

Animated eye pair as minimal AI avatar. Ghost controls via Channel.

## Setup

```bash
cd .moss_ws/apps/games/ai_eye
uv sync
```

Dependencies: `pygame>=2.5.0`, `PySide6>=6.5.0`, `ghoshell-moss[host]` (editable from workspace root). No extra models or services required.

## Window

The eye window is a **frameless, transparent PySide6 widget** — no title bar
or min/close/maximize buttons; only the two eyes float on the desktop (always
on top). Eye drawing stays in pygame, rendered to an offscreen SRCALPHA surface
and blitted into the Qt window each frame.

- **Move**: drag anywhere on the window body.
- **Resize**: drag the right edge, bottom edge, or bottom-right corner (8px grip).
  Eyes scale with the window.
- **Quit**: press `ESC` (or end the ghost session, which closes the window).

## Commands

- `look_at(x, y)` — gaze direction (0..1 normalized); disables auto-drift
- `dilate(amount)` — pupil size (0=pinhole, 1=fully dilated)
- `blink()` — trigger single blink
- `set_expression(name)` — preset: neutral/curious/surprised/focused/sleepy/thinking/speaking
- `thinking()` — shortcut: thinking expression + auto-gaze drift
- `speaking()` — shortcut: speaking expression (dilated, fast blinks)
- `idle()` — back to neutral with auto-gaze drift

## Auto-behavior

- Random blinks every 1.5-7s (interval varies by expression)
- Breathing pupil oscillation (~5s cycle) for "alive" look
- Idle gaze drift — eyes slowly wander when no target is set
- Gaze and dilation use smooth lerp interpolation
