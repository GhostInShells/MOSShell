"""
_led_sen_factory_showcase — LED 工厂动画双工 showcase.

场景:
  脚本运行后, 输入一行命令, 立刻在 G1 眼条 LED 上看到对应动画.
  预设了 7 个工厂 (solid/blink/breath/fade/pulse/police/rainbow) + off, 每个工厂
  都有"默认轨道"; 可以加 event/anim/bg 前缀强制落到指定轨道, 直接测三轨道优先级:
  比如在背景"蓝色呼吸"上抢一个"红闪", 看 event 显形完是否自动让位 background.

Usage:
  python -m ghoshell_moss_contrib.unitree.g1.runtime._led_sen_factory_showcase <nic>

  nic 示例: eth0 / enp3s0 — 见 docs/hardware.md.

前置:
  - G1 已开机 (任何模式; LED 不依赖运动模式).
  - 你能直接看到 G1 面部眼条 LED (面部中线位置的横向灯条).
  - 安全: LED 是被动反馈, 不触发任何 G1 动作, 无须特别准备.

预期 (脚本运行后输入 `?` 看完整命令清单):
  > breath blue                # 蓝色呼吸 background, 永久循环
  > event blink red            # 红闪 3 次, 抢占 background, 完后蓝色呼吸恢复
  > rainbow                    # 彩虹环 animation, 压在 background 上
  > clear anim                 # 清 animation, 背景重新显形
  > fade red blue 2000 smooth  # animation 渐变, 2 秒 smooth easing
  > police 400 5               # event 红蓝交替 5 次
  > off                        # 清所有轨, 灭灯
  > health                     # 看 sdk_call_count / 各 track 状态
  > q                          # 干净退出 (Ctrl+C 也行)

读完 docstring 还看不懂请回去读 runtime/README.md.
"""
from __future__ import annotations

import json as _json
import shlex
import sys

from prompt_toolkit import PromptSession, patch_stdout

from ghoshell_moss_contrib.unitree.g1.runtime import led
from ghoshell_moss_contrib.unitree.g1.sdk import bootstrap


# ── 帮助文档 ────────────────────────────────────────────────────────────

HELP_TEXT = """
LED Showcase 命令清单 (大小写不敏感):

  颜色工厂 (后跟参数; 默认轨道 = 该 factory 最自然的轨)
    solid <color>                             → background, 单色常亮
    blink <color> [count] [period_ms]         → event, 默认 3 次 500ms
    breath <color> [period_ms]                → background, 默认 2000ms 永久
    fade <c1> <c2> [duration_ms] [easing]     → animation, 默认 1000ms linear
                                                easing: linear/smooth/step/hsv
    pulse <color> [count] [period_ms]         → event, 默认 1 次 800ms
    police [period_ms] [count]                → event, 默认 600ms 永久 (count=-1)
    rainbow [period_ms] [count]               → animation, 默认 4000ms 永久

  轨道前缀 (覆盖默认轨道)
    event <factory ...>                       → 强制落 event 轨
    anim  <factory ...>                       → 强制落 animation 轨
    bg    <factory ...>                       → 强制落 background 轨

  清理
    off                                       → clear_all + 灭灯
    clear event | clear anim | clear bg | clear all

  调试
    set <color>                               → set_led 单帧 (绕 driver, 立刻被覆盖)
    health    (或 h)                          → 打印 health()
    schema                                    → 打印 Animation JSON schema

  其它
    ?  help                                   → 本帮助
    q  quit                                   → 退出

颜色:
  hex:   #rgb / #rrggbb
  命名:  red green blue yellow cyan magenta orange purple pink white off warm dim
"""


# ── factory 调度 — 每个 handler 接受 args list, 返回 Animation ──────────

def _factory_solid(args: list[str]):
    if not args:
        raise ValueError("solid 需要 <color>")
    return led.solid(args[0])


def _factory_blink(args: list[str]):
    if not args:
        raise ValueError("blink 需要 <color>")
    color = args[0]
    count = int(args[1]) if len(args) > 1 else 3
    period_ms = int(args[2]) if len(args) > 2 else 500
    return led.blink(color, count=count, period_ms=period_ms)


def _factory_breath(args: list[str]):
    if not args:
        raise ValueError("breath 需要 <color>")
    color = args[0]
    period_ms = int(args[1]) if len(args) > 1 else 2000
    return led.breath(color, period_ms=period_ms)


def _factory_fade(args: list[str]):
    if len(args) < 2:
        raise ValueError("fade 需要 <c1> <c2>")
    c1, c2 = args[0], args[1]
    duration_ms = int(args[2]) if len(args) > 2 else 1000
    easing = led.Easing(args[3]) if len(args) > 3 else led.Easing.LINEAR
    return led.fade(c1, c2, duration_ms=duration_ms, easing=easing)


def _factory_pulse(args: list[str]):
    if not args:
        raise ValueError("pulse 需要 <color>")
    color = args[0]
    count = int(args[1]) if len(args) > 1 else 1
    period_ms = int(args[2]) if len(args) > 2 else 800
    return led.pulse(color, count=count, period_ms=period_ms)


def _factory_police(args: list[str]):
    period_ms = int(args[0]) if len(args) > 0 else 600
    count = int(args[1]) if len(args) > 1 else -1
    return led.police(period_ms=period_ms, count=count)


def _factory_rainbow(args: list[str]):
    period_ms = int(args[0]) if len(args) > 0 else 4000
    count = int(args[1]) if len(args) > 1 else -1
    return led.rainbow(period_ms=period_ms, count=count)


# 命令名 → (handler, default_track)
FACTORIES = {
    "solid":   (_factory_solid,   "background"),
    "blink":   (_factory_blink,   "event"),
    "breath":  (_factory_breath,  "background"),
    "fade":    (_factory_fade,    "animation"),
    "pulse":   (_factory_pulse,   "event"),
    "police":  (_factory_police,  "event"),
    "rainbow": (_factory_rainbow, "animation"),
}

TRACK_DISPATCH = {
    "event":      led.play_event,
    "animation":  led.play_animation,
    "background": led.set_background,
}

TRACK_PREFIX = {"event": "event", "anim": "animation", "bg": "background"}


# ── 命令派发 ────────────────────────────────────────────────────────────

def _execute(line: str) -> None:
    """解析一行命令并执行. 异常往上抛, 由主循环捕获并 print 反馈."""
    tokens = shlex.split(line, comments=False)
    if not tokens:
        return
    cmd = tokens[0].lower()
    rest = tokens[1:]

    if cmd in ("q", "quit", "exit"):
        raise EOFError()
    if cmd in ("?", "help"):
        print(HELP_TEXT)
        return
    if cmd in ("h", "health"):
        print(_json.dumps(led.health(), indent=2, ensure_ascii=False))
        return
    if cmd == "schema":
        print(_json.dumps(led.animation_schema(), indent=2, ensure_ascii=False))
        return
    if cmd == "set":
        if not rest:
            raise ValueError("set 需要 <color>")
        led.set_led(rest[0])
        print(f"[set_led] {rest[0]} -> 已下发 (driver 下一 tick 会用 track 状态覆盖)")
        return
    if cmd == "off":
        led.clear_all()
        print("[off] 清所有轨道, 灭灯")
        return
    if cmd == "clear":
        if not rest:
            raise ValueError("clear 需要 event/anim/bg/all 之一")
        which = rest[0].lower()
        if which == "event":
            led.clear_event()
        elif which == "anim":
            led.clear_animation()
        elif which == "bg":
            led.clear_background()
        elif which == "all":
            led.clear_all()
        else:
            raise ValueError(f"clear 未知子项: {which}")
        print(f"[clear] {which}")
        return

    # 轨道前缀: event/anim/bg <factory ...>
    track_override = None
    if cmd in TRACK_PREFIX:
        if not rest:
            raise ValueError(f"{cmd} 需要后跟 factory")
        track_override = TRACK_PREFIX[cmd]
        cmd = rest[0].lower()
        rest = rest[1:]

    if cmd not in FACTORIES:
        raise ValueError(f"未知命令: {cmd!r}. 输入 ? 看帮助.")
    factory_fn, default_track = FACTORIES[cmd]
    track = track_override or default_track
    anim = factory_fn(rest)
    TRACK_DISPATCH[track](anim)
    print(f"[{track}] {cmd} {' '.join(rest)}  → loop={anim.loop} clips={len(anim.clips)}")


def main(nic: str) -> int:
    print(f"[1/3] sdk.bootstrap(nic={nic!r}) ...")
    bootstrap(nic)
    print("[2/3] led.start() ...")
    led.start()

    print()
    print("=" * 64)
    print(" LED Showcase — 输入命令, 看 G1 眼条 LED 实时变化")
    print(" ?  看帮助    q  退出    Ctrl+C  干净退出")
    print("=" * 64)
    print()

    session: PromptSession = PromptSession()
    try:
        with patch_stdout.patch_stdout(raw=True):
            while True:
                try:
                    line = session.prompt("led > ")
                except (KeyboardInterrupt, EOFError):
                    print()
                    break
                try:
                    _execute(line)
                except EOFError:
                    print()
                    break
                except Exception as e:
                    print(f"[error] {e}")
    finally:
        print("[3/3] led.stop() ...")
        led.stop()
        h = led.health()
        print()
        print("=" * 64)
        print(f" sdk_call_count={h['sdk_call_count']}  error_count={h['error_count']}")
        print("=" * 64)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
