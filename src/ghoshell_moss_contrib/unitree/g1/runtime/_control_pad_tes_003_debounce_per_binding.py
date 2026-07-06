"""
_control_pad_tes_003_debounce_per_binding — per-binding debounce.

验证:
  - 同一 binding 在 debounce_sec 内连续触发, 只第一次进 ring buffer / 触发 cb
  - debounce_sec 过后再次触发, 正常 fire
  - 不同 binding 独立 debounce (A 触发不重置 B 的 cooldown)
  - debounce_suppressed_count + fired_count 正确累计 (health() 报)
  - debounce_sec 可 per-binding 配置 (一个 short, 一个 long)

Usage:
  python -m ghoshell_moss_contrib.unitree.g1.runtime._control_pad_tes_003_debounce_per_binding

不依赖 G1 实机. 用 time.sleep 跨越 debounce 窗口.
"""
from __future__ import annotations

import logging
import sys
import time

from ghoshell_moss_contrib.unitree.g1.runtime import control_pad

logging.basicConfig(level=logging.ERROR, format="%(name)s %(levelname)s %(message)s")


def main() -> int:
    # global_min floor 设小, 让 binding 可以注册 0.05s debounce
    control_pad._configure_for_testing(
        default_debounce_sec=0.10,
        global_min_debounce_sec=0.05,
    )

    a_count = [0]
    b_count = [0]

    def on_a(_e):
        a_count[0] += 1

    def on_b(_e):
        b_count[0] += 1

    # a: 50ms debounce (短), b: 300ms debounce (长)
    control_pad.register_binding("a_binding", {"a"}, on_a, debounce_sec=0.05)
    control_pad.register_binding("b_binding", {"b"}, on_b, debounce_sec=0.30)

    # case 1: 连续 3 次触发 a → 只第一次 fire
    control_pad._dispatch_press_for_testing("a", {"a"})
    control_pad._dispatch_press_for_testing("a", {"a"})
    control_pad._dispatch_press_for_testing("a", {"a"})
    assert a_count[0] == 1, f"case1 expected 1, got {a_count[0]}"

    # case 2: sleep 跨过 a debounce, 再次触发
    time.sleep(0.08)
    control_pad._dispatch_press_for_testing("a", {"a"})
    assert a_count[0] == 2, f"case2 expected 2, got {a_count[0]}"

    # case 3: 不同 binding 独立 — 立刻触发 b 应该 fire (b 还没触发过)
    control_pad._dispatch_press_for_testing("b", {"b"})
    assert b_count[0] == 1, f"case3 b expected 1, got {b_count[0]}"

    # case 4: a 仍在 50ms debounce 内 (case 2 触发到现在 < 50ms), 应该静默
    control_pad._dispatch_press_for_testing("a", {"a"})
    assert a_count[0] == 2, f"case4 a expected still 2, got {a_count[0]}"

    # case 5: b 在 300ms debounce 内 (case 3 触发到现在 < 300ms), 应该静默
    control_pad._dispatch_press_for_testing("b", {"b"})
    assert b_count[0] == 1, f"case5 b expected still 1, got {b_count[0]}"

    # case 6: sleep 跨过 a debounce 但不跨 b debounce
    time.sleep(0.08)
    control_pad._dispatch_press_for_testing("a", {"a"})
    assert a_count[0] == 3, f"case6 a expected 3, got {a_count[0]}"
    control_pad._dispatch_press_for_testing("b", {"b"})
    # b debounce 300ms, 从 case 3 (~0.08s 前) 到现在 ~0.08s, 仍 debounce
    assert b_count[0] == 1, f"case6 b expected still 1, got {b_count[0]}"

    # case 7: health 计数对齐
    # fired: a 3 次 + b 1 次 = 4
    # suppressed: case1 (a 2) + case4 (a 1) + case5 (b 1) + case6 (b 1) = 5
    h = control_pad.health()
    assert h["fired_count"] == 4, f"fired_count expected 4, got {h['fired_count']}"
    assert h["debounce_suppressed_count"] == 5, \
        f"suppressed expected 5, got {h['debounce_suppressed_count']}"

    print("PASS: tes_003_debounce_per_binding")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
