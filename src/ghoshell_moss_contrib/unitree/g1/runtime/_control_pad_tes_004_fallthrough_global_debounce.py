"""
_control_pad_tes_004_fallthrough_global_debounce — fallthrough 全局 debounce.

验证:
  - 没注册 binding 时按下边沿触发 fallthrough listener
  - 全局 debounce: 不论按什么组合, 同一窗口内 fallthrough 只 fire 一次
  - fallthrough_debounce_sec 过后再 fire
  - 命中 binding 的边沿不触发 fallthrough (matched_any 跳过)
  - fallthrough event 进 ring buffer (is_fallthrough=True), 可被 drain 拿到
  - fallthrough_fired_count + 共享 debounce_suppressed_count 累计

Usage:
  python -m ghoshell_moss_contrib.unitree.g1.runtime._control_pad_tes_004_fallthrough_global_debounce

不依赖 G1 实机.
"""
from __future__ import annotations

import logging
import sys
import time

from ghoshell_moss_contrib.unitree.g1.runtime import control_pad

logging.basicConfig(level=logging.ERROR, format="%(name)s %(levelname)s %(message)s")


def main() -> int:
    # fallthrough debounce 设短, 测试方便
    control_pad._configure_for_testing(fallthrough_debounce_sec=0.10)

    ft_events: list[tuple[tuple[str, ...], bool]] = []

    def on_fallthrough(pressed_keys, event):
        ft_events.append((pressed_keys, event.is_fallthrough))

    control_pad.register_fallthrough_listener(on_fallthrough)

    # case 1: 没 binding 时按 a → fallthrough
    control_pad._dispatch_press_for_testing("a", {"a"})
    assert len(ft_events) == 1, f"case1 expected 1 ft, got {len(ft_events)}"
    assert ft_events[0] == (("a",), True)

    # case 2: 全局 debounce 内按其它键 → 静默 (即使是不同组合也共享一个 cooldown)
    control_pad._dispatch_press_for_testing("b", {"b"})
    control_pad._dispatch_press_for_testing("x", {"x", "y"})
    assert len(ft_events) == 1, f"case2 expected still 1, got {len(ft_events)}"

    # case 3: sleep 跨过 fallthrough debounce, 再次触发
    time.sleep(0.12)
    control_pad._dispatch_press_for_testing("y", {"y"})
    assert len(ft_events) == 2, f"case3 expected 2, got {len(ft_events)}"

    # case 4: 注册 binding 后, 匹配的边沿不触发 fallthrough
    time.sleep(0.12)
    control_pad.register_binding("a_binding", {"a"}, lambda e: None)
    control_pad._dispatch_press_for_testing("a", {"a"})  # 命中 binding, 不 fallthrough
    assert len(ft_events) == 2, f"case4 expected still 2, got {len(ft_events)}"

    # case 5: 不匹配 binding 的组合 → fallthrough (a binding 是 {a}, {a,b} 不匹配)
    control_pad._dispatch_press_for_testing("a", {"a", "b"})
    assert len(ft_events) == 3, f"case5 expected 3, got {len(ft_events)}"

    # case 6: fallthrough event 进 ring buffer
    batch = control_pad.drain()
    # 入队顺序: case1 ft, case3 ft, case4 binding-event, case5 ft = 4 条
    assert len(batch.items) == 4, f"expected 4 events in buffer, got {len(batch.items)}"
    ft_count = sum(1 for e in batch.items if e.is_fallthrough)
    assert ft_count == 3, f"expected 3 fallthrough in buffer, got {ft_count}"

    # case 7: health 计数
    h = control_pad.health()
    assert h["fallthrough_fired_count"] == 3, \
        f"fallthrough_fired_count expected 3, got {h['fallthrough_fired_count']}"
    # case2 (2 次静默) → debounce_suppressed_count
    assert h["debounce_suppressed_count"] == 2, \
        f"debounce_suppressed_count expected 2, got {h['debounce_suppressed_count']}"

    print("PASS: tes_004_fallthrough_global_debounce")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
