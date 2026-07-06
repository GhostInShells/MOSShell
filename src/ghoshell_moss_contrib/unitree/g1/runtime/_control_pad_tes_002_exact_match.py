"""
_control_pad_tes_002_exact_match — exact match 判定契约 (核心语义).

验证:
  - binding.keys == pressed_keys 才命中 (exact match, 不用 subset)
  - {l2} binding 不会被 {l2, b} 组合意外触发
  - {l2, b} binding 不会被 {l2} 单键触发
  - super-set {l2, b, a} 既不命中 {l2} 也不命中 {l2, b} (触发 fallthrough)
  - 顺序无关: 先按 b 单键 (fallthrough) 再按 l2 组合 → 最后边沿命中 {l2,b} binding

Usage:
  python -m ghoshell_moss_contrib.unitree.g1.runtime._control_pad_tes_002_exact_match

不依赖 G1 实机.
"""
from __future__ import annotations

import logging
import sys

from ghoshell_moss_contrib.unitree.g1.runtime import control_pad

logging.basicConfig(level=logging.ERROR, format="%(name)s %(levelname)s %(message)s")


def main() -> int:
    control_pad._configure_for_testing()

    fired: list[tuple[str, tuple[str, ...]]] = []

    def cb(e):
        fired.append((e.binding_name, e.pressed_keys))

    control_pad.register_binding("single_l2", {"l2"}, cb)
    control_pad.register_binding("combo_l2_b", {"l2", "b"}, cb)

    # case 1: 单按 l2 → 命中 single_l2, 不命中 combo_l2_b
    control_pad._dispatch_press_for_testing("l2", {"l2"})
    assert fired == [("single_l2", ("l2",))], \
        f"case1 expected single_l2 only, got {fired}"

    # case 2: l2+b 同按 → 命中 combo_l2_b, 不命中 single_l2 (exact match: {l2}!={l2,b})
    fired.clear()
    control_pad._dispatch_press_for_testing("b", {"l2", "b"})
    assert fired == [("combo_l2_b", ("b", "l2"))], \
        f"case2 expected combo_l2_b only, got {fired}"

    # case 3: super-set {l2, b, a} → 不命中任何 binding
    # (fallthrough 触发但不进 fired, 因为我们没注册 fallthrough listener)
    fired.clear()
    control_pad._dispatch_press_for_testing("a", {"l2", "b", "a"})
    assert fired == [], f"case3 expected no binding fire, got {fired}"

    # case 4: 顺序无关 — 先 b 单键 (no binding, fallthrough),
    # 再按 l2 加入 (l2 边沿时 pressed={l2,b}, 命中 combo_l2_b)
    import time
    time.sleep(0.25)  # 等 case 2 的 per-binding debounce 窗口过期
    fired.clear()
    control_pad._dispatch_press_for_testing("b", {"b"})  # fallthrough (no fired)
    control_pad._dispatch_press_for_testing("l2", {"l2", "b"})  # combo_l2_b
    assert fired == [("combo_l2_b", ("b", "l2"))], \
        f"case4 expected combo_l2_b in fired, got {fired}"

    # case 5: pressed_keys 字母序稳定 — 不论 set 内部顺序
    time.sleep(0.25)  # 等 case 4 的 per-binding debounce 窗口过期
    fired.clear()
    control_pad._dispatch_press_for_testing("b", frozenset(["l2", "b"]))
    control_pad._dispatch_press_for_testing("b", frozenset(["b", "l2"]))  # 同样的 set
    # 第一次 fire, 第二次 debounce 静默 (默认 200ms)
    assert len(fired) == 1, f"case5 expected 1 (second debounced), got {len(fired)}"
    assert fired[0] == ("combo_l2_b", ("b", "l2")), \
        f"case5 expected ('b', 'l2') sorted, got {fired[0]}"

    print("PASS: tes_002_exact_match")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
