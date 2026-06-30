"""
_control_pad_tes_005_listener_exception_isolation — cb 异常隔离.

验证 (三类 callback 都不能因一个 cb 抛异常而影响其它):
  - binding cb 抛异常: 不影响后续 binding 注册/触发
  - fallthrough listener 抛异常: 同一次 fallthrough, 其它 listener 仍被调
  - event listener 抛异常: 同一次事件, 其它 listener 仍被调

跑测试时会看到一些 ERROR log "raised (隔离)" — 这是预期信号, 不是测试失败.

Usage:
  python -m ghoshell_moss_contrib.unitree.g1.runtime._control_pad_tes_005_listener_exception_isolation

不依赖 G1 实机.
"""
from __future__ import annotations

import logging
import sys

from ghoshell_moss_contrib.unitree.g1.runtime import control_pad

# 预期会有 ERROR log, 抑制到 CRITICAL 让输出干净一点 (注释掉这行可看 log).
logging.basicConfig(level=logging.CRITICAL, format="%(name)s %(levelname)s %(message)s")


def main() -> int:
    control_pad._configure_for_testing()

    # ── case 1: binding cb 异常 → 不影响后续 binding ──────────────────
    bad_called = [0]
    good_called = [0]

    def bad_cb(_e):
        bad_called[0] += 1
        raise RuntimeError("intentional in bad_cb")

    def good_cb(_e):
        good_called[0] += 1

    control_pad.register_binding("bad", {"a"}, bad_cb)
    control_pad._dispatch_press_for_testing("a", {"a"})
    assert bad_called[0] == 1, f"bad_cb expected 1, got {bad_called[0]}"

    # 注册并触发 good — bad 之前抛异常没让模块挂
    control_pad.register_binding("good", {"b"}, good_cb)
    control_pad._dispatch_press_for_testing("b", {"b"})
    assert good_called[0] == 1, f"good_cb expected 1, got {good_called[0]}"

    # ── case 2: fallthrough listener 异常隔离 ─────────────────────────
    ft_bad_called = [0]
    ft_good_called = [0]

    def ft_bad(_pk, _e):
        ft_bad_called[0] += 1
        raise RuntimeError("intentional in ft_bad")

    def ft_good(_pk, _e):
        ft_good_called[0] += 1

    control_pad.register_fallthrough_listener(ft_bad)
    control_pad.register_fallthrough_listener(ft_good)

    # 按一个没匹配的键 → 同一次 fallthrough 触发两个 listener
    control_pad._dispatch_press_for_testing("x", {"x"})
    assert ft_bad_called[0] == 1, f"ft_bad expected 1, got {ft_bad_called[0]}"
    assert ft_good_called[0] == 1, \
        f"ft_good expected 1 (bad's exception should not block it), got {ft_good_called[0]}"

    # ── case 3: event listener 异常隔离 ───────────────────────────────
    ev_bad_called = [0]
    ev_good_called = [0]

    def ev_bad(_e):
        ev_bad_called[0] += 1
        raise RuntimeError("intentional in ev_bad")

    def ev_good(_e):
        ev_good_called[0] += 1

    control_pad.register_event_listener(ev_bad)
    control_pad.register_event_listener(ev_good)

    # 触发一个新 binding (b 已触发, c 没用过)
    control_pad.register_binding("c_binding", {"c"}, lambda e: None)
    control_pad._dispatch_press_for_testing("c", {"c"})
    assert ev_bad_called[0] == 1, f"ev_bad expected 1, got {ev_bad_called[0]}"
    assert ev_good_called[0] == 1, \
        f"ev_good expected 1 (bad's exception should not block it), got {ev_good_called[0]}"

    # ── case 4: 即使一个 cb 持续抛异常, ring buffer 仍正确入队 ───────
    batch = control_pad.drain()
    # 入队事件: case1 a (bad fire), case1 b (good fire), case2 x (fallthrough),
    #          case3 c (c_binding fire) = 4 条
    assert len(batch.items) == 4, \
        f"buffer expected 4 events (异常不影响入队), got {len(batch.items)}"

    print("PASS: tes_005_listener_exception_isolation")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
