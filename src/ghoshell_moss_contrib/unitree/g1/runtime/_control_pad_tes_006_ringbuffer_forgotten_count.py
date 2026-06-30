"""
_control_pad_tes_006_ringbuffer_forgotten_count — buffer 满时 forgotten 计数.

验证:
  - buffer_size=N 时, 入队 N+M 条, drain 返回最新 N 条 + forgotten=M
  - drain 后 forgotten 归零, 下批次重新累计
  - 多次 drain 跨批次, forgotten 不串扰
  - 空 buffer drain 返回 items=[] forgotten=0

避开 debounce 复杂性: 用多个不同 binding 各触发一次, 不需要 sleep + debounce 时间窗.

Usage:
  python -m ghoshell_moss_contrib.unitree.g1.runtime._control_pad_tes_006_ringbuffer_forgotten_count

不依赖 G1 实机.
"""
from __future__ import annotations

import logging
import sys

from ghoshell_moss_contrib.unitree.g1.runtime import control_pad

logging.basicConfig(level=logging.ERROR, format="%(name)s %(levelname)s %(message)s")


def _noop(_e):
    pass


def main() -> int:
    control_pad._configure_for_testing(buffer_size=3)

    # 第一批: 5 个不同 binding 各触发一次. buffer_size=3 → 挤掉 2 条, forgotten=2.
    keys_batch1 = ["a", "b", "x", "y", "up"]
    for k in keys_batch1:
        control_pad.register_binding(f"k_{k}", {k}, _noop)
    for k in keys_batch1:
        control_pad._dispatch_press_for_testing(k, {k})

    batch = control_pad.drain()
    assert len(batch.items) == 3, f"batch1 expected 3 items, got {len(batch.items)}"
    assert batch.forgotten == 2, f"batch1 expected forgotten=2, got {batch.forgotten}"

    # ring buffer 是 deque(maxlen=N) — 挤掉最旧, 保留最新.
    # 入队顺序 a, b, x, y, up; 挤掉 a, b; 留下 x, y, up.
    binding_names = [e.binding_name for e in batch.items]
    assert binding_names == ["k_x", "k_y", "k_up"], \
        f"batch1 expected newest 3, got {binding_names}"

    # 第二批: 4 个新 binding, buffer 空了重新入队. 4 - 3 = 1 forgotten.
    keys_batch2 = ["down", "left", "right", "select"]
    for k in keys_batch2:
        control_pad.register_binding(f"k_{k}", {k}, _noop)
    for k in keys_batch2:
        control_pad._dispatch_press_for_testing(k, {k})

    batch = control_pad.drain()
    assert len(batch.items) == 3, f"batch2 expected 3, got {len(batch.items)}"
    assert batch.forgotten == 1, \
        f"batch2 expected forgotten=1 (reset after drain), got {batch.forgotten}"

    # 第三批: 空 drain
    batch = control_pad.drain()
    assert len(batch.items) == 0, f"batch3 expected 0, got {len(batch.items)}"
    assert batch.forgotten == 0, f"batch3 expected forgotten=0, got {batch.forgotten}"

    # 第四批: 正好 buffer_size 条 — 不挤
    keys_batch4 = ["l1", "r1", "l2"]
    for k in keys_batch4:
        control_pad.register_binding(f"k_{k}", {k}, _noop)
    for k in keys_batch4:
        control_pad._dispatch_press_for_testing(k, {k})

    batch = control_pad.drain()
    assert len(batch.items) == 3
    assert batch.forgotten == 0, \
        f"batch4 (恰好填满) expected forgotten=0, got {batch.forgotten}"

    # health 一致性: forgotten_since_last_drain 在 drain 后归零
    h = control_pad.health()
    assert h["forgotten_since_last_drain"] == 0
    assert h["buffer_len"] == 0
    assert h["buffer_max"] == 3

    print("PASS: tes_006_ringbuffer_forgotten_count")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
