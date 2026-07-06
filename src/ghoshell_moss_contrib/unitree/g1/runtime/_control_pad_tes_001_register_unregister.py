"""
_control_pad_tes_001_register_unregister — 注册/反注册/list/invalid 契约.

验证 (assert + 退出码):
  - register_binding 返回非空 handle, list_bindings 看到
  - unregister_binding 清掉, 重复/未知 handle 静默 (不 raise)
  - keys 含非法按键名 → ValueError
  - keys 为空 → ValueError
  - name 为空 → ValueError
  - debounce_sec 低于 global_min_debounce_sec 被 floor 兜底
  - debounce_sec=None 走模块 default_debounce_sec

Usage:
  python -m ghoshell_moss_contrib.unitree.g1.runtime._control_pad_tes_001_register_unregister

不依赖 G1 实机 / monitor / nic. 走 _configure_for_testing hook 绕过 sdk.
"""
from __future__ import annotations

import logging
import sys

from ghoshell_moss_contrib.unitree.g1.runtime import control_pad

logging.basicConfig(level=logging.ERROR, format="%(name)s %(levelname)s %(message)s")


def _noop(_e):
    pass


def main() -> int:
    control_pad._configure_for_testing(
        default_debounce_sec=0.20,
        global_min_debounce_sec=0.05,
    )

    # 1. 注册返回非空 handle
    h1 = control_pad.register_binding("test1", {"f1"}, _noop)
    assert h1 and isinstance(h1, str), f"expected non-empty handle, got {h1!r}"

    # 2. list_bindings 含
    bindings = control_pad.list_bindings()
    assert h1 in bindings, f"handle {h1} not in list: {list(bindings)}"
    assert bindings[h1].name == "test1"
    assert bindings[h1].keys == frozenset({"f1"})

    # 3. 多个 binding 共存
    h2 = control_pad.register_binding("test2", {"f1", "f3"}, _noop)
    bindings = control_pad.list_bindings()
    assert h1 in bindings and h2 in bindings and len(bindings) == 2

    # 4. unregister 清掉, 其它保留
    control_pad.unregister_binding(h1)
    bindings = control_pad.list_bindings()
    assert h1 not in bindings and h2 in bindings and len(bindings) == 1

    # 5. 重复 unregister 静默 (允许 cleanup 安全调)
    control_pad.unregister_binding(h1)

    # 6. 未知 handle 静默
    control_pad.unregister_binding("nonexistent_handle")

    # 7. 非法按键名 raise
    try:
        control_pad.register_binding("bad", {"f1", "nonexistent_key"}, _noop)
        raise AssertionError("expected ValueError for invalid key")
    except ValueError as e:
        assert "nonexistent_key" in str(e), f"unexpected msg: {e}"

    # 8. keys 空 raise
    try:
        control_pad.register_binding("empty_keys", set(), _noop)
        raise AssertionError("expected ValueError for empty keys")
    except ValueError as e:
        assert "empty" in str(e).lower(), f"unexpected msg: {e}"

    # 9. name 空 raise
    try:
        control_pad.register_binding("", {"f1"}, _noop)
        raise AssertionError("expected ValueError for empty name")
    except ValueError as e:
        assert "empty" in str(e).lower(), f"unexpected msg: {e}"

    # 10. debounce_sec=0.001 被 floor 0.05 兜底
    h3 = control_pad.register_binding("low_db", {"a"}, _noop, debounce_sec=0.001)
    binding = control_pad.list_bindings()[h3]
    assert binding.debounce_sec >= 0.05, \
        f"expected floor 0.05, got {binding.debounce_sec}"

    # 11. debounce_sec=None 走 default 0.20
    h4 = control_pad.register_binding("default_db", {"b"}, _noop, debounce_sec=None)
    binding = control_pad.list_bindings()[h4]
    assert abs(binding.debounce_sec - 0.20) < 1e-6, \
        f"expected default 0.20, got {binding.debounce_sec}"

    # 12. debounce_sec 大于 default 不被压低
    h5 = control_pad.register_binding("high_db", {"x"}, _noop, debounce_sec=1.5)
    binding = control_pad.list_bindings()[h5]
    assert abs(binding.debounce_sec - 1.5) < 1e-6, \
        f"expected 1.5, got {binding.debounce_sec}"

    print("PASS: tes_001_register_unregister")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        sys.exit(1)
