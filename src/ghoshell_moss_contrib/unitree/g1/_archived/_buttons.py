"""
G1 遥控器按键 callback 注册接口.

设计思路:
  - bootstrap 跑起来后, monitor 在 reader 线程里持续解析 wireless_remote[40].
  - 用户(channel.startup 钩子 / 应用代码) 通过 register_button_callback 注册"按键边沿"回调.
  - 回调跑在 cyclonedds reader 线程 — 用户实现必须用 loop.call_soon_threadsafe 把信号推回自己的 event loop.

可用 button_name(参考 sdk/04 的 RemoteController 解析):
  a, b, x, y                — 四主键
  up, down, left, right     — 方向键
  l1, l2, r1, r2            — 肩键
  start, select, f1, f3     — 辅助键

不暴露摇杆边沿(用户用 state.remote() 自己读). 如果未来需要"推到死区外"信号, 单独加.

线程安全:
  - 注册 / 反注册用 lock
  - 触发 callback 时遍历快照(避免遍历过程中变更)
  - 每个 callback 调用包 try/except, 单个 callback 崩不能影响其他
"""

from __future__ import annotations

import logging
import threading
from typing import Callable
from uuid import uuid4

from ghoshell_moss_contrib.unitree.g1._archived import _monitor
from ghoshell_moss_contrib.unitree.g1._archived import state

logger = logging.getLogger(__name__)


ButtonCallback = Callable[[bool], None]
"""按键回调签名: cb(pressed: bool) -> None.
pressed=True 是按下边沿, False 是松开边沿.
回调跑在 cyclonedds reader 线程! 用户必须用 loop.call_soon_threadsafe.
"""


VALID_BUTTONS = frozenset([
    'a', 'b', 'x', 'y',
    'up', 'down', 'left', 'right',
    'l1', 'l2', 'r1', 'r2',
    'start', 'select', 'f1', 'f3',
])


class CallbackHandle:
    """注册返回的 handle. 用于 unregister."""
    __slots__ = ('id', 'button')

    def __init__(self, id: str, button: str):
        self.id = id
        self.button = button


# ═══════════════════════════════════════════════════════════════════════════════
# 内部状态
# ═══════════════════════════════════════════════════════════════════════════════

_lock = threading.Lock()
# button_name -> {handle_id -> callback}
_callbacks: dict[str, dict[str, ButtonCallback]] = {}
_wired_to_monitor: bool = False


def _on_edge(prev: state.RemoteState, curr: state.RemoteState) -> None:
    """monitor 调的回调. 跑在 reader 线程.

    遍历所有按键, 检测边沿, 调用注册的 callback.
    """
    # 拷贝当前注册表的快照, 释放锁后调用 callback. 避免 callback 内反注册导致死锁.
    with _lock:
        snapshots: list[tuple[str, ButtonCallback, bool]] = []
        for button in VALID_BUTTONS:
            prev_v = getattr(prev, button)
            curr_v = getattr(curr, button)
            if prev_v == curr_v:
                continue
            # 边沿. curr_v=True 是按下, curr_v=False 是松开.
            cbs = _callbacks.get(button, {})
            for cb in cbs.values():
                snapshots.append((button, cb, curr_v))

    for button, cb, pressed in snapshots:
        try:
            cb(pressed)
        except Exception:
            logger.exception("button callback for '%s' raised", button)


def _ensure_wired() -> None:
    """把 _on_edge 注册到 monitor. 幂等."""
    global _wired_to_monitor
    if _wired_to_monitor:
        return
    _monitor.set_button_edge_callback(_on_edge)
    _wired_to_monitor = True


# ═══════════════════════════════════════════════════════════════════════════════
# 公共 API
# ═══════════════════════════════════════════════════════════════════════════════


def register_button_callback(button: str, callback: ButtonCallback) -> CallbackHandle:
    """注册按键边沿回调.

    Args:
        button: 按键名(小写). 参考 VALID_BUTTONS.
        callback: 边沿触发时调用 cb(pressed: bool).
                  pressed=True 按下, False 松开.
                  **跑在 cyclonedds reader 线程**, 用户必须用 loop.call_soon_threadsafe.

    Returns:
        CallbackHandle, 用于 unregister.

    Raises:
        ValueError: 未知 button_name.
        RuntimeError: monitor 未启动.
    """
    if button not in VALID_BUTTONS:
        raise ValueError(f"unknown button '{button}'. valid: {sorted(VALID_BUTTONS)}")
    if not _monitor.is_started():
        raise RuntimeError(
            "g1 monitor not started; call bootstrap() before register_button_callback()"
        )

    _ensure_wired()

    handle_id = uuid4().hex
    handle = CallbackHandle(id=handle_id, button=button)

    with _lock:
        if button not in _callbacks:
            _callbacks[button] = {}
        _callbacks[button][handle_id] = callback

    logger.debug("registered button callback: button=%s handle=%s", button, handle_id)
    return handle


def unregister_button_callback(handle: CallbackHandle) -> None:
    """反注册. 幂等(重复调不报错)."""
    with _lock:
        cbs = _callbacks.get(handle.button)
        if cbs is None:
            return
        cbs.pop(handle.id, None)
    logger.debug("unregistered button callback: button=%s handle=%s", handle.button, handle.id)


def list_callbacks() -> dict[str, int]:
    """现场调试: 返回 button -> count."""
    with _lock:
        return {button: len(cbs) for button, cbs in _callbacks.items() if cbs}


def _clear_all_for_testing() -> None:
    """测试 hook: 清空所有注册."""
    global _wired_to_monitor
    with _lock:
        _callbacks.clear()
    _monitor.set_button_edge_callback(None)
    _wired_to_monitor = False
