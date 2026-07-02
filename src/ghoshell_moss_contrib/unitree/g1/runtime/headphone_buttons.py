"""
Headphone buttons runtime — 蓝牙耳机按键事件监听.

跟 sdk/_buttons.py 同范式: evdev 后台线程监听耳机按键,
按下边沿时调用注册的回调. 回调跑在 evdev reader 线程, 不能阻塞.

OpenRun by Shokz (AVRCP): 多功能中键是 AVRCP 状态感知 toggle, 交替发送
KEY_PLAYCD (200) / KEY_PAUSECD (201) — 耳机根据自己认为的"当前播放态"决定
code, 对 MOSS 来说都是同一个 toggle 语义信号, 两个 code 都触发 dispatch.
(实测于 2026-07-02 _headphone_buttons_probe)

Usage:
    from ghoshell_moss_contrib.unitree.g1.runtime import headphone_buttons

    headphone_buttons.start()
    handle = headphone_buttons.register_callback(lambda: print("pressed"))
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable
from uuid import uuid4

logger = logging.getLogger("moss.g1.runtime.headphone_buttons")

# ── 常量 ────────────────────────────────────────────────────────────────

# 蓝牙类设备名关键字, 用于自动发现耳机 evdev 设备
_BT_HINT_KEYWORDS = (
    "bluetooth", "bt", "airpods", "wireless", "headset", "buds",
    "beats", "avrcp", "hfp", "a2dp", "openrun", "shokz",
)

# 关注的按键 codes: AVRCP 中键交替发 KEY_PLAYCD(200) / KEY_PAUSECD(201),
# 都视为同一 toggle 语义. 用 frozenset 让 in 查询 O(1) 且不可变.
_TRIGGER_CODES: frozenset[int] = frozenset({200, 201})

# evdev 读超时. 200ms 平衡响应速度 vs CPU
_READ_TIMEOUT_MS = 200


# ── 模块级状态 ──────────────────────────────────────────────────────────

_running: bool = False
_listeners: dict[str, Callable[[], None]] = {}
_listeners_lock = threading.Lock()
_thread: threading.Thread | None = None
_stop_event: threading.Event | None = None
_device_name: str | None = None  # 实际打开的 evdev 设备名, 供 health 查看


# ── 公开接口 ────────────────────────────────────────────────────────────

def start(device_path: str | None = None) -> None:
    """启动 evdev 监听线程. 幂等.

    不传 device_path 则自动搜索蓝牙类 evdev 设备.
    若之前启动的线程已死 (设备断连等), 自动重启.
    """
    global _running, _thread, _stop_event

    with _listeners_lock:
        if _running and _thread is not None and _thread.is_alive():
            return
        # 线程已死或首次启动 — (重新)创建
        if _running:
            logger.info("headphone_buttons: thread dead, restarting.")
        _running = True
        _stop_event = threading.Event()

    _thread = threading.Thread(
        target=_evdev_loop,
        args=(device_path, _stop_event),
        name="g1-headphone-btns",
        daemon=True,
    )
    _thread.start()
    logger.info("headphone_buttons: started")


def stop(timeout: float = 2.0) -> None:
    """停止监听线程. 幂等."""
    global _running, _thread, _stop_event

    with _listeners_lock:
        if not _running:
            return
        _running = False

    evt = _stop_event
    if evt is not None:
        evt.set()

    th = _thread
    if th is not None and th.is_alive():
        th.join(timeout)
    logger.info("headphone_buttons: stopped")


def is_running() -> bool:
    return _running


def register_callback(cb: Callable[[], None]) -> str:
    """注册按键回调. 按下时触发一次. 返回 handle (uuid hex)."""
    handle = uuid4().hex
    with _listeners_lock:
        _listeners[handle] = cb
    return handle


def unregister_callback(handle: str) -> None:
    """反注册. 未知 handle 静默忽略."""
    with _listeners_lock:
        _listeners.pop(handle, None)


def health() -> dict:
    return {
        "running": _running,
        "device": _device_name,
        "listeners": len(_listeners),
    }


# ── 内部: evdev 监听循环 ────────────────────────────────────────────────

def _find_device() -> tuple[str, str] | tuple[None, None]:
    """返回 (path, name) 或 (None, None).

    搜索所有 /dev/input/event*, 返回首个蓝牙类设备.
    """
    try:
        import evdev
    except ImportError:
        logger.error("headphone_buttons: evdev not installed")
        return None, None

    for path in evdev.list_devices():
        try:
            dev = evdev.InputDevice(path)
            name = dev.name
            dev.close()
            if any(k in name.lower() for k in _BT_HINT_KEYWORDS):
                return path, name
        except Exception:
            continue
    return None, None


_DEVICE_RETRY_INTERVAL = 3.0  # 设备未找到 / 断连后重试间隔 (s)


def _evdev_loop(device_path: str | None, stop_evt: threading.Event) -> None:
    """evdev 读循环, 跑在 daemon 线程内.

    外层: 设备发现 retry 循环. 蓝牙 AVRCP evdev 设备在 bluetoothctl connect
    完成后还需数秒才注册进内核; 断连后设备消失. 两种情况都重试, 不退出 thread.
    内层: 事件读循环. OSError (设备断连) → break 回外层重新发现.
    """
    global _device_name

    try:
        import evdev
        from evdev import ecodes
    except ImportError:
        logger.error("headphone_buttons: evdev not installed")
        return

    while not stop_evt.is_set():
        # ── 设备发现 ──────────────────────────────────────────────────────
        if device_path is not None:
            path, name = device_path, device_path
        else:
            path, name = _find_device()
            if path is None:
                logger.debug(
                    "headphone_buttons: no device found, retry in %.0fs",
                    _DEVICE_RETRY_INTERVAL,
                )
                stop_evt.wait(_DEVICE_RETRY_INTERVAL)
                continue

        _device_name = name
        logger.info("headphone_buttons: auto-selected %s (%s)", path, name)

        # ── 打开设备 + 事件读循环 ─────────────────────────────────────────
        dev = None
        try:
            try:
                dev = evdev.InputDevice(path)
            except PermissionError:
                logger.error(
                    "headphone_buttons: no permission for %s. "
                    "user must be in 'input' group.", path
                )
                return  # 权限问题需手动修, 不 retry
            except Exception as e:
                logger.warning(
                    "headphone_buttons: cannot open %s: %s — retry in %.0fs",
                    path, e, _DEVICE_RETRY_INTERVAL,
                )
                _device_name = None
                stop_evt.wait(_DEVICE_RETRY_INTERVAL)
                continue

            _device_name = dev.name
            logger.info("headphone_buttons: listening on %s", dev.name)

            poll_s = _READ_TIMEOUT_MS / 1000.0
            while not stop_evt.is_set():
                try:
                    event = dev.read_one()
                except OSError as e:
                    logger.warning(
                        "headphone_buttons: read error: %s (device gone?) — retry in %.0fs",
                        e, _DEVICE_RETRY_INTERVAL,
                    )
                    break  # break 内层, 外层 retry

                if event is None:
                    stop_evt.wait(poll_s)
                    continue

                if event.type == ecodes.EV_KEY and event.code in _TRIGGER_CODES:
                    if event.value == 1:  # press edge
                        _dispatch()
                # ignore release (0) and hold (2)

        except Exception:
            logger.exception("headphone_buttons: evdev loop exception")
        finally:
            _device_name = None
            if dev is not None:
                try:
                    dev.close()
                except Exception:
                    pass

        # 断连后等一下再重新发现
        if not stop_evt.is_set():
            stop_evt.wait(_DEVICE_RETRY_INTERVAL)


def _dispatch() -> None:
    """通知全部 listener. 异常隔离."""
    with _listeners_lock:
        cbs = list(_listeners.values())
    for cb in cbs:
        try:
            cb()
        except Exception:
            logger.exception("headphone_buttons: listener raised (isolated)")
