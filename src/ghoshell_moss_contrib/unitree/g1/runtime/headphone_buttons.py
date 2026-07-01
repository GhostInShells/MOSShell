"""
Headphone buttons runtime — 蓝牙耳机按键事件监听.

跟 sdk/_buttons.py 同范式: evdev 后台线程监听耳机按键,
按下边沿时调用注册的回调. 回调跑在 evdev reader 线程, 不能阻塞.

OpenRun by Shokz (AVRCP): 单键 KEY_PLAYCD (code=200). 单击切换聆听开关.

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

# 关注的按键: KEY_PLAYCD = 200 (Shokz 单键)
_PLAYCD_CODE = 200

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


def _evdev_loop(device_path: str | None, stop_evt: threading.Event) -> None:
    """evdev 读循环, 跑在 daemon 线程内."""
    global _device_name

    # 设备发现
    if device_path is not None:
        path = device_path
    else:
        found = _find_device()
        if found[0] is None:
            logger.warning("headphone_buttons: no bluetooth input device found")
            return
        path, name = found
        _device_name = name
        logger.info("headphone_buttons: auto-selected %s (%s)", path, name)

    # 打开设备
    try:
        import evdev
        from evdev import ecodes
    except ImportError:
        logger.error("headphone_buttons: evdev not installed")
        return

    dev = None
    try:
        try:
            dev = evdev.InputDevice(path)
        except PermissionError:
            logger.error(
                "headphone_buttons: no permission for %s. "
                "user must be in 'input' group.", path
            )
            return
        except Exception as e:
            logger.error("headphone_buttons: cannot open %s: %s", path, e)
            return

        _device_name = _device_name or dev.name
        logger.info("headphone_buttons: listening on %s", dev.name)

        poll_s = _READ_TIMEOUT_MS / 1000.0
        while not stop_evt.is_set():
            try:
                event = dev.read_one()
            except OSError as e:
                logger.warning("headphone_buttons: read error: %s (device gone?)", e)
                break

            if event is None:
                # read_one 非阻塞, 无事件时 sleep 防 CPU 空转.
                # stop_evt 同时作为 sleep 的 timeout, shutdown 及时响应.
                stop_evt.wait(poll_s)
                continue

            if event.type == ecodes.EV_KEY and event.code == _PLAYCD_CODE:
                if event.value == 1:  # press edge
                    _dispatch()
            # ignore release (0) and hold (2)

    except Exception:
        logger.exception("headphone_buttons: evdev loop exception")
    finally:
        if dev is not None:
            try:
                dev.close()
            except Exception:
                pass


def _dispatch() -> None:
    """通知全部 listener. 异常隔离."""
    with _listeners_lock:
        cbs = list(_listeners.values())
    for cb in cbs:
        try:
            cb()
        except Exception:
            logger.exception("headphone_buttons: listener raised (isolated)")
