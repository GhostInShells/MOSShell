"""
G1 数据同步监控 — cyclonedds callback 写入 state.py.

设计:
  - 不自建线程. cyclonedds reader 线程已存在, 我们注册 callback 跑在它上面.
  - 每个 topic 一个 ChannelSubscriber, queueLen=1 (cyclonedds 丢旧帧).
  - LowState 一帧四件事(motion + joints + imu + remote), 在同一个 callback 里依次 _set_*.
  - callback 跑在 reader 线程! GIL 保证 frozen dataclass 构造 + 引用赋值原子.
  - 任何解析异常 log + 继续 — 不能让一个错帧崩掉整个 reader 线程.

按键边沿检测也在 LowState callback 里做: 拿到新 remote 后跟上一帧比较 → 触发 _buttons 的 callback.

关键: 此模块**不假设 SDK 已可 import**. import unitree_sdk2py 延迟到 start_monitor() 里.
原因: state.py 必须能在不 import SDK 的环境下 import(尽管最终只在 PC2 跑).
"""

from __future__ import annotations

import logging
import struct
import threading
from typing import Any, Callable

from ghoshell_moss_contrib.unitree.g1._archived import state

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 模块级状态
# ═══════════════════════════════════════════════════════════════════════════════

_subscribers: list[Any] = []  # ChannelSubscriber 实例, 仅启动后填充
_started: bool = False
_lock = threading.Lock()

# 健康统计
_lowstate_count: int = 0
_battery_count: int = 0
_health_count: int = 0
_last_lowstate_time: float = 0.0
_last_battery_time: float = 0.0
_last_health_time: float = 0.0

# 上一帧 RemoteState — 用于按键边沿检测.
# 写入: LowState callback 内, reader 线程, 单线程写.
_last_remote: state.RemoteState | None = None

# 按键边沿回调 (button_name -> list of callbacks).
# 由 _buttons.py 注册. 解耦: monitor 不知道 _buttons 的存在, _buttons 提供注册函数.
_button_edge_callback: Callable[[state.RemoteState, state.RemoteState], None] | None = None


def set_button_edge_callback(cb: Callable[[state.RemoteState, state.RemoteState], None] | None) -> None:
    """由 _buttons.py 调一次, 注册边沿检测回调.

    cb 签名: cb(prev: RemoteState, curr: RemoteState) -> None
    cb 跑在 reader 线程! 实现端必须用 loop.call_soon_threadsafe.
    """
    global _button_edge_callback
    _button_edge_callback = cb


# ═══════════════════════════════════════════════════════════════════════════════
# 按键字节解析
# ═══════════════════════════════════════════════════════════════════════════════

# wireless_remote[40] 布局, 参考 sdk/04 + sdk/17 脚本.
# bytes[2..3]: 按键 bitfield
# bytes[4..7,8..11,12..15,20..23]: 4 个摇杆轴 float32

_KEY_BITS_DATA1 = [
    ('r1', 0), ('l1', 1), ('start', 2), ('select', 3),
    ('r2', 4), ('l2', 5), ('f1', 6), ('f3', 7),
]
_KEY_BITS_DATA2 = [
    ('a', 0), ('b', 1), ('x', 2), ('y', 3),
    ('up', 4), ('right', 5), ('down', 6), ('left', 7),
]


def _parse_wireless(data: bytes) -> state.RemoteState:
    data1, data2 = data[2], data[3]
    keys: dict[str, bool] = {}
    for name, bit in _KEY_BITS_DATA1:
        keys[name] = bool((data1 >> bit) & 1)
    for name, bit in _KEY_BITS_DATA2:
        keys[name] = bool((data2 >> bit) & 1)
    lx = struct.unpack('<f', data[4:8])[0]
    rx = struct.unpack('<f', data[8:12])[0]
    ry = struct.unpack('<f', data[12:16])[0]
    ly = struct.unpack('<f', data[20:24])[0]

    return state.RemoteState(
        lx=lx, ly=ly, rx=rx, ry=ry,
        l1=keys['l1'], l2=keys['l2'], r1=keys['r1'], r2=keys['r2'],
        a=keys['a'], b=keys['b'], x=keys['x'], y=keys['y'],
        up=keys['up'], down=keys['down'], left=keys['left'], right=keys['right'],
        select=keys['select'], start=keys['start'],
        f1=keys['f1'], f3=keys['f3'],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Callback 实现 — 跑在 cyclonedds reader 线程
# ═══════════════════════════════════════════════════════════════════════════════


def _on_lowstate(msg: Any) -> None:
    """rt/lowstate callback. 一帧四件事: motion + joints + imu + remote."""
    global _lowstate_count, _last_lowstate_time, _last_remote

    try:
        import time as _time
        now = _time.monotonic()

        # 1. motion
        motion_snap = state.MotionState(
            fsm_mode=int(getattr(msg, 'mode_machine', 0)),
            tick=int(getattr(msg, 'tick', 0)),
        )

        # 2. joints (35 槽)
        joint_snaps = []
        for ms in msg.motor_state:
            joint_snaps.append(state.JointState(
                q=float(ms.q),
                dq=float(ms.dq),
                tau=float(getattr(ms, 'tau_est', 0.0)),
                mode=int(ms.mode),
            ))
        joints_snap = state.JointsState(joints=tuple(joint_snaps))

        # 3. IMU
        imu_msg = msg.imu_state
        imu_snap = state.IMUState(
            rpy=(float(imu_msg.rpy[0]), float(imu_msg.rpy[1]), float(imu_msg.rpy[2])),
            gyro=(float(imu_msg.gyroscope[0]), float(imu_msg.gyroscope[1]), float(imu_msg.gyroscope[2])),
            accel=(float(imu_msg.accelerometer[0]), float(imu_msg.accelerometer[1]), float(imu_msg.accelerometer[2])),
            quat=(float(imu_msg.quaternion[0]), float(imu_msg.quaternion[1]),
                  float(imu_msg.quaternion[2]), float(imu_msg.quaternion[3])),
        )

        # 4. remote
        remote_snap = _parse_wireless(bytes(msg.wireless_remote))

        # 原子写入(GIL 保证)
        state._set_motion(motion_snap)
        state._set_joints(joints_snap)
        state._set_imu(imu_snap)
        state._set_remote(remote_snap)
        state._touch()

        _lowstate_count += 1
        _last_lowstate_time = now

        # 按键边沿
        prev_remote = _last_remote
        _last_remote = remote_snap
        if prev_remote is not None and _button_edge_callback is not None:
            try:
                _button_edge_callback(prev_remote, remote_snap)
            except Exception:
                logger.exception("button edge callback raised")

    except Exception:
        logger.exception("lowstate callback failed (frame dropped)")


def _on_bmsstate(msg: Any) -> None:
    """rt/lf/bmsstate callback."""
    global _battery_count, _last_battery_time
    try:
        import time as _time
        battery_snap = state.BatteryState(
            soc=int(getattr(msg, 'soc', 0)),
            soh=int(getattr(msg, 'soh', 0)),
            voltage=float(getattr(msg, 'vol', 0.0)),
            current=float(getattr(msg, 'current', 0.0)),
            cycle=int(getattr(msg, 'cycle', 0)),
            temperature=tuple(int(t) for t in getattr(msg, 'bms_temperature', ())),
            cells=tuple(int(c) for c in getattr(msg, 'cell_vol', ())),
        )
        state._set_battery(battery_snap)
        state._touch()
        _battery_count += 1
        _last_battery_time = _time.monotonic()
    except Exception:
        logger.exception("bmsstate callback failed")


def _on_mainboardstate(msg: Any) -> None:
    """rt/lf/mainboardstate callback."""
    global _health_count, _last_health_time
    try:
        import time as _time
        health_snap = state.HealthState(
            board_temp=int(getattr(msg, 'board_temperature', 0)),
            fan_state=tuple(int(f) for f in getattr(msg, 'fan_state', ())),
            voltages=tuple(float(v) for v in getattr(msg, 'voltage', ())),
        )
        state._set_health(health_snap)
        state._touch()
        _health_count += 1
        _last_health_time = _time.monotonic()
    except Exception:
        logger.exception("mainboardstate callback failed")


# ═══════════════════════════════════════════════════════════════════════════════
# 公共 API
# ═══════════════════════════════════════════════════════════════════════════════


def start_monitor() -> None:
    """启动监控. 注册 3 个 ChannelSubscriber + cyclonedds callback.

    必须在 ChannelFactoryInitialize 之后调用(由 _bootstrap 保证).
    幂等.
    """
    global _started

    with _lock:
        if _started:
            return

        # 延迟 import SDK — 让 state.py 能在不带 SDK 的环境 import.
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, BmsState_, MainBoardState_

        # LowState — 高频, queueLen=1 丢旧帧
        sub_lowstate = ChannelSubscriber("rt/lowstate", LowState_)
        sub_lowstate.Init(_on_lowstate, 1)
        _subscribers.append(sub_lowstate)
        logger.info("monitor: subscribed rt/lowstate")

        # BmsState — 低频
        try:
            sub_bms = ChannelSubscriber("rt/lf/bmsstate", BmsState_)
            sub_bms.Init(_on_bmsstate, 1)
            _subscribers.append(sub_bms)
            logger.info("monitor: subscribed rt/lf/bmsstate")
        except Exception:
            logger.exception("monitor: failed to subscribe bmsstate (battery will be unavailable)")

        # MainBoardState — 低频
        try:
            sub_mb = ChannelSubscriber("rt/lf/mainboardstate", MainBoardState_)
            sub_mb.Init(_on_mainboardstate, 1)
            _subscribers.append(sub_mb)
            logger.info("monitor: subscribed rt/lf/mainboardstate")
        except Exception:
            logger.exception("monitor: failed to subscribe mainboardstate (health will be unavailable)")

        state._mark_started()
        _started = True
        logger.info("monitor: started")


def stop_monitor() -> None:
    """停止监控. 主要给测试反复 init 用. 生产模式 = 进程退出即停."""
    global _started

    with _lock:
        if not _started:
            return

        for sub in _subscribers:
            try:
                sub.Close()
            except Exception:
                logger.exception("monitor: failed to close a subscriber")
        _subscribers.clear()

        state._mark_stopped()
        _started = True if False else False  # noqa — 显式 False
        _started = False
        logger.info("monitor: stopped")


def is_started() -> bool:
    return _started


def get_health() -> dict:
    """现场调试: 返回 monitor 当前统计."""
    import time as _time
    now = _time.monotonic()
    return {
        'started': _started,
        'subscribers': len(_subscribers),
        'lowstate': {
            'count': _lowstate_count,
            'last_age_sec': (now - _last_lowstate_time) if _last_lowstate_time > 0 else None,
        },
        'battery': {
            'count': _battery_count,
            'last_age_sec': (now - _last_battery_time) if _last_battery_time > 0 else None,
        },
        'health': {
            'count': _health_count,
            'last_age_sec': (now - _last_health_time) if _last_health_time > 0 else None,
        },
    }
