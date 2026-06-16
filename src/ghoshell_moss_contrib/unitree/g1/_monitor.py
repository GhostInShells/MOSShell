"""
G1 DDS 状态监控线程 — 20Hz 轮询 rt/lowstate, 2Hz 轮询电池/主板.

由 bootstrap() 启动, 作为 daemon 线程运行直到进程退出.
解析 DDS 帧 → 构建 frozen dataclass → 原子写入 state 模块.

设计:
  - 单线程轮询, 不做回调 — 控制轮询速率, 避免 DDS 回调线程竞争.
  - lowstate 每 50ms 读一次 (20Hz). bmsstate/mainboardstate 每 500ms.
  - 解析失败不影响下一帧 — 日志 + 继续.
"""

from __future__ import annotations

import struct
import threading
import time
import logging
from typing import Optional

_log = logging.getLogger("moss.g1.monitor")


# ═══════════════════════════════════════════════════════════════════════════════
# 遥控器解析 — 复现 SDK example wireless_controller.py 的位布局
# ═══════════════════════════════════════════════════════════════════════════════


class _RemoteParser:
    """从 LowState_.wireless_remote[40] 字节数组解析摇杆和按键."""

    _BTN_MAP = [
        # (byte_idx, bit, attr)
        (2, 0, 'r1'), (2, 1, 'l1'), (2, 2, 'start'), (2, 3, 'select'),
        (2, 4, 'r2'), (2, 5, 'l2'), (2, 6, 'f1'), (2, 7, 'f3'),
        (3, 0, 'a'), (3, 1, 'b'), (3, 2, 'x'), (3, 3, 'y'),
        (3, 4, 'up'), (3, 5, 'right'), (3, 6, 'down'), (3, 7, 'left'),
    ]

    @staticmethod
    def parse(data: bytes):
        """返回 (lx, ly, rx, ry, button_dict)."""
        from ghoshell_moss_contrib.unitree.g1.state import RemoteState

        btn = {name: bool((data[idx] >> bit) & 1) for idx, bit, name in _RemoteParser._BTN_MAP}

        lx = struct.unpack('<f', data[4:8])[0]
        rx = struct.unpack('<f', data[8:12])[0]
        ry = struct.unpack('<f', data[12:16])[0]
        ly = struct.unpack('<f', data[20:24])[0]

        return RemoteState(
            lx=lx, ly=ly, rx=rx, ry=ry,
            **btn,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DDS 帧解析 — LowState_ / BmsState_ / MainBoardState_ → frozen dataclass
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_lowstate(msg) -> None:
    """解析一帧 LowState_ → MotionState + JointsState + IMUState + RemoteState."""
    from ghoshell_moss_contrib.unitree.g1.state import (
        MotionState, JointState, JointsState, IMUState,
        _set_motion, _set_joints, _set_imu, _set_remote, _touch,
    )

    _set_motion(MotionState(
        fsm_mode=getattr(msg, 'mode_machine', 0),
        tick=getattr(msg, 'tick', 0),
    ))

    motor_states = getattr(msg, 'motor_state', [])
    joint_list = []
    for ms in motor_states:
        joint_list.append(JointState(
            q=getattr(ms, 'q', 0.0),
            dq=getattr(ms, 'dq', 0.0),
            tau=getattr(ms, 'tau_est', 0.0),
            mode=getattr(ms, 'mode', 0),
        ))
    _set_joints(JointsState(joints=tuple(joint_list)))

    imu = getattr(msg, 'imu_state', None)
    if imu is not None:
        rpy = getattr(imu, 'rpy', [0.0, 0.0, 0.0])
        gyro = getattr(imu, 'gyroscope', [0.0, 0.0, 0.0])
        accel = getattr(imu, 'accelerometer', [0.0, 0.0, 0.0])
        quat = getattr(imu, 'quaternion', [1.0, 0.0, 0.0, 0.0])
        _set_imu(IMUState(
            rpy=(float(rpy[0]), float(rpy[1]), float(rpy[2])),
            gyro=(float(gyro[0]), float(gyro[1]), float(gyro[2])),
            accel=(float(accel[0]), float(accel[1]), float(accel[2])),
            quat=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
        ))

    remote_raw = getattr(msg, 'wireless_remote', None)
    if remote_raw is not None:
        _set_remote(_RemoteParser.parse(bytes(remote_raw)))

    _touch()


def _parse_bmsstate(msg) -> None:
    """解析一帧 BmsState_ → BatteryState."""
    from ghoshell_moss_contrib.unitree.g1.state import BatteryState, _set_battery

    cells = getattr(msg, 'cell_vol', [])
    temps = getattr(msg, 'temperature', [])
    voltage_raw = getattr(msg, 'bmsvoltage', [0, 0, 0])
    # bmsvoltage 是两个 16-bit 值: [mv_low, mv_high, reserved]
    voltage = (voltage_raw[0] + (voltage_raw[1] << 16)) / 1000.0 if len(voltage_raw) >= 2 else 0.0

    _set_battery(BatteryState(
        soc=getattr(msg, 'soc', 0),
        soh=getattr(msg, 'soh', 0),
        voltage=voltage,
        current=getattr(msg, 'current', 0) / 1000.0 if hasattr(msg, 'current') else 0.0,
        cycle=getattr(msg, 'cycle', 0),
        temperature=tuple(int(t) for t in temps[:12] if t != 0),
        cells=tuple(int(c) for c in cells if c != 0),
    ))


def _parse_mainboard(msg) -> None:
    """解析一帧 MainBoardState_ → HealthState."""
    from ghoshell_moss_contrib.unitree.g1.state import HealthState, _set_health

    _set_health(HealthState(
        board_temp=getattr(msg, 'temperature', [0])[0],
        fan_state=tuple(getattr(msg, 'fan_state', [])),
        voltages=tuple(getattr(msg, 'value', [])),
    ))


# ═══════════════════════════════════════════════════════════════════════════════
# 监控线程
# ═══════════════════════════════════════════════════════════════════════════════


class _MonitorThread:
    """G1 状态监控线程.

    单线程轮询 3 个 DDS topic. lowstate 高频 (20Hz), 电池/主板低频 (2Hz).
    解析帧 → 原子写入 state 模块. 异常不退出线程 — 日志 + 重试.
    """

    # 轮询间隔
    LOWSTATE_INTERVAL = 0.05    # 20Hz
    SLOW_INTERVAL = 0.5         # 2Hz 用于电池/主板

    def __init__(self, nic: str):
        self._nic = nic
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # DDS subscribers — 在 run() 中初始化 (需要 DDS domain 已就绪)
        self._lowstate_sub = None
        self._bms_sub = None
        self._mainboard_sub = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="g1-state-monitor")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _init_subs(self) -> None:
        """创建 DDS 订阅者. 必须在 ChannelFactoryInitialize 之后调用."""
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, BmsState_, MainBoardState_

        self._lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._lowstate_sub.Init()

        self._bms_sub = ChannelSubscriber("rt/lf/bmsstate", BmsState_)
        self._bms_sub.Init()

        self._mainboard_sub = ChannelSubscriber("rt/lf/mainboardstate", MainBoardState_)
        self._mainboard_sub.Init()

    def _run(self) -> None:
        """监控线程主循环."""
        try:
            self._init_subs()
        except Exception:
            _log.exception("G1 monitor: DDS subscriber init failed, thread exiting")
            return

        _log.info("G1 monitor started: lowstate@20Hz slow@2Hz nic=%s", self._nic)
        last_slow = 0.0

        while not self._stop.is_set():
            try:
                # ── lowstate (20Hz) ──
                msg = self._lowstate_sub.Read(timeout=50)
                if msg is not None:
                    _parse_lowstate(msg)

                # ── slow topics (2Hz) ──
                now = time.monotonic()
                if now - last_slow >= self.SLOW_INTERVAL:
                    last_slow = now
                    bms = self._bms_sub.Read(timeout=200)
                    if bms is not None:
                        _parse_bmsstate(bms)
                    mb = self._mainboard_sub.Read(timeout=200)
                    if mb is not None:
                        _parse_mainboard(mb)

            except Exception:
                _log.exception("G1 monitor: frame parse error, continuing")
                time.sleep(0.5)

        # 清理
        for sub in (self._lowstate_sub, self._bms_sub, self._mainboard_sub):
            try:
                if sub is not None:
                    sub.Close()
            except Exception:
                pass
        _log.info("G1 monitor stopped")


_monitor: Optional[_MonitorThread] = None


def start_monitor(nic: str) -> None:
    global _monitor
    if _monitor is not None and _monitor._thread is not None and _monitor._thread.is_alive():
        return
    _monitor = _MonitorThread(nic)
    _monitor.start()


def stop_monitor() -> None:
    global _monitor
    if _monitor is not None:
        _monitor.stop()
        _monitor = None