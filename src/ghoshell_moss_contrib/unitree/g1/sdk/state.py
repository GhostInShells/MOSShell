"""
G1 状态快照 — frozen dataclass + 模块级原子读 + 启动检查.

设计原则:
  - 所有状态由 _monitor.py 的 cyclonedds reader 线程原子写入 (引用赋值, GIL 保证原子)
  - 读取端不做 I/O, 不加锁, 不 await — 始终拿到一个一致的快照
  - **monitor 未启动时调读取函数 → raise RuntimeError**
    这一条是关键: 上一版返回默认零值, 调用方误以为收到了 DDS 数据. 必须早炸早死.

用法:
    from ghoshell_moss_contrib.unitree.g1 import bootstrap, motion, remote, battery

    bootstrap(nic)             # bootstrap 完成 = monitor 启动 + 收到首帧
    if remote().is_estop:      # L2+B 急停
        ...
    if motion().fsm_mode == 6: # Sport 模式
        ...

直接 import state 不 bootstrap 而调读取函数:
    motion()  # raise RuntimeError("g1 monitor not started; call bootstrap() first")
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable
from uuid import uuid4

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 数据模型 — frozen + slots, 零拷贝读
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class MotionState:
    """G1 运动模式快照 — 来自 rt/lowstate.

    fsm_mode 是 MOSS 命令可用性的核心门控:
      0 = Damp (急停阻尼)
      3 = Sit (落座)
      5 = Start (基础站立)
      6 = Sport (运控全开)
      其他值可能在调试模式或 ZeroTorque, 需实测确认 (见 sdk/24).
    """
    fsm_mode: int
    """FSM 当前模式."""
    tick: int
    """LowState 帧计数器, 单调递增."""


@dataclass(frozen=True, slots=True)
class JointState:
    """单个关节瞬时状态 — 来自 LowState_.motor_state[i]."""
    q: float
    """关节角度 (rad)."""
    dq: float
    """关节角速度 (rad/s)."""
    tau: float
    """估计力矩 (Nm)."""
    mode: int
    """0=空闲, 1=使能(active)."""


@dataclass(frozen=True, slots=True)
class JointsState:
    """G1 全身关节 — 35 槽, 其中 0-28 为 G1 23-DoF, 29-34 保留(2026-06-16 实测确认)."""
    joints: tuple[JointState, ...]
    """按槽位索引. 长度固定 35."""


@dataclass(frozen=True, slots=True)
class IMUState:
    """机身惯性测量 — 来自 LowState_.imu_state."""
    rpy: tuple[float, float, float]
    """横滚/俯仰/偏航 (rad)."""
    gyro: tuple[float, float, float]
    """角速度 (rad/s)."""
    accel: tuple[float, float, float]
    """线加速度 (m/s^2)."""
    quat: tuple[float, float, float, float]
    """姿态四元数 (w,x,y,z)."""


@dataclass(frozen=True, slots=True)
class RemoteState:
    """遥控器瞬时状态 — 来自 LowState_.wireless_remote[40].

    摇杆值域 [-1, 1], 中位为 0. 按键为 bool.
    布局参见 sdk/17_remote_keys_passthrough.py 的 KEY_BITS_DATA1/DATA2.
    """
    lx: float
    ly: float
    rx: float
    ry: float
    l1: bool
    l2: bool
    r1: bool
    r2: bool
    a: bool
    b: bool
    x: bool
    y: bool
    up: bool
    down: bool
    left: bool
    right: bool
    select: bool
    start: bool
    f1: bool
    f3: bool

    @property
    def is_estop(self) -> bool:
        """L2+B 双按 = 硬件急停. G1 FSM 直接进 Damp, 不可绕过."""
        return self.l2 and self.b


@dataclass(frozen=True, slots=True)
class BatteryState:
    """电池状态 — 来自 rt/lf/bmsstate."""
    soc: int
    """电量百分比 (0-100)."""
    soh: int
    """健康度 (0-100)."""
    voltage: float
    """总电压 (V)."""
    current: float
    """电流 (A). 负值 = 放电."""
    cycle: int
    """充放电循环次数."""
    temperature: tuple[int, ...]
    """电芯温度 (°C)."""
    cells: tuple[int, ...]
    """各电芯电压 (mV)."""


@dataclass(frozen=True, slots=True)
class HealthState:
    """主板/系统健康 — 来自 rt/lf/mainboardstate."""
    board_temp: int
    """主板温度 (°C)."""
    fan_state: tuple[int, ...]
    """风扇转速标志."""
    voltages: tuple[float, ...]
    """各路电压 (V)."""


# ═══════════════════════════════════════════════════════════════════════════════
# 模块级状态 — 由 _monitor 线程原子更新
# ═══════════════════════════════════════════════════════════════════════════════

# monitor 是否已启动. 决定读取函数是否 raise.
# 由 _mark_started() 写入. 这是 monitor 启动信号, 不是 LowState 到达信号 — 后者用 _last_update 判断.
_monitor_started: bool = False

# 各状态快照. 默认 None — monitor 未启动 raise, monitor 启动后但首帧未到也 raise.
# 用 None 而非默认零值, 是因为零值会诱使调用方"看似工作".
_current_motion: MotionState | None = None
_current_joints: JointsState | None = None
_current_imu: IMUState | None = None
_current_remote: RemoteState | None = None
_current_battery: BatteryState | None = None
_current_health: HealthState | None = None

# FSM mode from rt/sportmodestate (the real source). -1 = not received yet.
# mode_machine in LowState is DoF config (4=23Dof,5=29Dof,6=27Dof), NOT FSM.
# FEATURE.md §数据字段 documents this as a known trap from 2026-06-29.
_current_sport_mode: int = -1

# 最后一次更新时刻 (monotonic seconds). 用于判断数据新鲜度.
_last_update: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 读取函数 — O(1) 内存读, 零 I/O, 零锁, monitor 未启动 raise
# ═══════════════════════════════════════════════════════════════════════════════


def _check_started(field_name: str) -> None:
    if not _monitor_started:
        raise RuntimeError(
            f"g1 monitor not started; cannot read {field_name}. "
            f"call ghoshell_moss_contrib.unitree.g1.bootstrap() first."
        )


def motion() -> MotionState:
    """G1 当前运控模式. 命令可用性门控的核心输入.

    Raises:
        RuntimeError: monitor 未启动, 或启动但 LowState 首帧未到.
    """
    _check_started("motion")
    if _current_motion is None:
        raise RuntimeError("g1 monitor started but no LowState received yet")
    return _current_motion


def joints() -> JointsState:
    """G1 35 槽关节. 0-28 为 23-DoF, 29-34 保留."""
    _check_started("joints")
    if _current_joints is None:
        raise RuntimeError("g1 monitor started but no LowState received yet")
    return _current_joints


def imu() -> IMUState:
    """机身 IMU."""
    _check_started("imu")
    if _current_imu is None:
        raise RuntimeError("g1 monitor started but no LowState received yet")
    return _current_imu


def remote() -> RemoteState:
    """遥控器. remote().is_estop 判 L2+B."""
    _check_started("remote")
    if _current_remote is None:
        raise RuntimeError("g1 monitor started but no LowState received yet")
    return _current_remote


def battery() -> BatteryState:
    """电池. 注意低频, 启动后可能要等几秒才到首帧."""
    _check_started("battery")
    if _current_battery is None:
        raise RuntimeError("g1 monitor started but no bmsstate received yet")
    return _current_battery


def health() -> HealthState:
    """主板. 同 battery 低频."""
    _check_started("health")
    if _current_health is None:
        raise RuntimeError("g1 monitor started but no mainboardstate received yet")
    return _current_health


def sport_mode() -> int:
    """G1 真实 FSM 模式 ID. 来源 rt/sportmodestate, 不是 LowState 的 mode_machine.

    -1 = sportmodestate 首帧未到 (或订阅失败).
    已知值: 0=Damp/ZeroTorque, 3=Sit, 4=Stand, 5=Start, 6=Sport.
    """
    _check_started("sport_mode")
    return _current_sport_mode


def last_update() -> float:
    """最近一次状态刷新时刻 (time.monotonic). 0 = 尚未收到任何帧.

    这是一个"健康度"读取, 不 raise — 用于现场调试.
    """
    return _last_update


def is_started() -> bool:
    """monitor 是否已启动. 不 raise, 用于现场调试."""
    return _monitor_started


# ═══════════════════════════════════════════════════════════════════════════════
# 写入端 — 仅供 _monitor 调用. 模块私有约定, 不暴露到 __init__.py.
# ═══════════════════════════════════════════════════════════════════════════════


def _mark_started() -> None:
    """monitor 启动时调一次. 启动 raise 检查."""
    global _monitor_started
    _monitor_started = True


def _mark_stopped() -> None:
    """monitor 停止时调. 用于测试反复 init."""
    global _monitor_started
    _monitor_started = False


def _set_motion(m: MotionState) -> None:
    global _current_motion
    _current_motion = m


def _set_joints(js: JointsState) -> None:
    global _current_joints
    _current_joints = js


def _set_imu(i: IMUState) -> None:
    global _current_imu
    _current_imu = i


def _set_remote(r: RemoteState) -> None:
    global _current_remote
    _current_remote = r


def _set_battery(b: BatteryState) -> None:
    global _current_battery
    _current_battery = b


def _set_health(h: HealthState) -> None:
    global _current_health
    _current_health = h


def _set_sport_mode(mode: int) -> None:
    """由 _monitor sportmodestate callback 调. 仅在值变化时通知回调."""
    global _current_sport_mode
    old = _current_sport_mode
    if old == mode:
        return
    _current_sport_mode = mode
    _notify_sport_mode_callbacks(old, mode)


# ── sport_mode 回调注册 ────────────────────────────────────────────────────
# 回调在 cyclonedds reader 线程内同步触发 — 不能阻塞, 异常隔离.
# FSM 变化是秒级事件, 回调开销可忽略.

_sport_mode_lock = threading.Lock()
_sport_mode_callbacks: dict[str, Callable[[int, int], None]] = {}
# cb(old_mode: int, new_mode: int) -> None


def register_sport_mode_callback(cb: Callable[[int, int], None]) -> str:
    """注册 FSM 模式变化回调. 回调在 reader 线程内同步触发, 不能阻塞.

    返回 handle (str) 用于 unregister.
    注册时若已有已知 mode (≠-1), 立即以 (-1, current) 触发一次 cb.
    """
    handle = uuid4().hex
    with _sport_mode_lock:
        _sport_mode_callbacks[handle] = cb
    # 立即补发当前值 (lock 外, 避免 cb 内 register 死锁)
    current = _current_sport_mode
    if current >= 0:
        try:
            cb(-1, current)
        except Exception:
            logger.exception("sport_mode callback initial fire failed (isolated)")
    return handle


def unregister_sport_mode_callback(handle: str) -> None:
    """反注册. 未知 handle 静默忽略."""
    with _sport_mode_lock:
        _sport_mode_callbacks.pop(handle, None)


def _notify_sport_mode_callbacks(old: int, new: int) -> None:
    """通知所有注册回调. 在 _set_sport_mode 内调 (reader 线程)."""
    with _sport_mode_lock:
        snapshot = list(_sport_mode_callbacks.values())
    for cb in snapshot:
        try:
            cb(old, new)
        except Exception:
            logger.exception("sport_mode callback raised (isolated)")


def _touch() -> None:
    """每收到一帧调一次. 用于 last_update."""
    global _last_update
    _last_update = time.monotonic()


def _reset_all_for_testing() -> None:
    """测试 hook: 把所有状态归位到"未启动"初始态. 不暴露到 __init__.py."""
    global _monitor_started
    global _current_motion, _current_joints, _current_imu
    global _current_remote, _current_battery, _current_health
    global _last_update

    _monitor_started = False
    _current_motion = None
    _current_joints = None
    _current_imu = None
    _current_remote = None
    _current_battery = None
    _current_health = None
    _current_sport_mode = -1
    _last_update = 0.0
