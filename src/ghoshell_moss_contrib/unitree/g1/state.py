"""
G1 状态快照 — frozen dataclass + 模块级无锁读。

所有状态由后台 _monitor 线程原子写入 (引用赋值). 读取端不做任何 I/O,
不加锁, 不 await — 始终拿到一个一致的快照.

设计约束:
  - frozen + slots: 构造零 malloc (Python 3.11+ __init__ 在 C 层完成).
  - 引用赋值原子性: GIL 保证 `_current = new_snapshot()` 不会被读到中间态.
  - 读取函数绝不抛异常: 即使 monitor 未启动, 也返回默认空值.

用法:
  from ghoshell_moss_contrib.unitree.g1.state import motion, remote, battery
  if remote().is_estop:          # L2+B 急停
      ...
  if motion().fsm_mode == 6:     # Sport 模式
      ...
  if battery().soc < 20:         # 低电量
      ...
"""

from __future__ import annotations

from dataclasses import dataclass

# ═══════════════════════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class MotionState:
    """G1 运动模式快照 — 来自 rt/lowstate.

    fsm_mode 是 MOSS 命令可用性的核心门控:
      0=Damp(急停阻尼) 3=Sit(落座) 5=Start(启动) 6=Sport(运控).
      手臂动作仅 Sport 模式可用. 移动仅 Sport/Start.
    """

    fsm_mode: int
    """FSM 当前模式. 0=Damp, 3=Sit, 5=Start, 6=Sport."""
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
    """G1 全身关节 — 35 槽, 其中 0-28 为 G1 23-DoF, 29-34 保留."""

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
    """

    lx: float; ly: float
    rx: float; ry: float
    l1: bool; l2: bool; r1: bool; r2: bool
    a: bool; b: bool; x: bool; y: bool
    up: bool; down: bool; left: bool; right: bool
    select: bool; start: bool; f1: bool; f3: bool

    @property
    def is_estop(self) -> bool:
        """L2+B 双按 = 硬件急停. G1 FSM 进入 Damp, 不可绕过."""
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
    """电芯温度 (°C), 最多 12 路."""
    cells: tuple[int, ...]
    """各电芯电压 (mV), 最多 40 路."""


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

# 默认值确保 monitor 未启动时读取不抛异常, 返回语义明确的零值/空值.
# frozen 实例构造后不可变 — 读取端拿到的引用永远指向完整快照.

_current_motion = MotionState(fsm_mode=0, tick=0)
_current_joints = JointsState(joints=())
_current_imu = IMUState(
    rpy=(0.0, 0.0, 0.0),
    gyro=(0.0, 0.0, 0.0),
    accel=(0.0, 0.0, 0.0),
    quat=(1.0, 0.0, 0.0, 0.0),
)
_current_remote = RemoteState(
    lx=0.0, ly=0.0, rx=0.0, ry=0.0,
    l1=False, l2=False, r1=False, r2=False,
    a=False, b=False, x=False, y=False,
    up=False, down=False, left=False, right=False,
    select=False, start=False, f1=False, f3=False,
)
_current_battery = BatteryState(
    soc=0, soh=0, voltage=0.0, current=0.0,
    cycle=0, temperature=(), cells=(),
)
_current_health = HealthState(
    board_temp=0, fan_state=(), voltages=(),
)

# 最后一次更新时刻 (monotonic seconds). 用于判断数据新鲜度.
_last_update: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 读取函数 — O(1) 内存读, 零 I/O, 零锁
# ═══════════════════════════════════════════════════════════════════════════════


def motion() -> MotionState:
    """G1 当前运控模式. 命令可用性门控的核心输入."""
    return _current_motion


def joints() -> JointsState:
    """G1 23-DoF 关节角度/速度/力矩. 槽位索引 0-28, 29-34 保留."""
    return _current_joints


def imu() -> IMUState:
    """机身 IMU — 横滚/俯仰/偏航, 角速度, 线加速度."""
    return _current_imu


def remote() -> RemoteState:
    """遥控器摇杆/按键. remote().is_estop 判 L2+B 急停."""
    return _current_remote


def battery() -> BatteryState:
    """电池 SOC/电压/温度/循环."""
    return _current_battery


def health() -> HealthState:
    """主板温度/风扇/电压."""
    return _current_health


def last_update() -> float:
    """最近一次状态刷新时刻 (time.monotonic). 0 = 尚未收到任何帧."""
    return _last_update


# ═══════════════════════════════════════════════════════════════════════════════
# 写入端 — 仅供 _monitor.py 调用
# ═══════════════════════════════════════════════════════════════════════════════


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


def _touch() -> None:
    global _last_update
    import time
    _last_update = time.monotonic()