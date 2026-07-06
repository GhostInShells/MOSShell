"""
IMU runtime — G1 机身姿态当前快照 + 定时采样轨迹.

设计纪律见同目录 README.md. 范式样例: asr.py.

数据语义:
  - 当前态: sdk.state.imu() 持续维护. read_current() 拿当前样本.
  - 历史轨迹: IMU 高频连续流, 不可能全留. 2Hz 降采样, ring buffer 默认 10 帧 (5s window).
  - quat 不存 — LLM 无法解读四元数, 静止时也不需要它做姿态推断. 内部计算另开接口.

物理事实 (来自 sdk/state.py + 实测):
  - rpy 来自 LowState_.imu_state, 单位 rad. 转 deg 给 LLM 更友好.
  - **rpy 零位 / 正向坐标系未实测校准**. 本期 helper 仅以差分形态喂 LLM ("姿态变化了多少"),
    绝对解读 ("现在朝东 / 前倾") 不可靠.
  - accel 含重力, 静止站立时 |a| ≈ 9.8 m/s². 偏离用于异常检测.
  - gyro 静止时近 0.

调用样例:
    from ghoshell_moss_contrib.unitree.g1.sdk import bootstrap
    from ghoshell_moss_contrib.unitree.g1.runtime import imu

    bootstrap(nic="eth0")
    imu.start()                          # 2Hz 定时采样
    snap = imu.read_current()            # 当前一帧
    batch = imu.drain()                  # 取走 ring buffer
    imu.stop()
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from typing import Callable, Optional

from pydantic import BaseModel, Field

from ghoshell_moss.message import Message, unique_id

from ghoshell_moss_contrib.unitree.g1 import sdk

logger = logging.getLogger("moss.g1.runtime.imu")


# ── 数据契约 ──────────────────────────────────────────────────────────────

class ImuSample(BaseModel):
    """G1 IMU 一帧采样. 字段单位都是 SI (rad / rad·s⁻¹ / m·s⁻²)."""

    id: str = Field(
        default_factory=unique_id,
        description="本条采样的 ulid. runtime 自生成 (G1 IMU 不带 id).",
    )
    roll_rad: float = Field(
        ...,
        description="机身横滚角 (rad). **零位 / 正向未实测校准, 仅作差分使用**.",
    )
    pitch_rad: float = Field(
        ...,
        description="机身俯仰角 (rad). **零位 / 正向未实测校准, 仅作差分使用**.",
    )
    yaw_rad: float = Field(
        ...,
        description="机身偏航角 (rad), 类似指南针角度. **相对开机时的参考, 会漂移**, "
                    "不可作为绝对方位使用.",
    )
    gyro_xyz: tuple[float, float, float] = Field(
        ...,
        description="三轴角速度 (rad/s). 静止时近 0. 模长大 = 在剧烈转动.",
    )
    accel_xyz: tuple[float, float, float] = Field(
        ...,
        description="三轴线加速度 (m/s², 含重力). 静止站立时 z≈9.8, x≈y≈0. "
                    "偏离 9.8 m/s² → 有外力或运动加速.",
    )
    captured_at: float = Field(
        default_factory=time.time,
        description="本帧采样的本地时间 (time.time() 秒).",
    )
    source: str = Field(default="g1.imu", description="数据来源固定常量.")


class ImuHistoryBatch(BaseModel):
    """drain 一次的返回. samples 按采样时间升序."""

    current: ImuSample = Field(..., description="drain 时刻的当前一帧.")
    samples: list[ImuSample] = Field(
        default_factory=list,
        description="自上次 drain 起累积的采样, 按时间升序.",
    )
    window_seconds: float = Field(
        ...,
        description="samples 覆盖的时间跨度 (秒).",
    )
    sample_rate_hz: float = Field(
        ...,
        description="实际采样频率. 跟 start(sample_rate_hz=...) 一致.",
    )


# ── 模块级私有状态 ────────────────────────────────────────────────────────

_state_lock = threading.Lock()
_listeners_lock = threading.Lock()

_dq: deque[ImuSample] = deque(maxlen=10)
_listeners: dict[str, Callable[[ImuSample], None]] = {}

_thread: Optional[threading.Thread] = None
_running: bool = False
_stop_event: Optional[threading.Event] = None

_sample_interval: float = 0.5  # 2 Hz
_sample_rate_hz: float = 2.0
_first_drain_at: float = 0.0
_error_count: int = 0


# ── 公开接口 ─────────────────────────────────────────────────────────────

def start(*, sample_rate_hz: float = 2.0, buffer_size: int = 10) -> None:
    """启动 imu runtime. 幂等.

    前置: sdk.bootstrap() 已完成 + 首帧 LowState 已收到.

    :param sample_rate_hz: 采样频率. 默认 2Hz.
    :param buffer_size: ring buffer 容量 (帧数). 默认 10 = 5s window @ 2Hz.
    """
    global _dq, _thread, _running, _stop_event
    global _sample_interval, _sample_rate_hz
    global _first_drain_at, _error_count

    with _state_lock:
        if _running:
            logger.debug("start() 重入 — 已在运行, 跳过.")
            return

        if not sdk.is_started():
            raise RuntimeError(
                "g1 sdk monitor not started. call bootstrap() first."
            )

        # 探测首帧 — 否则 raise 让调用方知道 bootstrap 没真起来.
        sdk.imu()

        _dq = deque(maxlen=buffer_size)
        _sample_rate_hz = sample_rate_hz
        _sample_interval = 1.0 / sample_rate_hz
        _first_drain_at = time.time()
        _error_count = 0
        _stop_event = threading.Event()
        _running = True

        _thread = threading.Thread(
            target=_poll_loop,
            name="g1-imu-sampler",
            daemon=True,
        )
        _thread.start()
        logger.info(
            "imu runtime started (sample_rate=%.1fHz, buffer_size=%d)",
            sample_rate_hz, buffer_size,
        )


def stop(timeout: float = 2.0) -> None:
    """停止 imu runtime. 幂等."""
    global _thread, _running, _stop_event

    with _state_lock:
        if not _running:
            logger.debug("stop() 重入 — 未在运行, 跳过.")
            return
        _running = False
        if _stop_event is not None:
            _stop_event.set()
        thread = _thread

    if thread is not None:
        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.warning("imu sampler 未在 %.1fs 内 join 完成.", timeout)

    with _state_lock:
        _thread = None
        _stop_event = None

    logger.info("imu runtime stopped.")


def is_running() -> bool:
    with _state_lock:
        return _running


def read_current() -> ImuSample:
    """读当前 IMU 一帧. 不出栈, 不影响 ring buffer.

    Raises:
        RuntimeError: sdk monitor 未启动或首帧未到.
    """
    i = sdk.imu()
    return ImuSample(
        roll_rad=i.rpy[0],
        pitch_rad=i.rpy[1],
        yaw_rad=i.rpy[2],
        gyro_xyz=i.gyro,
        accel_xyz=i.accel,
    )


def drain() -> ImuHistoryBatch:
    """取走采样 buffer + 附带当前快照."""
    global _first_drain_at

    now = time.time()
    with _state_lock:
        samples = list(_dq)
        _dq.clear()
        window = now - _first_drain_at
        _first_drain_at = now
        rate = _sample_rate_hz

    return ImuHistoryBatch(
        current=read_current(),
        samples=samples,
        window_seconds=round(window, 3),
        sample_rate_hz=rate,
    )


def peek_latest() -> Optional[ImuSample]:
    """看 buffer 末尾一条, 不出栈."""
    with _state_lock:
        if not _dq:
            return None
        return _dq[-1]


def register_listener(cb: Callable[[ImuSample], None]) -> str:
    """注册 sample 回调. cb 在 sampler 线程内同步触发, 不能阻塞.

    注意: 2Hz 采样下 listener 每 500ms 被触发一次, 实机看显示是密集刷屏的.
    场景脚本里建议在 listener 内自行 throttle (例如每 5 条打印 1 条) 或只
    打印显著变化 — runtime 不替 listener 做语义判断.
    """
    handle = unique_id()
    with _listeners_lock:
        _listeners[handle] = cb
    return handle


def unregister_listener(handle: str) -> None:
    with _listeners_lock:
        _listeners.pop(handle, None)


def health() -> dict:
    with _state_lock:
        return {
            "running": _running,
            "sample_rate_hz": _sample_rate_hz,
            "buffer_len": len(_dq),
            "buffer_max": _dq.maxlen,
            "error_count": _error_count,
            "seconds_since_first_drain": round(time.time() - _first_drain_at, 3),
        }


# ── 内部: sampler 线程 ───────────────────────────────────────────────────

def _poll_loop() -> None:
    """定时采样 sdk.state.imu() 入 ring buffer.

    异常隔离纪律: log + 短 sleep + 继续.
    """
    global _error_count

    logger.info("imu sampler loop entered.")
    stop_event = _stop_event
    while not stop_event.is_set():
        try:
            sample = read_current()
            _enqueue(sample)
        except Exception:
            _error_count += 1
            logger.exception("imu sampler 异常 (累计 %d).", _error_count)
            time.sleep(0.1)
        stop_event.wait(_sample_interval)
    logger.info("imu sampler loop exited.")


def _enqueue(sample: ImuSample) -> None:
    """入 buffer + 触发 listeners."""
    with _state_lock:
        _dq.append(sample)

    with _listeners_lock:
        snapshot = list(_listeners.values())
    for cb in snapshot:
        try:
            cb(sample)
        except Exception:
            logger.exception("imu listener 回调异常 (隔离).")


# ── 无状态 helper (channel 层用) ─────────────────────────────────────────
# 给 LLM 的表达原则:
#   - 角度 rad → deg (LLM 友好, 数字直观)
#   - gyro / accel 三轴聚合为模长 (LLM 不需要三轴细节判断"是否在动")
#   - 同值列折叠 (yaw 不变时不重复列)
#   - 不做 "is_stable / is_rotating" boolean — 让 channel 层按业务阈值判断

def _gyro_magnitude(s: ImuSample) -> float:
    gx, gy, gz = s.gyro_xyz
    return math.sqrt(gx * gx + gy * gy + gz * gz)


def _accel_magnitude(s: ImuSample) -> float:
    ax, ay, az = s.accel_xyz
    return math.sqrt(ax * ax + ay * ay + az * az)


def _rad_to_deg(rad: float) -> float:
    return rad * 180.0 / math.pi


def sample_to_xml(s: ImuSample) -> str:
    """单帧 → 紧凑 XML."""
    roll_d = _rad_to_deg(s.roll_rad)
    pitch_d = _rad_to_deg(s.pitch_rad)
    yaw_d = _rad_to_deg(s.yaw_rad)
    return (
        f'<{s.source} id="{s.id}" ts="{s.captured_at:.3f}" '
        f'roll="{roll_d:+.1f}°" pitch="{pitch_d:+.1f}°" yaw="{yaw_d:+.1f}°" '
        f'|accel|="{_accel_magnitude(s):.2f}" |gyro|="{_gyro_magnitude(s):.3f}"/>'
    )


def batch_to_xml(b: ImuHistoryBatch) -> str:
    """轨迹 batch → 表格化 XML. 折叠 yaw 等列同值."""
    lines = [
        f'<g1.imu window="{b.window_seconds:.1f}s" '
        f'rate="{b.sample_rate_hz:.1f}Hz" samples="{len(b.samples)}">',
    ]

    # current 行
    cur = b.current
    lines.append(
        f'  current: roll={_rad_to_deg(cur.roll_rad):+.1f}° '
        f'pitch={_rad_to_deg(cur.pitch_rad):+.1f}° '
        f'yaw={_rad_to_deg(cur.yaw_rad):+.1f}° '
        f'|accel|={_accel_magnitude(cur):.2f} '
        f'|gyro|={_gyro_magnitude(cur):.3f}'
    )

    if not b.samples:
        lines.append('  (no samples in window)')
        lines.append('</g1.imu>')
        return "\n".join(lines)

    # 历史表 — 是否折叠 yaw 列
    yaws = [_rad_to_deg(s.yaw_rad) for s in b.samples]
    yaw_static = max(yaws) - min(yaws) < 0.5  # 漂移 <0.5° 视为不变

    if yaw_static:
        avg_yaw = sum(yaws) / len(yaws)
        lines.append(f'  recent samples (yaw≈{avg_yaw:+.1f}° throughout):')
        lines.append('    t       roll     pitch    |accel|  |gyro|')
        for s in b.samples:
            rel = s.captured_at - cur.captured_at
            lines.append(
                f'    {rel:+5.1f}s  {_rad_to_deg(s.roll_rad):+6.1f}°  '
                f'{_rad_to_deg(s.pitch_rad):+6.1f}°  '
                f'{_accel_magnitude(s):5.2f}    {_gyro_magnitude(s):5.3f}'
            )
    else:
        lines.append('  recent samples:')
        lines.append('    t       roll     pitch    yaw      |accel|  |gyro|')
        for s in b.samples:
            rel = s.captured_at - cur.captured_at
            lines.append(
                f'    {rel:+5.1f}s  {_rad_to_deg(s.roll_rad):+6.1f}°  '
                f'{_rad_to_deg(s.pitch_rad):+6.1f}°  '
                f'{_rad_to_deg(s.yaw_rad):+6.1f}°  '
                f'{_accel_magnitude(s):5.2f}    {_gyro_magnitude(s):5.3f}'
            )

    lines.append('  note: roll/pitch zero & positive direction NOT calibrated; '
                 'use deltas, not absolutes.')
    lines.append('</g1.imu>')
    return "\n".join(lines)


def sample_to_message(s: ImuSample) -> Message:
    """单帧 → Message."""
    return Message.new(
        tag=s.source,
        attributes={
            "id": s.id,
            "roll_deg": round(_rad_to_deg(s.roll_rad), 1),
            "pitch_deg": round(_rad_to_deg(s.pitch_rad), 1),
            "yaw_deg": round(_rad_to_deg(s.yaw_rad), 1),
        },
        timestamp=True,
    ).with_content(sample_to_xml(s))


def batch_to_message(b: ImuHistoryBatch) -> Message:
    return Message.new(
        tag="g1.imu",
        attributes={
            "samples": len(b.samples),
            "window_seconds": round(b.window_seconds, 1),
            "sample_rate_hz": round(b.sample_rate_hz, 1),
        },
        timestamp=True,
    ).with_content(batch_to_xml(b))
