"""
Arm Joints runtime — G1 双臂 10 关节当前快照 + 定时采样轨迹.

设计纪律见同目录 README.md. 范式样例: asr.py.

数据语义:
  - 关节范围: G1 23-DoF 中双臂 10 关节 (左右肩 pitch/roll/yaw + 肘 + 腕 roll).
              槽位映射来自 `design/2026-06-30_g1_arms_animation.md` §4.5.
  - 当前态: sdk.state.joints() 提供 35 槽 frozen tuple, 这里只提取 10 个手臂槽.
  - 历史轨迹: 2Hz 降采样, ring buffer 默认 6 帧 (3s window).
  - rad 单位跟 arms keyframe API 一致 (设计文档 §2.1), 不转 deg —
    采样轨迹可直接喂回 save_animation 学习闭环.
  - **rad 零位 / 正反向未实测校准**, helper 在文末标注, 仅作差分使用.

调用样例:
    from ghoshell_moss_contrib.unitree.g1.sdk import bootstrap
    from ghoshell_moss_contrib.unitree.g1.runtime import arm_joints

    bootstrap(nic="eth0")
    arm_joints.start()
    snap = arm_joints.read_current()        # 当前 10 关节 q + dq + active
    batch = arm_joints.drain()              # 取走历史 samples
    arm_joints.stop()
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable, Optional

from pydantic import BaseModel, Field

from ghoshell_moss.message import Message, unique_id

from ghoshell_moss_contrib.unitree.g1 import sdk

logger = logging.getLogger("moss.g1.runtime.arm_joints")


# ── 槽位 ↔ 关节名映射 ─────────────────────────────────────────────────────
# 来自 design/2026-06-30_g1_arms_animation.md §4.5.
# LLM 永远只接触关节名, 不接触 index 数字.

ARM_JOINTS_BY_INDEX: dict[int, str] = {
    15: "left_shoulder_pitch",
    16: "left_shoulder_roll",
    17: "left_shoulder_yaw",
    18: "left_elbow",
    19: "left_wrist_roll",
    22: "right_shoulder_pitch",
    23: "right_shoulder_roll",
    24: "right_shoulder_yaw",
    25: "right_elbow",
    26: "right_wrist_roll",
}

ARM_JOINT_NAMES: tuple[str, ...] = tuple(ARM_JOINTS_BY_INDEX.values())

# 关节"在动"判定阈值. 静止时 dq 在传感器噪声范围内.
_DQ_MOVING_THRESHOLD: float = 0.05  # rad/s

# 静止关节折叠阈值. delta_q < 此值的关节不出现在历史段.
_HISTORY_DELTA_THRESHOLD: float = 0.05  # rad


# ── 数据契约 ──────────────────────────────────────────────────────────────

class ArmJointsSample(BaseModel):
    """G1 双臂 10 关节一帧采样. 键是关节名 (左右肩 pitch/roll/yaw + 肘 + 腕 roll)."""

    id: str = Field(
        default_factory=unique_id,
        description="本帧的 ulid. runtime 自生成.",
    )
    q: dict[str, float] = Field(
        ...,
        description="关节名 → 当前角度 (rad). 10 个关节固定: "
                    "left/right_shoulder_pitch/roll/yaw + left/right_elbow + left/right_wrist_roll. "
                    "**零位 / 正方向未实测校准**, 仅作差分参考. "
                    "单位与 arms keyframe API 一致, 可直接喂回 save_animation.",
    )
    dq: dict[str, float] = Field(
        ...,
        description="关节名 → 当前角速度 (rad/s). 静止时近 0.",
    )
    active: dict[str, bool] = Field(
        ...,
        description="关节名 → motor mode (True=engaged/active, False=free/idle). "
                    "Damp 模式下通常全 False (手臂可被手动推动); Sport 模式下通常全 True.",
    )
    captured_at: float = Field(
        default_factory=time.time,
        description="本帧采样的本地时间 (time.time() 秒).",
    )
    source: str = Field(default="g1.arm_joints", description="数据来源固定常量.")


class ArmJointsHistoryBatch(BaseModel):
    """drain 返回. samples 按时间升序."""

    current: ArmJointsSample = Field(
        ...,
        description="drain 时刻的当前一帧.",
    )
    samples: list[ArmJointsSample] = Field(
        default_factory=list,
        description="自上次 drain 起累积的采样, 按时间升序.",
    )
    window_seconds: float = Field(
        ...,
        description="samples 覆盖的时间跨度 (秒).",
    )
    sample_rate_hz: float = Field(
        ...,
        description="实际采样频率.",
    )


# ── 模块级私有状态 ────────────────────────────────────────────────────────

_state_lock = threading.Lock()
_listeners_lock = threading.Lock()

_dq_buffer: deque[ArmJointsSample] = deque(maxlen=6)
_listeners: dict[str, Callable[[ArmJointsSample], None]] = {}

_thread: Optional[threading.Thread] = None
_running: bool = False
_stop_event: Optional[threading.Event] = None

_sample_interval: float = 0.5  # 2 Hz
_sample_rate_hz: float = 2.0
_first_drain_at: float = 0.0
_error_count: int = 0


# ── 公开接口 ─────────────────────────────────────────────────────────────

def start(*, sample_rate_hz: float = 2.0, buffer_size: int = 6) -> None:
    """启动 arm_joints runtime. 幂等.

    :param sample_rate_hz: 采样频率. 默认 2Hz.
    :param buffer_size: ring buffer 容量. 默认 6 = 3s window @ 2Hz.
    """
    global _dq_buffer, _thread, _running, _stop_event
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

        sdk.joints()  # 探测首帧

        _dq_buffer = deque(maxlen=buffer_size)
        _sample_rate_hz = sample_rate_hz
        _sample_interval = 1.0 / sample_rate_hz
        _first_drain_at = time.time()
        _error_count = 0
        _stop_event = threading.Event()
        _running = True

        _thread = threading.Thread(
            target=_poll_loop,
            name="g1-arm-joints-sampler",
            daemon=True,
        )
        _thread.start()
        logger.info(
            "arm_joints runtime started (sample_rate=%.1fHz, buffer_size=%d)",
            sample_rate_hz, buffer_size,
        )


def stop(timeout: float = 2.0) -> None:
    """停止 arm_joints runtime. 幂等."""
    global _thread, _running, _stop_event

    with _state_lock:
        if not _running:
            return
        _running = False
        if _stop_event is not None:
            _stop_event.set()
        thread = _thread

    if thread is not None:
        thread.join(timeout=timeout)
        if thread.is_alive():
            logger.warning("arm_joints sampler 未在 %.1fs 内 join 完成.", timeout)

    with _state_lock:
        _thread = None
        _stop_event = None

    logger.info("arm_joints runtime stopped.")


def is_running() -> bool:
    with _state_lock:
        return _running


def read_current() -> ArmJointsSample:
    """读当前 10 关节一帧. 不出栈."""
    js = sdk.joints()
    q_map: dict[str, float] = {}
    dq_map: dict[str, float] = {}
    active_map: dict[str, bool] = {}
    for idx, name in ARM_JOINTS_BY_INDEX.items():
        joint = js.joints[idx]
        q_map[name] = joint.q
        dq_map[name] = joint.dq
        active_map[name] = bool(joint.mode == 1)
    return ArmJointsSample(q=q_map, dq=dq_map, active=active_map)


def drain() -> ArmJointsHistoryBatch:
    """取走 samples buffer + 附带当前快照."""
    global _first_drain_at

    now = time.time()
    with _state_lock:
        samples = list(_dq_buffer)
        _dq_buffer.clear()
        window = now - _first_drain_at
        _first_drain_at = now
        rate = _sample_rate_hz

    return ArmJointsHistoryBatch(
        current=read_current(),
        samples=samples,
        window_seconds=round(window, 3),
        sample_rate_hz=rate,
    )


def peek_latest() -> Optional[ArmJointsSample]:
    """看 buffer 末尾一条."""
    with _state_lock:
        if not _dq_buffer:
            return None
        return _dq_buffer[-1]


def register_listener(cb: Callable[[ArmJointsSample], None]) -> str:
    """注册 sample 回调. cb 在 sampler 线程内同步触发, 不能阻塞.

    2Hz 默认采样 → 每 500ms 触发一次. 场景脚本里通常需要 throttle 或只在
    显著变化 (运动中) 时打印.
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
            "buffer_len": len(_dq_buffer),
            "buffer_max": _dq_buffer.maxlen,
            "error_count": _error_count,
            "seconds_since_first_drain": round(time.time() - _first_drain_at, 3),
        }


# ── 内部: sampler 线程 ───────────────────────────────────────────────────

def _poll_loop() -> None:
    """定时采样 sdk.state.joints() 提取手臂槽, 入 ring buffer."""
    global _error_count

    logger.info("arm_joints sampler loop entered.")
    stop_event = _stop_event
    while not stop_event.is_set():
        try:
            sample = read_current()
            _enqueue(sample)
        except Exception:
            _error_count += 1
            logger.exception("arm_joints sampler 异常 (累计 %d).", _error_count)
            time.sleep(0.1)
        stop_event.wait(_sample_interval)
    logger.info("arm_joints sampler loop exited.")


def _enqueue(sample: ArmJointsSample) -> None:
    """入 buffer + 触发 listeners."""
    with _state_lock:
        _dq_buffer.append(sample)

    with _listeners_lock:
        snapshot = list(_listeners.values())
    for cb in snapshot:
        try:
            cb(sample)
        except Exception:
            logger.exception("arm_joints listener 回调异常 (隔离).")


# ── 无状态 helper (channel 层用) ─────────────────────────────────────────

def _moving_joints(sample: ArmJointsSample) -> list[str]:
    """当前帧 |dq| > 阈值的关节名列表."""
    return [name for name, v in sample.dq.items() if abs(v) > _DQ_MOVING_THRESHOLD]


def _active_count(sample: ArmJointsSample) -> int:
    return sum(1 for v in sample.active.values() if v)


def sample_to_xml(s: ArmJointsSample) -> str:
    """当前一帧 → 多行 XML. 关节按"左行/右行"两段, q 用 rad."""
    left_parts = []
    right_parts = []
    for name in ARM_JOINT_NAMES:
        q = s.q[name]
        suffix = name.split("_", 1)[1]  # left_shoulder_pitch → shoulder_pitch
        text = f'{suffix}={q:+.2f}'
        if name.startswith("left_"):
            left_parts.append(text)
        else:
            right_parts.append(text)

    moving = _moving_joints(s)
    active = _active_count(s)

    lines = [
        f'<{s.source} id="{s.id}" ts="{s.captured_at:.3f}" unit="rad">',
        f'  left  ' + ' '.join(left_parts),
        f'  right ' + ' '.join(right_parts),
        f'  motors_active={active}/10' + (
            f', moving_now={moving}' if moving else ', all_still'
        ),
        f'</{s.source}>',
    ]
    return "\n".join(lines)


def batch_to_xml(b: ArmJointsHistoryBatch) -> str:
    """轨迹 batch → 表格化 XML. 静止关节折叠, 只列变化超过阈值的."""
    lines = [
        f'<g1.arm_joints window="{b.window_seconds:.1f}s" '
        f'rate="{b.sample_rate_hz:.1f}Hz" samples="{len(b.samples)}" unit="rad">',
    ]

    cur = b.current

    # current 段 — 完整 10 关节, 两行
    left_parts = []
    right_parts = []
    for name in ARM_JOINT_NAMES:
        q = cur.q[name]
        suffix = name.split("_", 1)[1]
        text = f'{suffix}={q:+.2f}'
        if name.startswith("left_"):
            left_parts.append(text)
        else:
            right_parts.append(text)
    lines.append(f'  current pose:')
    lines.append(f'    left  ' + ' '.join(left_parts))
    lines.append(f'    right ' + ' '.join(right_parts))

    moving = _moving_joints(cur)
    active = _active_count(cur)
    if moving:
        lines.append(f'    moving now: {", ".join(moving)}')
    lines.append(f'    motors: {active}/10 engaged')

    # 历史段 — 折叠静止关节
    if not b.samples:
        lines.append('  (no samples in window)')
        lines.append('</g1.arm_joints>')
        return "\n".join(lines)

    # 找出在 window 内变化 > 阈值的关节
    changed_joints: list[str] = []
    for name in ARM_JOINT_NAMES:
        qs = [s.q[name] for s in b.samples]
        if max(qs) - min(qs) > _HISTORY_DELTA_THRESHOLD:
            changed_joints.append(name)

    if not changed_joints:
        lines.append(
            f'  recent ({len(b.samples)} samples over {b.window_seconds:.1f}s): '
            f'all 10 joints stationary (max delta < {_HISTORY_DELTA_THRESHOLD} rad)'
        )
    else:
        lines.append(
            f'  recent samples (joints with delta > {_HISTORY_DELTA_THRESHOLD} rad shown; '
            f'others stationary):'
        )
        # 表头
        header = '    t      ' + '  '.join(f'{n:>22}' for n in changed_joints)
        lines.append(header)
        for s in b.samples:
            rel = s.captured_at - cur.captured_at
            row = f'    {rel:+5.1f}s ' + '  '.join(
                f'{s.q[n]:+22.2f}' for n in changed_joints
            )
            lines.append(row)

    lines.append(
        '  note: rad zero & positive-direction NOT calibrated; '
        'use deltas, not absolutes. unit matches arms keyframe API '
        '— values can feed directly into save_animation.'
    )
    lines.append('</g1.arm_joints>')
    return "\n".join(lines)


def sample_to_message(s: ArmJointsSample) -> Message:
    return Message.new(
        tag=s.source,
        attributes={
            "id": s.id,
            "moving_count": len(_moving_joints(s)),
            "active_count": _active_count(s),
        },
        timestamp=True,
    ).with_content(sample_to_xml(s))


def batch_to_message(b: ArmJointsHistoryBatch) -> Message:
    return Message.new(
        tag="g1.arm_joints",
        attributes={
            "samples": len(b.samples),
            "window_seconds": round(b.window_seconds, 1),
            "sample_rate_hz": round(b.sample_rate_hz, 1),
        },
        timestamp=True,
    ).with_content(batch_to_xml(b))
