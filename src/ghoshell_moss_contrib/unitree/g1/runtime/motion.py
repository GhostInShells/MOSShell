"""
Motion runtime — G1 FSM 模式当前快照 + 事件触发的轨迹.

设计纪律见同目录 README.md. 范式样例: asr.py.

数据语义:
  - 当前态: sdk.state.motion() 已经维护, 这里只做语义化包装 (fsm_mode → mode_name).
  - 历史轨迹: FSM 模式是离散事件 (Sit → Start → Sport), 99% 时间不变.
              定时采样无意义. runtime daemon 以 10Hz 轮询 sdk.state.motion(),
              检测 fsm_mode 变化时入 ring buffer.

调用样例:
    from ghoshell_moss_contrib.unitree.g1.sdk import bootstrap
    from ghoshell_moss_contrib.unitree.g1.runtime import motion

    bootstrap(nic="eth0")
    motion.start()
    snap = motion.read_current()          # 当前 FSM 模式 + 名称
    batch = motion.drain()                # 取走 transitions, 含 current 一并返回
    motion.stop()
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
from ghoshell_moss_contrib.unitree.g1.runtime.story_202607_fsm import FsmMode

logger = logging.getLogger("moss.g1.runtime.motion")


# ── FSM 名称映射 ──────────────────────────────────────────────────────────
# story_202607_fsm.FsmMode 用 IntEnum, 但 DAMP / ZERO_TORQUE 共享 value 0,
# Python enum 只有第一个有 .name. 这里手动维护 id → name 表, 多 ID 别名按
# "物理 ID 解析为最常见语义" 原则: 0 默认 Damp (急停形态), ZeroTorque 仅在
# 操作者主动切入时才能区分, runtime 层无从感知, 故合并表达.

_FSM_NAMES: dict[int, str] = {
    FsmMode.UNKNOWN.value: "Unknown",
    FsmMode.DAMP.value: "Damp",       # 0, 与 ZeroTorque 共享 ID
    FsmMode.SIT.value: "Sit",         # 3
    FsmMode.STAND.value: "Stand",     # 4
    FsmMode.START.value: "Start",     # 5
    FsmMode.SPORT.value: "Sport",     # 6
}


def _fsm_name(fsm_mode: int) -> str:
    """FSM ID → 语义化名称. 未知 ID 返回 'Unknown(<id>)'."""
    name = _FSM_NAMES.get(fsm_mode)
    if name is not None:
        return name
    return f"Unknown({fsm_mode})"


# ── 数据契约 ──────────────────────────────────────────────────────────────
# Field description 直接进 LLM prompt (channel 层 model_json_schema 时).

class MotionSnapshot(BaseModel):
    """G1 当前 FSM 模式快照 — sdk.state.motion() 的语义化包装."""

    fsm_mode: int = Field(
        ...,
        description="原始 FSM ID. 已知值: 0=Damp/ZeroTorque, 3=Sit, 4=Stand, 5=Start, 6=Sport. "
                    "Dance/Debug 等模式 ID 待实测.",
    )
    mode_name: str = Field(
        ...,
        description="FSM 模式的语义化名称 (Damp/Sit/Stand/Start/Sport). 未知 ID 返回 'Unknown(<id>)'.",
    )
    tick: int = Field(
        ...,
        description="LowState 帧计数器, 单调递增. 一般 LLM 不需要读这个字段.",
    )
    captured_at: float = Field(
        default_factory=time.time,
        description="本快照的本地时间 (time.time() 秒).",
    )
    source: str = Field(
        default="g1.motion",
        description="数据来源固定常量.",
    )


class MotionTransition(BaseModel):
    """G1 FSM 模式发生变化的事件 — runtime daemon 检测到 fsm_mode 改变时记录."""

    id: str = Field(
        default_factory=unique_id,
        description="本事件的 ulid. runtime 自生成.",
    )
    from_mode: int = Field(..., description="变化前的 FSM ID.")
    from_name: str = Field(..., description="变化前的 FSM 名称.")
    to_mode: int = Field(..., description="变化后的 FSM ID.")
    to_name: str = Field(..., description="变化后的 FSM 名称.")
    at: float = Field(..., description="变化发生时刻 (time.time() 秒).")


class MotionHistoryBatch(BaseModel):
    """drain 一次的返回. transitions 按发生时间升序, 配合 current 表达完整时序."""

    current: MotionSnapshot = Field(
        ...,
        description="drain 时刻的当前 FSM 快照. 配合 transitions 看完整时序.",
    )
    transitions: list[MotionTransition] = Field(
        default_factory=list,
        description="自上次 drain 起捕获的 FSM 模式变化事件, 按发生时间升序.",
    )
    window_seconds: float = Field(
        ...,
        description="transitions 覆盖的时间跨度 (秒). 0 表示无变化事件.",
    )


# ── 模块级私有状态 ────────────────────────────────────────────────────────

_state_lock = threading.Lock()
_listeners_lock = threading.Lock()

_dq: deque[MotionTransition] = deque(maxlen=64)
_listeners: dict[str, Callable[[MotionTransition], None]] = {}

_thread: Optional[threading.Thread] = None
_running: bool = False
_stop_event: Optional[threading.Event] = None

_last_fsm_mode: int = FsmMode.UNKNOWN.value
_first_drain_at: float = 0.0
_error_count: int = 0
_poll_interval: float = 0.1  # 10 Hz


# ── 公开接口 ─────────────────────────────────────────────────────────────

def start(*, buffer_size: int = 64, poll_hz: float = 10.0) -> None:
    """启动 motion runtime. 幂等.

    前置: sdk.bootstrap() 已完成 — 否则 sdk.state.motion() raise.

    :param buffer_size: ring buffer 容量 (transitions 数). 满则自动挤掉最旧.
    :param poll_hz: 轮询 fsm_mode 变化的频率. 10Hz 默认足够 (FSM 变化是秒级事件).
    """
    global _dq, _thread, _running, _stop_event
    global _last_fsm_mode, _first_drain_at, _error_count, _poll_interval

    with _state_lock:
        if _running:
            logger.debug("start() 重入 — 已在运行, 跳过.")
            return

        # 检查 sdk 已就绪 — 早炸早死, 不留下默认零值掩盖问题.
        if not sdk.is_started():
            raise RuntimeError(
                "g1 sdk monitor not started. call ghoshell_moss_contrib.unitree.g1.sdk.bootstrap() first."
            )

        # 拿首帧 fsm_mode 作为 baseline. monitor 起来但首帧未到时 motion() 也会 raise,
        # 这里让它 raise 出去 — 调用方需要等到 bootstrap 真正收到帧.
        baseline = sdk.motion()
        _last_fsm_mode = baseline.fsm_mode

        _dq = deque(maxlen=buffer_size)
        _first_drain_at = time.time()
        _error_count = 0
        _poll_interval = 1.0 / poll_hz
        _stop_event = threading.Event()
        _running = True

        _thread = threading.Thread(
            target=_poll_loop,
            name="g1-motion-poller",
            daemon=True,
        )
        _thread.start()
        logger.info(
            "motion runtime started (buffer_size=%d, poll_hz=%.1f, baseline=%s)",
            buffer_size, poll_hz, _fsm_name(_last_fsm_mode),
        )


def stop(timeout: float = 2.0) -> None:
    """停止 motion runtime. 幂等."""
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
            logger.warning(
                "motion poller 未在 %.1fs 内 join 完成 (daemon, 随进程退出).",
                timeout,
            )

    with _state_lock:
        _thread = None
        _stop_event = None

    logger.info("motion runtime stopped.")


def is_running() -> bool:
    with _state_lock:
        return _running


def read_current() -> MotionSnapshot:
    """读当前 FSM 快照. 不出栈, 不影响 ring buffer.

    Raises:
        RuntimeError: sdk monitor 未启动或首帧未到 (透传 sdk.state.motion() 的异常).
    """
    m = sdk.motion()  # raise on not started / no frame
    return MotionSnapshot(
        fsm_mode=m.fsm_mode,
        mode_name=_fsm_name(m.fsm_mode),
        tick=m.tick,
    )


def drain() -> MotionHistoryBatch:
    """取走 transitions buffer + 附带当前快照.

    window_seconds = 从上次 drain (或 start) 到现在的真实跨度, 跟 transitions
    是否存在无关 — 用来告知模型 "我看了多久".
    """
    global _first_drain_at

    now = time.time()
    with _state_lock:
        transitions = list(_dq)
        _dq.clear()
        window = now - _first_drain_at
        _first_drain_at = now

    current = read_current()
    return MotionHistoryBatch(
        current=current,
        transitions=transitions,
        window_seconds=round(window, 3),
    )


def peek_latest_transition() -> Optional[MotionTransition]:
    """看 buffer 末尾一条 transition, 不出栈. 无数据返回 None."""
    with _state_lock:
        if not _dq:
            return None
        return _dq[-1]


def register_listener(cb: Callable[[MotionTransition], None]) -> str:
    """注册 transition 回调. cb 在 poller 线程内同步触发, 不能阻塞.

    跨线程需求 (推回 asyncio loop / queue) 由 cb 自行处理.
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
            "last_seen_fsm_mode": _last_fsm_mode,
            "last_seen_fsm_name": _fsm_name(_last_fsm_mode),
            "buffer_len": len(_dq),
            "buffer_max": _dq.maxlen,
            "error_count": _error_count,
            "poll_interval": _poll_interval,
            "seconds_since_first_drain": round(time.time() - _first_drain_at, 3),
        }


# ── 内部: poller 线程 ────────────────────────────────────────────────────

def _poll_loop() -> None:
    """轮询 sdk.state.motion(), 检测 fsm_mode 变化即入 buffer.

    异常隔离纪律 (runtime README §4): 任何异常 log + 短 sleep + 继续.
    """
    global _last_fsm_mode, _error_count

    logger.info("motion poller loop entered.")
    stop_event = _stop_event
    while not stop_event.is_set():
        try:
            m = sdk.motion()
            if m.fsm_mode != _last_fsm_mode:
                transition = MotionTransition(
                    from_mode=_last_fsm_mode,
                    from_name=_fsm_name(_last_fsm_mode),
                    to_mode=m.fsm_mode,
                    to_name=_fsm_name(m.fsm_mode),
                    at=time.time(),
                )
                _enqueue(transition)
                _last_fsm_mode = m.fsm_mode
        except Exception:
            _error_count += 1
            logger.exception("motion poller 异常 (累计 %d).", _error_count)
            time.sleep(0.1)
        stop_event.wait(_poll_interval)
    logger.info("motion poller loop exited.")


def _enqueue(transition: MotionTransition) -> None:
    """入 buffer + 触发 listeners. 在 poller 线程内调用."""
    with _state_lock:
        _dq.append(transition)

    with _listeners_lock:
        snapshot = list(_listeners.values())
    for cb in snapshot:
        try:
            cb(transition)
        except Exception:
            logger.exception("motion listener 回调异常 (隔离).")


# ── 无状态 helper (channel 层用) ─────────────────────────────────────────

def snapshot_to_xml(s: MotionSnapshot) -> str:
    """当前快照 → XML 行. 用 channel 偶尔点查."""
    return (
        f'<{s.source} mode="{s.mode_name}" fsm_id="{s.fsm_mode}" '
        f'ts="{s.captured_at:.3f}"/>'
    )


def batch_to_xml(b: MotionHistoryBatch) -> str:
    """轨迹 batch → 多行 XML. 含 current + transitions."""
    lines = [
        f'<g1.motion window="{b.window_seconds:.1f}s" '
        f'transitions="{len(b.transitions)}">',
        f'  current mode={b.current.mode_name} (fsm_id={b.current.fsm_mode})',
    ]
    if b.transitions:
        lines.append('  recent transitions:')
        for t in b.transitions:
            rel = t.at - b.current.captured_at  # 负数 = 过去多少秒
            lines.append(
                f'    T{rel:+.1f}s: {t.from_name} → {t.to_name} '
                f'(fsm {t.from_mode} → {t.to_mode})'
            )
    else:
        lines.append('  no transitions in window.')
    lines.append('</g1.motion>')
    return "\n".join(lines)


def snapshot_to_message(s: MotionSnapshot) -> Message:
    """当前快照 → Message. channel context_messages 用."""
    return Message.new(
        tag=s.source,
        attributes={
            "mode": s.mode_name,
            "fsm_id": s.fsm_mode,
        },
        timestamp=True,
    ).with_content(snapshot_to_xml(s))


def batch_to_message(b: MotionHistoryBatch) -> Message:
    """batch → Message. channel pop 进 context_messages."""
    return Message.new(
        tag="g1.motion",
        attributes={
            "mode": b.current.mode_name,
            "transitions": len(b.transitions),
            "window_seconds": round(b.window_seconds, 1),
        },
        timestamp=True,
    ).with_content(batch_to_xml(b))
