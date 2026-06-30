"""
Motion runtime — G1 FSM 模式变化事件. RPC 轮询 (2Hz).

数据源: sdk.get_fsm_id() — LocoClient._Call(7001) RPC. 2026-07-01 实机验证.
FsmMode 常量在 sdk._fsm, 官方文档 ID 表.

设计纪律见同目录 README.md.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable, Optional

from pydantic import BaseModel, Field

from ghoshell_moss.message import Message, unique_id

from ghoshell_moss_contrib.unitree.g1.sdk import get_fsm_id, FsmMode

logger = logging.getLogger("moss.g1.runtime.motion")


_FSM_NAMES: dict[int, str] = {
    FsmMode.UNKNOWN.value: "Unknown",
    FsmMode.ZERO_TORQUE.value: "ZeroTorque",
    FsmMode.DAMP.value: "Damp",
    FsmMode.SIT.value: "Sit",
    FsmMode.STAND.value: "Stand",
    FsmMode.REGULAR.value: "Regular",
    FsmMode.WALK_RUN.value: "WalkRun",
}


def _fsm_name(fsm_id: int) -> str:
    return _FSM_NAMES.get(fsm_id, f"Unknown({fsm_id})")


class MotionSnapshot(BaseModel):
    fsm_id: int = Field(...)
    mode_name: str = Field(...)
    captured_at: float = Field(default_factory=time.time)
    source: str = Field(default="g1.motion")


class MotionTransition(BaseModel):
    id: str = Field(default_factory=unique_id)
    from_id: int = Field(...)
    from_name: str = Field(...)
    to_id: int = Field(...)
    to_name: str = Field(...)
    at: float = Field(...)


class MotionHistoryBatch(BaseModel):
    current: MotionSnapshot = Field(...)
    transitions: list[MotionTransition] = Field(default_factory=list)
    window_seconds: float = Field(...)


_state_lock = threading.Lock()
_listeners_lock = threading.Lock()

_dq: deque[MotionTransition] = deque(maxlen=64)
_listeners: dict[str, Callable[[MotionTransition], None]] = {}

_thread: Optional[threading.Thread] = None
_running: bool = False
_stop_event: Optional[threading.Event] = None

_last_fsm_id: int = FsmMode.UNKNOWN.value
_first_drain_at: float = 0.0
_error_count: int = 0
_transition_count: int = 0


def start(*, buffer_size: int = 64, poll_hz: float = 2.0) -> None:
    global _dq, _thread, _running, _stop_event
    global _last_fsm_id, _first_drain_at, _error_count, _transition_count

    with _state_lock:
        if _running:
            return
        try:
            _last_fsm_id = get_fsm_id()
        except Exception:
            _last_fsm_id = FsmMode.UNKNOWN.value
        _dq = deque(maxlen=buffer_size)
        _first_drain_at = time.time()
        _error_count = 0
        _transition_count = 0
        _stop_event = threading.Event()
        _running = True
        _thread = threading.Thread(
            target=_poll_loop, name="g1-motion-poller",
            args=(1.0 / poll_hz,), daemon=True,
        )
        _thread.start()
        logger.info("motion started (%.1fHz, baseline=%s).", poll_hz, _fsm_name(_last_fsm_id))


def stop(timeout: float = 2.0) -> None:
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
            logger.warning("motion poller join timeout.")
    with _state_lock:
        _thread = None
        _stop_event = None
    logger.info("motion stopped.")


def is_running() -> bool:
    with _state_lock:
        return _running


def read_current() -> MotionSnapshot:
    mode = get_fsm_id()
    return MotionSnapshot(fsm_id=mode, mode_name=_fsm_name(mode))


def drain() -> MotionHistoryBatch:
    global _first_drain_at
    now = time.time()
    with _state_lock:
        transitions = list(_dq)
        _dq.clear()
        window = now - _first_drain_at
        _first_drain_at = now
    return MotionHistoryBatch(
        current=read_current(), transitions=transitions,
        window_seconds=round(window, 3),
    )


def peek_latest_transition() -> Optional[MotionTransition]:
    with _state_lock:
        return _dq[-1] if _dq else None


def register_listener(cb: Callable[[MotionTransition], None]) -> str:
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
            "fsm_id": _last_fsm_id, "fsm_name": _fsm_name(_last_fsm_id),
            "buffer_len": len(_dq), "buffer_max": _dq.maxlen,
            "transition_count": _transition_count, "error_count": _error_count,
            "window_sec": round(time.time() - _first_drain_at, 3),
        }


# ── poller ─────────────────────────────────────────────────────

def _poll_loop(interval: float) -> None:
    global _last_fsm_id, _error_count, _transition_count
    logger.info("motion poller (interval=%.2fs).", interval)
    stop_event = _stop_event
    while not stop_event.is_set():
        try:
            mode = get_fsm_id()
            if mode != _last_fsm_id:
                t = MotionTransition(
                    from_id=_last_fsm_id, from_name=_fsm_name(_last_fsm_id),
                    to_id=mode, to_name=_fsm_name(mode), at=time.time(),
                )
                _enqueue(t)
                _last_fsm_id = mode
        except Exception:
            _error_count += 1
            logger.exception("motion poller error (%d).", _error_count)
            time.sleep(0.1)
        stop_event.wait(interval)
    logger.info("motion poller exited.")


def _enqueue(t: MotionTransition) -> None:
    global _transition_count
    with _state_lock:
        _dq.append(t)
        _transition_count += 1
    with _listeners_lock:
        snapshot = list(_listeners.values())
    for cb in snapshot:
        try:
            cb(t)
        except Exception:
            logger.exception("motion listener error.")


# ── helpers ─────────────────────────────────────────────────────

def snapshot_to_xml(s: MotionSnapshot) -> str:
    return f'<{s.source} mode="{s.mode_name}" fsm_id="{s.fsm_id}" ts="{s.captured_at:.3f}"/>'


def batch_to_xml(b: MotionHistoryBatch) -> str:
    lines = [
        f'<g1.motion window="{b.window_seconds:.1f}s" transitions="{len(b.transitions)}">',
        f'  current mode={b.current.mode_name} (fsm_id={b.current.fsm_id})',
    ]
    if b.transitions:
        lines.append('  transitions:')
        for t in b.transitions:
            rel = t.at - b.current.captured_at
            lines.append(f'    T{rel:+.1f}s: {t.from_name}→{t.to_name} ({t.from_id}→{t.to_id})')
    else:
        lines.append('  no transitions in window.')
    lines.append('</g1.motion>')
    return "\n".join(lines)


def snapshot_to_message(s: MotionSnapshot) -> Message:
    return Message.new(
        tag=s.source, attributes={"mode": s.mode_name, "fsm_id": s.fsm_id},
        timestamp=True,
    ).with_content(snapshot_to_xml(s))


def batch_to_message(b: MotionHistoryBatch) -> Message:
    return Message.new(
        tag="g1.motion",
        attributes={"mode": b.current.mode_name, "transitions": len(b.transitions)},
        timestamp=True,
    ).with_content(batch_to_xml(b))
