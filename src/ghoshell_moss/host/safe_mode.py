"""SafeMode 实现 — GhostRuntime 的人工审批闸口, 与 PauseController 同级同风格.

设计要点 (对齐 ghost-runtime-safemode FEATURE):
    - 单一 pending, uuid 对齐防批错帧, 并发裁决用锁串行化.
    - Future 用 ``concurrent.futures.Future``: TUI 线程 set_result, async 侧
      ``asyncio.wrap_future`` await (决策 3).
    - Verdict 三态: approved / rejected / cancelled, ``message`` 字段供 approve-note
      与 reject-reason 共享 (决策 12/13).
    - 无历史记录, 无 pending changed 队列, 一读一回调 (决策 9/10).
"""

import threading
import uuid as _uuid
from concurrent.futures import Future
from typing import Callable

from ghoshell_moss.core.blueprint.host import PendingApproval, SafeMode, Verdict


class _Pending:
    """内部结构 — 承载 uuid, logos, future. 对外暴露只读 ``PendingApproval`` 快照."""

    __slots__ = ('uuid', 'logos', 'future')

    def __init__(self, logos: str) -> None:
        self.uuid: str = _uuid.uuid4().hex
        self.logos: str = logos
        self.future: Future[Verdict] = Future()

    def snapshot(self) -> PendingApproval:
        return {'uuid': self.uuid, 'logos': self.logos}


class SafeModeImpl(SafeMode):
    """SafeMode 具体实现 — 幂等 setter + uuid 对齐的三态裁决."""

    def __init__(self) -> None:
        self._enabled: bool = False
        self._pending: _Pending | None = None
        self._on_pending_changed: Callable[[], None] | None = None
        # 锁保护 _pending 的写入 + Future.set_result 的唯一性.
        # 裁决可能来自 TUI 线程, cancel_current 来自 articulate loop.
        self._lock = threading.Lock()

    # -- enabled --

    def is_enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool = True) -> bool:
        if self._enabled == enabled:
            return False
        self._enabled = enabled
        return True

    # -- pending 快照 --

    def pending(self) -> PendingApproval | None:
        p = self._pending
        return p.snapshot() if p is not None else None

    # -- 拦截侧: 提交 & 幂等收尾 --

    def submit(self, logos: str) -> Future[Verdict]:
        with self._lock:
            if self._pending is not None:
                # 上一个未结算 — articulate loop 串行不该并发 submit.
                # 抛出即声明契约破裂, 而非默认接受.
                raise RuntimeError(
                    "SafeMode.submit called while a pending approval is in flight"
                )
            self._pending = _Pending(logos)
            future = self._pending.future
        self._fire_pending_changed()
        return future

    def cancel_current(self) -> bool:
        return self._resolve(uuid=None, verdict=Verdict(kind='cancelled'))

    # -- TUI 侧裁决 --

    def approve(self, uuid: str, note: str = '') -> bool:
        return self._resolve(uuid=uuid, verdict=Verdict(kind='approved', message=note))

    def reject(self, uuid: str, reason: str) -> bool:
        return self._resolve(uuid=uuid, verdict=Verdict(kind='rejected', message=reason))

    # -- 回调 --

    def on_pending_changed(self, callback: Callable[[], None] | None) -> None:
        self._on_pending_changed = callback

    # -- 内部 --

    def _resolve(self, uuid: str | None, verdict: Verdict) -> bool:
        """结算 pending. uuid=None 表示 cancel_current 无差别命中当前 pending.

        锁内完成 uuid 比对 + Future.set_result + 清 pending, 保证并发裁决只有一个赢.
        """
        with self._lock:
            p = self._pending
            if p is None:
                return False
            if uuid is not None and p.uuid != uuid:
                return False
            # 已被其它路径结算 (罕见: set_result 抢先) — 保持 no-op.
            if p.future.done():
                self._pending = None
                return False
            p.future.set_result(verdict)
            self._pending = None
        self._fire_pending_changed()
        return True

    def _fire_pending_changed(self) -> None:
        cb = self._on_pending_changed
        if cb is not None:
            cb()
