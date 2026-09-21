"""两个方位共享的单一 store：channel 读它写它，surface 读它、通过 settle 改授权状态。

授权格 = type → auto（整个命令族自动放行）。公共域（capabilities 里 identity=platform）
天然放行，不进格；私域（access_secret_owner）默认 pending，人类单条通过或把整个 type
设 auto。授权状态由 surface 经 ``settle`` 写入，CLI 执行仍归 channel —— 这里只存事实。

单线程事件循环上跑，plain dict 安全。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ActionRecord", "ZhihuStore"]

STATE_PENDING = "pending"
STATE_APPROVED = "approved"
STATE_REJECTED = "rejected"
STATE_RUNNING = "running"
STATE_DONE = "done"
STATE_ERROR = "error"


@dataclass
class ActionRecord:
    id: int
    type: str
    args: dict[str, Any]
    transform: str | None
    render: str | None
    state: str
    at: float
    identity: str = ""
    result: dict[str, Any] | None = None
    error: str | None = None
    settled: asyncio.Future | None = field(default=None, repr=False)


_PLATFORM_PREFIXES = ("search", "hot", "answer")


def _guess_identity(type_: str) -> str:
    for prefix in _PLATFORM_PREFIXES:
        if type_.startswith(prefix):
            return "platform"
    return "access_secret_owner"


class ZhihuStore:
    def __init__(self) -> None:
        self._actions: dict[int, ActionRecord] = {}
        self._auto: set[str] = set()
        self._counter = 0
        self.auth_configured = False
        self.identity = ""
        self.disabled = False
        self.capabilities: dict[str, str] = {}  # type -> identity

    # -- identity ----------------------------------------------------------

    def identity_of(self, type_: str) -> str:
        return self.capabilities.get(type_, "")

    def is_auto(self, type_: str) -> bool:
        return type_ in self._auto

    def set_auto(self, type_: str, on: bool) -> None:
        if on:
            self._auto.add(type_)
        else:
            self._auto.discard(type_)

    def set_capabilities(self, mapping: dict[str, str]) -> None:
        self.capabilities = mapping

    def set_disabled(self, on: bool) -> None:
        self.disabled = on

    # -- actions -----------------------------------------------------------

    def add(self, action: Any) -> ActionRecord:
        self._counter += 1
        rec = ActionRecord(
            id=self._counter,
            type=action.type,
            args=action.args,
            transform=action.transform,
            render=action.render,
            state=STATE_PENDING,
            at=time.time(),
            identity=self.capabilities.get(action.type) or _guess_identity(action.type),
            settled=asyncio.get_running_loop().create_future(),
        )
        self._actions[rec.id] = rec
        return rec

    def get(self, action_id: int) -> ActionRecord | None:
        return self._actions.get(action_id)

    def actions(self) -> list[ActionRecord]:
        return sorted(self._actions.values(), key=lambda r: r.id)

    def awaiting(self) -> list[ActionRecord]:
        return [r for r in self._actions.values() if r.state == STATE_PENDING]

    def settle(self, action_id: int, verdict: str) -> bool:
        """人类裁决。verdict: 'approve' | 'reject'。唤醒 channel 的 waiter。"""
        rec = self._actions.get(action_id)
        if rec is None or rec.state != STATE_PENDING:
            return False
        rec.state = STATE_APPROVED if verdict == "approve" else STATE_REJECTED
        if rec.settled is not None and not rec.settled.done():
            rec.settled.set_result(verdict)
        return True

    def mark_running(self, action_id: int) -> None:
        rec = self._actions.get(action_id)
        if rec is not None:
            rec.state = STATE_RUNNING

    def complete(self, action_id: int, result: dict[str, Any]) -> None:
        rec = self._actions.get(action_id)
        if rec is not None:
            rec.state = STATE_DONE
            rec.result = result

    def fail(self, action_id: int, error: str) -> None:
        rec = self._actions.get(action_id)
        if rec is not None:
            rec.state = STATE_ERROR
            rec.error = error
