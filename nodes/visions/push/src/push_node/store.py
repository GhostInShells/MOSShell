"""PushStore — the single source of truth both faces read and write.

The channel (model face) and the web surface (human face) share this store and
never talk to each other directly. Verdicts are the one rendezvous: the channel
parks on a future the store hands out, and the surface settles it — exactly the
terminal store's waiter/settle shape, but over sessions instead of cards.

Sessions are ephemeral: the store is in-memory and never replayed. A restart
therefore cannot resurrect a live stream — the approval boundary is the session
boundary.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from .session import PushSession, SessionState

__all__ = ["PushStore", "AcceptAll"]

_MAX_SESSIONS = 100


class AcceptAll:
    """The human's "accept everything" posture — a front-end flag, not persisted.

    It auto-approves *future* pending sessions only; sessions already pending are
    not retroactively accepted (the human has to click them, or the toggle was
    already on when they arrived).
    """

    def __init__(self) -> None:
        self.enabled = False


class PushStore:
    def __init__(self) -> None:
        self._sessions: dict[int, PushSession] = {}
        self._order: list[int] = []
        self._counter = 0
        self._waiters: dict[int, asyncio.Future] = {}
        self._verdicts: dict[int, str] = {}

    # -- sessions ------------------------------------------------------------

    def request(
        self,
        source: str,
        *,
        label: str = "",
        owner: str = "model",
        description: str = "",
        fps: float = 10.0,
        max_width: int = 1280,
        quality: int = 5,
    ) -> PushSession:
        self._counter += 1
        session = PushSession(
            id=self._counter,
            source=source,
            label=label,
            owner=owner,
            description=description,
            fps=fps,
            max_width=max_width,
            quality=quality,
        )
        self._sessions[session.id] = session
        self._order.append(session.id)
        self._trim()
        return session

    def get(self, session_id: int) -> Optional[PushSession]:
        return self._sessions.get(session_id)

    def sessions(self) -> list[PushSession]:
        return [self._sessions[i] for i in self._order if i in self._sessions]

    def pending(self) -> list[PushSession]:
        return [s for s in self.sessions() if s.state == SessionState.PENDING]

    def live(self) -> list[PushSession]:
        return [s for s in self.sessions() if s.state == SessionState.LIVE]

    def set_state(self, session_id: int, state: str) -> PushSession:
        session = self._require(session_id)
        session.settle(state)
        self._trim()
        return session

    def attach_process(self, session_id: int, process_index: int) -> PushSession:
        session = self._require(session_id)
        session.process_index = process_index
        session.touch()
        return session

    def mark_frame(self, session_id: int) -> PushSession:
        import time

        session = self._require(session_id)
        session.last_frame_at = time.time()
        return session

    def fail(self, session_id: int, reason: str) -> PushSession:
        session = self._require(session_id)
        session.failure = reason
        session.settle(SessionState.FAILED)
        return session

    # -- verdicts ------------------------------------------------------------

    def waiter(self, session_id: int) -> asyncio.Future:
        """The future the channel parks on while a session awaits its verdict."""
        future = self._waiters.get(session_id)
        if future is None:
            future = asyncio.get_running_loop().create_future()
            self._waiters[session_id] = future
            verdict = self._verdicts.get(session_id)
            if verdict is not None:
                future.set_result(verdict)
        return future

    def settle(self, session_id: int, verdict: str) -> bool:
        """Hand down a verdict. False = already decided (debounce guard).

        ``verdict`` is ``"accept"`` or ``"deny"``.
        """
        session = self._sessions.get(session_id)
        if session is None or session.state != SessionState.PENDING:
            return False
        self._verdicts[session_id] = verdict
        future = self._waiters.get(session_id)
        if future is not None and not future.done():
            future.set_result(verdict)
        return True

    # -- helpers -------------------------------------------------------------

    def _require(self, session_id: int) -> PushSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"no session {session_id}")
        return session

    def _trim(self) -> None:
        while len(self._order) > _MAX_SESSIONS:
            for i, sid in enumerate(self._order):
                session = self._sessions.get(sid)
                if session is not None and session.state in (
                    SessionState.STOPPED,
                    SessionState.DENIED,
                    SessionState.FAILED,
                ):
                    self._order.pop(i)
                    self._sessions.pop(sid, None)
                    self._waiters.pop(sid, None)
                    self._verdicts.pop(sid, None)
                    break
            else:
                return
