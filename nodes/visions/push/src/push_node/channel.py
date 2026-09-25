"""The model-facing channel: requests become sessions, sessions become producers.

The pose is the terminal's: a request returns a receipt immediately — the model
never blocks on a human's response time. Whatever happens next (approved,
denied, stopped) reaches the model as a signal, not as a return value.

Ownership is the one asymmetry written into the command surface: the model may
stop sessions it requested, never one the human opened. That is the trust
boundary made concrete — shared visibility, with the human holding the final
hand on everything.

When a session goes live, the model is signalled its MJPEG URL; it then opens a
*separate* stream node on that address to consume the vision. The push node only
pushes — it does not feed frames into model context itself.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Optional

from ghoshell_moss.contracts.subprocesses import Subprocesses
from ghoshell_moss.core.blueprint.channel_builder import CommandUtil, new_channel
from ghoshell_moss.core.blueprint.mindflow import Priority
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.message import Message
from ghoshell_moss.signals import NotifySignalMeta

from .ffmpeg import source_summary
from .producer import ProducerManager
from .session import PushSession, SessionState
from .store import AcceptAll, PushStore
from .surface import PushHandles, PushSurface

__all__ = ["build_push_channel"]

_LEVELS: dict[str, Priority] = {
    "background": Priority.BACKGROUND,
    "info": Priority.INFO,
    "warning": Priority.WARNING,
}


def build_push_channel(
    store: PushStore,
    accept_all: AcceptAll,
    processes: Subprocesses,
    *,
    surface: PushSurface | None = None,
    handles: PushHandles | None = None,
    signaler: Callable[[Any], None] | None = None,
    surface_url: str | Callable[[], str] | None = None,
    name: str = "push",
) -> Channel:
    handles = handles or PushHandles()
    producer = ProducerManager(
        processes,
        store,
        on_failed=lambda sid, reason: _on_failed(sid, reason),
    )
    if surface is not None:
        surface.bind_frame(producer.frame)

    async def _emit(frame: dict[str, Any]) -> None:
        if surface is not None:
            await surface.broadcast(frame)

    async def _emit_session(session: PushSession) -> None:
        await _emit({"type": "session", "session": session.view()})

    def _url() -> str:
        if surface_url is None:
            return ""
        return surface_url() if callable(surface_url) else surface_url

    def _stream_url(session_id: int) -> str:
        base = _url()
        return f"{base}/stream/{session_id}" if base else ""

    def _signal(text: str, level: str = "info") -> None:
        if signaler is None:
            return
        signal = NotifySignalMeta(next=True).to_signal(
            Message.new(tag=name, name=name).with_content(text),
            description=text[:120],
            priority=_LEVELS.get(level, Priority.INFO),
        )
        signaler(signal)

    async def _on_failed(session_id: int, reason: str) -> None:
        session = store.get(session_id)
        if session is not None:
            await _emit_session(session)
            _signal(f"[{name} #{session_id}] stream failed: {reason}", "warning")

    async def _await_verdict(session: PushSession) -> None:
        verdict = await store.waiter(session.id)
        if store.get(session.id) is None or store.get(session.id).state is not SessionState.PENDING:
            return
        if verdict == "accept":
            store.set_state(session.id, SessionState.LIVE)
            await _emit_session(session)
            await producer.start(session)
            _signal(
                f"[{name} #{session.id}] '{session.label or session.source}' approved — "
                f"consume it with a stream node at {_stream_url(session.id)}",
            )
        else:
            store.set_state(session.id, SessionState.DENIED)
            await _emit_session(session)
            _signal(f"[{name} #{session.id}] '{session.label or session.source}' denied")

    async def _spawn_live(session: PushSession) -> None:
        store.set_state(session.id, SessionState.LIVE)
        await _emit_session(session)
        await producer.start(session)

    # -- channel ------------------------------------------------------------

    chan = new_channel(
        name=name,
        description=(
            "push a local visual stream (screen / camera) as a shared object. "
            "request() asks the human; a live stream is a URL you consume with a "
            "stream node. You may stop your own streams, never the human's."
        ),
    )

    @chan.build.instruction
    def instruction() -> str:
        return (
            "There is a local push surface (its URL is in this channel's `url` notice) "
            "where a human approves and watches every visual stream. request() returns a "
            "receipt at once and never waits on a person — the verdict reaches you as a "
            f"signal. Sources: {source_summary()}. When a stream goes live you are told its "
            "MJPEG URL; open a *separate* stream node on that address to see it (this node "
            "only pushes, it does not feed frames into context). stop(id) closes one of your "
            "own live streams; stop_all() closes all of yours. You cannot stop a stream the "
            "human opened — that is theirs."
        )

    @chan.build.command(name="request", always_observe=False)
    async def request(
        source: str,
        label: str = "",
        description: str = "",
        fps: float = 10.0,
        max_width: int = 1280,
        quality: int = 5,
    ) -> str:
        """Ask to push a local visual stream. Returns a receipt immediately.

        ``source`` is one of the enumerated sources (see instruction). ``label``
        is your handle for it; ``description`` is a one-line caption for the human.
        ``fps`` / ``max_width`` / ``quality`` tune the stream (clamped to safe
        bounds). The human approves or denies on the push surface; you are
        signalled either way. When approved, the stream URL is signalled so you
        can consume it with a stream node.
        """
        try:
            session = store.request(
                source,
                label=label,
                owner="model",
                description=description,
                fps=fps,
                max_width=max_width,
                quality=quality,
            )
        except ValueError as e:
            CommandUtil.raise_observe(str(e))
            return ""
        await _emit_session(session)
        if accept_all.enabled:
            CommandUtil.create_task(_spawn_live(session))
            return (
                f"[{name} #{session.id}] {source} approved (accept-all) — "
                f"consume at {_stream_url(session.id)}"
            )
        CommandUtil.create_task(_await_verdict(session))
        return (
            f"[{name} #{session.id}] {source} awaiting approval — the human decides "
            f"on the push surface; you will be signalled"
        )

    @chan.build.command(name="sessions", always_observe=True)
    async def sessions() -> str:
        """List every push session, newest last, with owner and state."""
        out = []
        for s in store.sessions():
            out.append(
                f"#{s.id} {s.state} [{s.owner}] {s.source}"
                + (f" {s.label!r}" if s.label else "")
                + (f" — {s.failure}" if s.failure else "")
            )
        return "\n".join(out) if out else "(no sessions)"

    @chan.build.command(name="read", always_observe=True)
    async def read(session_id: int) -> str:
        """Read one session's state and, if live, its stream URL."""
        session = store.get(session_id)
        if session is None:
            CommandUtil.raise_observe(f"no session #{session_id} — sessions() to list them")
        lines = [f"#{session.id} {session.state} [{session.owner}] {session.source}"]
        if session.label:
            lines.append(f"label: {session.label}")
        if session.description:
            lines.append(f"desc: {session.description}")
        if session.failure:
            lines.append(f"failure: {session.failure}")
        if session.state == SessionState.LIVE:
            lines.append(f"stream: {_stream_url(session.id)}")
        elif session.state == SessionState.PENDING:
            lines.append("(awaiting the human's verdict on the push surface)")
        return "\n".join(lines)

    async def _stop_own(session_id: int) -> str:
        session = store.get(session_id)
        if session is None:
            return f"[{name} #{session_id}] no such session"
        if session.owner != "model":
            return f"[{name} #{session_id}] not yours to stop (owner={session.owner})"
        if session.state != SessionState.LIVE:
            return f"[{name} #{session_id}] not live (state={session.state})"
        await producer.stop(session_id)
        store.set_state(session_id, SessionState.STOPPED)
        await _emit_session(session)
        _signal(f"[{name} #{session_id}] '{session.label or session.source}' stopped")
        return f"[{name} #{session_id}] stopped"

    async def _stop_all_own() -> str:
        targets = [s for s in store.live() if s.owner == "model"]
        for s in targets:
            await producer.stop(s.id)
            store.set_state(s.id, SessionState.STOPPED)
            await _emit_session(s)
        _signal(f"[{name}] stopped {len(targets)} of your stream(s)")
        return f"[{name}] stopped {len(targets)} stream(s)"

    # The surface's human-owned buttons (stop any / stop all / open) reach the
    # channel through this holder; the model's commands stay inside the channel.
    async def _human_stop(session_id: int) -> None:
        session = store.get(session_id)
        if session is None or session.state != SessionState.LIVE:
            return
        await producer.stop(session_id)
        store.set_state(session_id, SessionState.STOPPED)
        await _emit_session(session)
        _signal(f"[{name} #{session_id}] stopped by human")

    async def _human_stop_all() -> None:
        for s in store.live():
            await producer.stop(s.id)
            store.set_state(s.id, SessionState.STOPPED)
            await _emit_session(s)
        _signal(f"[{name}] human stopped every live stream")

    async def _human_open(source: str, label: str) -> int:
        session = store.request(source, label=label, owner="human")
        await _spawn_live(session)
        return session.id

    handles.stop = _human_stop
    handles.stop_all = _human_stop_all
    handles.open_human = _human_open

    @chan.build.command(name="stop", always_observe=False)
    async def stop(session_id: int) -> str:
        """Stop one of your own live streams. The human's streams are not yours."""
        return await _stop_own(session_id)

    @chan.build.command(name="stop_all", always_observe=False)
    async def stop_all() -> str:
        """Stop every live stream you own."""
        return await _stop_all_own()

    # -- context ------------------------------------------------------------

    @chan.build.named_notices
    def notices() -> dict[str, str]:
        out: dict[str, str] = {}
        url = _url()
        if url:
            out["url"] = url
        pending = len(store.pending())
        live = len(store.live())
        out[name] = f"pending: {pending} | live: {live}"
        return out

    @chan.build.context_messages
    def context() -> list[str]:
        hot = [s for s in store.sessions() if s.state in (SessionState.PENDING, SessionState.LIVE)]
        if not hot:
            return []
        lines = [f"[{name}] streams in flight (read(id) for detail):"]
        for s in hot:
            lines.append(f"  #{s.id} {s.state} [{s.owner}] {s.source}" + (f" {s.label!r}" if s.label else ""))
        return ["\n".join(lines)]

    @chan.build.close
    async def _close() -> None:
        await producer.stop_all()

    return chan
