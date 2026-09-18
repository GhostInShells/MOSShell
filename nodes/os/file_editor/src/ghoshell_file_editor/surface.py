"""The human-facing web surface: cards in, verdicts and questions out.

One port serves the page and the WebSocket. Downlink carries the card stream
(``action`` / ``threads`` / ``detail``) plus a full ``snapshot`` on connect.
Uplink carries the human's three moves — accept, deny, ask — plus detail
requests (content is fetched when a card is clicked, never streamed with it) and
the enable toggle.

Verdicts are handed down, never written here: ``store.settle`` wakes the
channel's waiter, and the channel is the only place a thread moves. What this
module does own is the signal to the ghost — a human's question is a fact the
model must learn, and it travels as a must-not-lose notify that guarantees a
turn; a verdict rides an aside, because it is a fact the ghost notices when it
is free rather than an interruption.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from websockets.asyncio.server import ServerConnection, serve

from ghoshell_moss.message import Message
from ghoshell_moss.signals import AsideSignalMeta, NotifySignalMeta

from .projection import action_view, detail_view, snapshot, thread_view
from .store import DocStore

__all__ = ["FileEditorSurface"]

_DEBOUNCE_SECONDS = 0.4
"""A verdict is one decision. Repeated clicks inside this window are the same
decision arriving twice, not a second one."""


class FileEditorSurface:
    def __init__(
        self,
        store: DocStore,
        *,
        send_signal: Callable[[Any], None],
        self_identity: str,
        on_toggle: Callable[[bool], None] | None = None,
        host: str = "127.0.0.1",
        port: int = 8767,
        html_path: Path,
    ) -> None:
        self._store = store
        self._send_signal = send_signal
        self._identity = self_identity
        self._on_toggle = on_toggle
        self._host = host
        self._html_path = html_path
        self._clients: set[ServerConnection] = set()
        self._server: Any = None
        self._decided: set[tuple[str, int]] = set()
        self._last: dict[tuple[str, int, str], float] = {}
        self.port = port

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self.port}"

    # -- downlink -----------------------------------------------------------

    async def broadcast(self, frame: dict[str, Any]) -> None:
        if not self._clients:
            return
        payload = json.dumps(frame, ensure_ascii=False)
        for client in list(self._clients):
            try:
                await client.send(payload)
            except Exception:
                self._clients.discard(client)

    # -- uplink -------------------------------------------------------------

    def _debounced(self, thread: str, n: int, action: str) -> bool:
        now = time.monotonic()
        key = (thread, n, action)
        if now - self._last.get(key, 0.0) < _DEBOUNCE_SECONDS:
            return True
        self._last[key] = now
        return False

    async def _verdict(self, thread: str, n: int, verdict: str) -> None:
        key = (thread, n)
        if key in self._decided or self._debounced(thread, n, verdict):
            return
        if not self._store.settle(thread, n, verdict):
            return
        self._decided.add(key)
        where = "accepted — writing it now" if verdict == "accept" else "denied"
        self._aside(f"export of {thread!r} #{n} was {where}")

    async def _ask(self, thread: str, n: int, text: str) -> None:
        if self._debounced(thread, n, "ask"):
            return
        try:
            action = self._store.say(thread, n, "u", text or "(wants to talk)")
        except KeyError:
            await self.broadcast({"type": "error", "text": f"no card {thread}#{n}"})
            return
        target = self._store.get(thread)
        if target is not None:
            await self.broadcast(
                {"type": "action", "thread": thread, **action_view(action)}
            )
        self._notify(f"the human asks about {thread!r} #{n}: {text}")

    async def _detail(self, thread: str, n: int) -> None:
        target = self._store.get(thread)
        if target is None:
            await self.broadcast({"type": "error", "text": f"no thread {thread!r}"})
            return
        action = target.get(n)
        if action is None:
            await self.broadcast(
                {"type": "error", "text": f"no card {thread}#{n}"}
            )
            return
        await self.broadcast({"type": "detail", **detail_view(target, action)})

    async def _set_thread_auto(self, thread: str, auto: bool) -> None:
        try:
            self._store.set_thread_auto(thread, auto)
        except (KeyError, ValueError) as e:
            await self.broadcast({"type": "error", "text": str(e)})
            return
        await self.broadcast({
            "type": "threads",
            "threads": [thread_view(t) for t in self._store.threads()],
        })

    async def _set_enabled(self, enabled: bool) -> None:
        if self._on_toggle is not None:
            self._on_toggle(enabled)

    def _notify(self, text: str) -> None:
        if self._send_signal is None:
            return
        signal = NotifySignalMeta(next=True).to_signal(
            Message.new(tag="file_editor", name=self._identity).with_content(
                f"[file_editor] {text}"
            ),
            description=text[:120],
        )
        self._send_signal(signal)

    def _aside(self, text: str) -> None:
        if self._send_signal is None:
            return
        signal = AsideSignalMeta().to_signal(
            Message.new(tag="file_editor", name=self._identity).with_content(
                f"[file_editor] {text}"
            ),
            description=text[:120],
        )
        self._send_signal(signal)

    # -- ws server ----------------------------------------------------------

    def _process_request(self, connection: ServerConnection, request: Any):
        path = request.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            try:
                html = self._html_path.read_text(encoding="utf-8")
            except OSError:
                return connection.respond(500, "index.html not found")
            response = connection.respond(200, html)
            del response.headers["Content-Type"]
            response.headers["Content-Type"] = "text/html; charset=utf-8"
            return response
        if path == "/ws":
            return None
        return connection.respond(404, "not found")

    async def _handler(self, connection: ServerConnection) -> None:
        self._clients.add(connection)
        try:
            await connection.send(
                json.dumps(snapshot(self._store), ensure_ascii=False)
            )
            async for raw in connection:
                try:
                    frame = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                kind = frame.get("type")
                thread = str(frame.get("thread", ""))
                n = int(frame.get("n", 0))
                if kind == "accept":
                    await self._verdict(thread, n, "accept")
                elif kind == "deny":
                    await self._verdict(thread, n, "deny")
                elif kind == "ask":
                    await self._ask(thread, n, str(frame.get("text", "")))
                elif kind == "detail":
                    await self._detail(thread, n)
                elif kind == "auto":
                    await self._set_thread_auto(thread, bool(frame.get("auto")))
                elif kind == "toggle":
                    await self._set_enabled(bool(frame.get("enabled", True)))
        finally:
            self._clients.discard(connection)

    async def start(self) -> None:
        self._server = await serve(
            self._handler,
            self._host,
            self.port,
            process_request=self._process_request,
        )
        sockets = getattr(self._server, "sockets", None)
        if sockets:
            self.port = sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
