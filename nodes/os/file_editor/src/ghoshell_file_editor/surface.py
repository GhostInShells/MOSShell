"""file_editor surface — the human-facing web surface (axis 3).

One port serves the page and the WebSocket. Downlink carries the action stream
(``action.head`` / ``action.delta`` / ``action.full``) plus a full ``state``
snapshot on connect. Uplink carries the human's verdicts and dialogue
(``confirm`` / ``reject`` / ``reply``) and the global ``toggle``.

Human events mutate the same store the channel drives, then signal the ghost:
``confirm`` / ``reject`` ride ``NotifySignalMeta`` (must not be lost), ``reply``
rides ``AsideSignalMeta`` (silent, aggregated). The store stays the single
source of truth — the signal is a ping, the ghost pulls detail via the channel's
query commands.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from websockets.asyncio.server import ServerConnection, serve

from ghoshell_moss.message import Message
from ghoshell_moss.signals import AsideSignalMeta, NotifySignalMeta

from .projection import action_view, snapshot
from .store import ThreadStore

__all__ = ["FileEditorSurface"]


class FileEditorSurface:
    def __init__(
        self,
        store: ThreadStore,
        *,
        send_signal: Callable[[Any], None],
        self_identity: str,
        on_toggle: Callable[[bool], None],
        host: str,
        port: int,
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
        self.port = port

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self.port}"

    # -- downlink ---------------------------------------------------

    async def broadcast(self, frame: dict[str, Any]) -> None:
        if not self._clients:
            return
        payload = json.dumps(frame, ensure_ascii=False)
        await self._fanout(payload)

    async def _fanout(self, payload: str) -> None:
        targets = [c for c in self._clients]
        for c in targets:
            try:
                await c.send(payload)
            except Exception:
                self._clients.discard(c)

    # -- uplink -----------------------------------------------------

    async def _confirm(self, thread: str, n: int) -> None:
        try:
            flipped = self._store.confirm(thread, n, by="u")
        except (KeyError, ValueError) as e:
            await self.broadcast({"type": "error", "text": str(e)})
            return
        for action in flipped:
            await self.broadcast({"type": "action.full", **action_view(thread, action)})
        self._signal(NotifySignalMeta, f"confirmed {thread} through {n}", thread, n)

    async def _reject(self, thread: str, n: int) -> None:
        try:
            cascaded = self._store.reject(thread, n, by="u")
        except (KeyError, ValueError) as e:
            await self.broadcast({"type": "error", "text": str(e)})
            return
        for seq in cascaded:
            action = self._store.get_action(thread, seq.n)
            await self.broadcast({"type": "action.full", **action_view(thread, action)})
        self._signal(NotifySignalMeta, f"rejected {thread} from {n}", thread, n)

    async def _reply(self, thread: str, n: int, text: str, anchor: str) -> None:
        try:
            self._store.reply(thread, n, "u", anchor, text=text)
        except (KeyError, ValueError) as e:
            await self.broadcast({"type": "error", "text": str(e)})
            return
        action = self._store.get_action(thread, n)
        await self.broadcast({"type": "action.full", **action_view(thread, action)})
        self._signal(AsideSignalMeta, f"reply on {thread}:{n}", thread, n)

    def _signal(self, meta_cls, text: str, thread: str, seq: int) -> None:
        msg = Message.new(
            tag="file_editor",
            name=self._identity,
            attributes={"thread": thread, "seq": str(seq)},
        ).with_content(text)
        signal = meta_cls().to_signal(msg, description=f"file_editor:{thread}")
        self._send_signal(signal)

    # -- ws server --------------------------------------------------

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
            await connection.send(json.dumps(snapshot(self._store), ensure_ascii=False))
            async for raw in connection:
                try:
                    frame = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                kind = frame.get("type")
                if kind == "confirm":
                    await self._confirm(frame.get("thread", ""), frame.get("n", 0))
                elif kind == "reject":
                    await self._reject(frame.get("thread", ""), frame.get("n", 0))
                elif kind == "reply":
                    await self._reply(
                        frame.get("thread", ""), frame.get("n", 0),
                        frame.get("text", ""), frame.get("anchor", "intent"),
                    )
                elif kind == "toggle":
                    self._on_toggle(bool(frame.get("enabled", True)))
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
