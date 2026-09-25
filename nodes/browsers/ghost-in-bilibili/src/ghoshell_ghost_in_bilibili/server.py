"""Local WS edge between the Chrome extension and the shared ``BridgeModel``.

One WS per browser session: the extension's service worker holds it, and every
tab of that session multiplexes over it. Downlink carries ``cmd`` (blocking,
resolved by cid) and ``say`` (fire-and-forget); uplink mutates the model and, for
human panel input, raises an ``input`` signal to the ghost.

The label is a node-internal abstraction — the wire talks in ``tab`` ids. The SW
stamps ``sender.tab.id`` on every frame; the node maps tab → label via the model.

Security: a WS bound to 127.0.0.1 is reachable from any page the human visits, so
``origins`` should be pinned to the extension id (websockets enforces it when
passed). The id is known only after install; the node reads it from config.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable
from typing import Any

from websockets.asyncio.server import ServerConnection, serve
from websockets.protocol import State

from ghoshell_moss.message import Message
from ghoshell_moss.signals import InputSignalMeta

from .model import BridgeModel

__all__ = ["BilibiliServer"]

_CMD_TIMEOUT = 15.0
_HELLO_TIMEOUT = 10.0


class BilibiliServer:
    def __init__(
        self,
        model: BridgeModel,
        *,
        send_signal: Callable[[Any], None] | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        origins: list[str] | None = None,
    ) -> None:
        self._model = model
        self._send_signal = send_signal
        self._host = host
        self._port = port
        self._origins = origins
        self._server = None
        self.port = port
        self._conn: dict[str, ServerConnection] = {}  # session -> live conn
        self._pending: dict[str, asyncio.Future] = {}  # cid -> result future

    @property
    def url(self) -> str:
        return f"ws://{self._host}:{self.port}"

    async def start(self) -> None:
        self._server = await serve(
            self._handle, self._host, self.port, origins=self._origins
        )
        sockets = getattr(self._server, "sockets", None)
        if sockets:
            self.port = sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    # -- downlink (channel -> extension) ---------------------------------

    async def send_action(self, label: str, action: str, value=None) -> dict:
        loc = self._model.location_of(label)
        if loc is None:
            return {"ok": False, "error": f"no page {label!r}"}
        session, tab = loc
        conn = self._conn.get(session)
        if conn is None or conn.state is not State.OPEN:
            return {"ok": False, "error": f"{label} offline"}

        cid = uuid.uuid4().hex[:8]
        future = asyncio.get_running_loop().create_future()
        self._pending[cid] = future
        try:
            await conn.send(json.dumps({
                "type": "cmd", "cid": cid, "tab": tab,
                "action": action, "value": value,
            }))
            return await asyncio.wait_for(future, timeout=_CMD_TIMEOUT)
        except asyncio.TimeoutError:
            return {"ok": False, "error": "timeout"}
        finally:
            self._pending.pop(cid, None)

    async def say(self, label: str, text: str) -> None:
        loc = self._model.location_of(label)
        if loc is None:
            return
        session, tab = loc
        conn = self._conn.get(session)
        if conn is None or conn.state is not State.OPEN:
            return
        await conn.send(json.dumps({"type": "say", "tab": tab, "text": text}))

    # -- connection handling ---------------------------------------------

    async def _handle(self, connection: ServerConnection) -> None:
        session = None
        try:
            first = await asyncio.wait_for(connection.recv(), timeout=_HELLO_TIMEOUT)
            hello = json.loads(first)
            if hello.get("type") != "hello":
                await connection.close(code=1008, reason="first frame must be hello")
                return
            session = hello.get("session") or uuid.uuid4().hex[:8]
            self._conn[session] = connection
            self._model.on_hello(session)

            async for raw in connection:
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                await self._on_frame(session, msg)
        except Exception:
            pass
        finally:
            if session is not None:
                if self._conn.get(session) is connection:
                    self._conn.pop(session, None)
                self._model.on_disconnect(session)
            self._fail_pending()

    async def _on_frame(self, session: str, msg: dict) -> None:
        kind = msg.get("type")

        # ``result`` carries only a cid — no tab. Handle it before the tab guard.
        if kind == "result":
            future = self._pending.pop(msg.get("cid"), None)
            if future is not None and not future.done():
                future.set_result({
                    "ok": msg.get("ok", False),
                    "result": msg.get("result"),
                    "error": msg.get("error"),
                })
            return

        tab = msg.get("tab")
        if tab is None:
            return
        if kind == "content":
            self._model.update_content(
                session, tab, msg.get("bvid"), msg.get("title", ""), msg.get("url", "")
            )
        elif kind == "auth":
            group = msg.get("group")
            on = bool(msg.get("on"))
            if group is None:
                self._model.set_presence(session, tab, on)
            else:
                self._model.set_grant(session, tab, group, on)
        elif kind == "state":
            self._model.update_state(
                session, tab,
                t=msg.get("t", 0.0),
                paused=msg.get("paused", True),
                rate=msg.get("rate", 1.0),
                duration=msg.get("duration", 0.0),
            )
        elif kind == "input":
            self._on_input(session, tab, msg.get("text", ""))
        elif kind == "bye":
            self._model.close_tab(session, tab)

    def _on_input(self, session: str, tab: int, text: str) -> None:
        if not text or self._send_signal is None:
            return
        label = self._model.label_of(session, tab) or "?"
        signal = InputSignalMeta().to_signal(
            Message.new(tag="ghost_in_bilibili", name=label).with_content(text),
            description=f"[{label}] {text[:120]}",
        )
        self._send_signal(signal)

    def _fail_pending(self) -> None:
        pending = list(self._pending.items())
        self._pending.clear()
        for _cid, future in pending:
            if not future.done():
                future.set_result({"ok": False, "error": "disconnected"})
