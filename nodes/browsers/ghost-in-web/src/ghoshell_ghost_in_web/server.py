"""Local WS edge between the Chrome extension and the shared ``PageModel``.

One WS per browser session: the extension's service worker holds it and every tab
of that session multiplexes over it. Downlink carries ``cmd`` (blocking, resolved
by cid) and ``say`` (fire-and-forget). Uplink mutates the model and, for the two
human-initiated pushes, raises a ghost signal:

- ``input`` — the human typed in the page panel → ``input`` signal (a message
  that expects an answer, so the ghost turns toward the human).
- ``event`` — the human clicked a satellite. Today the only satellite is
  screenshot: the SW captures the visible tab and ships a data URL, which is
  decoded to a PIL image and pushed as an ``aside`` signal (notice without
  interrupting). **The ghost can never pull a screenshot** — no click, no image.

The label is a node-internal abstraction; the wire talks in ``tab`` ids. The SW
stamps ``sender.tab.id`` on every frame.

Security: a WS bound to 127.0.0.1 is reachable from any page the human visits, so
``origins`` should be pinned to the extension id (websockets enforces it when
passed). The id is known only after install; the node reads it from config.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
import uuid
from collections.abc import Callable
from typing import Any

from PIL import Image
from websockets.asyncio.server import ServerConnection, serve
from websockets.protocol import State

from ghoshell_moss.core.mindflow.aside_nucleus import new_aside_signal
from ghoshell_moss.message import Base64Image, Message
from ghoshell_moss.signals import InputSignalMeta

from .model import PageModel

__all__ = ["WebServer", "decode_data_url"]

_CMD_TIMEOUT = 20.0
_HELLO_TIMEOUT = 10.0


def decode_data_url(data: str) -> Image.Image:
    """Decode ``data:image/png;base64,…`` into a PIL image. Raises ValueError on
    anything that is not a well-formed base64 image data URL."""
    if not data or "," not in data:
        raise ValueError("not a data URL")
    head, _, payload = data.partition(",")
    if "base64" not in head:
        raise ValueError("data URL is not base64")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"bad base64 payload: {e}") from e
    return Image.open(io.BytesIO(raw))


def _downscale_jpeg(
    image: Image.Image, *, max_dim: int = 1280, quality: int = 85
) -> bytes:
    """截屏是 Retina 全分辨率,直接进上下文太贵。等比缩到最长边 max_dim,
    转 JPEG(quality>=85 下截图文字可读,体积远小于 PNG)。"""
    image = image.convert("RGB")
    image.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class WebServer:
    def __init__(
        self,
        model: PageModel,
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
            self._handle, self._host, self._port, origins=self._origins
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
        self._model.log(label, "behavior", _describe(action, value), ok=None)
        future = asyncio.get_running_loop().create_future()
        self._pending[cid] = future
        try:
            await conn.send(json.dumps({
                "type": "cmd", "cid": cid, "tab": tab,
                "action": action, "value": value,
            }))
            result = await asyncio.wait_for(future, timeout=_CMD_TIMEOUT)
            self._model.log(
                label, "behavior", _describe(action, value),
                ok=bool(result.get("ok")),
            )
            return result
        except asyncio.TimeoutError:
            self._model.log(label, "behavior", _describe(action, value), ok=False)
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
        self._model.log(label, "dialog", f"→ {text}")
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
                    "accepted": msg.get("accepted"),
                })
            return

        tab = msg.get("tab")
        if tab is None:
            return
        if kind == "content":
            self._model.update_content(session, tab, msg.get("title", ""), msg.get("url", ""))
        elif kind == "auth":
            self._model.set_perceived(session, tab, bool(msg.get("on")))
        elif kind == "event":
            self._on_event(session, tab, msg)
        elif kind == "input":
            self._on_input(session, tab, msg.get("text", ""))
        elif kind == "bye":
            self._model.close_tab(session, tab)

    def _on_event(self, session: str, tab: int, msg: dict) -> None:
        if msg.get("kind") == "error":
            label = self._model.label_of(session, tab) or "?"
            self._model.log(label, "event", f"错误: {msg.get('detail', '')}", ok=False)
            return
        if msg.get("kind") != "screenshot":
            return
        label = self._model.label_of(session, tab) or "?"
        try:
            image = decode_data_url(msg.get("data", ""))
        except ValueError as e:
            self._model.log(label, "event", f"截图解码失败: {e}", ok=False)
            return
        width, height = image.size
        jpeg = _downscale_jpeg(image)
        b64 = base64.b64encode(jpeg).decode()
        self._model.log(
            label, "event",
            f"人类点击截图 ({width}×{height} → {len(jpeg) // 1024}KB JPEG)",
            ok=True, image_b64=b64,
        )
        if self._send_signal is None:
            return
        signal = new_aside_signal(
            f"[{label}] 人类截了这张图",
            Base64Image.from_binary("image/jpeg", jpeg),
            description=f"[{label}] 人类主动发送的截图",
        )
        self._send_signal(signal)

    def _on_input(self, session: str, tab: int, text: str) -> None:
        if not text or self._send_signal is None:
            return
        label = self._model.label_of(session, tab) or "?"
        self._model.log(label, "dialog", f"← {text}")
        signal = InputSignalMeta().to_signal(
            Message.new(tag="ghost_in_web", name=label).with_content(text),
            description=f"[{label}] {text[:120]}",
        )
        self._send_signal(signal)

    def _fail_pending(self) -> None:
        pending = list(self._pending.items())
        self._pending.clear()
        for _cid, future in pending:
            if not future.done():
                future.set_result({"ok": False, "error": "disconnected"})


def _describe(action: str, value: object) -> str:
    if value is None:
        return action
    return f"{action}({value!r})"
