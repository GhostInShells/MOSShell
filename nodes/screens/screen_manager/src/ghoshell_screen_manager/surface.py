"""The human-facing web surface: window frames down, steering up.

One port serves the page and the WebSocket. Downlink carries the window stream
(``snapshot`` on connect, then ``state`` / ``open`` / ``close`` / ``activate`` /
``focus`` / ``veil``), the 5Hz audio samples, and a ``notice`` line for the
human's activity log.

Uplink carries the human's moves: switch to a group or the desktop, toggle
focus on an item, and dismiss (send a window back to the desktop).

A human move is handed to the store; the store stays the only authority, and this
module re-broadcasts the change so every client stays in step. The move also
reaches the ghost as an aside — a fact it notices when free, not an interruption.

**One window is the screen at a time.** Any tab pointed at this url connects, and
with more than one live the page stops being *a* place: the same frames render at
two different window sizes, and a gesture the ghost draws lands somewhere the
human is not looking. So a page announces itself (``hello`` with a per-tab uid) and
this surface appoints exactly one *current* window; every other one is told it is
not the screen and masks itself. The page that opens from ``?claim=1`` — the one
the node opened at startup — takes the screen when it says hello, so the human
never has to pick after a restart; every other page holds a button that claims the
screen by hand. Uplink from a masked page is refused, not silently obeyed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.http11 import Response

from ghoshell_moss.message import Message
from ghoshell_moss.signals import AsideSignalMeta

from . import projection as P
from .audio import AudioSource
from .model import ScreenModel

__all__ = ["ScreenSurface"]

_AUDIO_HZ = 5.0

# The stage's own asset folder — the few images the veil may paint. Served over
# the same port as the page, because a page from http:// cannot read file://.
_ASSET_DIR = "assets"
_LOCAL_TOKEN = re.compile(r"^[A-Za-z0-9]{6,32}$")


def _workspace_root(start: Path) -> Path:
    """The project's own tree — the boundary a local file may not be read outside of.

    "Serve a local page" is a request that must not turn into "serve any file", so the
    boundary is discovered from the code rather than taken from the caller.
    """
    for d in [start, *start.parents]:
        if (d / ".git").exists() or (d / "pyproject.toml").exists():
            return d
    return start
_ASSET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_ASSET_TYPES = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
    "svg": "image/svg+xml",
}


class ScreenSurface:
    def __init__(
        self,
        model: ScreenModel,
        audio: AudioSource,
        *,
        send_signal: Callable[[Any], None],
        self_identity: str,
        host: str = "127.0.0.1",
        port: int = 0,
        html_path: Path,
    ) -> None:
        self._model = model
        self._audio = audio
        self._send_signal = send_signal
        self._identity = self_identity
        self._host = host
        self._html_path = html_path
        self._clients: set[ServerConnection] = set()
        self._current: ServerConnection | None = None
        """The one window that *is* the screen. Every other page is told to mask."""
        self._uids: dict[ServerConnection, str] = {}
        """Per-tab identity, as the page reported it. Lets us name the current tab
        without pretending a connection object is a place."""
        self._server: Any = None
        self._locals: dict[str, Path] = {}
        """token -> local html file. A page served from here shares this origin, which
        is the only way an iframe can show a file on the machine."""
        self._audio_task: asyncio.Task | None = None
        self._rail = ""
        """The human's sidebar state, reported by the page. Empty until it says."""
        self._viewport = ""
        """The current window's size, as ``WxH@dpr`` — reported by the page that
        holds the screen. Empty until it says; because only the current page gets
        through, a change in this value *is* the human moving the screen."""
        self.port = port

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self.port}"

    # -- downlink -----------------------------------------------------------

    async def broadcast(self, frame: dict[str, Any]) -> None:
        """Downlink goes to the one current window — never to a masked page.

        A masked page receiving frames would draw the same scene at its own size,
        which is exactly the confusion the current-screen rule exists to remove.
        """
        await self._send(self._current, frame)

    async def _send(self, connection: ServerConnection | None, frame: dict[str, Any]) -> None:
        if connection is None:
            return
        try:
            await connection.send(json.dumps(frame, ensure_ascii=False))
        except Exception:
            # A dead socket is not worth a traceback here; the connection's own
            # finally-clause clears the registry.
            if connection is self._current:
                self._current = None

    async def _set_current(self, connection: ServerConnection) -> None:
        """Appoint the screen, and demote whoever held it."""
        previous = self._current
        self._current = connection
        await self._send(connection, {"type": "current"})
        # The winner needs the whole world, not the delta it never saw.
        await self._send(connection, P.snapshot(self._model))
        if previous is not None and previous is not connection:
            await self._mask(previous, "another window took over as the screen")

    async def _mask(self, connection: ServerConnection | None, reason: str) -> None:
        await self._send(connection, {"type": "mask", "reason": reason})

    async def _mask_everyone_but(self, keeper: ServerConnection | None, reason: str) -> None:
        for client in list(self._clients):
            if client is not keeper:
                await self._mask(client, reason)

    # -- uplink -------------------------------------------------------------

    def rail_state(self) -> str:
        """How the human's sidebar stands — the channel reflects this as a notice.

        Value domain is two non-empty zeros (``expanded`` / ``collapsed``), never
        ``""``: in a named fragment ``""`` means *unchanged*, so a blank state has
        to be said out loud to be announced when it is entered or left.
        """
        return self._rail

    async def _set_rail(self, collapsed: bool) -> None:
        self._rail = "collapsed" if collapsed else "expanded"

    def viewport_state(self) -> str:
        """The current window's size — what the human chose to look through.

        One fragment, two meanings: the size a gesture has to fit, and the fact
        that the screen moved (only the current page reports, so the value
        changing means someone pressed the button on another window).
        """
        return self._viewport

    async def _set_viewport(self, frame: dict[str, Any]) -> None:
        try:
            w = int(frame.get("w", 0))
            h = int(frame.get("h", 0))
        except (TypeError, ValueError):
            return
        if w <= 0 or h <= 0:
            return
        dpr = frame.get("dpr", 1)
        self._viewport = f"{w}x{h}@{dpr}"

    async def _switch_view(self, group: str) -> None:
        """A human picked a place to look. No aside: the fact is *state* (``arena``
        + ``view_by``), and it also cancels any focus — two view-setters must not
        stack, or clicking home while focused looks like a dead button."""
        try:
            self._model.activate(group, by="human")
        except (KeyError, ValueError) as e:
            await self.broadcast({"type": "error", "text": str(e)})
            return
        await self.broadcast(P.activate_frame(self._model, by_model=False))

    async def _focus(self, item_id: str) -> None:
        """The human looked at one item alone (or left it).

        A *view* event: it never touches groups. Same item_id twice = leave, so a
        second click on the tile you are looking at closes it.
        """
        new = None if (not item_id or self._model.focus() == item_id) else item_id
        try:
            self._model.set_focus(new, by="human")
        except (KeyError, ValueError) as e:
            await self.broadcast({"type": "error", "text": str(e)})
            return
        await self.broadcast(P.focus_frame(new))
        await self.broadcast(P.state_frame(self._model))

    async def _dismiss(self, item_id: str) -> None:
        if self._model.dismiss(item_id) is None:
            return
        self._model.note_view_by("human")
        await self.broadcast(P.state_frame(self._model))

    async def _tap(self, item_id: str) -> None:
        """A human tapped a desktop proxy — interest, not a move.

        This is the one human move that changes **no** state, so there is nothing
        for a state fragment to carry: the aside is the only channel it has. Keep
        it as a tap on the shoulder, knowing it may be dropped while the ghost is
        busy — which is exactly what "aside" means.
        """
        self._aside(f"human tapped #{item_id} on the desktop")

    def _aside(self, text: str) -> None:
        if self._send_signal is None:
            return
        signal = AsideSignalMeta().to_signal(
            Message.new(tag="webview_screen", name=self._identity).with_content(
                f"[webview_screen] {text}"
            ),
            description=text[:120],
        )
        self._send_signal(signal)

    # -- audio pump ---------------------------------------------------------

    async def _audio_loop(self) -> None:
        while True:
            await asyncio.sleep(1 / _AUDIO_HZ)
            if self._current is None:
                continue
            await self.broadcast(P.audio_frame(self._audio.sample()))

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
            # 这个页面在开发期一直在改：绝不能让浏览器吃缓存，
            # 否则"刷新"看起来生效了、其实加载的还是旧版本。
            response.headers["Cache-Control"] = "no-store, max-age=0"
            return response
        if path == "/__ver":
            response = connection.respond(200, json.dumps({"ver": self._page_ver()}))
            del response.headers["Content-Type"]
            response.headers["Content-Type"] = "application/json"
            response.headers["Cache-Control"] = "no-store, max-age=0"
            return response
        if path == "/ws":
            return None
        if path.startswith("/local/"):
            return self._serve_local(connection, path[len("/local/"):])
        if path.startswith("/asset/"):
            return self._serve_asset(connection, path[len("/asset/"):])
        return connection.respond(404, "not found")

    def add_local(self, path: Path) -> str:
        """Register a local html file and return a url that serves it from this origin.

        A relative path resolves against the workspace root; anything that lands outside
        it is refused, and only html is accepted — this is "show me a page I wrote", not
        a file browser.
        """
        root = _workspace_root(Path(__file__).resolve())
        target = path if path.is_absolute() else (root / path)
        try:
            target = target.resolve()
            target.relative_to(root)
        except (OSError, ValueError):
            raise ValueError(f"not inside the workspace: {path}") from None
        if not target.is_file():
            raise ValueError(f"no such file: {target}")
        if target.suffix.lower() not in (".html", ".htm"):
            raise ValueError(f"only .html pages can be opened, got {target.suffix!r}")
        token = uuid.uuid4().hex[:12]
        self._locals[token] = target
        return f"{self.url}/local/{token}"

    def _serve_local(self, connection: ServerConnection, token: str):
        token = (token or "").split("?", 1)[0]
        if not _LOCAL_TOKEN.match(token) or token not in self._locals:
            return connection.respond(404, "not found")
        try:
            body = self._locals[token].read_bytes()
        except OSError:
            return connection.respond(404, "not found")
        response = connection.respond(200, body.decode("utf-8", "replace"))
        del response.headers["Content-Type"]
        response.headers["Content-Type"] = "text/html; charset=utf-8"
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    def _serve_asset(self, connection: ServerConnection, name: str):
        """One image the stage draws, by file name.

        The veil can paint anything it can load, but a page served over http
        cannot read a local file, and a base64 blob does not belong in a CTML
        attribute. So the node serves the few assets it owns. A name only — no
        directories, no traversal, an extension whitelist — because "serve a
        file" is exactly the request that must not become "serve any file".
        """
        if not _ASSET_NAME.match(name or ""):
            return connection.respond(404, "not found")
        suffix = name.rsplit(".", 1)[-1].lower()
        if suffix not in _ASSET_TYPES:
            return connection.respond(404, "not found")
        try:
            body = (self._html_path.parent / _ASSET_DIR / name).read_bytes()
        except OSError:
            return connection.respond(404, "not found")
        # ``respond`` only builds text bodies; an image needs the response object
        # itself, because going through ``respond`` would re-encode the bytes.
        headers = Headers()
        headers["Content-Type"] = _ASSET_TYPES[suffix]
        headers["Content-Length"] = str(len(body))
        headers["Cache-Control"] = "no-store, max-age=0"
        return Response(200, "OK", headers, body)

    def _page_ver(self) -> str:
        """The page's fingerprint — so an open window can tell it runs old code.

        A window left open across an edit keeps reporting the old world: we spent a
        round reading stale ``[client]`` lines that came from an old tab. Pull-based
        and cheap: the client asks, the server hashes the file, nobody has to
        remember to hit refresh.
        """
        try:
            return hashlib.sha1(self._html_path.read_bytes()).hexdigest()[:10]
        except OSError:
            return ""

    async def _handler(self, connection: ServerConnection) -> None:
        self._clients.add(connection)
        try:
            async for raw in connection:
                try:
                    frame = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                kind = frame.get("type")
                if kind == "hello":
                    await self._hello(
                        connection,
                        str(frame.get("uid", "")),
                        bool(frame.get("claim")),
                    )
                    continue
                if kind == "claim":
                    # Before the guard below: the one page that must be able to talk
                    # is the masked one asking to become the screen.
                    await self._set_current(connection)
                    continue
                if connection is not self._current:
                    # A masked page keeps talking — its moves must not land. Answer
                    # with the mask again so it settles back where it belongs.
                    await self._mask(connection, "this window is not the screen")
                    continue
                if kind == "switch_view":
                    await self._switch_view(str(frame.get("group", "")))
                elif kind == "focus":
                    await self._focus(str(frame.get("id", "")))
                elif kind == "dismiss":
                    await self._dismiss(str(frame.get("id", "")))
                elif kind == "tap":
                    await self._tap(str(frame.get("id", "")))
                elif kind == "rail":
                    # 侧栏的展开/收起由人在页面里做，服务端看不到 —— 所以由页面上报。
                    # 这条不是 debug：它改变舞台的可用区，是要进 notice 的状态。
                    await self._set_rail(bool(frame.get("collapsed")))
                elif kind == "viewport":
                    # 当前屏的画幅同理：也只有页面知道。它一变就说明屏幕换了人。
                    await self._set_viewport(frame)
                elif kind == "debug":
                    # 前端自报几何与画法状态。
                    # 动机：分屏下"贴底那层看不见"这类问题，根因在客户端，而服务端够不着
                    # 客户端。先前只能让人念控制台 —— 人不该当模型的眼睛。让页面自己把
                    # 数报回来，落在 stdout 上，模型读 read_output 就行。
                    payload = frame.get("payload")
                    print(
                        f"[client] {json.dumps(payload, ensure_ascii=False)}",
                        flush=True,
                    )
        finally:
            self._clients.discard(connection)
            self._uids.pop(connection, None)
            if connection is self._current:
                # The screen left the room. Nobody is promoted behind the human's
                # back — the others show their button and wait to be told.
                self._current = None
                await self._mask_everyone_but(None, "the screen window closed")

    async def _hello(self, connection: ServerConnection, uid: str, claim: bool) -> None:
        """A page says who it is. Exactly one of them gets to be the screen.

        ``claim`` comes from the url the node opened itself with: the window we just
        asked the human to look at wins without anyone clicking. Everything else is
        masked until a human presses its button — an old tab must never quietly
        steal the screen.
        """
        self._uids[connection] = uid
        if claim or self._current is None:
            await self._set_current(connection)
        else:
            await self._mask(connection, "another window is the screen")

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
        self._audio_task = asyncio.create_task(self._audio_loop())

    async def stop(self) -> None:
        if self._audio_task is not None:
            self._audio_task.cancel()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
