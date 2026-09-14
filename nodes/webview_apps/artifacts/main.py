"""Artifacts — a streamed, code-driven visual surface (webview node).

A model streams an artifact's source into a live page: the first token announces
the artifact, later tokens accumulate its source visibly, and the end finalizes
it into a rendered result beside the source. `kind` only decides the finalize
step, so the mechanism is general — canvas / mermaid / style share one lifecycle.

Start:  moss nodes run nodes/webview_apps/artifacts
Debug:  python main.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

from websockets.asyncio.server import ServerConnection, serve

from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.blueprint.matrix import Matrix

_NODE_DIR = Path(__file__).resolve().parent
_INDEX_HTML = _NODE_DIR / "index.html"

HOST = os.getenv("MOSS_ARTIFACTS_HOST", "127.0.0.1")


def _port_from_argv(argv: list[str]) -> int | None:
    if "--port" in argv:
        i = argv.index("--port")
        if i + 1 < len(argv):
            return int(argv[i + 1])
    return None


PORT = _port_from_argv(sys.argv[1:]) or int(os.getenv("MOSS_ARTIFACTS_PORT", "8766"))


class ArtifactStore:
    """In-memory artifacts. `_order` is the append-only history; `_items` is the fold.

    `_order` never rewrites — a re-used label is moved to the tail, so the tail is
    always the most recently touched artifact (the LRU order history() reads).
    """

    def __init__(self) -> None:
        self._order: list[str] = []
        self._items: dict[str, dict[str, Any]] = {}
        self._user_actions: deque = deque(maxlen=20)

    def start(self, label: str, kind: str, description: str) -> None:
        # style is a meta-artifact — it applies globally, so it never enters the
        # displayable order (history / tabs) but stays in _items for read() recall.
        if label in self._order:
            self._order.remove(label)
        if kind != "style":
            self._order.append(label)
        self._items[label] = {
            "label": label,
            "kind": kind,
            "description": description,
            "source": "",
            "created_at": time.time(),
        }

    def append(self, label: str, text: str) -> None:
        item = self._items.get(label)
        if item is not None:
            item["source"] += text

    def end(self, label: str, source: str) -> None:
        item = self._items.get(label)
        if item is not None:
            item["source"] = source

    def read(self, label: str) -> str | None:
        item = self._items.get(label)
        return item["source"] if item is not None else None

    def get(self, label: str) -> dict[str, Any] | None:
        return self._items.get(label)

    def history(self, n: int) -> list[dict[str, Any]]:
        labels = self._order[-n:] if n > 0 else []
        return [self._items[label] for label in reversed(labels)]

    def snapshot(self) -> list[dict[str, Any]]:
        # displayable artifacts in order, then styles (so the page can re-apply them).
        items = [self._items[label] for label in self._order]
        for item in self._items.values():
            if item["kind"] == "style":
                items.append(item)
        return items

    def remove(self, label: str) -> bool:
        if label not in self._items:
            return False
        self._items.pop(label, None)
        if label in self._order:
            self._order.remove(label)
        return True

    def clear(self) -> int:
        count = len(self._order)
        self._order.clear()
        self._items.clear()
        return count

    def log_user_action(self, action: str, detail: str = "") -> None:
        self._user_actions.append((time.time(), action, detail))

    def recent_user_actions(self, n: int) -> list[str]:
        lines = []
        for at, action, detail in list(self._user_actions)[-n:]:
            ts = time.strftime("%H:%M:%S", time.localtime(at))
            lines.append(f"[{ts}] {action}" + (f" {detail}" if detail else ""))
        return lines


class SurfaceServer:
    """One port serves both the page and the WebSocket, so the node needs no extra deps.

    A plain GET returns index.html via websockets' `process_request` hook; `/ws`
    returns None to let the WebSocket handshake proceed.
    """

    def __init__(self, store: ArtifactStore, *, host: str, port: int, html_path: Path) -> None:
        self._store = store
        self._host = host
        self._html_path = html_path
        self._clients: set[ServerConnection] = set()
        self._server: Any = None
        self.port = port

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self.port}"

    def _process_request(self, connection: ServerConnection, request: Any):
        path = request.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            try:
                html = self._html_path.read_text(encoding="utf-8")
            except OSError:
                return connection.respond(500, "index.html not found")
            response = connection.respond(200, html)
            # respond() defaults Content-Type to text/plain and __setitem__ appends,
            # so delete the default first or the page renders as plain text.
            del response.headers["Content-Type"]
            response.headers["Content-Type"] = "text/html; charset=utf-8"
            return response
        if path == "/ws":
            return None
        return connection.respond(404, "not found")

    async def _handler(self, connection: ServerConnection) -> None:
        self._clients.add(connection)
        try:
            await connection.send(self._encode({
                "type": "state",
                "artifacts": self._store.snapshot(),
            }))
            async for raw in connection:
                try:
                    frame = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                if frame.get("type") == "user.action":
                    self._store.log_user_action(
                        frame.get("action", ""), frame.get("detail", "")
                    )
        finally:
            self._clients.discard(connection)

    @staticmethod
    def _encode(frame: dict[str, Any]) -> str:
        return json.dumps(frame, ensure_ascii=False)

    async def _send_to(self, connection: ServerConnection, payload: str) -> None:
        try:
            await connection.send(payload)
        except Exception:
            self._clients.discard(connection)

    async def broadcast(self, frame: dict[str, Any]) -> None:
        if not self._clients:
            return
        payload = self._encode(frame)
        await asyncio.gather(*(self._send_to(c, payload) for c in list(self._clients)))

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
            self._server = None


def new_artifacts_channel(store: ArtifactStore, server: SurfaceServer):
    chan = new_channel(
        name="artifacts",
        description=(
            "streamed visual artifacts. write(kind, label, chunks__) streams a "
            "source into a live page — announced, accumulated visibly, then "
            "finalized by kind (canvas / mermaid / markdown / image / hls / term / html / style). "
            "read/history/display recall it."
        ),
    )

    @chan.build.instruction
    def instruction() -> str:
        return (
            f"surface: {server.url} — open it in a browser to watch artifacts appear live.\n"
            "stream the artifact source as the tag body, wrapped in <![CDATA[ ... ]]> when "
            "it holds XML-like characters. the body streams token by token and must not "
            "contain CTML.\n"
            "kinds: canvas (JavaScript on a 480x300 2D canvas, `ctx` and `canvas` in scope), "
            "mermaid (diagram source), markdown (markdown text), image (image or MJPEG URL), "
            "hls (an HLS .m3u8 stream URL), term (monospace blackboard text), "
            "html (a full document in a sandboxed iframe), "
            "style (CSS injected globally — a meta-artifact, never listed).\n"
            "label is the handle: read(label) recalls the source, display(label) switches to it, "
            "remove(label) deletes it."
        )

    @chan.build.notice
    def notice() -> str:
        actions = store.recent_user_actions(5)
        if not actions:
            return "no human actions yet"
        return "recent human actions: " + "; ".join(actions)

    @chan.build.command(always_observe=False)
    async def write(kind: str, label: str, chunks__: str, description: str = "", duration: float = 0) -> str:
        """Stream a source into the artifacts surface and finalize it by kind.

        The body streams token by token: the head announces the artifact, the
        deltas accumulate its source in the page, the tail finalizes it into a
        rendered result. `kind` picks the renderer and the meaning of the source:

        - canvas: JavaScript drawn on a 480x300 2D canvas; `ctx` and `canvas` in scope.
        - mermaid: a mermaid diagram source (flowchart, sequence, class, state, ...).
        - markdown: markdown text, rendered to HTML.
        - image: an image URL or data URL (also MJPEG streams) shown full-frame.
        - hls: an HLS (.m3u8) stream URL, played in an autoplaying video.
        - term: plain text rendered as a green-on-black monospace blackboard.
        - html: a full HTML document rendered in a sandboxed iframe (escape hatch).
        - style: CSS injected globally — a meta-artifact, never listed.

        :param kind: canvas | mermaid | markdown | image | hls | term | html | style
        :param label: stable name; the handle for read(label) / display(label) / remove(label)
        :param description: one-line caption shown in the card
        :param duration: seconds to hold the artifact after it finalizes, measured from
            command start; 0 (default) finishes without blocking. Use it to pace a
            sequence of artifacts without a `sleep` primitive between them.
        """
        start = time.monotonic()
        started = False
        parts: list[str] = []
        async for chunk in chunks__:
            if not started:
                started = True
                store.start(label, kind, description)
                await server.broadcast({
                    "type": "artifact.start", "label": label,
                    "kind": kind, "description": description,
                })
            parts.append(chunk)
            store.append(label, chunk)
            await server.broadcast({"type": "artifact.chunk", "label": label, "text": chunk})
        if not started:
            store.start(label, kind, description)
            await server.broadcast({
                "type": "artifact.start", "label": label,
                "kind": kind, "description": description,
            })
        source = "".join(parts)
        store.end(label, source)
        await server.broadcast({"type": "artifact.end", "label": label, "source": source})
        if duration > 0:
            remaining = duration - (time.monotonic() - start)
            if remaining > 0:
                await asyncio.sleep(remaining)
        return f"artifact '{label}' ({kind}), {len(source)} chars"

    @chan.build.command(always_observe=True)
    async def read(label: str) -> str:
        """Return the stored source of an artifact by label."""
        source = store.read(label)
        if source is None:
            return f"no artifact named '{label}'"
        return source

    @chan.build.command(always_observe=True)
    async def history(n: int = 10) -> str:
        """List recent artifacts, most recent first.

        :param n: how many entries to show.
        """
        items = store.history(n)
        if not items:
            return "no artifacts"
        return "\n".join(
            f"{i['label']}: {i['description']} [{i['kind']}]" for i in items
        )

    @chan.build.command(always_observe=False)
    async def display(label: str) -> str:
        """Switch the surface to this artifact, full-screen."""
        item = store.get(label)
        if item is None:
            return f"no artifact named '{label}'"
        if item["kind"] == "style":
            return f"'{label}' is a style — it applies globally, not displayable"
        await server.broadcast({"type": "artifact.focus", "label": label})
        return f"displayed '{label}'"

    @chan.build.command(always_observe=False)
    async def source(show: bool) -> str:
        """Show or hide the source pane on the surface.

        :param show: True shows the code pane, False hides it for a full-screen render.
        """
        await server.broadcast({"type": "source", "show": show})
        return f"source pane {'shown' if show else 'hidden'}"

    @chan.build.command(always_observe=False)
    async def remove(label: str) -> str:
        """Remove a single artifact from the surface and memory."""
        if store.get(label) is None:
            return f"no artifact named '{label}'"
        store.remove(label)
        await server.broadcast({"type": "artifact.remove", "label": label})
        return f"removed '{label}'"

    @chan.build.command(always_observe=False)
    async def clear() -> str:
        """Remove every artifact from the surface and from memory."""
        count = store.clear()
        await server.broadcast({"type": "clear"})
        return f"cleared {count} artifacts"

    @chan.build.startup
    async def _startup() -> None:
        await server.start()

    @chan.build.close
    async def _close() -> None:
        await server.stop()

    return chan


async def main(matrix: Matrix) -> None:
    store = ArtifactStore()
    server = SurfaceServer(store, host=HOST, port=PORT, html_path=_INDEX_HTML)
    channel = new_artifacts_channel(store, server)
    await matrix.provide_channel(channel)


if __name__ == "__main__":
    Matrix.discover().run(main)
