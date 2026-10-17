"""Search research surface — one shared page for tasks, sources and their graph.

This is the prototype node for the search-visualization direction: the page in
``index.html`` carries mock data that shows the product end state, while the
channel exposes the surface to the ghost — its URL, a focus pointer, replay
control, and the human actions that flow back as warm notices.

Start:  moss nodes run nodes/research/web_search
Debug:  python main.py
Port:   --port N, else MOSS_SEARCH_SURFACE_PORT, else an ephemeral port (0)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

from websockets.asyncio.server import ServerConnection, serve

from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.blueprint.matrix import Matrix

_NODE_DIR = Path(__file__).resolve().parent
_INDEX_HTML = _NODE_DIR / "index.html"

HOST = os.getenv("MOSS_SEARCH_SURFACE_HOST", "127.0.0.1")


def resolve_port() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--port", type=int, default=0)
    args, _ = parser.parse_known_args()
    if args.port:
        return args.port
    return int(os.getenv("MOSS_SEARCH_SURFACE_PORT", "0"))


class SearchSurface:
    """One port serves both the page and the WebSocket.

    The page is self-contained (its demo graph lives in ``index.html``), so it
    also opens straight from disk. When served by this node it connects to
    ``/ws``: the channel can push ``focus`` / ``step`` / ``view`` frames down,
    and the human's clicks come back up as ``user.action`` frames.
    """

    def __init__(self, *, host: str, port: int, html_path: Path) -> None:
        self._host = host
        self._html_path = html_path
        self._clients: set[ServerConnection] = set()
        self._actions: deque[str] = deque(maxlen=20)
        self._server: Any = None
        self.port = port

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self.port}"

    def recent_actions(self, n: int = 5) -> list[str]:
        return list(self._actions)[-n:]

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
            await connection.send(self._encode({"type": "state", "prototype": True}))
            async for raw in connection:
                try:
                    frame = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                if frame.get("type") != "user.action":
                    continue
                action = str(frame.get("action", "")).strip()
                if not action:
                    continue
                ts = time.strftime("%H:%M:%S", time.localtime())
                self._actions.append(f"[{ts}] {action}")
        finally:
            self._clients.discard(connection)

    @staticmethod
    def _encode(frame: dict[str, Any]) -> str:
        return json.dumps(frame, ensure_ascii=False)

    async def broadcast(self, frame: dict[str, Any]) -> None:
        if not self._clients:
            return
        payload = self._encode(frame)
        for connection in list(self._clients):
            try:
                await connection.send(payload)
            except Exception:
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
            self._server = None


def build_search_channel(surface: SearchSurface):
    chan = new_channel(
        name="web_search",
        description=(
            "a shared research surface for web search — search tasks are nodes, "
            "sources are evidence inside them, and typed edges form the reproducible "
            "path. The page is a prototype with mock data; focus/replay/view drive "
            "the human's view, and human clicks return in notice."
        ),
    )

    @chan.build.instruction
    def instruction() -> str:
        return (
            "There is a live research surface (its URL is in this channel's `url` notice). "
            "Open it to see the task graph, the selected task's motivation/actions/sources/"
            "artifacts, and a source reader.\n"
            "This node is a prototype: the graph data is embedded in the page as mock data. "
            "The channel exists to validate the shared surface — `focus(task_id)` points the "
            "human's graph at a task, `replay(step)` scrubs the research timeline, and "
            "`view(name)` switches the side panel (task / reader / artifacts). Human clicks "
            "come back in `notice`; they are not yet a search execution surface."
        )

    @chan.build.named_notices
    def named_notices() -> dict[str, str]:
        return {"url": surface.url}

    @chan.build.notice
    def notice() -> str:
        actions = surface.recent_actions(5)
        if not actions:
            return "no human actions yet"
        return "recent human actions: " + "; ".join(actions)

    @chan.build.command(always_observe=False)
    async def focus(task_id: str) -> str:
        """Point the human's research surface at one task node.

        :param task_id: the task node id, e.g. "t5"
        """
        await surface.broadcast({"type": "focus", "task_id": task_id})
        return f"focused {task_id}"

    @chan.build.command(always_observe=False)
    async def replay(step: int) -> str:
        """Move the human's research timeline to a step (0-based).

        :param step: the timeline step; the page clamps it to its own range
        """
        await surface.broadcast({"type": "step", "step": step})
        return f"replay step {step}"

    @chan.build.command(always_observe=False)
    async def view(name: str) -> str:
        """Switch the human's side panel.

        :param name: task | reader | artifacts
        """
        if name not in ("task", "reader", "artifacts"):
            return f"unknown view '{name}' (task | reader | artifacts)"
        await surface.broadcast({"type": "view", "name": name})
        return f"view {name}"

    return chan


async def main(matrix: Matrix) -> None:
    surface = SearchSurface(host=HOST, port=resolve_port(), html_path=_INDEX_HTML)
    channel = build_search_channel(surface)
    await surface.start()
    print(f"[web_search] at {surface.url}", flush=True)
    await matrix.provide_channel(channel)


if __name__ == "__main__":
    Matrix.discover().run(main)
