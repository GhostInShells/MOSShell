
"""Discuss — one conversation, drawn as branches over time.

The problem this answers is not "remember everything". It is that a long
conversation forks constantly and the human carries the forks in his head: a
sentence can throw out three directions, two of them never come back, and nothing
about the conversation records that they were ever raised.

So the unit here is not a summary of the discussion. It is three small facts:

- ``fork``  — a direction leaves the main line here, named in one sentence;
- ``note``  — something worth remembering happened on that direction;
- ``state`` — it landed, it was dropped, or it is still open.

Everything else (which branch, how far, whether it came back) is derived. The
record is append-only and the picture is a projection of it, the same shape the
svg works node uses: the log is the truth, the drawing is what it looks like.

Not tied to memento commits on purpose: a single sentence can fork three branches
that never get discussed, and a commit is far too coarse an instrument for that.

Start:  moss nodes run nodes/webview_apps/discuss
Debug:  python main.py
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from websockets.asyncio.server import ServerConnection, serve

from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.services.webview import WebViewDeclaration, WebViewServer

_NODE_DIR = Path(__file__).resolve().parent
_INDEX_HTML = _NODE_DIR / "index.html"

HOST = os.getenv("MOSS_DISCUSS_HOST", "127.0.0.1")
DISCUSS_DIR = Path(os.getenv("MOSS_DISCUSS_DIR", str(_NODE_DIR / "sessions")))
SUFFIX = ".jsonl"
STATES = ("open", "landed", "dropped")


def _port_from_argv(argv: list[str]) -> int | None:
    if "--port" in argv:
        i = argv.index("--port")
        if i + 1 < len(argv):
            return int(argv[i + 1])
    return None


PORT = _port_from_argv(sys.argv[1:]) or int(os.getenv("MOSS_DISCUSS_PORT", "0"))


@dataclass
class Rec:
    t: float
    kind: str           # fork | note | state
    id: str             # branch id; empty = the main line
    text: str

    def frame(self) -> dict[str, Any]:
        return {"t": self.t, "kind": self.kind, "id": self.id, "text": self.text}


@dataclass
class Board:
    topic: str = ""
    recs: list[Rec] = field(default_factory=list)
    _next: int = 0

    def add(self, kind: str, id_: str, text: str) -> Rec:
        rec = Rec(t=round(time.time(), 3), kind=kind, id=id_, text=text)
        self.recs.append(rec)
        return rec

    def fork(self, text: str) -> Rec:
        self._next += 1
        return self.add("fork", f"b{self._next}", text)

    def state_of(self, id_: str) -> str:
        """A branch's state is the last thing said about it — not a field to keep."""
        for rec in reversed(self.recs):
            if rec.kind == "state" and rec.id == id_:
                return rec.text
        return "open"

    def branches(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for rec in self.recs:
            if rec.kind == "fork":
                out.append({"id": rec.id, "text": rec.text, "t": rec.t,
                            "state": self.state_of(rec.id)})
        return out

    def snapshot(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "recs": [r.frame() for r in self.recs],
            "branches": self.branches(),
            "now": round(time.time(), 3),
        }

    # ---- export ---------------------------------------------------------
    def dump(self) -> list[dict[str, Any]]:
        head = {"type": "discuss", "version": 1, "topic": self.topic}
        return [head] + [{"type": "rec", **r.frame()} for r in self.recs]

    def load(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row.get("type") == "discuss":
                self.topic = row.get("topic", self.topic)
            elif row.get("type") == "rec":
                self.recs.append(Rec(t=float(row.get("t", 0)), kind=row.get("kind", "note"),
                                     id=row.get("id", ""), text=row.get("text", "")))
        self._next = sum(1 for r in self.recs if r.kind == "fork")


def session_path(name: str) -> Path:
    raw = Path(name).expanduser()
    base = raw if raw.is_absolute() else DISCUSS_DIR / raw
    if not base.name.endswith(SUFFIX):
        base = base.with_name(base.name + SUFFIX)
    if not raw.is_absolute():
        root = DISCUSS_DIR.resolve()
        try:
            base.resolve().relative_to(root)
        except ValueError as exc:
            raise ValueError(f"path escapes {root}: {name}") from exc
    return base


class SurfaceServer:
    """One port serves the page and the socket — same shape as the other webview nodes."""

    def __init__(self, board: Board, *, host: str, port: int, html_path: Path) -> None:
        self._board = board
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
            del response.headers["Content-Type"]
            response.headers["Content-Type"] = "text/html; charset=utf-8"
            response.headers["Cache-Control"] = "no-store, max-age=0"
            return response
        if path == "/ws":
            return None
        return connection.respond(404, "not found")

    async def _handler(self, connection: ServerConnection) -> None:
        self._clients.add(connection)
        try:
            await connection.send(self._encode({"type": "state", **self._board.snapshot()}))
            async for raw in connection:
                # The human may drag the timeline; that is a view and needs no reply.
                try:
                    json.loads(raw)
                except (TypeError, ValueError):
                    continue
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
        self._server = await serve(self._handler, self._host, self.port,
                                   process_request=self._process_request)
        sockets = getattr(self._server, "sockets", None)
        if sockets:
            self.port = sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None


def new_discuss_channel(board: Board, server: SurfaceServer, matrix: Matrix):
    chan = new_channel(
        name="discuss",
        description=(
            "one conversation seen as branches over time. fork() opens a direction in "
            "one sentence, note() adds a line to it, state() says whether it landed, was "
            "dropped, or is still open. the picture is a projection of the record."
        ),
    )

    @chan.build.instruction
    def instruction() -> str:
        return (
            "a timeline of the conversation we are having (its URL is in the `url` notice): "
            "the main line runs left to right, every direction that forked off it runs on its "
            "own lane.\n"
            "you are the secretary: `fork(text)` when a direction leaves the main line, "
            "`note(id, text)` when something on it is worth keeping, `state(id, …)` when it "
            "lands, is dropped, or opens again. one sentence each — the detail stays in the "
            "conversation, this is only the index that makes it findable later.\n"
            "fork when the point is NOT on the line being discussed; if it gets settled in "
            "place, mark it landed and leave no lane.\n"
            "the topic names the session; several can coexist.\n"
            "the human does not operate this surface — they watch it and click a point to read "
            "that one line."
        )

    @chan.build.named_notices
    def named_notices() -> dict[str, str]:
        return {"url": server.url}

    @chan.build.notice
    def notice() -> str:
        br = board.branches()
        if not br:
            return f"{board.topic or 'no topic'} · no branches yet"
        groups: dict[str, list[str]] = {}
        for b in br:
            groups.setdefault(b["state"], []).append(b["id"])
        parts = [f"{k}: {len(v)} ({', '.join(v)})" for k, v in groups.items()]
        return f"{board.topic or 'untitled'} · " + " · ".join(parts)

    @chan.build.command(always_observe=False)
    async def topic(name: str) -> str:
        """Name this discussion — one line, the thing we are talking about.

        :param name: the topic; also the default export name
        """
        board.topic = name
        await server.broadcast({"type": "state", **board.snapshot()})
        return f"topic: {name}"

    @chan.build.command(always_observe=False)
    async def fork(text: str) -> str:
        """Open a direction that leaves the main line here.

        One sentence is enough. It does not need to be a topic with a conclusion — a
        branch that is raised and never discussed is a real event, and it is exactly
        what the human is carrying in his head.

        :param text: one sentence — what this direction is
        """
        rec = board.fork(text)
        await server.broadcast({"type": "rec", "rec": rec.frame(), "branches": board.branches()})
        return f"forked {rec.id}: {text}"

    @chan.build.command(always_observe=False)
    async def note(id: str, text: str) -> str:
        """Add one line to an existing branch — the clue future-you needs to find it.

        :param id: the branch handle (``list()`` shows them)
        :param text: one sentence
        """
        if not any(b["id"] == id for b in board.branches()):
            return f"no branch {id!r}"
        rec = board.add("note", id, text)
        await server.broadcast({"type": "rec", "rec": rec.frame(), "branches": board.branches()})
        return f"{id} + {text}"

    @chan.build.command(always_observe=False)
    async def state(id: str, state: str) -> str:
        """Say where a branch stands: ``open``, ``landed`` or ``dropped``.

        "Landed" means the conclusion is somewhere it can be found again — a feature, a
        journal entry, code, a file. "Dropped" is a decision, not a failure: the record
        keeps it, so "we raised this and did not do it" stays visible.

        :param id: the branch handle
        :param state: open | landed | dropped
        """
        if state not in STATES:
            return f"state must be one of {', '.join(STATES)}"
        if not any(b["id"] == id for b in board.branches()):
            return f"no branch {id!r}"
        rec = board.add("state", id, state)
        await server.broadcast({"type": "rec", "rec": rec.frame(), "branches": board.branches()})
        return f"{id} → {state}"

    @chan.build.command(always_observe=False)
    async def list() -> str:
        """List the branches and where each one stands."""
        br = board.branches()
        if not br:
            return "no branches yet"
        return "\n".join(f"{b['id']} [{b['state']}] {b['text']}" for b in br)

    @chan.build.command(always_observe=False)
    async def export(path: str = "") -> str:
        """Write the session out: the record as jsonl, the minutes as markdown.

        The jsonl is the session itself and can be loaded back; the markdown is where a
        human reads it, and it is written from the record rather than by hand — a summary
        that disagrees with the log would be a second truth.

        :param path: target under ``sessions/``; empty = the topic
        """
        target = session_path(path or board.topic or "session")
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = board.dump()
        target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                          encoding="utf-8")
        md = target.with_suffix(".md")
        md.write_text(_minutes(board), encoding="utf-8")
        return f"{board.topic or 'session'} → {target} ({len(board.recs)} recs) + {md.name}"

    @chan.build.command(always_observe=False)
    async def load(path: str) -> str:
        """Read a session back from its jsonl and show it.

        :param path: the file under ``sessions/``
        """
        target = session_path(path)
        if not target.exists():
            return f"no such session: {target}"
        rows = []
        for i, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                return f"{target.name} line {i}: not JSON — nothing loaded"
        board.recs.clear()
        board.load(rows)
        await server.broadcast({"type": "state", **board.snapshot()})
        return f"loaded {len(board.recs)} recs from {target}"

    @chan.build.startup
    async def _startup() -> None:
        await server.start()
        await WebViewServer.serve(
            matrix,
            WebViewDeclaration(url=server.url, title="discuss", icon="discuss"),
        )

    @chan.build.close
    async def _close() -> None:
        await server.stop()

    return chan


def _minutes(board: Board) -> str:
    day = datetime.date.today().isoformat()
    lines = [f"---", f"topic: {board.topic or 'untitled'}", f"date: {day}",
             f"branches: {len(board.branches())}", "---", "",
             f"# {board.topic or 'untitled'}", "",
             "<!-- 由记录生成，不手写：改了记录再导出，不要在这里另写一份真相。 -->", ""]
    for b in board.branches():
        lines.append(f"- **{b['id']}** [{b['state']}] {b['text']}")
        for rec in board.recs:
            if rec.kind == "note" and rec.id == b["id"]:
                lines.append(f"  - {rec.text}")
    return "\n".join(lines) + "\n"


async def main(matrix: Matrix) -> None:
    board = Board()
    server = SurfaceServer(board, host=HOST, port=PORT, html_path=_INDEX_HTML)
    channel = new_discuss_channel(board, server, matrix)
    await matrix.provide_channel(channel)


if __name__ == "__main__":
    Matrix.discover().run(main)
