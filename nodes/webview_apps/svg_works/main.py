"""SVG Works — vector works that stay mutable, and can be replayed to any step.

The difference from `artifacts` is what is kept. Artifacts keeps **results**: each
artifact is one final document, so changing anything means writing the whole thing
again — and re-writing reloads the page, which throws away scroll, animation and any
in-flight state. This node keeps **actions**: a work is an append-only list of small
structural operations, and the DOM on screen is the projection of that list. That
makes "change one node" a cheap, local operation, and makes `checkout` possible —
going back to step n is replaying the first n operations, not restoring a snapshot.

Two rules make replay honest:

- an op must be structural and side-effect free (no timers, no network, no random),
  or replaying it twice would not give the same document;
- streaming is live, but the record is not the stream: a streamed write lands on
  screen token by token, and is recorded once, as one op with its final text.

Start:  moss nodes run nodes/webview_apps/svg_works
Debug:  python main.py
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from websockets.asyncio.server import ServerConnection, serve

from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.services.webview import WebViewDeclaration, WebViewServer

_NODE_DIR = Path(__file__).resolve().parent
_INDEX_HTML = _NODE_DIR / "index.html"

WORKS_DIR = Path(os.getenv("MOSS_SVG_WORKS_DIR", str(_NODE_DIR / "works")))
"""Where a relative export path lands. A work file is `*.jsonl`: a header line, then one
op per line — append-friendly, diff-friendly, and short enough to read with `head`."""

WORK_SUFFIX = ".jsonl"
NOTE_SUFFIX = ".md"
_OP_KINDS = {"insert", "set", "remove"}


def work_path(path: str) -> Path:
    """Resolve an export/import target.

    A relative path lands in ``WORKS_DIR`` (which is gitignored — a shared example is
    committed on purpose, not by accident). The suffix is normalised to ``.jsonl`` so
    the same handle can be used for both halves of a work.
    """
    raw = Path(path).expanduser()
    base = raw if raw.is_absolute() else WORKS_DIR / raw
    if not base.name.endswith(WORK_SUFFIX):
        base = base.with_name(base.name + WORK_SUFFIX)
    if not raw.is_absolute():
        root = WORKS_DIR.resolve()
        try:
            base.resolve().relative_to(root)
        except ValueError as exc:  # a relative path may not climb out of the works dir
            raise ValueError(f"path escapes {root}: {path}") from exc
    return base


def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            meta: dict[str, str] = {}
            for line in parts[1].strip().splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    meta[key.strip()] = value.strip()
            return meta, parts[2].lstrip("\n")
    return {}, text


def write_note(note_path: Path, work: Work) -> None:
    """Keep a work's Markdown half in step with its ops — without touching the prose.

    The prose half is the thinking behind the drawing: mine to write, the human's to
    edit. The machine half is counters and dates. Rewriting the whole file would erase
    the only part that cannot be regenerated from the ops, so an existing body is kept
    as it is.
    """
    today = datetime.date.today().isoformat()
    if note_path.exists():
        meta, body = split_frontmatter(note_path.read_text(encoding="utf-8"))
        meta.setdefault("created", today)
    else:
        meta = {"node": "webview_apps/svg_works", "created": today}
        body = (
            f"# {work.description or work.name}\n\n"
            "<!-- 创作思路写在这里：为什么这样画、哪一笔是后来改的、下次想怎么改。\n"
            "     导出只更新上面 frontmatter 里机器那几行，这一节不会被覆盖。 -->\n"
        )
    meta["name"] = work.name
    meta["ops"] = str(len(work.ops))
    meta["updated"] = today
    if not meta.get("description"):
        meta["description"] = work.description
    order = ["name", "description", "ops", "node", "created", "updated"]
    head = "\n".join(f"{k}: {meta[k]}" for k in order if k in meta)
    note_path.write_text(f"---\n{head}\n---\n\n{body}", encoding="utf-8")

HOST = os.getenv("MOSS_SVG_WORKS_HOST", "127.0.0.1")


def _port_from_argv(argv: list[str]) -> int | None:
    if "--port" in argv:
        i = argv.index("--port")
        if i + 1 < len(argv):
            return int(argv[i + 1])
    return None


PORT = _port_from_argv(sys.argv[1:]) or int(os.getenv("MOSS_SVG_WORKS_PORT", "0"))
"""0 = bind an ephemeral port; the real one is reported back as the warm ``url`` notice."""


@dataclass
class Op:
    """One structural operation — the unit of both editing and replay."""

    n: int                      # 1-based position in the work
    kind: str                   # insert | set | remove
    addr: str                   # CSS selector the op lands on
    payload: dict[str, Any] = field(default_factory=dict)
    note: str = ""              # the motive half of the card title; only the model can give it

    def as_frame(self) -> dict[str, Any]:
        return {"n": self.n, "kind": self.kind, "addr": self.addr,
                "payload": self.payload, "note": self.note}


@dataclass
class Work:
    """A work is a thread: an addressable object with its own action history."""

    name: str
    description: str = ""
    ops: list[Op] = field(default_factory=list)
    cursor: int = 0             # how many ops are currently applied (0..len(ops))

    def add(self, kind: str, addr: str, payload: dict[str, Any], note: str) -> Op:
        op = Op(n=len(self.ops) + 1, kind=kind, addr=addr, payload=payload, note=note)
        self.ops.append(op)
        self.cursor = len(self.ops)
        return op

    def head(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "ops": len(self.ops), "cursor": self.cursor}


class WorkStore:
    """All works. Each one is append-only; `cursor` is where the reader stands."""

    def __init__(self) -> None:
        self._works: dict[str, Work] = {}
        self._order: list[str] = []
        self.current: str = ""

    # ---- works -------------------------------------------------------------
    def open(self, name: str, description: str = "") -> Work:
        work = self._works.get(name)
        if work is None:
            work = Work(name=name, description=description)
            self._works[name] = work
            self._order.append(name)
        elif description and not work.description:
            work.description = description
        self.current = name
        return work

    def get(self, name: str = "") -> Work | None:
        return self._works.get(name or self.current)

    def works(self) -> list[dict[str, Any]]:
        return [self._works[n].head() for n in self._order]

    def works_names(self) -> list[str]:
        return list(self._order)

    # ---- ops ---------------------------------------------------------------
    def record(self, name: str, kind: str, addr: str, payload: dict[str, Any], note: str) -> Op | None:
        work = self.get(name)
        if work is None:
            return None
        return work.add(kind, addr, payload, note)

    def checkout(self, name: str, n: int) -> Work | None:
        work = self.get(name)
        if work is None:
            return None
        work.cursor = max(0, min(n, len(work.ops)))
        return work

    # ---- export / import ---------------------------------------------------
    def dump(self, name: str = "") -> list[dict[str, Any]]:
        """The work as lines: one header, then one op per line."""
        work = self.get(name)
        if work is None:
            return []
        lines: list[dict[str, Any]] = [{
            "type": "work", "version": 1,
            "name": work.name, "description": work.description,
        }]
        lines += [{"type": "op", **op.as_frame()} for op in work.ops]
        return lines

    def load(self, name: str, rows: list[dict[str, Any]], description: str = "") -> Work:
        """Rebuild a work from its lines — the ops, not a rendered document.

        Numbering is rebuilt from position rather than trusted from the file: a hand
        edit that leaves a hole in `n` must not create a work replay cannot address.
        """
        work = Work(name=name, description=description)
        for row in rows:
            if row.get("type") != "op":
                continue
            kind = row.get("kind", "")
            if kind not in _OP_KINDS:
                raise ValueError(f"unknown op kind {kind!r} — file refused, nothing imported")
            work.ops.append(Op(
                n=len(work.ops) + 1, kind=kind, addr=row.get("addr", ""),
                payload=row.get("payload") or {}, note=row.get("note", ""),
            ))
        work.cursor = len(work.ops)
        self._works[name] = work
        if name not in self._order:
            self._order.append(name)
        self.current = name
        return work

    def snapshot(self) -> dict[str, Any]:
        work = self.get()
        return {
            "works": self.works(),
            "current": self.current,
            "ops": [op.as_frame() for op in work.ops] if work else [],
            "cursor": work.cursor if work else 0,
        }


class SurfaceServer:
    """One port serves both the page and the WebSocket.

    A plain GET returns index.html through websockets' `process_request` hook; `/ws`
    returns None so the handshake proceeds. Same shape as the artifacts node.
    """

    def __init__(self, store: WorkStore, *, host: str, port: int, html_path: Path) -> None:
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
            # respond() defaults to text/plain; drop it or the page shows as source.
            del response.headers["Content-Type"]
            response.headers["Content-Type"] = "text/html; charset=utf-8"
            return response
        if path == "/ws":
            return None
        return connection.respond(404, "not found")

    async def _handler(self, connection: ServerConnection) -> None:
        self._clients.add(connection)
        try:
            await connection.send(self._encode({"type": "state", **self._store.snapshot()}))
            async for raw in connection:
                try:
                    frame = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                # The page decides when a human steps the reader back: that is a
                # view action, so it is allowed to be cheap and reversible.
                if frame.get("type") == "user.checkout":
                    work = self._store.checkout("", int(frame.get("n", 0)))
                    if work is not None:
                        await self.broadcast({"type": "cursor", "n": work.cursor})
                # 切换作品是人的"看哪个"，和切分组同一类动作：它改视图，不改任何作品的内容。
                elif frame.get("type") == "user.open":
                    name = str(frame.get("name") or "")
                    if name and (name in self._store.works_names()):
                        self._store.open(name)
                        await self.broadcast({"type": "state", **self._store.snapshot()})
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
            self._handler, self._host, self.port, process_request=self._process_request,
        )
        sockets = getattr(self._server, "sockets", None)
        if sockets:
            self.port = sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None


def new_svg_works_channel(store: WorkStore, server: SurfaceServer, matrix: Matrix):
    chan = new_channel(
        name="svg_works",
        description=(
            "vector works that stay mutable. a work is an addressable object with an "
            "append-only action list; write/set/remove change it locally, checkout "
            "replays the first n actions. nothing here reloads the page."
        ),
    )

    @chan.build.instruction
    def instruction() -> str:
        return (
            "a live surface (its URL is in this channel's `url` notice) shows one work at a "
            "time: the drawing on the left, the action column on the right.\n"
            "a work is a thread — `open_work(name)` starts or selects one; never rewrite a "
            "whole document to change one node.\n"
            "every change is an **op** addressing a node by CSS selector: `write(name, addr, "
            "chunks__)` streams a fragment under that address, `set_attr(name, addr, attr, "
            "value)` changes one attribute, `remove_node(name, addr)` drops a subtree.\n"
            "the op list is the truth and the DOM is its projection: `checkout(name, n)` puts "
            "the reader back at the first n ops, and the page replays them.\n"
            "streaming is live but the record is not the stream — what lands on screen token by "
            "token is recorded once, as a single op with its final text.\n"
            "keep ops structural and free of side effects: an op that reads time, network or "
            "random would not replay to the same document.\n"
            "`note` is the card's title on the action column — say why, not what."
        )

    @chan.build.named_notices
    def named_notices() -> dict[str, str]:
        return {"url": server.url}

    @chan.build.notice
    def notice() -> str:
        work = store.get()
        if work is None:
            return "no work open — open_work(name) starts one"
        return (
            f"current: {work.name} · {len(work.ops)} ops"
            + (f" · reader at {work.cursor}" if work.cursor != len(work.ops) else "")
        )

    # ---- works -------------------------------------------------------------
    @chan.build.command(always_observe=False)
    async def open_work(name: str, description: str = "") -> str:
        """Start a work, or select an existing one.

        A work is the unit a reader can come back to: its own drawing, its own action
        history, its own cursor. Switching works never touches another work's ops.

        :param name: stable handle; also the address other commands use
        :param description: one line saying what this work is, for the human
        """
        work = store.open(name, description)
        await server.broadcast({"type": "state", **store.snapshot()})
        return f"work '{work.name}' — {len(work.ops)} ops, reader at {work.cursor}"

    @chan.build.command(always_observe=False)
    async def works() -> str:
        """List the works and where each reader stands."""
        rows = store.works()
        if not rows:
            return "no works yet"
        return "\n".join(
            f"{w['name']} — {w['ops']} ops, reader at {w['cursor']}"
            + (f" — {w['description']}" if w["description"] else "")
            for w in rows
        )

    # ---- ops ---------------------------------------------------------------
    @chan.build.command(always_observe=False)
    async def write(name: str, addr: str, chunks__: str, note: str = "") -> str:
        """Stream a fragment into the node at `addr` (appended at the end of it).

        The tag body streams token by token: the page shows it landing right away, while
        the record stays one op with the final text. Nothing reloads, so scroll,
        animation and anything else already on screen keep going.

        The fragment is parsed as SVG when `addr` points inside the drawing, and as HTML
        otherwise — no need to say which.

        :param name: the work to change
        :param addr: CSS selector for the parent node, e.g. `#stage` or `[data-addr="arm"]`
        :param note: the card's title — say the motive, not the mechanics
        """
        work = store.get(name)
        if work is None:
            return f"no work '{name}' — open_work first"
        parts: list[str] = []
        async for chunk in chunks__:
            parts.append(chunk)
            await server.broadcast({"type": "stream", "addr": addr, "text": chunk})
        text = "".join(parts)
        op = store.record(name, "insert", addr, {"html": text}, note)
        await server.broadcast({"type": "op", "op": op.as_frame()})
        return f"op {op.n} · insert into {addr} ({len(text)} chars)"

    @chan.build.command(always_observe=False)
    async def set_attr(name: str, addr: str, attr: str, value: str, note: str = "") -> str:
        """Set one attribute on the node at `addr` — the smallest possible change.

        No re-render of anything else: this is the operation that makes "画到一半能改"
        true, because it does not touch the rest of the drawing.

        :param name: the work to change
        :param addr: CSS selector for the node
        :param attr: attribute name, e.g. `fill`, `data-addr`, `transform`
        :param value: new value
        :param note: the card's title
        """
        if store.get(name) is None:
            return f"no work '{name}' — open_work first"
        op = store.record(name, "set", addr, {"attr": attr, "value": value}, note)
        await server.broadcast({"type": "op", "op": op.as_frame()})
        return f"op {op.n} · {addr}[{attr}] = {value}"

    @chan.build.command(always_observe=False)
    async def remove_node(name: str, addr: str, note: str = "") -> str:
        """Remove the node at `addr` and its subtree.

        :param name: the work to change
        :param addr: CSS selector for the node
        :param note: the card's title
        """
        if store.get(name) is None:
            return f"no work '{name}' — open_work first"
        op = store.record(name, "remove", addr, {}, note)
        await server.broadcast({"type": "op", "op": op.as_frame()})
        return f"op {op.n} · remove {addr}"

    # ---- replay ------------------------------------------------------------
    @chan.build.command(always_observe=False)
    async def history(name: str = "", n: int = 20) -> str:
        """Read the action list — the index, not the drawing.

        This is what makes the work addressable in time: each line is an op you can
        checkout to.

        :param name: the work; empty = the current one
        :param n: how many of the most recent ops to list
        """
        work = store.get(name)
        if work is None:
            return "no work open"
        rows = work.ops[-n:]
        if not rows:
            return f"'{work.name}' has no ops yet"
        return "\n".join(
            f"{op.n}. {op.kind} {op.addr}"
            + (f" — {op.note}" if op.note else "")
            + ("  ← reader" if op.n == work.cursor else "")
            for op in rows
        )

    @chan.build.command(always_observe=False)
    async def checkout(name: str = "", n: int = -1) -> str:
        """Put the reader at the n-th op and have the page replay to there.

        Replay, not undo: the ops after n are kept, so moving forward again costs
        nothing. This is a view action — cheap and reversible by design.

        :param name: the work; empty = the current one
        :param n: how many ops to apply (0 = skeleton only, -1 = everything)
        """
        work = store.get(name)
        if work is None:
            return "no work open"
        target = len(work.ops) if n < 0 else n
        store.checkout(work.name, target)
        await server.broadcast({"type": "replay", "n": work.cursor,
                                "ops": [op.as_frame() for op in work.ops[:work.cursor]]})
        return f"'{work.name}' reader at {work.cursor}/{len(work.ops)}"

    @chan.build.command(always_observe=False)
    async def export_work(name: str = "", path: str = "") -> str:
        """Save a work as a pair: `<name>.jsonl` (its ops) and `<name>.md` (its notes).

        The `.jsonl` **is** the work — the action list, which is all a replay needs. The
        `.md` beside it carries the source information and the thinking behind the
        drawing; an export refreshes only the machine fields there, so anything written
        in its body survives.

        A relative path lands in this node's `works/` directory (gitignored: a work worth
        shipping is committed on purpose, not by accident).

        :param name: the work; empty = the current one
        :param path: target path; a bare name is enough, the suffix is normalised
        """
        work = store.get(name)
        if work is None:
            return "no work open"
        target = work_path(path or work.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = store.dump(work.name)
        target.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in lines),
            encoding="utf-8",
        )
        note = target.with_name(target.name[: -len(WORK_SUFFIX)] + NOTE_SUFFIX)
        write_note(note, work)
        return f"'{work.name}' → {target} ({len(lines) - 1} ops) + {note.name}"

    @chan.build.command(always_observe=False)
    async def import_work(path: str, name: str = "") -> str:
        """Load a work from a `.jsonl` file and show it.

        Only the ops are read — the drawing is rebuilt by replaying them, the same path
        `checkout` takes. The `.md` beside it (if present) supplies the description, so
        the notes travel with the file.

        :param path: the `.jsonl` to read; relative paths resolve under `works/`
        :param name: override the work's name; empty = the file's own name
        """
        target = work_path(path)
        if not target.exists():
            return f"no such work file: {target}"
        rows: list[dict[str, Any]] = []
        for i, raw in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except ValueError:
                return f"{target.name} line {i}: not JSON — nothing imported"
        header = next((r for r in rows if r.get("type") == "work"), {})
        work_name = name or header.get("name") or target.stem
        note = target.with_name(target.name[: -len(WORK_SUFFIX)] + NOTE_SUFFIX)
        description = header.get("description", "")
        if note.exists():
            meta, _ = split_frontmatter(note.read_text(encoding="utf-8"))
            description = description or meta.get("description", "")
        try:
            work = store.load(work_name, rows, description)
        except ValueError as exc:
            return str(exc)
        await server.broadcast({"type": "state", **store.snapshot()})
        return f"imported '{work.name}' — {len(work.ops)} ops from {target}"

    @chan.build.command(always_observe=False)
    async def panel(show: bool = True) -> str:
        """Show or collapse the action column: full screen for the drawing.

        What collapses is the **cards**, not the record — the strip that remains still
        carries the reader's position and still opens on a click. A view action, so it
        is the human's to undo as well.

        :param show: True shows the column, False gives the drawing the whole width
        """
        await server.broadcast({"type": "panel", "show": bool(show)})
        return "action column shown" if show else "action column collapsed — the record is still there"

    @chan.build.startup
    async def _startup() -> None:
        await server.start()
        await WebViewServer.serve(
            matrix,
            WebViewDeclaration(url=server.url, title="svg works", icon="svg works"),
        )

    @chan.build.close
    async def _close() -> None:
        await server.stop()

    return chan


async def main(matrix: Matrix) -> None:
    store = WorkStore()
    server = SurfaceServer(store, host=HOST, port=PORT, html_path=_INDEX_HTML)
    channel = new_svg_works_channel(store, server, matrix)
    await matrix.provide_channel(channel)


if __name__ == "__main__":
    Matrix.discover().run(main)
