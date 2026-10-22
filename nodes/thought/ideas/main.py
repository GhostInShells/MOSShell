"""Ideas — thinking governance: an append-only event log, projected onto a force graph.

The log is the truth. Every idea is a sequence of events in `<root>/ideas/ideas.jsonl`;
the current state is a fold of them. The page is handed the raw events and folds them
itself, so the timeline is an exact prefix of the record rather than a saved snapshot.

Start:  moss nodes run nodes/thought/ideas
Debug:  python main.py --root <dir>
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

from pydantic import BaseModel, Field
from websockets.asyncio.server import ServerConnection, serve

from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.blueprint.matrix import Matrix

_NODE_DIR = Path(__file__).resolve().parent
_INDEX_HTML = _NODE_DIR / "index.html"


def _workspace_root() -> Path:
    """Relative roots resolve against the project, never against this node's own folder.

    A node is started with its own directory as cwd, so ``open(".moss/ghosts/moss")``
    would otherwise land inside ``nodes/thought/ideas/`` — which is exactly the kind of
    silent misplacement that makes a record look lost.
    """
    env = os.getenv("MOSS_IDEAS_WORKSPACE", "")
    if env:
        return Path(env).expanduser().resolve()
    for parent in [_NODE_DIR, *_NODE_DIR.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd().resolve()

HOST = os.getenv("MOSS_IDEAS_HOST", "127.0.0.1")


def _port_from_argv(argv: list[str]) -> int | None:
    if "--port" in argv:
        i = argv.index("--port")
        if i + 1 < len(argv):
            return int(argv[i + 1])
    return None


def _roots_from_argv(argv: list[str]) -> list[str]:
    roots: list[str] = []
    for i, arg in enumerate(argv):
        if arg == "--root" and i + 1 < len(argv):
            roots.append(argv[i + 1])
    env = os.getenv("MOSS_IDEAS_ROOTS", "")
    roots += [p for p in env.split(os.pathsep) if p]
    return roots


PORT = _port_from_argv(sys.argv[1:]) or int(os.getenv("MOSS_IDEAS_PORT", "0"))
"""0 = bind an ephemeral port; the real one shows up in the channel's `url` notice."""

# --------------------------------------------------------------------------- models


class IdeaEvent(BaseModel):
    """One line of the log. The op decides what `payload` means."""

    t: str                      # ISO timestamp, when it happened
    id: str                     # which idea
    op: str                     # born | state | priority | tag | untag | link | unlink | note
    payload: dict[str, Any] = Field(default_factory=dict)


class IdeaLink(BaseModel):
    src: str
    dst: str
    kind: str = "related"       # related | depend  (a → b means a depends on b)
    weight: float = 1.0


class Idea(BaseModel):
    id: str
    root: str = ""              # which `ideas/` directory it came from
    text: str = ""              # the first sentence — the whole thought
    label: str = ""             # what the bubble shows; empty = fall back to text
    icon: str = ""              # a glyph, or a URL to a small image, drawn inside the bubble
    born: str = ""
    priority: float = 1.0       # size
    state: str = "spark"        # spark | active | waiting | done | retired
    tags: list[str] = Field(default_factory=list)
    links: list[IdeaLink] = Field(default_factory=list)
    notes: list[dict[str, str]] = Field(default_factory=list)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _apply(ideas: dict[str, Idea], ev: IdeaEvent, root: str) -> None:
    """Fold one event into the current state. Kept deliberately boring."""
    item = ideas.get(ev.id)
    if ev.op == "born":
        ideas[ev.id] = Idea(
            id=ev.id,
            root=root,
            text=str(ev.payload.get("text", "")),
            born=ev.t,
            priority=float(ev.payload.get("priority", 1.0)),
            state=str(ev.payload.get("state", "spark")),
            tags=list(ev.payload.get("tags", [])),
        )
        return
    if item is None:                      # an event for an idea we never saw born
        item = ideas[ev.id] = Idea(id=ev.id, root=root, born=ev.t)
    if ev.op == "state":
        item.state = str(ev.payload.get("state", item.state))
    elif ev.op == "label":
        # 气泡上那几个字是**可治理的元数据**，不是正文的截断。
        # 正文说自己是什么，label 说它在图上叫什么——后者随时可以改。
        item.label = str(ev.payload.get("label", ""))
    elif ev.op == "icon":
        # 一个字符，或者一张小图的地址。'是什么'由 label 说，'长什么样'由它说。
        item.icon = str(ev.payload.get("icon", ""))
    elif ev.op == "priority":
        item.priority = float(ev.payload.get("priority", item.priority))
    elif ev.op == "tag":
        tag = str(ev.payload.get("tag", ""))
        if tag and tag not in item.tags:
            item.tags.append(tag)
    elif ev.op == "untag":
        tag = str(ev.payload.get("tag", ""))
        item.tags = [x for x in item.tags if x != tag]
    elif ev.op == "link":
        link = IdeaLink(src=ev.id, dst=str(ev.payload.get("dst", "")),
                        kind=str(ev.payload.get("kind", "related")),
                        weight=float(ev.payload.get("weight", 1.0)))
        if all(not (l.dst == link.dst and l.kind == link.kind) for l in item.links):
            item.links.append(link)
    elif ev.op == "unlink":
        dst = str(ev.payload.get("dst", ""))
        item.links = [l for l in item.links if l.dst != dst]
    elif ev.op == "note":
        item.notes.append({"t": ev.t, "text": str(ev.payload.get("text", ""))})


# ---------------------------------------------------------------------------- store


class IdeaStore:
    """One append-only log per root; the whole picture is their fold."""

    def __init__(self) -> None:
        self.roots: list[Path] = []
        self.events: list[IdeaEvent] = []      # every event, in arrival order
        self._owner: list[str] = []            # parallel to events: which root wrote it
        self.ideas: dict[str, Idea] = {}

    # -- roots ---------------------------------------------------------------
    def add_root(self, raw: str) -> Path:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = _workspace_root() / path
        path = path.resolve()
        if path not in self.roots:
            self.roots.append(path)
            (path / "ideas").mkdir(parents=True, exist_ok=True)
        return path

    def _log_of(self, root: Path) -> Path:
        return root / "ideas" / "ideas.jsonl"

    # -- read ----------------------------------------------------------------
    def load(self) -> None:
        self.events.clear()
        self._owner.clear()
        self.ideas.clear()
        for root in self.roots:
            log = self._log_of(root)
            if not log.exists():
                continue
            for line in log.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = IdeaEvent.model_validate_json(line)
                except Exception:
                    continue                   # a broken line must not eat the record
                self.events.append(ev)
                self._owner.append(str(root))
                _apply(self.ideas, ev, str(root))

    # -- write ---------------------------------------------------------------
    def append(self, root: Path, op: str, idea_id: str, payload: dict[str, Any]) -> IdeaEvent:
        ev = IdeaEvent(t=_now(), id=idea_id, op=op, payload=payload)
        log = self._log_of(root)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as fh:
            fh.write(ev.model_dump_json() + "\n")
        self.events.append(ev)
        self._owner.append(str(root))
        _apply(self.ideas, ev, str(root))
        self._materialise(root, idea_id)
        return ev

    def _materialise(self, root: Path, idea_id: str) -> None:
        """Keep `<id>.md` as the human face of the same record."""
        item = self.ideas.get(idea_id)
        if item is None:
            return
        lines = [
            "---",
            f"id: {item.id}",
            f"born: {item.born}",
            f"priority: {item.priority}",
            f"state: {item.state}",
            f"tags: [{', '.join(item.tags)}]",
            "---",
            "",
            f"# {item.text}",
            "",
        ]
        for note in item.notes:
            lines.append(f"- {note['t']}　{note['text']}")
        if item.links:
            lines.append("")
            lines.append("## 连着")
            for link in item.links:
                lines.append(f"- --{link.kind}--> {link.dst}")
        target = root / "ideas" / f"{idea_id}.md"
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # -- read models ---------------------------------------------------------
    def root_of(self, idea_id: str) -> Path | None:
        item = self.ideas.get(idea_id)
        if item is None or not item.root:
            return None
        return Path(item.root)

    def snapshot(self) -> dict[str, Any]:
        # 每条事件都带上它出自哪个目录：多目录挂在一起时，页面得能分出哪颗球
        # 属于哪个簇。events 与 _owner 一直平行增长，这里只是把那条影子说出口。
        return {
            "roots": [str(r) for r in self.roots],
            "events": [{**ev.model_dump(), "root": root}
                       for ev, root in zip(self.events, self._owner)],
        }


# --------------------------------------------------------------------------- server


class IdeasServer:
    """One port serves the page and the WebSocket."""

    def __init__(self, store: IdeaStore, *, host: str, port: int, html_path: Path) -> None:
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
            del response.headers["Content-Type"]
            response.headers["Content-Type"] = "text/html; charset=utf-8"
            return response
        if path == "/ws":
            return None
        if path.startswith("/idea/") and path.endswith(".md"):
            # 一颗 idea 的人面文档：`<root>/ideas/<id>.md`。它是投影，不是源——
            # 每次追加事件都会重写，所以这里只读；写要走事件。
            idea_id = path[len("/idea/"):-len(".md")]
            if not idea_id or not all(c.isalnum() or c in "-_" for c in idea_id):
                return connection.respond(400, "bad idea id")
            root = self._store.root_of(idea_id)
            doc = (root / "ideas" / f"{idea_id}.md") if root else None
            if doc is None or not doc.exists():
                return connection.respond(404, "no such idea")
            try:
                body = doc.read_text(encoding="utf-8")
            except OSError:
                return connection.respond(500, "unreadable")
            response = connection.respond(200, body)
            del response.headers["Content-Type"]
            response.headers["Content-Type"] = "text/markdown; charset=utf-8"
            return response
        return connection.respond(404, "not found")

    async def _handler(self, connection: ServerConnection) -> None:
        self._clients.add(connection)
        try:
            await connection.send(json.dumps({"type": "state", **self._store.snapshot()}))
            async for raw in connection:
                try:
                    frame = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                # The human stepping the timeline back is a *view*: the page folds
                # locally, so nothing here has to move any cursor.
                if frame.get("type") == "ping":
                    await connection.send(json.dumps({"type": "pong"}))
        finally:
            self._clients.discard(connection)

    async def broadcast(self, frame: dict[str, Any]) -> None:
        if not self._clients:
            return
        payload = json.dumps(frame)
        dead = []
        for client in list(self._clients):
            try:
                await client.send(payload)
            except Exception:
                dead.append(client)
        for client in dead:
            self._clients.discard(client)

    async def start(self) -> None:
        self._server = await serve(
            self._handler, self._host, self.port,
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


# -------------------------------------------------------------------------- channel


_STATES = ("spark", "active", "waiting", "done", "retired")
_KINDS = ("related", "depend")


def new_ideas_channel(store: IdeaStore, server: IdeasServer, matrix: Matrix):
    chan = new_channel(
        name="ideas",
        description=(
            "thinking governance. ideas live in `<root>/ideas/ideas.jsonl` as an "
            "append-only event log; the current state is a fold of it, and the surface "
            "shows that fold as a force graph over time."
        ),
    )

    @chan.build.instruction
    def instruction() -> str:
        return (
            "capture first, structure later. `spark(text)` writes one line and nothing else "
            "— no tags, no priority, no judgement. That is the whole point: an idea with no "
            "cheap home keeps coming back and crowds out the work in progress.\n"
            "the log is the truth: state, priority, tags and links are all *events*, so an "
            "idea's history is never overwritten, only appended to. Current state = fold.\n"
            "directories are the shards: `open(root)` mounts a directory that carries an "
            "`ideas/`; several can be mounted at once (one directory = one cluster).\n"
            "relations start small: `related` (default) and `depend` (a depends on b). Do not "
            "invent more words until real ones are missing.\n"
            "the surface draws size = priority, ring = state; dragging its timeline steps "
            "through the record, and it never rebuilds the layout."
        )

    @chan.build.named_notices
    def named_notices() -> dict[str, str]:
        return {"url": server.url}

    @chan.build.notice
    def notice() -> str:
        if not store.roots:
            return "no root mounted — open(path_to_a_directory) first"
        by_state: dict[str, int] = {}
        for item in store.ideas.values():
            by_state[item.state] = by_state.get(item.state, 0) + 1
        counts = " · ".join(f"{k} {v}" for k, v in sorted(by_state.items())) or "empty"
        return (
            f"{len(store.ideas)} ideas in {len(store.roots)} root(s) · {len(store.events)} events\n"
            f"states: {counts}"
        )

    # ---- roots -------------------------------------------------------------
    @chan.build.command(always_observe=True)
    async def open(root: str) -> str:
        """Mount a directory that carries (or will carry) an `ideas/` subdirectory.

        Mounting several is normal: one directory is one cluster, and clusters can be
        linked. Re-mounting an already-open root just reloads it.

        :param root: a directory path (absolute, or relative to the working directory)
        """
        path = store.add_root(root)
        store.load()
        await server.broadcast({"type": "state", **store.snapshot()})
        return f"mounted {path} — {len(store.ideas)} ideas, {len(store.events)} events so far"

    @chan.build.command(always_observe=False)
    async def roots() -> str:
        """List the mounted roots, with how many ideas each one holds."""
        if not store.roots:
            return "no root mounted"
        rows = []
        for root in store.roots:
            n = sum(1 for item in store.ideas.values() if item.root == str(root))
            rows.append(f"{root} — {n} ideas")
        return "\n".join(rows)

    # ---- capture -----------------------------------------------------------
    @chan.build.command(always_observe=True)
    async def spark(text: str, priority: float = 1.0, root: str = "") -> str:
        """Write one idea down. This is the whole capture verb — nothing else is asked.

        The pressure leaves because the idea is now durably held somewhere, not because it
        was acted on. Triage (tags, priority changes, promotion) is a different rhythm.

        :param text: the idea, in one sentence — it becomes the bubble's label
        :param priority: how much it weighs; decides the bubble's size, adjustable later
        :param root: which mounted root to write into; empty = the first one
        """
        if not store.roots:
            return "no root mounted — open(path) first"
        target = store.add_root(root) if root else store.roots[0]
        idea_id = f"idea-{int(time.time() * 1000) % 10_000_000:07d}"
        ev = store.append(target, "born", idea_id, {"text": text, "priority": priority})
        await server.broadcast({"type": "event", "event": ev.model_dump(), "root": str(target)})
        return f"{idea_id} born at {ev.t} — {text}"

    @chan.build.command(always_observe=True)
    async def note(id: str, text__: str) -> str:
        """Append a line of content to an idea — the record keeps growing, nothing is replaced.

        :param id: the idea's id
        :param text__: what to add
        """
        root = store.root_of(id)
        if root is None:
            return f"no idea {id!r}"
        ev = store.append(root, "note", id, {"text": text__})
        await server.broadcast({"type": "event", "event": ev.model_dump(), "root": str(root)})
        return f"{id}: +1 note ({len(text__)} chars)"

    # ---- governance --------------------------------------------------------
    @chan.build.command(always_observe=True)
    async def set_state(id: str, state: str) -> str:
        """Move an idea's state. A move is an event, so the trajectory stays readable.

        :param id: the idea
        :param state: spark | active | waiting | done | retired
        """
        if state not in _STATES:
            return f"unknown state {state!r} — one of {', '.join(_STATES)}"
        root = store.root_of(id)
        if root is None:
            return f"no idea {id!r}"
        ev = store.append(root, "state", id, {"state": state})
        await server.broadcast({"type": "event", "event": ev.model_dump(), "root": str(root)})
        return f"{id} → {state}"

    @chan.build.command(always_observe=True)
    async def relabel(id: str, label: str) -> str:
        """Give an idea the few words its bubble shows.

        The full text says what the thought *is*; the label says what it is called on the
        graph. They are different questions — a sentence rarely fits in a circle, and
        truncating it is not naming it. A label is metadata: change it as often as the
        picture needs, and the change is an event like every other.

        :param id: the idea
        :param label: a few characters; empty = fall back to the text
        """
        root = store.root_of(id)
        if root is None:
            return f"no idea {id!r}"
        ev = store.append(root, "label", id, {"label": label})
        await server.broadcast({"type": "event", "event": ev.model_dump(), "root": str(root)})
        return f"{id} label = {label!r}"

    @chan.build.command(always_observe=True)
    async def set_icon(id: str, icon: str) -> str:
        """Give an idea a face: one glyph, or the address of a small image.

        Same contract as `relabel`: the picture is a reading of the record, so what it
        shows is metadata that can be changed at any time — never a thing baked into
        the text.

        :param id: the idea
        :param icon: a character (e.g. 一个 emoji) or an http(s)/data URL to an image
        """
        root = store.root_of(id)
        if root is None:
            return f"no idea {id!r}"
        ev = store.append(root, "icon", id, {"icon": icon})
        await server.broadcast({"type": "event", "event": ev.model_dump(), "root": str(root)})
        return f"{id} icon = {icon!r}"

    @chan.build.command(always_observe=True)
    async def set_priority(id: str, priority: float) -> str:
        """Change how much an idea weighs — the bubble resizes, the record remembers when.

        :param id: the idea
        :param priority: a positive number; bigger draws bigger
        """
        root = store.root_of(id)
        if root is None:
            return f"no idea {id!r}"
        ev = store.append(root, "priority", id, {"priority": priority})
        await server.broadcast({"type": "event", "event": ev.model_dump(), "root": str(root)})
        return f"{id} priority = {priority}"

    @chan.build.command(always_observe=True)
    async def set_tag(id: str, tag: str) -> str:
        """Tag an idea. Tags are applied *after* capture, never at capture time.

        :param id: the idea
        :param tag: the tag word; empty string removes nothing
        """
        root = store.root_of(id)
        if root is None:
            return f"no idea {id!r}"
        if not tag:
            return "empty tag"
        ev = store.append(root, "tag", id, {"tag": tag})
        await server.broadcast({"type": "event", "event": ev.model_dump(), "root": str(root)})
        return f"{id} +tag {tag}"

    @chan.build.command(always_observe=True)
    async def link(src: str, dst: str, kind: str = "related", weight: float = 1.0) -> str:
        """Connect two ideas. A link is an event, and it can carry meaning rather than decoration.

        :param src: the idea the link starts from
        :param dst: the idea it points at
        :param kind: related | depend  (a --depend--> b means a depends on b)
        :param weight: how strong; draws as line thickness
        """
        if kind not in _KINDS:
            return f"unknown kind {kind!r} — one of {', '.join(_KINDS)}"
        root = store.root_of(src)
        if root is None:
            return f"no idea {src!r}"
        if dst not in store.ideas:
            return f"no idea {dst!r}"
        ev = store.append(root, "link", src, {"dst": dst, "kind": kind, "weight": weight})
        await server.broadcast({"type": "event", "event": ev.model_dump(), "root": str(root)})
        return f"{src} --{kind}--> {dst}"

    @chan.build.command(always_observe=True)
    async def unlink(src: str, dst: str) -> str:
        """Drop the link from `src` to `dst`. The old link stays in the record — this is a new event.

        :param src: the idea the link starts from
        :param dst: the idea it used to point at
        """
        root = store.root_of(src)
        if root is None:
            return f"no idea {src!r}"
        ev = store.append(root, "unlink", src, {"dst": dst})
        await server.broadcast({"type": "event", "event": ev.model_dump(), "root": str(root)})
        return f"{src} -/-> {dst}"

    # ---- read --------------------------------------------------------------
    @chan.build.command(always_observe=True)
    async def list(state: str = "", tag: str = "", root: str = "", limit: int = 40) -> str:
        """Read the current fold — the ideas as they stand right now, newest first.

        :param state: keep only this state; empty = all
        :param tag: keep only ideas carrying this tag; empty = all
        :param root: keep only ideas from this root; empty = all
        :param limit: how many to show
        """
        wanted = ""
        if root:
            probe = Path(root).expanduser()
            wanted = str((probe if probe.is_absolute() else _workspace_root() / probe).resolve())
        rows = []
        for item in sorted(store.ideas.values(), key=lambda x: x.born, reverse=True):
            if state and item.state != state:
                continue
            if tag and tag not in item.tags:
                continue
            if wanted and item.root != wanted:
                continue
            rows.append(
                f"{item.id} · {item.state} · p{item.priority:g} · {item.text}"
                + (f" · [{', '.join(item.tags)}]" if item.tags else "")
            )
            if len(rows) >= limit:
                break
        return "\n".join(rows) if rows else "nothing matches"

    @chan.build.command(always_observe=True)
    async def show(id: str) -> str:
        """Read one idea in full: its state, its content lines, and what it is connected to.

        :param id: the idea
        """
        item = store.ideas.get(id)
        if item is None:
            return f"no idea {id!r}"
        lines = [
            f"# {item.text}",
            f"{id} · {item.state} · priority {item.priority:g} · born {item.born}",
            f"root: {item.root}",
        ]
        if item.tags:
            lines.append("tags: " + ", ".join(item.tags))
        if item.notes:
            lines.append("——")
            lines += [f"{n['t']}　{n['text']}" for n in item.notes]
        if item.links:
            lines.append("——")
            lines += [f"--{l.kind}--> {l.dst} (w{l.weight:g})" for l in item.links]
        return "\n".join(lines)

    return chan


# ----------------------------------------------------------------------------- main


async def main(matrix: Matrix) -> None:
    store = IdeaStore()
    for raw in _roots_from_argv(sys.argv[1:]):
        store.add_root(raw)
    if not store.roots:
        default = Path.cwd()
        if (default / "ideas").exists():
            store.add_root(str(default))
    store.load()
    server = IdeasServer(store, host=HOST, port=PORT, html_path=_INDEX_HTML)
    await server.start()
    channel = new_ideas_channel(store, server, matrix)
    await matrix.provide_channel(channel)
    try:
        await asyncio.Event().wait()
    finally:
        await server.stop()


if __name__ == "__main__":
    Matrix.discover().run(main)
