"""Ideas — the surface over plain files.

The mechanism is `*.idea.md` files in first-level subdirectories of a root; `cli.py`
is the implementation that edits them; this process serves the page and exposes the
channel. Every command here is a thin wrapper over that same script, so there is one
implementation of "write an idea" and the node is never the only way in.

Start:  moss nodes run nodes/thought/ideas
Debug:  python main.py --root <dir>
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any

from websockets.asyncio.server import ServerConnection, serve

from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.blueprint.matrix import Matrix

from store import README, SUFFIX, Store, resolve_root

_NODE_DIR = Path(__file__).resolve().parent
_INDEX_HTML = _NODE_DIR / "index.html"
_CLI = _NODE_DIR / "cli.py"


def _argv_value(flag: str) -> str | None:
    argv = sys.argv[1:]
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None


HOST = os.getenv("MOSS_IDEAS_HOST", "127.0.0.1")
PORT = int(_argv_value("--port") or os.getenv("MOSS_IDEAS_PORT", "0") or 0)
"""0 = bind an ephemeral port; the real one shows up in the channel's `url` notice."""

WATCH_INTERVAL = float(os.getenv("MOSS_IDEAS_WATCH", "2.0"))


# --------------------------------------------------------------------------- server


class IdeasServer:
    """One port serves the page, the markdown docs and the WebSocket."""

    def __init__(self, store: Store, *, host: str, port: int, html_path: Path) -> None:
        self._store = store
        self._host = host
        self._html_path = html_path
        self._clients: set[ServerConnection] = set()
        self._server: Any = None
        self.port = port

    @property
    def url(self) -> str:
        return f"http://{self._host}:{self.port}"

    def _doc(self, path: str) -> tuple[int, str, str] | None:
        """`/doc/<group>/<name>.md` → (status, body, content_type)."""
        parts = [p for p in path.strip("/").split("/") if p]
        if len(parts) == 2 and parts[0] == "readme":
            group = self._store.groups.get(parts[1])
            if group is None:
                return 404, f"no group {parts[1]}", "text/plain; charset=utf-8"
            return 200, group.readme or f"# {group.name}\n\n(no README.md in this group)", "text/markdown; charset=utf-8"
        if len(parts) == 2 and parts[0] == "doc":
            idea = self._store.find(parts[1])
            if idea is None:
                return 404, f"no idea {parts[1]}", "text/plain; charset=utf-8"
            return 200, idea.path.read_text(encoding="utf-8", errors="replace"), "text/markdown; charset=utf-8"
        return None

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
        if path.startswith("/readme/") or path.startswith("/doc/"):
            hit = self._doc(path)
            if hit is None:
                return connection.respond(404, "not found")
            status, body, ctype = hit
            response = connection.respond(status, body)
            del response.headers["Content-Type"]
            response.headers["Content-Type"] = ctype
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

    async def publish(self) -> None:
        await self.broadcast({"type": "state", **self._store.snapshot()})

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


# --------------------------------------------------------------------------- channel


def _signature(store: Store) -> tuple:
    """Cheap change detection: the files themselves, not an event log."""
    out = []
    for group in store.groups.values():
        out.append((group.name, (group.path / README).stat().st_mtime if (group.path / README).exists() else 0))
        for idea in group.ideas.values():
            out.append((group.name, idea.name, idea.mtime))
    return tuple(sorted(out))


async def _watch(store: Store, server: IdeasServer) -> None:
    """Pick up edits made by anybody — a hand, a CLI, another agent."""
    while True:
        await asyncio.sleep(WATCH_INTERVAL)
        before = _signature(store)
        store.reload()
        if _signature(store) != before:
            await server.publish()


def new_ideas_channel(store: Store, server: IdeasServer, matrix: Matrix):
    chan = new_channel(
        name="ideas",
        description=(
            "thinking governance. ideas are plain files: first-level subdirectories of "
            "the root are groups, each `*.idea.md` is one idea, and frontmatter carries "
            "state / priority / tags / links. The page is one view of those files."
        ),
    )

    async def run_cli(argv: list[str]) -> str:
        """One implementation, called the way a person would call it."""
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(_CLI), *argv,
            cwd=str(_NODE_DIR),
            env={**os.environ, "MOSS_IDEAS_ROOT": str(store.root)},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        text = (out.decode() + err.decode()).strip()
        store.reload()
        await server.publish()
        if not text:
            return f"exit {proc.returncode}"
        return text

    @chan.build.instruction
    def instruction() -> str:
        return (
            "capture first, structure later. `spark(text)` writes one file and asks nothing "
            "else — an idea with no cheap home keeps coming back and crowds out the work.\n"
            "the file is the truth: state / priority / tags / links live in its frontmatter, "
            "the thinking lives in its body. Editing the file by hand is a normal move.\n"
            "a group is a directory: first-level subdirectories of the root, each with a "
            "README.md that is that group's own context. The name is the identity and it is "
            "semantic — `duplex-gap.idea.md` says what it is before anything is opened.\n"
            "`rename` rewrites the links that pointed at the old name; the graph draws a dead "
            "link as a cut line rather than hiding it.\n"
            "`run(text)` is the escape hatch: any `cli.py` verb, exactly as in a shell."
        )

    @chan.build.named_notices
    def named_notices() -> dict[str, str]:
        return {"url": server.url, "root": str(store.root)}

    @chan.build.notice
    def notice() -> str:
        if not store.groups:
            return f"no group under {store.root} — mkdir one, or spark into a group"
        total = sum(len(g.ideas) for g in store.groups.values())
        by_state: dict[str, int] = {}
        for idea in store.all_ideas():
            by_state[idea.state] = by_state.get(idea.state, 0) + 1
        counts = " · ".join(f"{k} {v}" for k, v in sorted(by_state.items())) or "empty"
        dead = sum(len(i.dead_links(store.known())) for i in store.all_ideas())
        tail = f"\n{total} ideas in {len(store.groups)} group(s): {', '.join(store.groups)}, {counts}"
        if dead:
            tail += f"\n{dead} dead link(s) — the graph draws them as cut lines"
        if store.broken:
            tail += f"\n{len(store.broken)} broken file(s): " + "; ".join(store.broken[:3])
        return tail.lstrip("\n")

    # ---- read --------------------------------------------------------------
    @chan.build.command(always_observe=True)
    async def groups() -> str:
        """List the groups (first-level directories of the root) with how many ideas each holds."""
        return await run_cli(["groups"])

    @chan.build.command(always_observe=True)
    async def list(group: str = "", state: str = "", tag: str = "", limit: int = 40) -> str:
        """Read the ideas as they stand, most recently touched first.

        :param group: keep only this group; empty = all
        :param state: keep only this state; empty = all
        :param tag: keep only ideas carrying this tag; empty = all
        :param limit: how many to show
        """
        argv = ["list", "--limit", str(limit)]
        if group:
            argv += ["--group", group]
        if state:
            argv += ["--state", state]
        if tag:
            argv += ["--tag", tag]
        return await run_cli(argv)

    @chan.build.command(always_observe=True)
    async def show(name: str) -> str:
        """Read one idea's file exactly as stored — frontmatter and body.

        :param name: the filename stem (what a link points at)
        """
        return await run_cli(["show", name])

    # ---- write -------------------------------------------------------------
    @chan.build.command(always_observe=True)
    async def spark(text: str, group: str = "", name: str = "", priority: float = 1.0) -> str:
        """Capture one idea. Write the sentence; nothing else is asked.

        :param text: the idea, in one sentence — it becomes the body, and the filename
        :param group: which group; empty = the first one
        :param name: filename stem; empty = a slug of the text
        :param priority: how much it weighs — the bubble's size
        """
        argv = ["spark", text, "--priority", str(priority)]
        if group:
            argv += ["--group", group]
        if name:
            argv += ["--name", name]
        return await run_cli(argv)

    @chan.build.command(always_observe=True)
    async def note(name: str, text__: str) -> str:
        """Append a paragraph to an idea's body.

        :param name: the idea
        :param text__: what to add
        """
        return await run_cli(["note", name, text__])

    @chan.build.command(always_observe=True)
    async def set_state(name: str, state: str) -> str:
        """Move an idea's state: spark | active | waiting | done | retired.

        :param name: the idea
        :param state: where it goes
        """
        return await run_cli(["state", name, state])

    @chan.build.command(always_observe=True)
    async def set_priority(name: str, priority: float) -> str:
        """Change how much an idea weighs.

        :param name: the idea
        :param priority: a positive number; bigger draws bigger
        """
        return await run_cli(["priority", name, str(priority)])

    @chan.build.command(always_observe=True)
    async def tag(name: str, word: str) -> str:
        """Tag an idea.

        :param name: the idea
        :param word: the tag
        """
        return await run_cli(["tag", name, word])

    @chan.build.command(always_observe=True)
    async def untag(name: str, word: str) -> str:
        """Remove a tag from an idea.

        :param name: the idea
        :param word: the tag to drop
        """
        return await run_cli(["untag", name, word])

    @chan.build.command(always_observe=True)
    async def link(src: str, dst: str, kind: str = "related") -> str:
        """Connect two ideas. A link is a line in the source file's frontmatter.

        :param src: the idea the link starts from
        :param dst: the idea it points at
        :param kind: related | depend  (a --depend--> b means a depends on b)
        """
        return await run_cli(["link", src, dst, "--kind", kind])

    @chan.build.command(always_observe=True)
    async def unlink(src: str, dst: str) -> str:
        """Drop a connection.

        :param src: the idea the link starts from
        :param dst: the idea it used to point at
        """
        return await run_cli(["unlink", src, dst])

    @chan.build.command(always_observe=True)
    async def rename(old: str, new: str) -> str:
        """Rename an idea's file and rewrite every link that pointed at it.

        The name is the identity, so this is the one verb that sweeps the root.

        :param old: the current filename stem
        :param new: the new one (slugified)
        """
        return await run_cli(["rename", old, new])

    # ---- escape hatch ------------------------------------------------------
    @chan.build.command(always_observe=True)
    async def run(text__: str) -> str:
        """Run any `cli.py` verb, exactly as a shell would — the long tail lives here.

        The body is the argument line, e.g. `list --group voice --state spark`.
        `--help` works, and is the way to re-read the verb list without leaving CTML.

        :param text__: the argument line
        """
        argv = shlex.split(text__ or "")
        if not argv:
            argv = ["--help"]
        return await run_cli(argv)

    return chan


# ------------------------------------------------------------------------------ main


async def main(matrix: Matrix) -> None:
    root = resolve_root(_argv_value("--root") or "", node_dir=_NODE_DIR)
    store = Store(root)
    store.reload()
    server = IdeasServer(store, host=HOST, port=PORT, html_path=_INDEX_HTML)
    await server.start()
    channel = new_ideas_channel(store, server, matrix)
    await matrix.provide_channel(channel)
    watcher = asyncio.create_task(_watch(store, server))
    try:
        await asyncio.Event().wait()
    finally:
        watcher.cancel()
        await server.stop()


if __name__ == "__main__":
    Matrix.discover().run(main)
