"""Ideas — the surface over a graph of plain JSON.

The mechanism is one `graph.json` per group; `cli.py` is the implementation that edits
it; this process serves the page and exposes the channel. Every command here is a thin
wrapper over that same script, so there is one implementation of "change the graph" and
the node is never the only way in.

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

from store import GRAPH, GraphStore, resolve_root, spine

_NODE_DIR = Path(__file__).resolve().parent
_INDEX_HTML = _NODE_DIR / "index.html"
_CLI = _NODE_DIR / "cli.py"

# `list` is also a command name inside new_ideas_channel — it shadows the builtin
# in that scope, so a bare `isinstance(x, list)` there checks against a function.
# Keep the type under a private alias for runtime use.
_LIST = list


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


# --------------------------------------------------------------------------- server


class IdeasServer:
    """One port serves the page, the raw graph files and the WebSocket."""

    def __init__(self, store: GraphStore, *, host: str, port: int, html_path: Path) -> None:
        self._store = store
        self._host = host
        self._html_path = html_path
        self._clients: set[ServerConnection] = set()
        self._server: Any = None
        self.port = port
        # 屏上是谁在看哪一组——这是共享感知的一部分，不是页面自己的私有状态。
        self.view = {"group": "", "selected": ""}

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
        if path == "/state":
            # 谁在磁盘上改了图，页面下一次问就到了——新鲜度是读的人的事。
            self._store.reload()
            return connection.respond(200, json.dumps({"type": "state", **self.snapshot()}))
        if path == "/ws":
            return None
        if path.startswith("/raw/"):
            name = path[len("/raw/"):]
            group = self._store.groups.get(name)
            if group is None or not group.graph_path.exists():
                return connection.respond(404, "no such group")
            response = connection.respond(200, group.graph_path.read_text(encoding="utf-8"))
            del response.headers["Content-Type"]
            response.headers["Content-Type"] = "application/json; charset=utf-8"
            return response
        return connection.respond(404, "not found")

    def snapshot(self) -> dict:
        focus = self._store.focus_group()
        return {
            "root": str(self._store.root),
            "focus": focus.name if focus else "",
            "view": dict(self.view),
            "groups": [g.snapshot() for g in self._store.groups.values()],
            "broken": list(self._store.broken),
        }

    async def _handler(self, connection: ServerConnection) -> None:
        self._clients.add(connection)
        try:
            await connection.send(json.dumps({"type": "state", **self.snapshot()}))
            async for raw in connection:
                try:
                    frame = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                kind = frame.get("type")
                if kind == "ping":
                    await connection.send(json.dumps({"type": "pong"}))
                    continue
                # 人的手在画布上：建、改、连、删、标红。落盘的是同一份 graph.json。
                if kind in ("view-group", "view-select"):
                    # 人这边在看什么，就是我要对齐的那个"什么"。页面报，我读。
                    if kind == "view-group":
                        self.view["group"] = str(frame.get("group", ""))
                        self.view["selected"] = ""
                    else:
                        self.view["selected"] = str(frame.get("id", ""))
                    await self.broadcast({"type": "eye", **self.view})
                    continue
                if kind in ("add", "set", "link", "unlink", "rm", "concept", "op"):
                    # 页面的每一帧都是同一个 op 协议的一行——不是第二条写路径。
                    try:
                        self._store.reload()
                        if kind == "op":
                            payload = frame.get("ops")
                        else:
                            one = {k: v for k, v in frame.items() if k != "type"}
                            one["op"] = kind
                            if kind == "add" and "from" in one:
                                one["parent"] = one.pop("from")   # the page says `from`
                            payload = [one]
                        if isinstance(payload, dict):
                            payload = [payload]
                        added = None
                        if kind == "add":
                            before = {n.id for g in self._store.groups.values() for n in g.nodes.values()}
                        out = self._store.apply_ops(payload)
                        if kind == "add":
                            after = {n.id: n for g in self._store.groups.values() for n in g.nodes.values()}
                            fresh = [nid for nid in after if nid not in before]
                            added = after[fresh[0]].id if fresh else None
                        await self.publish()
                        if added:
                            await connection.send(json.dumps({"type": "added", "id": added}))
                        elif kind == "rm":
                            await connection.send(json.dumps({"type": "removed", "node": frame.get("id")}))
                        else:
                            await connection.send(json.dumps({"type": "ok", "result": out}))
                    except Exception as exc:  # noqa: BLE001 — tell the page, never swallow
                        await connection.send(json.dumps({"type": "error", "message": f"{exc!r}"}))
        finally:
            self._clients.discard(connection)

    async def _ws_add(self, connection: ServerConnection, frame: dict) -> None:
        self._store.reload()
        group = self._store.require_group(str(frame.get("group", "")))
        label = str(frame.get("label", "") or "").strip() or "新的一颗"
        node = self._store.add_node(
            group, label,
            body=str(frame.get("body", "") or ""),
            concept=str(frame.get("concept", "") or ""),
            parent=str(frame.get("from", "") or ""),
        )
        # 先说「建好了」，再广播状态：反过来的话，页面收到状态时还不知道该选哪一颗。
        await connection.send(json.dumps({"type": "added", "group": group.name, "id": node.id}))
        await self.publish()

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
        self._store.reload()
        await self.broadcast({"type": "state", **self.snapshot()})

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


def new_ideas_channel(store: GraphStore, server: IdeasServer, matrix: Matrix):
    chan = new_channel(
        name="ideas",
        description=(
            "thinking governance as a graph. one `graph.json` per group: nodes carry "
            "label / body / concept / weight / flag, edges are entities, and `rev` moves "
            "on every change. the page is one view of the same file."
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
        await server.publish()
        return text or f"exit {proc.returncode}"

    @chan.build.instruction
    def instruction() -> str:
        return (
            "the graph is the truth: one `graph.json` per group. `tree()` is the cold read — "
            "the whole structure in one call; `diff(since)` says what moved since a rev.\n"
            "a node is `{label, body, concept, weight, flag}`. weight decays with generation "
            "(child = parent × 0.8, floor 0.2), so depth is size: further out *is* more detail.\n"
            "an edge is an entity `{id, from, to, kind}`; kind is detail | depend | related. "
            "labels are display names, ids never change, so renaming cannot break an edge.\n"
            "concepts are the colour scale in `meta.concepts` — a shared vocabulary, not an enum "
            "in code. `flag` is the human's red mark: the node he says has a problem.\n"
            "`critical()` answers where the spine is (longest chain + branch points); "
            "`path(a,b)` and `leaves()` are the pruning questions. `run(text)` is the escape hatch.\n"
            "two planes, kept apart: **data** (`op`, `add`, `set`, `link`, `rm`, `save`) changes "
            "graph.json; **presentation** (`spot`, `walk`, `focus`, `pulse`, `caption`, "
            "`overview`, `view_clear`) never touches it — a narration is a reading, not a fact. "
            "A presentation command's own duration is the effect's duration, so a CTML scope is "
            "how long a slide stays on screen."
        )

    @chan.build.named_notices
    def named_notices() -> dict[str, str]:
        return {"url": server.url, "root": str(store.root)}

    @chan.build.notice
    def notice() -> str:
        if not store.groups:
            return f"no group under {store.root} — mkdir one"
        seen = server.view.get("group")
        focus = store.focus_group()
        lines: list[str] = []
        # 先说屏上那一组：我的感知面和人看到的是同一个东西，否则我说什么都在错图上。
        if seen and seen in store.groups:
            lines.append(f"屏上: {seen}" + (f" · 选中 {server.view.get('selected')}"
                                           if server.view.get("selected") else ""))
        if focus is not None and focus.name != seen:
            lines.append(f"焦点组: {focus.name}（箱里最大的活图）")
        counts = " · ".join(
            f"{g.name}{'*' if g.name == seen else ''} {len(g.nodes)}/{len(g.edges)}"
            for g in store.groups.values() if g.status == "active")
        if counts:
            lines.append("活图: " + counts)
        # 主路只在最大的那一组上算：第一组是例子，不是他正在看的那张图。
        main = store.groups.get(seen) if seen in store.groups else store.focus_group()
        if main is not None and main.nodes:
            chain = spine(main)
            if chain:
                lines.append(f"主路[{main.name}]: " + " → ".join(
                    main.nodes[n].name for n in chain[:5] if n in main.nodes))
        dead = [n.name for g in store.groups.values() for n in g.nodes.values() if n.flag]
        if dead:
            lines.append("标红: " + ", ".join(dead[:5]))
        if store.broken:
            lines.append("坏图: " + "; ".join(store.broken[:2]))
        return "\n".join(lines)

    # ---- read --------------------------------------------------------------
    @chan.build.command(always_observe=True)
    async def groups() -> str:
        """List the groups with node / edge counts and their revision."""
        return await run_cli(["groups"])

    @chan.build.command(always_observe=True)
    async def tree(group: str = "") -> str:
        """The whole graph as a tree, one line per node — the cold read.

        :param group: which group; empty = all of them
        """
        return await run_cli(["tree"] + (["--group", group] if group else []))

    @chan.build.command(always_observe=True)
    async def list(group: str = "", concept: str = "", flagged: bool = False) -> str:
        """Flat list of nodes: id, label, concept, weight, flag.

        :param group: only this group
        :param concept: only this concept
        :param flagged: only the ones the human marked red
        """
        argv = ["list"]
        if group:
            argv += ["--group", group]
        if concept:
            argv += ["--concept", concept]
        if flagged:
            argv += ["--flagged"]
        return await run_cli(argv)

    @chan.build.command(always_observe=True)
    async def show(key: str, group: str = "") -> str:
        """One node in full: fields, body, and the edges around it.

        :param key: the node's id or its label
        :param group: which group; empty = search all
        """
        return await run_cli(["show", key] + (["--group", group] if group else []))

    @chan.build.command(always_observe=True)
    async def diff(since: int = 0, group: str = "") -> str:
        """What changed since a revision — the account, not the truth.

        :param since: only ops after this rev
        :param group: which group
        """
        return await run_cli(["diff", "--since", str(since)] + (["--group", group] if group else []))

    @chan.build.command(always_observe=True)
    async def critical(group: str = "") -> str:
        """Where the spine is: the longest chain, then the branch points.

        :param group: which group
        """
        return await run_cli(["critical"] + (["--group", group] if group else []))

    @chan.build.command(always_observe=True)
    async def path(src: str, dst: str, group: str = "") -> str:
        """Shortest path between two nodes.

        :param src: start node (id or label)
        :param dst: end node (id or label)
        :param group: which group
        """
        return await run_cli(["path", src, dst] + (["--group", group] if group else []))

    @chan.build.command(always_observe=True)
    async def leaves(group: str = "") -> str:
        """Leaf nodes — the pruning candidates.

        :param group: which group
        """
        return await run_cli(["leaves"] + (["--group", group] if group else []))

    # ---- write -------------------------------------------------------------
    @chan.build.command(always_observe=True)
    async def add(label: str, group: str = "", body: str = "", concept: str = "", parent: str = "") -> str:
        """Add a node; with `parent`, it grows out of that node and inherits the edge.

        :param label: the few characters the bubble draws
        :param group: which group; empty = the first one
        :param body: what it is, if it is more than a label
        :param concept: the colour vocabulary word
        :param parent: grow it out of this node (id or label)
        """
        argv = ["add", label]
        if group:
            argv += ["--group", group]
        if body:
            argv += ["--body", body]
        if concept:
            argv += ["--concept", concept]
        if parent:
            argv += ["--from", parent]
        return await run_cli(argv)

    @chan.build.command(always_observe=True)
    async def set_node(key: str, group: str = "", label: str = "", body: str = "",
                       concept: str = "", weight: float = -1.0, flag: bool = False) -> str:
        """Change one node. `flag` is the human's red mark: "this point has a problem".

        :param key: the node's id or label
        :param group: which group
        :param label: new label
        :param body: new body
        :param concept: new concept word
        :param weight: new weight; negative = leave it
        :param flag: mark it red
        """
        argv = ["set", key]
        if group:
            argv += ["--group", group]
        if label:
            argv += ["--label", label]
        if body:
            argv += ["--body", body]
        if concept:
            argv += ["--concept", concept]
        if weight >= 0:
            argv += ["--weight", str(weight)]
        if flag:
            argv += ["--flag"]
        return await run_cli(argv)

    @chan.build.command(always_observe=True)
    async def link(spec: str, group: str = "") -> str:
        """Connect nodes: `a -> b` (detail), `a => b` (depend), `a ~ b` (related).

        :param spec: the edge line, e.g. `存储策略 -> md, 存储策略 => 版本控制`
        :param group: which group
        """
        return await run_cli(["link", spec] + (["--group", group] if group else []))

    @chan.build.command(always_observe=True)
    async def unlink(spec: str, group: str = "") -> str:
        """Drop edges: `a -> b, c ~ d`.

        :param spec: the edge line
        :param group: which group
        """
        return await run_cli(["unlink", spec] + (["--group", group] if group else []))

    @chan.build.command(always_observe=True)
    async def rm(key: str, group: str = "") -> str:
        """Drop a node and every edge touching it.

        :param key: the node's id or label
        :param group: which group
        """
        return await run_cli(["rm", key] + (["--group", group] if group else []))

    @chan.build.command(always_observe=True)
    async def status(group: str, value: str) -> str:
        """Set a group's sign: active | waiting | archived. The graph itself never moves.

        :param group: the group
        :param value: active / waiting / archived
        """
        return await run_cli(["status", group, value])

    @chan.build.command(always_observe=True)
    async def concept(name: str, color: str, group: str = "") -> str:
        """Define or override a concept's colour in a group's scale.

        :param name: the word
        :param color: a hex colour
        :param group: which group
        """
        return await run_cli(["concept", name, color] + (["--group", group] if group else []))


    @chan.build.command(always_observe=True)
    async def ops(chunks__) -> str:
        """Apply ops as a stream: one JSON object per line, landing as each line arrives.

        `chunks__` is an async iterator of text, so the body is *streamed*, not parsed
        as a value — which is why the payload lives here and not in an attribute.
        Each completed line is applied and published immediately, so an audience watches
        the structure being written rather than the result of writing it.

            {"op": "add", "label": "判据", "parent": "ideas"}
            {"op": "link", "src": "ideas", "dst": "判据"}
        """
        buf = ""
        applied = 0
        failed: list[str] = []

        def take(line: str) -> int:
            text = line.strip()
            if not text or text.startswith("#"):
                return 0
            try:
                data = json.loads(text)
            except ValueError as exc:
                failed.append(f"{exc}")
                return 0
            batch = data if isinstance(data, _LIST) else [data]
            store.reload()
            try:
                store.apply_ops(batch)
            except Exception as exc:  # noqa: BLE001 — one bad line must not kill the batch
                failed.append(f"{exc}")
                return 0
            return len(batch)

        async for chunk in chunks__:
            buf += chunk
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                got = take(line)
                if got:
                    applied += got
                    await server.publish()
        if buf.strip():
            got = take(buf)
            if got:
                applied += got
                await server.publish()
        store.reload()
        tail = "; ".join(failed[:2])
        return f"{applied} op(s) applied" + (f" · {tail}" if tail else "")

    # ---- presentation ------------------------------------------------------
    # 展示控制面：这些命令**从不碰 graph.json**。讲述是读法，不是数据。
    # 命令的物理时长就是效果的时长——CTML 的 scope 因此就是"这张幻灯片放多久"。

    def _seen() -> Group | None:
        """The group the human is actually looking at — his eye is the reference."""
        if server.view.get("group"):
            g = store.groups.get(server.view["group"])
            if g is not None:
                return g
        return store.focus_group()

    def _resolve(spec: str) -> list[str]:
        """Spec tokens: node id, node label, or concept word. `_all` clears."""
        group = _seen()
        if group is None or not spec or spec.strip() in ("_all", "all", "*"):
            return []
        ids: list[str] = []
        concepts = group.concepts()
        for token in [t.strip() for t in spec.split(",") if t.strip()]:
            node = group.find(token)
            if node is not None:
                ids.append(node.id)
            elif token in concepts:
                ids.extend(n.id for n in group.nodes.values() if n.concept == token)
        return ids

    async def _view(payload: dict) -> None:
        await server.broadcast({"type": "view", **payload})

    @chan.build.command(always_observe=False)
    async def spot(spec: str = "", dim: float = 0.15, duration: float = 0.0) -> str:
        """Light these up and dim the rest; everything returns to full when the command ends.

        Duration is the command's own life: with `duration` it waits, then clears; with
        0 it waits for the next `overview`/`clear`/`spot`. Contrast beats emphasis —
        dimming the rest is what moves an eye, not colouring one thing.

        :param spec: ids / labels / concept words, comma separated; empty = clear
        :param dim: how dark the rest goes (0..1)
        :param duration: seconds to hold; 0 = until replaced
        """
        ids = _resolve(spec)
        g = _seen()
        await _view({"op": "spot", "ids": ids, "dim": dim, "group": g.name if g else ""})
        if duration > 0:
            await asyncio.sleep(duration)
            await _view({"op": "spot", "ids": [], "dim": dim})
        return f"spot {len(ids)} node(s)" + (f" for {duration:g}s" if duration > 0 else "")

    @chan.build.command(always_observe=False)
    async def walk(spec: str, beat: float = 2.4, dim: float = 0.15) -> str:
        """Move through nodes one at a time, camera following — progressive disclosure.

        The whole graph first, then one node per beat. Speaking a line per beat is the
        point: the visual change lands on the word.

        :param spec: ids / labels / concept words, comma separated, in order
        :param beat: seconds per node
        :param dim: how dark the rest goes
        """
        ids = _resolve(spec)
        group = _seen()
        if group is None or not ids:
            return "walk: nothing to show"
        await _view({"op": "overview"})
        await asyncio.sleep(0.5)
        for nid in ids:
            node = group.nodes.get(nid)
            await _view({"op": "spot", "ids": [nid], "dim": dim})
            await _view({"op": "focus", "id": nid})
            await asyncio.sleep(beat)
        await _view({"op": "overview"})
        await _view({"op": "spot", "ids": [], "dim": dim})
        return f"walk {len(ids)} node(s)"

    @chan.build.command(always_observe=False)
    async def focus(id: str, zoom: float = 1.8) -> str:
        """Push the camera onto one node.

        :param id: node id or label
        :param zoom: how close
        """
        group = _seen()
        node = group.find(id) if group else None
        if node is None:
            return f"focus: no node {id!r}"
        await _view({"op": "focus", "id": node.id, "zoom": zoom})
        await asyncio.sleep(0.65)
        return f"focus {node.name}"

    @chan.build.command(always_observe=False)
    async def overview() -> str:
        """Pull the camera back to the whole graph — how a story ends."""
        await _view({"op": "overview"})
        await asyncio.sleep(0.7)
        return "overview"

    @chan.build.command(always_observe=False)
    async def pulse(id: str, times: int = 2) -> str:
        """Beat a node a few times. Motion rather than colour: an onset grabs the eye.

        :param id: node id or label
        :param times: how many beats
        """
        group = _seen()
        node = group.find(id) if group else None
        if node is None:
            return f"pulse: no node {id!r}"
        times = max(1, min(6, times))
        await _view({"op": "pulse", "id": node.id, "times": times})
        await asyncio.sleep(0.45 * times)
        return f"pulse {node.name} x{times}"

    @chan.build.command(always_observe=False)
    async def caption(text: str, seconds: float = 3.0) -> str:
        """One line along the bottom — for what is not worth saying out loud.

        :param text: the line
        :param seconds: how long it stays
        """
        await _view({"op": "caption", "text": text, "seconds": seconds})
        await asyncio.sleep(seconds)
        return f"caption {seconds:g}s"

    @chan.build.command(always_observe=False)
    async def relayout(pause: float = 0.45) -> str:
        """Re-plan the whole layout — the tail op of a batch of changes.

        While ops stream in, new nodes sit next to their parent; the graph would end up
        piled up without this. Emitting `relayout` after the batch is what makes the
        picture settle: one gesture, one reflow.

        :param pause: seconds before it fires, so the last bubbles are visible first
        """
        await asyncio.sleep(max(0.0, pause))
        await _view({"op": "relayout"})
        await asyncio.sleep(0.8)
        return "relayout"

    @chan.build.command(always_observe=False)
    async def view_clear() -> str:
        """Take every temporary effect off — the graph as it was."""
        await _view({"op": "clear"})
        return "view clear"

    # ---- escape hatch ------------------------------------------------------
    @chan.build.command(always_observe=True)
    async def run(text__: str) -> str:
        """Run any `cli.py` verb exactly as a shell would — the long tail lives here.

        :param text__: the argument line, e.g. `tree --group ideas-design`
        """
        lines = [ln for ln in (text__ or "").splitlines() if ln.strip()]
        if not lines:
            return await run_cli(["--help"])
        # 多行就是一批：治理是通读一遍，不是一条一条来。
        if len(lines) > 1:
            return await run_cli(["batch", *lines])
        return await run_cli(shlex.split(lines[0]))

    return chan


# ------------------------------------------------------------------------------ main


async def main(matrix: Matrix) -> None:
    root = resolve_root(_argv_value("--root") or "", node_dir=_NODE_DIR)
    store = GraphStore(root)
    store.reload()
    server = IdeasServer(store, host=HOST, port=PORT, html_path=_INDEX_HTML)
    await server.start()
    channel = new_ideas_channel(store, server, matrix)
    # `provide_channel` resolves only when the channel is disposed — it is the last await.
    try:
        await matrix.provide_channel(channel)
    finally:
        await server.stop()


if __name__ == "__main__":
    Matrix.discover().run(main)
