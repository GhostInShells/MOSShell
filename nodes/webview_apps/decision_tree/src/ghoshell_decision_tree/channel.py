"""The decision-tree channel — the model's control surface over the graph.

The channel is the only writer of ``tree.jsonl`` (K4/K11): the human clicks on
the web surface and the model drives structure through these commands; nobody
hand-edits the structure files. Content lives in a node's directory, edited with
any tool — outside this channel.
"""

from __future__ import annotations

import subprocess
import time
from collections import deque
from typing import Any, Callable

from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.message import Message
from ghoshell_moss.signals import NotifySignalMeta

from .meta import (
    CreateNodeSeed,
    CreateTreeSeed,
    LinkNodeSeed,
    UpdateNodeSeed,
    instruction_schema,
)
from .store import DecisionTreeStore

__all__ = ["build_decision_tree_channel"]


def build_decision_tree_channel(
    store: DecisionTreeStore,
    *,
    surface: Any,
    signaler: Callable | None = None,
) -> Any:
    chan = new_channel(
        name="decision_tree",
        description=(
            "a decision-tree outline for guided creation — a visualized, layered "
            "state machine the human and model shape together. create_node/link_node "
            "grow the tree, update_node moves a node's status; the web surface "
            "(left: tree, right: detail) is the human's view of the same graph."
        ),
    )

    opened: dict[str, Any] = {}
    recent: deque[str] = deque(maxlen=20)

    def _key(rel: str) -> str:
        return str(store.tree_root(rel))

    def _ensure_open(rel: str) -> None:
        key = _key(rel)
        if key not in opened:
            meta, _ = store.load_tree(rel)
            opened[key] = meta

    def _state_payload() -> dict[str, Any]:
        return {"trees": [{"rel": rel, "snapshot": store.snapshot(rel)} for rel in opened]}

    async def _refresh() -> None:
        await surface.broadcast({"type": "state", **_state_payload()})

    async def _push_detail(rel: str, name: str) -> None:
        snap = store.snapshot(rel)
        node = next((n for n in snap["nodes"] if n["name"] == name), None)
        if node is None:
            return
        d = store.node_dir(rel, name)
        exists = d.is_dir()
        entries = [
            {"name": e["name"], "dir": e["dir"], "path": str(d / e["name"])}
            for e in store.ls(rel, name)
        ] if exists else []
        await surface.broadcast({
            "type": "detail", "rel": _key(rel), "name": name,
            "dir": str(d), "exists": exists, "ls": entries,
        })

    def _note(text: str) -> None:
        recent.append(f"[{time.strftime('%H:%M:%S')}] {text}")

    def _notify(text: str) -> None:
        if signaler is None:
            return
        signal = NotifySignalMeta(next=True).to_signal(
            Message.new(tag="decision_tree", name="decision_tree").with_content(text),
            description=text[:120],
        )
        signaler(signal)

    async def _on_action(frame: dict[str, Any]) -> None:
        action = frame.get("action", "")
        rel = frame.get("rel", "")
        name = frame.get("name", "")
        if action == "select":
            _note(f"human opened {name or rel}")
            if name:
                await _push_detail(rel, name)
        elif action == "confirm":
            _note(f"human confirmed {name or rel}")
            _notify(f"[decision_tree] human confirmed {name or rel}")
        elif action == "open_path":
            await _open_path(frame.get("path", ""))

    async def _open_path(path: str) -> None:
        try:
            p = store.resolve(path)
        except ValueError as e:
            _note(f"open_path refused: {e}")
            return
        subprocess.run(["open", str(p)])
        _note(f"opened {p}")

    surface.get_state = _state_payload
    surface.on_action = _on_action

    # -- instruction / notice ----------------------------------------------

    @chan.build.instruction
    def instruction() -> str:
        return (
            "There is a live web surface (its URL is in this channel's `url` notice) — "
            "open it to see the tree on the left and the detail pane on the right.\n"
            "This channel drives a decision tree on disk. The tree's root meta "
            "(DECISION_TREE_ROOT.md) declares the statuses and edge vocabulary; the nodes "
            "and their state live in an append-only tree.jsonl; a node's own directory is "
            "created lazily, only when it has material.\n"
            "Grow the tree with create_node / link_node, move a node's state with "
            "update_node. The human sees every change and clicks to confirm on the surface. "
            "`read` returns the tree meta or a node's state + directory listing; `focus` "
            "switches the surface. text__ bodies are JSON — one schema per command:\n\n"
            + instruction_schema()
        )

    @chan.build.named_notices
    def named_notices() -> dict[str, str]:
        return {"url": surface.url}

    @chan.build.notice
    def notice() -> str:
        parts: list[str] = []
        if opened:
            parts.append("trees: " + ", ".join(opened))
        if recent:
            parts.append("recent: " + "; ".join(list(recent)[-5:]))
        return "\n".join(parts) if parts else "no trees open"

    # -- graph commands -----------------------------------------------------

    @chan.build.command()
    async def create_tree(text__: str) -> str:
        """Create a decision tree: its root meta (DECISION_TREE_ROOT.md) + an empty log.

        `text__` is a JSON string of CreateTreeSeed — see the instruction for the schema.
        """
        seed = CreateTreeSeed.model_validate_json(text__)
        meta = store.create_tree(seed)
        opened[_key(seed.root)] = meta
        _note(f"created tree {seed.root}")
        await _refresh()
        return f"tree '{meta.name}' created at {seed.root}"

    @chan.build.command()
    async def open_tree(root: str) -> str:
        """Open an existing tree (discovered by its DECISION_TREE_ROOT.md) for viewing."""
        _ensure_open(root)
        await _refresh()
        return f"tree '{root}' open"

    @chan.build.command(always_observe=True)
    async def trees() -> str:
        """List the open trees."""
        if not opened:
            return "no trees open"
        return "\n".join(f"{rel}: {m.title}" for rel, m in opened.items())

    # -- node commands ------------------------------------------------------

    @chan.build.command()
    async def create_node(text__: str) -> str:
        """Create a node hanging off an existing node (or the graph root).

        `text__` is a JSON string of CreateNodeSeed — see the instruction for the schema.
        """
        seed = CreateNodeSeed.model_validate_json(text__)
        _ensure_open(seed.tree)
        ev = store.create_node(seed.tree, seed)
        _note(f"create_node {seed.name} under {seed.from_node or 'root'}")
        await _refresh()
        await surface.broadcast({
            "type": "card", "rel": _key(seed.tree), "name": seed.name,
            "action": "create_node", "detail": ev,
        })
        return f"node '{seed.name}' created"

    @chan.build.command()
    async def link_node(text__: str) -> str:
        """Link two existing nodes with an edge (cross-edge; unused for a plain tree).

        `text__` is a JSON string of LinkNodeSeed — see the instruction for the schema.
        """
        seed = LinkNodeSeed.model_validate_json(text__)
        _ensure_open(seed.tree)
        ev = store.link_node(seed.tree, seed)
        _note(f"link_node {seed.node} -> {seed.linked_node}")
        await _refresh()
        return f"linked {seed.node} -> {seed.linked_node} [{seed.edge}]"

    @chan.build.command()
    async def update_node(text__: str) -> str:
        """Change a node's status / note / pruned flag / title / description.

        `text__` is a JSON string of UpdateNodeSeed — see the instruction for the schema.
        """
        seed = UpdateNodeSeed.model_validate_json(text__)
        _ensure_open(seed.tree)
        ev = store.update_node(seed.tree, seed)
        _note(f"update_node {seed.name}")
        await _refresh()
        await surface.broadcast({
            "type": "card", "rel": _key(seed.tree), "name": seed.name,
            "action": "update_node", "detail": ev,
        })
        return f"node '{seed.name}' updated"

    # -- observation --------------------------------------------------------

    @chan.build.command(always_observe=True)
    async def read(root: str, name: str = "") -> str:
        """Read the tree meta (name empty) or a node's folded state + directory listing.

        :param root: graph root dir
        :param name: node name; empty reads the tree root meta
        """
        _ensure_open(root)
        if not name:
            meta, body = store.load_tree(root)
            lines = [
                f"tree {meta.name}: {meta.title}",
                meta.description,
                "statuses: " + ", ".join(f"{k}({v})" for k, v in meta.statuses.items()),
                "edges: " + ", ".join(f"{k}: {v}" for k, v in meta.edges.items()),
            ]
            if body.strip():
                lines.append("---\n" + body.strip())
            return "\n".join(lines)
        snap = store.snapshot(root)
        node = next((n for n in snap["nodes"] if n["name"] == name), None)
        if node is None:
            return f"no node named '{name}'"
        state = f"{node['status']} ({node['level']})" + (" PRUNED" if node["pruned"] else "")
        lines = [f"{node['name']} — {node['title']}", f"status: {state}"]
        if node["status_note"]:
            lines.append(f"note: {node['status_note']}")
        if node["description"]:
            lines.append(node["description"])
        children = snap["children"].get(name, [])
        if children:
            lines.append("children: " + ", ".join(f"{c['name']}[{c['edge']}]" for c in children))
        entries = store.ls(root, name)
        if entries:
            lines.append("dir: " + ", ".join(("d:" if e["dir"] else "") + e["name"] for e in entries))
        return "\n".join(lines)

    @chan.build.command(always_observe=False)
    async def focus(root: str, name: str = "") -> str:
        """Switch the web surface's focus to a tree (name empty) or a node."""
        _ensure_open(root)
        await surface.broadcast({"type": "focus", "rel": _key(root), "name": name})
        if name:
            await _push_detail(root, name)
        return f"focused {'tree' if not name else name}"

    @chan.build.command(always_observe=True)
    async def history(root: str, name: str) -> str:
        """Return the event-log lines that mention a node (its structural history)."""
        events = store.history(root, name)
        if not events:
            return f"no events for '{name}'"
        return "\n".join(
            f"{e.get('at','')} {e.get('ev')} {e.get('status') or e.get('edge') or ''}".strip()
            for e in events
        )

    # -- lifecycle ----------------------------------------------------------

    @chan.build.close
    async def _close() -> None:
        await surface.stop()

    return chan
