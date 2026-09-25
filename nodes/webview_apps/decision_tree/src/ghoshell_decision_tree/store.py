"""The decision-tree store — one process's view over the three storage tiers.

The channel is the only writer of ``tree.jsonl`` (single writer, no lock needed);
every mutation appends one event line and the current tree is a *fold* over the
log (K4/K5). Node directories are created lazily — a structural node needs no
directory until it has material (K4).

Path boundary reuses file_editor's rule: a target must live inside the project
home or the system temp dir, else it is refused (K16).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .meta import (
    DEFAULT_EDGE,
    DEFAULT_STATUS,
    LOG_FILENAME,
    NODES_DIRNAME,
    PRUNED_LEVEL,
    ROOT_META_FILENAME,
    Event,
    NodeState,
    TreeMeta,
)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    end: int | None = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return {}, text
    fm = "\n".join(lines[1:end])
    body = "\n".join(lines[end + 1:])
    return yaml.safe_load(fm) or {}, body


class DecisionTreeStore:
    def __init__(self, *, root: str | Path):
        self._root = Path(root).resolve()
        self._tempdir = Path(tempfile.gettempdir()).resolve()

    # -- path boundary -------------------------------------------------------

    @staticmethod
    def _inside(p: Path, root: Path) -> bool:
        return p == root or root in p.parents

    def resolve(self, path: str) -> Path:
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = self._root / p
        r = p.resolve()
        if self._inside(r, self._root) or self._inside(r, self._tempdir):
            return r
        raise ValueError(f"{path!r} escapes the allowed roots ({self._root}, {self._tempdir})")

    def tree_root(self, rel: str) -> Path:
        rel = rel.strip()
        if not rel or rel in (".", ".."):
            raise ValueError(f"tree root must be a path under the project home, got {rel!r}")
        p = Path(rel)
        r = p.resolve() if p.is_absolute() else (self._root / p).resolve()
        if r == self._root or not self._inside(r, self._root):
            raise ValueError(f"tree root {rel!r} escapes the project home")
        return r

    # -- tree meta -----------------------------------------------------------

    def create_tree(self, seed: Any) -> TreeMeta:
        root = self.tree_root(seed.root)
        if (root / ROOT_META_FILENAME).exists():
            raise ValueError(f"tree already exists at {seed.root!r}")
        meta = TreeMeta(
            name=seed.name,
            title=seed.title,
            description=seed.description,
            statuses=seed.statuses,
            edges=seed.edges,
            created=_now(),
            updated=_now(),
        )
        root.mkdir(parents=True, exist_ok=True)
        (root / NODES_DIRNAME).mkdir(exist_ok=True)
        self._write_tree_meta(root / ROOT_META_FILENAME, meta, seed.description)
        (root / LOG_FILENAME).touch()
        return meta

    def _write_tree_meta(self, path: Path, meta: TreeMeta, body: str) -> None:
        data = meta.model_dump(exclude_none=True)
        fm = yaml.safe_dump(data, sort_keys=False, allow_unicode=True).strip()
        path.write_text(f"---\n{fm}\n---\n\n{body}\n", encoding="utf-8")

    def load_tree(self, rel: str) -> tuple[TreeMeta, str]:
        root = self.tree_root(rel)
        text = (root / ROOT_META_FILENAME).read_text(encoding="utf-8")
        meta, body = _split_frontmatter(text)
        return TreeMeta(**meta), body

    # -- event log -----------------------------------------------------------

    def _log_path(self, rel: str) -> Path:
        return self.tree_root(rel) / LOG_FILENAME

    def _read_events(self, rel: str) -> list[dict[str, Any]]:
        path = self._log_path(rel)
        if not path.is_file():
            return []
        events: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
        return events

    def _append(self, rel: str, event: dict[str, Any]) -> None:
        with self._log_path(rel).open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    # -- node mutations ------------------------------------------------------

    def create_node(self, rel: str, seed: Any) -> dict[str, Any]:
        meta, _ = self.load_tree(rel)
        self._check_edge(meta, seed.edge)
        nodes, _ = self.fold(rel)
        if seed.name in nodes:
            raise ValueError(f"node {seed.name!r} already exists")
        if seed.from_node and seed.from_node not in nodes:
            raise ValueError(f"parent node {seed.from_node!r} does not exist")
        event = Event(
            ev="create",
            name=seed.name,
            from_node=seed.from_node,
            edge=seed.edge,
            title=seed.title,
            description=seed.description,
            status=DEFAULT_STATUS,
            at=_now(),
        )
        self._append(rel, event.model_dump())
        return event.model_dump()

    def link_node(self, rel: str, seed: Any) -> dict[str, Any]:
        meta, _ = self.load_tree(rel)
        self._check_edge(meta, seed.edge)
        nodes, _ = self.fold(rel)
        if seed.node not in nodes or seed.linked_node not in nodes:
            raise ValueError(f"both nodes must exist to link ({seed.node!r}, {seed.linked_node!r})")
        event = Event(ev="link", name=seed.node, from_node=seed.node, to=seed.linked_node, edge=seed.edge, at=_now())
        self._append(rel, event.model_dump())
        return event.model_dump()

    def update_node(self, rel: str, seed: Any) -> dict[str, Any]:
        meta, _ = self.load_tree(rel)
        if seed.status:
            self._check_status(meta, seed.status)
        event = Event(
            ev="update",
            name=seed.name,
            status=seed.status,
            status_note=seed.status_note,
            pruned=seed.pruned,
            title=seed.title,
            description=seed.description,
            at=_now(),
        )
        self._append(rel, event.model_dump())
        return event.model_dump()

    @staticmethod
    def _check_edge(meta: TreeMeta, edge: str) -> None:
        if edge not in meta.edges:
            raise ValueError(f"edge {edge!r} not declared in this tree; declared: {sorted(meta.edges)}")

    @staticmethod
    def _check_status(meta: TreeMeta, status: str) -> None:
        if status not in meta.statuses:
            raise ValueError(f"status {status!r} not declared in this tree; declared: {sorted(meta.statuses)}")

    # -- fold + snapshot -----------------------------------------------------

    def fold(self, rel: str) -> tuple[dict[str, NodeState], dict[str, list[dict[str, str]]]]:
        nodes: dict[str, NodeState] = {}
        children: dict[str, list[dict[str, str]]] = {}
        for ev in self._read_events(rel):
            if ev["ev"] == "create":
                nodes[ev["name"]] = NodeState(
                    name=ev["name"],
                    title=ev.get("title", ""),
                    description=ev.get("description", ""),
                    status=ev.get("status") or DEFAULT_STATUS,
                    created=ev.get("at", ""),
                    updated=ev.get("at", ""),
                )
                if ev.get("from_node"):
                    children.setdefault(ev["from_node"], []).append(
                        {"name": ev["name"], "edge": ev.get("edge") or DEFAULT_EDGE}
                    )
            elif ev["ev"] == "link":
                children.setdefault(ev.get("from_node", ""), []).append(
                    {"name": ev.get("to", ""), "edge": ev.get("edge") or DEFAULT_EDGE}
                )
            elif ev["ev"] == "update":
                node = nodes.get(ev["name"])
                if node is None:
                    continue
                if ev.get("title"):
                    node.title = ev["title"]
                if ev.get("description"):
                    node.description = ev["description"]
                if ev.get("status"):
                    node.status = ev["status"]
                if ev.get("status_note"):
                    node.status_note = ev["status_note"]
                if ev.get("pruned") is not None:
                    node.pruned = ev["pruned"]
                node.updated = ev.get("at", node.updated)
        return nodes, children

    def snapshot(self, rel: str) -> dict[str, Any]:
        meta, body = self.load_tree(rel)
        nodes, children = self.fold(rel)
        targeted = {child["name"] for lst in children.values() for child in lst}
        roots = [name for name in nodes if name not in targeted]
        node_payloads = []
        for node in nodes.values():
            level = PRUNED_LEVEL if node.pruned else meta.statuses.get(node.status, "info")
            node_payloads.append({**node.model_dump(), "level": level})
        return {
            "tree": meta.model_dump(),
            "tree_body": body,
            "nodes": node_payloads,
            "children": children,
            "roots": roots,
        }

    # -- read / ls -----------------------------------------------------------

    def read_tree(self, rel: str) -> str:
        return (self.tree_root(rel) / ROOT_META_FILENAME).read_text(encoding="utf-8")

    def node_dir(self, rel: str, name: str) -> Path:
        return self.tree_root(rel) / NODES_DIRNAME / name

    def history(self, rel: str, name: str) -> list[dict[str, Any]]:
        return [ev for ev in self._read_events(rel) if ev.get("name") == name or ev.get("to") == name]

    def ls(self, rel: str, name: str) -> list[dict[str, Any]]:
        d = self.node_dir(rel, name)
        if not d.is_dir():
            return []
        entries = []
        for e in sorted(os.scandir(d), key=lambda x: x.name):
            if self._ignored(Path(e.path)):
                continue
            entries.append({"name": e.name, "dir": e.is_dir()})
        return entries

    def _ignored(self, path: Path) -> bool:
        if path.name.startswith("."):
            return True
        try:
            r = subprocess.run(
                ["git", "check-ignore", "-q", "--", str(path)],
                cwd=self._root,
                capture_output=True,
            )
            return r.returncode == 0
        except OSError:
            return False
