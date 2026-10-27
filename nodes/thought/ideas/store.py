"""The ideas graph — one `graph.json` per group, edges as entities.

**The graph is the truth.** A group is a directory; inside it one JSON holds the whole
picture: `meta` (the concept→colour scale), `nodes` (id → label / body / concept /
weight / flag), `edges` (each with its own id, direction and kind), and a monotone
`rev` that every change increments.

Why JSON and not one markdown per idea: a graph is read as a whole. A reader — model or
human — should be able to take the entire structure in one look, and topology
(what depends on what, where the bottleneck is, which branch can be cut) is a question
about the whole, not about a file.

**`ops.jsonl` is the account, not the truth.** Every write appends one line: what
changed, and at which rev. It exists for `diff` (a cold reader asking "what moved since
I last looked") and it may be deleted at any time — the graph would lose its history and
nothing else. There is no replay: `graph.json` never has to be rebuilt from ops.

**Nodes are identified by opaque ids.** `n7` never changes; `label` is what it is called
today. Renaming therefore cannot break an edge — what an edge points at is the id.

**Weight decays with generation.** A child weighs `parent × 0.8` (floor 0.2), so
"further out is more detail" is a fact of the data rather than a matter of taste, and
cutting a branch becomes a decision about size instead of an argument.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

try:  # POSIX only; a missing lock degrades to "write and hope", never to a crash
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

GRAPH = "graph.json"          # the published copy — what the repo carries
LIVE_GRAPH = ".graph.json"     # the working state — what every write touches
IDEAS = "IDEAS.md"
OPS = "ops.jsonl"
LOCK = ".lock"
TRASH = ".trash"
VERSIONS = ".versions"
VERSION_OP = "version"
LEGACY_SUFFIX = ".idea.md"

DECAY = 0.8
FLOOR = 0.2
DEFAULT_CONCEPT = "gray"
DEFAULT_KINDS = ("detail", "depend", "related")
KIND = "detail"

GROUP_STATUS = ("active", "waiting", "archived")

UNDO_DEPTH = 40
_UNDO_DEPTH = UNDO_DEPTH

IDEAS_TEMPLATE = """---
status: {status}
graph: {graph}
updated: {updated}
---

# {name}

One line on what this group is — for whoever opens it next.

- the graph lives in `{graph}`: nodes, edges, the concept scale, rev.
- how to read it: `ideas tree --group {name}` for the whole thing;
  `ideas show <label>` for one node.

> This file is a **sign**: what this is, and whether it is alive. It **does not list
> nodes** — they live in the graph, and a second copy would drift.
"""

# The scale is a fixed vocabulary of colour names, and a node's `concept` is one of
# these words. There is no palette to edit and no word to add: picking a colour *is*
# the whole gesture, and the word a node wears is the colour it draws in.
DEFAULT_CONCEPTS: dict[str, str] = {
    "gray": "#8aa89a",     # unclassified
    "green": "#7ee787",    # execution, implementation
    "blue": "#58a6ff",     # architecture, data structures
    "amber": "#f0b429",    # perception, interaction
    "red": "#d0715c",      # the main contradiction, risk
    "white": "#e8f5ea",    # the trunk: motive, purpose
}


# --------------------------------------------------------------------------- models


@dataclass
class Node:
    id: str
    label: str = ""
    body: str = ""
    concept: str = DEFAULT_CONCEPT
    weight: float = 1.0
    flag: bool = False          # the human's red mark: "I have a problem with this one"
    created: str = ""

    def to_dict(self) -> dict:
        out = {
            "label": self.label,
            "body": self.body,
            "concept": self.concept,
            "weight": round(float(self.weight), 3),
        }
        if self.flag:
            out["flag"] = True
        if self.created:
            out["created"] = self.created
        return out

    @property
    def name(self) -> str:
        """What reads well in a line of text: the label, else the id."""
        return self.label or self.id


@dataclass
class Edge:
    id: str
    src: str
    dst: str
    kind: str = KIND
    label: str = ""

    def to_dict(self) -> dict:
        out = {"id": self.id, "from": self.src, "to": self.dst, "kind": self.kind}
        if self.label:
            out["label"] = self.label
        return out


@dataclass
class Group:
    name: str
    path: Path
    rev: int = 0
    status: str = "active"
    saved_rev: int = 0          # the newest version's rev, read from the sign — never the log
    saved_name: str = ""
    meta: dict = field(default_factory=lambda: {"concepts": {}})
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    # -- read ---------------------------------------------------------------
    @property
    def graph_path(self) -> Path:
        return self.path / LIVE_GRAPH

    @property
    def published_path(self) -> Path:
        """The copy that enters git: rewritten only on save, never on a gesture."""
        return self.path / GRAPH

    @property
    def ops_path(self) -> Path:
        return self.path / OPS

    @property
    def ideas_path(self) -> Path:
        return self.path / IDEAS

    @property
    def versions_path(self) -> Path:
        return self.path / VERSIONS

    def version_files(self) -> dict[int, tuple[str, Path]]:
        """Every checkpoint on disk, keyed by rev — the filename *is* the index.

        No index file: an index is a fourth thing that can drift from the three it
        describes. A directory scan is cheap and cannot lie.
        """
        out: dict[int, tuple[str, Path]] = {}
        if not self.versions_path.is_dir():
            return out
        for path in self.versions_path.glob("r*.json"):
            head, _, tail = path.stem[1:].partition("-")
            if not head.isdigit():
                continue
            out[int(head)] = (tail.replace("_", " ") or "checkpoint", path)
        return out

    def default_concept(self) -> str:
        """The word a new node wears when nobody says otherwise.

        It is the word used the last time a node was created — expressed by using it,
        not by a separate setting nobody remembers to change.
        """
        scale = self.concepts()
        name = str(self.meta.get("default_concept", "") or "")
        return name if name in scale else next(iter(scale), DEFAULT_CONCEPT)

    def concepts(self) -> dict[str, str]:
        """The colour scale: six slots. Their *words* are ours to rename.

        Once a group has touched its scale the map is authoritative — otherwise the
        defaults would keep re-appearing under a name someone already renamed.
        """
        own = {str(k): str(v) for k, v in (self.meta.get("concepts") or {}).items()}
        return own or dict(DEFAULT_CONCEPTS)

    def find(self, key: str) -> Node | None:
        """Accept an id or a label — the human thinks in labels, the file speaks ids."""
        if key in self.nodes:
            return self.nodes[key]
        for node in self.nodes.values():
            if node.label == key:
                return node
        return None

    def children(self, nid: str) -> list[Edge]:
        return [e for e in self.edges if e.src == nid]

    def parents(self, nid: str) -> list[Edge]:
        return [e for e in self.edges if e.dst == nid]

    # -- write --------------------------------------------------------------
    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "path": str(self.path),
            "rev": self.rev,
            "status": self.status,
            "saved_rev": self.saved_rev,
            "saved_name": self.saved_name,
            "default_concept": self.default_concept(),
            "meta": {"concepts": self.concepts()},
            "nodes": [{"id": n.id, **n.to_dict()} for n in self._sorted_nodes()],
            "edges": [e.to_dict() for e in self.edges],
        }

    def _sorted_nodes(self) -> list[Node]:
        def key(n: Node) -> int:
            return int(n.id[1:]) if n.id[1:].isdigit() else 0
        return sorted(self.nodes.values(), key=key)

    def to_json(self) -> str:
        payload = {
            "rev": self.rev,
            "name": self.name,
            "meta": {"concepts": dict(self.meta.get("concepts") or {}),
                     "default_concept": str(self.meta.get("default_concept", "") or "")},
            "nodes": {n.id: n.to_dict() for n in self._sorted_nodes()},
            "edges": [e.to_dict() for e in self.edges],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    def write(self, note: str = "") -> None:
        """Whole-graph save. `rev` moves here so a save is never an anonymous write."""
        self.rev += 1
        write_atomic(self.graph_path, self.to_json())
        if note:
            with self.ops_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"rev": self.rev, "t": _now(), "op": "init", "note": note},
                                    ensure_ascii=False) + "\n")


# ------------------------------------------------------------------------ topology


def roots_of(group: Group) -> list[str]:
    """Nodes nobody points at with a `detail` edge — where the reading starts."""
    pointed = {e.dst for e in group.edges if e.kind == KIND}
    return [n.id for n in group._sorted_nodes() if n.id not in pointed]


def subtree_size(group: Group, nid: str) -> int:
    seen: set[str] = set()
    stack = [nid]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(e.dst for e in group.children(cur) if e.kind == KIND)
    return len(seen) - 1


def shortest_path(group: Group, src: str, dst: str) -> list[str] | None:
    a, b = group.find(src), group.find(dst)
    if a is None or b is None:
        return None
    prev: dict[str, str] = {}
    seen = {a.id}
    queue = [a.id]
    while queue:
        cur = queue.pop(0)
        if cur == b.id:
            path = [b.id]
            while path[-1] != a.id:
                path.append(prev[path[-1]])
            return list(reversed(path))
        neighbours = [e.dst for e in group.children(cur)] + [e.src for e in group.parents(cur)]
        for nxt in neighbours:
            if nxt not in seen:
                seen.add(nxt)
                prev[nxt] = cur
                queue.append(nxt)
    return None


def spine(group: Group) -> list[str]:
    """The longest detail chain — the main contradiction, read as what everything else hangs off.

    Ranked by *length of the longest chain*, not by subtree size: a root always owns
    everything, so counting descendants would answer "where am I" and never "what is
    the spine".
    """
    best: list[str] = []

    def walk(nid: str, trail: list[str], seen: set[str]) -> None:
        nonlocal best
        if len(trail) > len(best):
            best = list(trail)
        for edge in group.children(nid):
            if edge.kind != KIND or edge.dst in seen:
                continue
            walk(edge.dst, trail + [edge.dst], seen | {edge.dst})

    for nid in roots_of(group):
        walk(nid, [nid], {nid})
    return best


def junctions(group: Group) -> list[tuple[str, int]]:
    """Branch points, by degree — where cutting one edge changes the most."""
    deg = {nid: len(group.children(nid)) + len(group.parents(nid)) for nid in group.nodes}
    fan = {nid: len(group.children(nid)) for nid in group.nodes}
    ranked = sorted(group.nodes, key=lambda nid: (-(fan[nid] * 2 + deg[nid]), nid))
    return [(nid, fan[nid]) for nid in ranked]


def centrality(group: Group) -> list[tuple[str, int, int]]:
    """Kept for callers that want a single number: (node, degree, descendants)."""
    deg = {nid: len(group.children(nid)) + len(group.parents(nid)) for nid in group.nodes}
    desc = {nid: subtree_size(group, nid) for nid in group.nodes}
    ranked = sorted(group.nodes, key=lambda nid: (-(desc[nid] * 2 + deg[nid]), nid))
    return [(nid, deg[nid], desc[nid]) for nid in ranked]


def leaves(group: Group) -> list[str]:
    return [n.id for n in group._sorted_nodes() if not group.children(n.id)]


# --------------------------------------------------------------------------- store


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _slug(name: str) -> str:
    """A filename-safe version name: `release 1` → `release_1`."""
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in str(name or "").strip()]
    return "".join(keep).strip("_") or "checkpoint"


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class GraphStore:
    """Every first-level directory under the root is a group with its own graph."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.groups: dict[str, Group] = {}
        self.broken: list[str] = []
        # A short in-memory history per group, for undo. Durability is the versions'
        # job; this only has to survive the last few gestures.
        self._undo: dict[str, list[dict]] = {}

    # -- discovery ----------------------------------------------------------
    def reload(self) -> None:
        self.groups = {}
        self.broken = []
        if not self.root.exists():
            return

        for child in sorted(self.root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            if not (child / IDEAS).is_file():
                # Discovery is the marker, not the position: a directory is a group
                # because it carries the sign. A stray graph without one is still
                # reported — a group that silently vanishes is worse than a broken one.
                if (child / LIVE_GRAPH).exists() or (child / GRAPH).exists():
                    self.broken.append(
                        f"{child.name}: no {IDEAS} — a group is a directory carrying one")
                continue

            group = Group(name=child.name, path=child)
            try:
                self._load_group(group)
            except Exception as exc:  # noqa: BLE001 — a broken graph is reported, never hidden
                self.broken.append(f"{child.name}: {exc}")
            self.groups[child.name] = group

    def _load_group(self, group: Group) -> None:
        if group.ideas_path.exists():
            meta, _body = split_frontmatter(group.ideas_path.read_text(encoding="utf-8", errors="replace"))
            group.status = str(meta.get("status", "active")) or "active"
            saved = str(meta.get("saved", "") or "")
            head, _, saved_name = saved.partition(" ")
            if head.startswith("r") and head[1:].isdigit():
                group.saved_rev, group.saved_name = int(head[1:]), saved_name.strip()
        live = group.graph_path if group.graph_path.exists() else group.published_path
        if not live.exists():
            legacy = sorted(group.path.glob(f"*{LEGACY_SUFFIX}"))
            if legacy:
                self._import_legacy(group, legacy)
                return
            group.write()          # a brand-new, empty graph on disk
            if not group.ideas_path.exists():
                write_ideas_sign(group)
            self.save(group, "init")   # a group starts from a version: history needs a base
            return
        data = json.loads(live.read_text(encoding="utf-8"))
        if live == group.published_path:
            # First run on a published group: the working state is born from the
            # published copy, so the next gesture has somewhere to land.
            write_atomic(group.graph_path, live.read_text(encoding="utf-8"))
        self._load_data(group, data)

    def _load_data(self, group: Group, data: dict) -> None:
        """Fill a group from a graph.json body — shared by the loader and by restore."""
        group.rev = int(data.get("rev", 0))
        raw_meta = data.get("meta") or {}
        group.meta = {"concepts": dict(raw_meta.get("concepts") or {})}
        if raw_meta.get("default_concept"):
            group.meta["default_concept"] = str(raw_meta["default_concept"])
        group.nodes = {}
        for nid, raw in (data.get("nodes") or {}).items():
            group.nodes[nid] = Node(
                id=nid,
                label=str(raw.get("label", "")),
                body=str(raw.get("body", "")),
                concept=str(raw.get("concept", DEFAULT_CONCEPT)),
                weight=float(raw.get("weight", 1.0)),
                flag=bool(raw.get("flag", False)),
                created=str(raw.get("created", "")),
            )
        group.edges = [
            Edge(id=str(raw.get("id", f"e{i}")), src=str(raw.get("from", "")),
                 dst=str(raw.get("to", "")), kind=str(raw.get("kind", KIND)),
                 label=str(raw.get("label", "")))
            for i, raw in enumerate(data.get("edges") or [])
        ]
        if not group.ideas_path.exists():
            write_ideas_sign(group)

    def _import_legacy(self, group: Group, files: list[Path]) -> None:
        """The markdown generation: one file per idea, links in frontmatter.

        Imported once into `graph.json`; the old files move to `.trash/` so the import
        cannot run twice and nothing is destroyed.
        """
        ids: dict[str, str] = {}
        parsed: list[tuple[Path, dict, str]] = []
        for i, file in enumerate(files, start=1):
            raw = file.read_text(encoding="utf-8", errors="replace")
            meta, body = split_frontmatter(raw)
            parsed.append((file, meta, body))
            ids[file.name[: -len(LEGACY_SUFFIX)]] = f"n{i}"
            group.nodes[f"n{i}"] = Node(
                id=f"n{i}",
                label=str(meta.get("label", "") or file.name[: -len(LEGACY_SUFFIX)]),
                body=body.strip(),
                concept=str(meta.get("concept", DEFAULT_CONCEPT)),
                weight=float(meta.get("priority", 1.0) or 1.0),
            )
        edge_n = 1
        for file, meta, _body in parsed:
            src = ids[file.name[: -len(LEGACY_SUFFIX)]]
            for item in meta.get("links") or []:
                if isinstance(item, dict):
                    dst, kind = str(item.get("to", "")), str(item.get("kind", KIND))
                else:
                    dst, _, kind = str(item).partition(":")
                    kind = kind or KIND
                target = ids.get(dst.strip())
                if target and target != src:
                    group.edges.append(Edge(id=f"e{edge_n}", src=src, dst=target,
                                            kind=kind if kind in DEFAULT_KINDS else KIND))
                    edge_n += 1
        group.write(note=f"imported {len(files)} markdown ideas")
        if not group.ideas_path.exists():
            write_ideas_sign(group)
        self.save(group, "imported")
        trash = group.path / TRASH
        trash.mkdir(exist_ok=True)
        for file, _meta, _body in parsed:
            os.replace(file, trash / file.name)


    def create_group(self, path: str) -> Group:
        """Create a group at `path` — writing the sign is the whole act.

        A group is a directory carrying `IDEAS.md`, so creating one is writing that
        sign; the empty graph follows on the next reload. `path` resolves against the
        root, so a plain name stays the common case. An absolute path lets a group
        live anywhere: the root is a default discovery radius, not a boundary.
        """
        clean = str(path or "").strip()
        if not clean:
            raise ValueError("new needs a path")
        target = Path(clean).expanduser()
        if not target.is_absolute():
            target = self.root / clean
        target = target.resolve()
        if target == self.root:
            raise ValueError("the root itself is not a group")
        if target.exists() and not target.is_dir():
            raise ValueError(f"{target} is not a directory")
        if self.root in target.parents and target.parent != self.root:
            raise ValueError(
                f"{target} is nested under the root — the topology is (root) -> (groups), "
                "one level deep. Put the group directly under the root, or outside it and "
                "name it by path.")
        target.mkdir(parents=True, exist_ok=True)
        write_ideas_sign(Group(name=target.name, path=target))
        self.reload()
        group = self.groups.get(target.name)
        if group is not None and group.path.resolve() == target:
            return group
        return self._group_at(target)

    def _group_at(self, path: Path) -> Group:
        """Resolve a group by its address — the door for a group outside the root."""
        if not (path / IDEAS).is_file():
            raise KeyError(f"{path} is not a group (no {IDEAS})")
        group = Group(name=path.name, path=path)
        self._load_group(group)
        return group
    # -- lookup -------------------------------------------------------------
    def group_of(self, name: str = "") -> Group | None:
        """No name = the group this run is *about*: the biggest active one.

        Alphabetical order is how a filesystem sorts, not how a person reads — an
        archived example must never become the default subject of a sentence.
        """
        if not name:
            return self.focus_group()
        return self.groups.get(name)

    def focus_group(self) -> Group | None:
        """The biggest **active** group — an archived one is never the subject by accident."""
        alive = [g for g in self.groups.values() if g.status == "active"]
        if not alive:
            return None
        return max(alive, key=lambda g: len(g.nodes))


    def require_group(self, name: str = "") -> Group:
        group = self.group_of(name)
        if group is not None:
            return group
        group = self._group_by_path(name)
        if group is not None:
            return group
        known = ", ".join(sorted(self.groups)) or "(none)"
        raise KeyError(f"no group {name!r} — known under {self.root}: {known}")

    def _group_by_path(self, raw: str) -> Group | None:
        """A group named by address rather than by name under the root.

        `--root` decides what is discovered, not what may be used: a path carrying
        `IDEAS.md` is a group wherever it lives. (Group names hold no slash, so the
        presence of one is what makes this a path.)

        A relative path resolves against the root, not the process cwd — the root is
        the frame of reference for every name; an absolute path names its own frame.
        """
        text = str(raw or "").strip()
        if not text or "/" not in text:
            return None
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        candidate = candidate.resolve()
        if not (candidate / IDEAS).is_file():
            return None
        return self._group_at(candidate)
    def find(self, key: str, group: str = "") -> tuple[Group | None, Node | None]:
        order = [self.require_group(group)] if group else list(self.groups.values())
        for g in order:
            node = g.find(key)
            if node is not None:
                return g, node
        return None, None

    def all_nodes(self) -> Iterable[tuple[Group, Node]]:
        for g in self.groups.values():
            for n in g._sorted_nodes():
                yield g, n

    # -- writes -------------------------------------------------------------
    def apply(self, group: Group, op: dict) -> None:
        """One mutation: bump rev, append the op, then write the graph down."""
        with self._lock(group):
            group.rev += 1
            write_atomic(group.graph_path, group.to_json())
            with group.ops_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"rev": group.rev, "t": _now(), **op}, ensure_ascii=False) + "\n")
        self._remember(group)

    def _remember(self, group: Group) -> None:
        """Keep the last few states in memory — enough for Ctrl+Z, nothing more."""
        stack = self._undo.setdefault(group.name, [])
        stack.append(json.loads(group.to_json()))
        del stack[:-_UNDO_DEPTH]

    def undo(self, group: Group) -> dict:
        """Step back one write.

        In-memory on purpose: what must survive a restart is a version, not an undo
        step. It appends an `undo` op like any other write, so the account stays honest.
        """
        stack = self._undo.get(group.name) or []
        if len(stack) < 2:
            raise KeyError("nothing to undo")
        stack.pop()
        data = stack[-1]
        with self._lock(group):
            self._load_data(group, data)
            group.rev += 1
            write_atomic(group.graph_path, group.to_json())
            with group.ops_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"rev": group.rev, "t": _now(), "op": "undo"},
                                    ensure_ascii=False) + "\n")
        # The last element of the stack *is* the current state, so the restored one is
        # already there — popping is the whole bookkeeping, and the stack runs out.
        return {"rev": group.rev, "nodes": len(group.nodes)}

    @contextmanager
    def _lock(self, group: Group):
        """Single writer: the node and the CLI both take this before touching the file."""
        path = group.path / LOCK
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+")
        try:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    def _next_id(self, group: Group, prefix: str) -> str:
        used = {int(k[1:]) for k in group.nodes if k[1:].isdigit()}
        used |= {int(e.id[1:]) for e in group.edges if e.id[1:].isdigit()}
        n = 1
        while n in used:
            n += 1
        return f"{prefix}{n}"

    def add_node(self, group: Group, label: str, *, body: str = "", concept: str = "",
                 weight: float | None = None, parent: str = "") -> Node:
        node_id = self._next_id(group, "n")
        parent_node = group.find(parent) if parent else None
        if weight is None:
            weight = max(FLOOR, (parent_node.weight * DECAY) if parent_node else 1.0)
        weight = round(float(weight), 3)
        chosen = concept.strip()
        if chosen:
            group.meta["default_concept"] = chosen      # the last word used becomes the default
        node = Node(id=node_id, label=label.strip(), body=body.strip(),
                    concept=chosen or group.default_concept(),
                    weight=weight, created=_now())
        group.nodes[node_id] = node
        edge = None
        if parent_node is not None:
            edge = Edge(id=self._next_id(group, "e"), src=parent_node.id, dst=node_id, kind=KIND)
            group.edges.append(edge)
        with self._lock(group):
            group.rev += 1
            write_atomic(group.graph_path, group.to_json())
            with group.ops_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"rev": group.rev, "t": _now(), "op": "add",
                                     "node": node_id, "label": node.label}, ensure_ascii=False) + "\n")
                if edge is not None:
                    fh.write(json.dumps({"rev": group.rev, "t": _now(), "op": "edge",
                                         "edge": edge.id, "from": edge.src, "to": edge.dst,
                                         "kind": edge.kind}, ensure_ascii=False) + "\n")
        self._remember(group)
        return node

    def set_node(self, group: Group, key: str, **fields) -> Node:
        node = group.find(key)
        if node is None:
            raise KeyError(key)
        changed: dict = {}
        for name in ("label", "body", "concept"):
            if fields.get(name) is not None and getattr(node, name) != str(fields[name]):
                setattr(node, name, str(fields[name]))
                changed[name] = str(fields[name])
        if fields.get("weight") is not None and float(node.weight) != float(fields["weight"]):
            node.weight = float(fields["weight"])
            changed["weight"] = node.weight
        if fields.get("flag") is not None and node.flag != bool(fields["flag"]):
            node.flag = bool(fields["flag"])
            changed["flag"] = node.flag
        if changed:
            self.apply(group, {"op": "set", "node": node.id, **changed})
        return node

    def link(self, group: Group, src: str, dst: str, kind: str = KIND, label: str = "") -> Edge:
        a, b = group.find(src), group.find(dst)
        if a is None:
            raise KeyError(src)
        if b is None:
            raise KeyError(dst)
        edge = Edge(id=self._next_id(group, "e"), src=a.id, dst=b.id,
                    kind=kind if kind in DEFAULT_KINDS else KIND, label=label)
        group.edges.append(edge)
        self.apply(group, {"op": "edge", "edge": edge.id, "from": a.id, "to": b.id, "kind": edge.kind})
        return edge

    def unlink(self, group: Group, src: str, dst: str) -> int:
        a, b = group.find(src), group.find(dst)
        if a is None or b is None:
            raise KeyError(src if a is None else dst)
        kept = [e for e in group.edges if not (e.src == a.id and e.dst == b.id)]
        dropped = len(group.edges) - len(kept)
        if dropped:
            group.edges = kept
            self.apply(group, {"op": "unedge", "from": a.id, "to": b.id})
        return dropped

    def remove_node(self, group: Group, key: str) -> dict:
        node = group.find(key)
        if node is None:
            raise KeyError(key)
        touched = [e for e in group.edges if e.src == node.id or e.dst == node.id]
        group.nodes.pop(node.id, None)
        group.edges = [e for e in group.edges if e.src != node.id and e.dst != node.id]
        self.apply(group, {"op": "rm", "node": node.id, "label": node.label, "edges": len(touched)})
        return {"node": node.id, "label": node.label, "edges": len(touched)}

    # -- versions -----------------------------------------------------------
    def save(self, group: Group, name: str) -> dict:
        """Anchor this rev as a **version** — a real copy, not a marker.

        `versions/r{rev}-{name}.json` is the whole state at that moment, so going back
        is a file copy, not a replay. The account keeps one line for the record;
        nothing reads the log to find out which version is current.
        """
        name = (name or "").strip() or "checkpoint"
        with self._lock(group):
            group.versions_path.mkdir(parents=True, exist_ok=True)
            path = group.versions_path / f"r{group.rev}-{_slug(name)}.json"
            write_atomic(path, group.to_json())
            # The published copy: same content, stable name, enters git. It only moves
            # when a version is made, so the repo's delta is one version at a time.
            write_atomic(group.published_path, group.to_json())
            with group.ops_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"rev": group.rev, "t": _now(), "op": VERSION_OP,
                                     "name": name}, ensure_ascii=False) + "\n")
        self._mark_saved(group, name)
        return {"rev": group.rev, "name": name, "file": str(path)}

    def _mark_saved(self, group: Group, name: str) -> None:
        """The sign carries the current version, so listing groups never reads the log."""
        group.saved_rev, group.saved_name = group.rev, name
        if not group.ideas_path.exists():
            return
        meta, body = split_frontmatter(group.ideas_path.read_text(encoding="utf-8"))
        meta["saved"] = f"r{group.rev} {name}"
        meta["updated"] = _now()
        write_atomic(group.ideas_path, join_frontmatter(meta, body))

    def versions(self, group: Group) -> list[dict]:
        """The checkpoints on disk, oldest first — a directory scan, no index."""
        return [{"rev": rev, "name": name, "file": str(path)}
                for rev, (name, path) in sorted(group.version_files().items())]

    def read_version(self, group: Group, ref: str) -> tuple[int, str, dict]:
        """Cold read a version: by name, by `r18`, or by `18`."""
        found = group.version_files()
        if not found:
            raise KeyError(f"no versions in {group.name}")
        ref = str(ref or "").strip()
        if ref[:1] == "r" and ref[1:].isdigit():
            rev = int(ref[1:])
        elif ref.isdigit():
            rev = int(ref)
        else:
            hits = [r for r, (n, _p) in found.items() if n == ref]
            if not hits:
                raise KeyError(f"no version {ref!r} in {group.name} — versions() lists them")
            rev = max(hits)
        if rev not in found:
            raise KeyError(f"no version at r{rev} in {group.name}")
        name, path = found[rev]
        return rev, name, json.loads(path.read_text(encoding="utf-8"))

    def restore(self, group: Group, ref: str) -> dict:
        """Go back to a version — appended to the account, never a rewrite of history."""
        rev, name, data = self.read_version(group, ref)
        with self._lock(group):
            self._load_data(group, data)
            group.rev += 1
            write_atomic(group.graph_path, group.to_json())
            with group.ops_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"rev": group.rev, "t": _now(), "op": "restore",
                                     "from": rev, "name": name}, ensure_ascii=False) + "\n")
        self._remember(group)
        return {"rev": group.rev, "from": rev, "name": name}

    def segment_report(self, group: Group) -> list[str]:
        """Which stretches of the log are readable — the never-silent half of recovery.

        A `version` line opens a segment; a broken line inside it invalidates the whole
        segment, because half a stretch of changes is worse than none of it. Nothing is
        rewritten here: this reports, and `graph.json` stays the truth either way.
        """
        if not group.ops_path.exists():
            return ["no ops recorded"]
        rows: list[str] = []
        start, broken = 0, 0
        for line in group.ops_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                op = json.loads(line)
                rev = int(op.get("rev", 0))
            except (ValueError, TypeError):
                broken += 1
                continue
            if op.get("op") == VERSION_OP:
                if broken:
                    rows.append(f"after r{start}: {broken} broken line(s) — that whole "
                                f"segment is ignored")
                start, broken = rev, 0
                rows.append(f"r{rev} ★ {op.get('name')} — readable")
        if broken:
            rows.append(f"after r{start}: {broken} broken line(s) — that whole segment is ignored")
        return rows

    def last_save(self, group: Group) -> tuple[int, str]:
        """The current version, read from the sign — the log is never opened for this."""
        return group.saved_rev, group.saved_name

    def ops_since_save(self, group: Group) -> int:
        return max(0, group.rev - group.saved_rev)

    def _graph_at(self, group: Group, ref: str) -> tuple[str, dict]:
        if not ref:
            return f"r{group.rev} (now)", json.loads(group.to_json())
        rev, name, data = self.read_version(group, ref)
        return f"r{rev} {name}", data

    def diff_versions(self, group: Group, from_ref: str = "", to_ref: str = "") -> list[str]:
        """What changed between two versions — the question the log used to answer.

        Empty `from_ref` means the current version, empty `to_ref` means now; neither
        reads the ops log.
        """
        if not from_ref and group.saved_rev:
            from_ref = f"r{group.saved_rev}"
        a_name, a = self._graph_at(group, from_ref)
        b_name, b = self._graph_at(group, to_ref)
        na, nb = a.get("nodes") or {}, b.get("nodes") or {}
        rows = [f"{a_name} → {b_name}"]
        for nid in sorted(set(nb) - set(na)):
            rows.append(f"+ {nb[nid].get('label') or nid}")
        for nid in sorted(set(na) - set(nb)):
            rows.append(f"− {na[nid].get('label') or nid}")
        for nid in sorted(set(na) & set(nb)):
            if na[nid] != nb[nid]:
                rows.append(f"~ {nb[nid].get('label') or nid}")
        ea = {str(e.get("id")): e for e in a.get("edges") or []}
        eb = {str(e.get("id")): e for e in b.get("edges") or []}
        for eid in sorted(set(eb) - set(ea)):
            rows.append(f"→ {eb[eid].get('from')}→{eb[eid].get('to')}")
        for eid in sorted(set(ea) - set(eb)):
            rows.append(f"✂ {ea[eid].get('from')}→{ea[eid].get('to')}")
        if len(rows) == 1:
            rows.append("no change")
        return rows

    def set_status(self, group: Group, status: str) -> None:
        """The sign's own field: alive, waiting, or put away. The graph never moves."""
        if status not in GROUP_STATUS:
            raise ValueError(f"status must be one of {GROUP_STATUS}")
        meta, body = split_frontmatter(group.ideas_path.read_text(encoding="utf-8"))
        meta["status"] = status
        meta["updated"] = _now()
        meta.setdefault("graph", GRAPH)
        group.status = status
        write_atomic(group.ideas_path, join_frontmatter(meta, body) if body.strip() else
                     _ideas_body(group.name))
        self.apply(group, {"op": "status", "group": group.name, "status": status})

    def rename_concept(self, group: Group, old: str, new: str) -> dict:
        """Rename a concept word — the colour slot stays, the label changes.

        Renaming moves the key in the scale and every node wearing it, in one write,
        so nothing is left pointing at a word that no longer exists.
        """
        old, new = str(old or "").strip(), str(new or "").strip()
        if not old or not new:
            raise ValueError("rename needs two words")
        concepts = dict(group.concepts())
        if old not in concepts:
            raise KeyError(f"no concept {old!r} in {group.name}")
        if new in concepts and new != old:
            raise ValueError(f"concept {new!r} already exists in {group.name}")
        if old != new:
            # A rename must not reshuffle the legend: the slot keeps its place in the
            # scale, only its word changes.
            group.meta["concepts"] = {
                (new if key == old else key): value for key, value in concepts.items()}
        moved = 0
        for node in group.nodes.values():
            if node.concept == old:
                node.concept = new
                moved += 1
        self.apply(group, {"op": "concept-rename", "from": old, "to": new, "nodes": moved})
        return {"from": old, "to": new, "nodes": moved}

    def set_concept(self, group: Group, name: str, color: str) -> None:
        concepts = dict(group.meta.get("concepts") or {})
        concepts[name] = color
        group.meta["concepts"] = concepts
        self.apply(group, {"op": "concept", "name": name, "color": color})

    # -- the op protocol ----------------------------------------------------
    def apply_ops(self, ops: list[dict]) -> list[str]:
        """The single write door: a list of op objects, applied in order.

        Every other surface (CLI verb, CTML command, page WebSocket) speaks this JSON,
        so "one implementation" is not a slogan — it is this function. Each op names its
        own group; an op that omits it lands in the first group.

            {"op": "add",  "group": "…", "label": "storage", "parent": "architecture"}
            {"op": "link", "src": "storage", "dst": "md", "kind": "detail"}
            {"op": "link", "src": "存储策略", "dst": "md", "kind": "detail"}
            {"op": "rm",   "id": "n7"}
        """
        out: list[str] = []
        for raw in ops:
            if not isinstance(raw, dict):
                raise ValueError(f"an op must be an object, got {type(raw).__name__}")
            name = str(raw.get("op", "")).strip()
            group = self.require_group(str(raw.get("group", "") or ""))
            if name == "add":
                node = self.add_node(
                    group, str(raw.get("label", "") or "new"),
                    body=str(raw.get("body", "") or ""),
                    concept=str(raw.get("concept", "") or ""),
                    weight=raw.get("weight"),
                    parent=str(raw.get("parent", "") or ""),
                )
                out.append(f"+ {group.name}/{node.id} {node.label}")
            elif name == "set":
                node = self.set_node(
                    group, str(raw.get("id", "")),
                    label=raw.get("label"), body=raw.get("body"),
                    concept=raw.get("concept"), weight=raw.get("weight"), flag=raw.get("flag"),
                )
                out.append(f"~ {node.id} {node.label}")
            elif name == "link":
                edge = self.link(group, str(raw.get("src", "")), str(raw.get("dst", "")),
                                 str(raw.get("kind", KIND)))
                out.append(f"→ {edge.src}→{edge.dst}")
            elif name == "unlink":
                dropped = self.unlink(group, str(raw.get("src", "")), str(raw.get("dst", "")))
                out.append(f"✂ {dropped}")
            elif name == "rm":
                gone = self.remove_node(group, str(raw.get("id", "")))
                out.append(f"− {gone['label']}")
            elif name == "concept":
                self.set_concept(group, str(raw.get("name", "")), str(raw.get("color", "")))
                out.append(f"◧ {raw.get('name')}")
            elif name == "status":
                self.set_status(group, str(raw.get("value", "")))
                out.append(f"⚑ {group.name} {raw.get('value')}")
            else:
                raise ValueError(f"unknown op {name!r} — one of add/set/link/unlink/rm/concept/status")
        return out

    # -- reading helpers ----------------------------------------------------
    def tree_lines(self, group: Group) -> list[str]:
        """The whole graph, compact enough to take in one look."""
        lines: list[str] = []
        seen: set[str] = set()

        def walk(nid: str, prefix: str) -> None:
            if nid in seen or nid not in group.nodes:
                return
            seen.add(nid)
            node = group.nodes[nid]
            mark = " !" if node.flag else ""
            lines.append(f"{prefix}{node.label or nid}{mark} [{node.concept} {node.weight:g}]")
            kids = group.children(nid)
            for i, edge in enumerate(kids):
                last = i == len(kids) - 1
                branch = "└─ " if last else "├─ "
                arrow = {"depend": "⇒ ", "related": "~ "}.get(edge.kind, "")
                walk(edge.dst, prefix.replace("├─ ", "│  ").replace("└─ ", "   ") + branch + arrow)

        for nid in roots_of(group):
            walk(nid, "")
        rest = [n.id for n in group._sorted_nodes() if n.id not in seen]
        if rest:
            lines.append("— islands with no upstream —")
            for nid in rest:
                walk(nid, "")
        return lines

    def diff_lines(self, group: Group, since: int = 0) -> list[str]:
        if not group.ops_path.exists():
            return [f"no ops recorded (rev {group.rev})"]
        rows: list[str] = []
        for line in group.ops_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                op = json.loads(line)
            except ValueError:
                continue
            if int(op.get("rev", 0)) <= since:
                continue
            what = op.get("op")
            node = group.nodes.get(str(op.get("node", "")))
            who = node.name if node is not None else (op.get("label") or op.get("node"))
            if what == "add":
                rows.append(f"r{op['rev']}  + {who}")
            elif what == "set":
                fields = ", ".join(f"{k}={v}" for k, v in op.items()
                                   if k not in ("rev", "t", "op", "node"))
                rows.append(f"r{op['rev']}  ~ {who} ({fields})")
            elif what == "edge":
                rows.append(f"r{op['rev']}  → {op.get('from')} → {op.get('to')}")
            elif what == "unedge":
                rows.append(f"r{op['rev']}  ✂ {op.get('from')} → {op.get('to')}")
            elif what == "rm":
                rows.append(f"r{op['rev']}  − {who}")
            elif what == "concept":
                rows.append(f"r{op['rev']}  ◧ {op.get('name')} = {op.get('color')}")
            elif what == "save":
                rows.append(f"r{op['rev']}  ★ save 「{op.get('name')}」")
            elif what == "init":
                rows.append(f"r{op['rev']}  ⚑ {op.get('note')}")
            else:
                rows.append(f"r{op['rev']}  ? {what}")
        return rows or [f"nothing since r{since} (now r{group.rev})"]


# ---------------------------------------------------------------------- root helpers


def split_frontmatter(raw: str) -> tuple[dict, str]:
    """Legacy markdown only: `--- yaml ---` + body."""
    if not raw.startswith("---"):
        return {}, raw
    lines = raw.splitlines(keepends=True)
    if lines[0].strip() != "---":
        return {}, raw
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            import yaml
            data = yaml.safe_load("".join(lines[1:i])) or {}
            return (data if isinstance(data, dict) else {}), "".join(lines[i + 1:])
    return {}, raw


def _ideas_body(name: str) -> str:
    return (
        f"# {name}\n\nOne line on what this group is — for whoever opens it next.\n\n"
        f"- the graph lives in `{GRAPH}`: nodes, edges, the concept scale, rev.\n"
        f"- how to read it: `ideas tree --group {name}` for the whole thing;"
        f" `ideas show <label>` for one node.\n\n"
        "> This file is a **sign**: what this is, and whether it is alive. It **does not\n"
        "> list nodes** — they live in the graph, and a second copy would drift.\n"
    )


def write_ideas_sign(group: Group) -> None:
    """Every group carries a sign: what it is, and whether it is alive."""
    if group.ideas_path.exists():
        return
    write_atomic(group.ideas_path, IDEAS_TEMPLATE.format(
        status=group.status, graph=GRAPH, updated=_now(), name=group.name))


def join_frontmatter(meta: dict, body: str) -> str:
    import yaml
    header = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).strip()
    return f"---\n{header}\n---\n\n{body.lstrip()}\n"


def default_root(node_dir: Path) -> Path:
    return node_dir / "ideas"


def resolve_root(raw: str = "", *, node_dir: Path) -> Path:
    """`--root` > `MOSS_IDEAS_ROOT` > the node's own `ideas/`; relative = caller's cwd."""
    candidate = raw or os.getenv("MOSS_IDEAS_ROOT", "")
    if not candidate:
        return default_root(node_dir).resolve()
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()
