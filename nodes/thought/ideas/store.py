"""The ideas store — plain files, one idea per `*.idea.md`.

**The file is the truth.** Nothing else holds an idea's state: frontmatter carries
`state` / `priority` / `tags` / `links`, the body carries the thinking itself. Editing
one of these files by hand is a first-class operation, not a workaround.

**A group is a directory.** The root has first-level subdirectories; each one is a
group, and its `*.idea.md` files are the ideas. `README.md` inside a group is that
group's shared context — what it is for, what to read first.

**The name is the identity and it is semantic.** `duplex-gap-between-clicks.idea.md`
says what it is before anything is opened; `idea-10384843.idea.md` says only that it
exists. Renaming therefore rewrites references — `rename()` sweeps the whole root.

No log, no fold, no replay: the timeline was a mechanism the problem never asked for.
If a trace is ever wanted it is a disposable file beside the data, never a second
truth.
"""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

SUFFIX = ".idea.md"
README = "README.md"

STATES = ("spark", "active", "waiting", "done", "retired")
KINDS = ("related", "depend")

_UNSAFE = re.compile(r"[\s/\\:*?\"<>|]+")


def slugify(text: str, limit: int = 24) -> str:
    """A filename that still reads like the idea it holds."""
    head = (text or "").strip().splitlines()[0] if text and text.strip() else "idea"
    head = _UNSAFE.sub("-", head).strip("-.")
    head = re.sub(r"-{2,}", "-", head)
    return (head[:limit] or "idea").rstrip("-.")


# --------------------------------------------------------------------- frontmatter


def split_frontmatter(raw: str) -> tuple[dict, str]:
    """`--- {...} ---` + body; a file without frontmatter is all body."""
    if not raw.startswith("---"):
        return {}, raw
    lines = raw.splitlines(keepends=True)
    if lines[0].strip() != "---":
        return {}, raw
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            header = "".join(lines[1:i])
            body = "".join(lines[i + 1:])
            data = yaml.safe_load(header) or {}
            if not isinstance(data, dict):
                raise ValueError("frontmatter is not a mapping")
            return data, body.lstrip("\n")
    raise ValueError("unterminated frontmatter")


def join_frontmatter(meta: dict, body: str) -> str:
    header = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).strip()
    return f"---\n{header}\n---\n\n{body.rstrip()}\n"


@dataclass
class Link:
    name: str
    kind: str = "related"

    def to_line(self) -> str:
        return self.name if self.kind == "related" else f"{self.name}:{self.kind}"


@dataclass
class Idea:
    name: str                      # filename stem = identity = what a link points at
    group: str
    path: Path
    state: str = "spark"
    priority: float = 1.0
    tags: list[str] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    body: str = ""
    mtime: float = 0.0

    # -- read ---------------------------------------------------------------
    @property
    def title(self) -> str:
        for line in self.body.splitlines():
            line = line.strip().lstrip("#").strip()
            if line:
                return line
        return self.name

    def live_links(self, known: set[str]) -> list[Link]:
        return [l for l in self.links if l.name in known]

    def dead_links(self, known: set[str]) -> list[Link]:
        return [l for l in self.links if l.name not in known]

    # -- write --------------------------------------------------------------
    def meta(self) -> dict:
        out: dict = {"state": self.state, "priority": self.priority}
        if self.tags:
            out["tags"] = list(self.tags)
        if self.links:
            out["links"] = [l.to_line() for l in self.links]
        return out

    def to_text(self) -> str:
        return join_frontmatter(self.meta(), self.body)

    def save(self) -> None:
        write_atomic(self.path, self.to_text())
        self.mtime = self.path.stat().st_mtime

    def to_dict(self, known: set[str]) -> dict:
        return {
            "name": self.name,
            "group": self.group,
            "state": self.state,
            "priority": self.priority,
            "tags": self.tags,
            "mtime": self.mtime,
            "title": self.title,
            "body": self.body,
            "links": [
                {"name": l.name, "kind": l.kind, "dead": l.name not in known}
                for l in self.links
            ],
        }


@dataclass
class Group:
    name: str
    path: Path
    ideas: dict[str, Idea] = field(default_factory=dict)
    readme: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "path": str(self.path), "readme": self.readme}


def write_atomic(path: Path, text: str) -> None:
    """Half-written files are read by the page and by `list`; never let one exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- store


class Store:
    """One root: its first-level directories are groups, their files are ideas."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.groups: dict[str, Group] = {}
        self.broken: list[str] = []      # files that could not be parsed — shown, not skipped

    # -- discovery ----------------------------------------------------------
    def reload(self) -> None:
        self.groups = {}
        self.broken = []
        if not self.root.exists():
            seed_if_missing(self.root)
        if not self.root.exists():
            return
        for child in sorted(self.root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            group = Group(name=child.name, path=child)
            readme = child / README
            if readme.exists():
                group.readme = readme.read_text(encoding="utf-8", errors="replace")
            for file in sorted(child.glob(f"*{SUFFIX}")):
                name = file.name[: -len(SUFFIX)]
                try:
                    idea = self._read_idea(file, name, group.name)
                except Exception as exc:  # noqa: BLE001 — a bad file must not vanish
                    self.broken.append(f"{group.name}/{file.name}: {exc}")
                    continue
                group.ideas[name] = idea
            self.groups[group.name] = group

    @staticmethod
    def _read_idea(path: Path, name: str, group: str) -> Idea:
        meta, body = split_frontmatter(path.read_text(encoding="utf-8", errors="replace"))
        state = str(meta.get("state", "spark")).strip() or "spark"
        if state not in STATES:
            state = "spark"
        raw_links = meta.get("links") or []
        links: list[Link] = []
        for item in raw_links:
            if isinstance(item, dict):
                dst = str(item.get("to") or item.get("name") or "").strip()
                kind = str(item.get("kind", "related")).strip() or "related"
            else:
                dst, _, kind = str(item).partition(":")
                dst, kind = dst.strip(), (kind.strip() or "related")
            if not dst:
                continue
            links.append(Link(name=dst, kind=kind if kind in KINDS else "related"))
        return Idea(
            name=name,
            group=group,
            path=path,
            state=state,
            priority=float(meta.get("priority", 1.0) or 1.0),
            tags=[str(t) for t in (meta.get("tags") or [])],
            links=links,
            body=body,
            mtime=path.stat().st_mtime,
        )

    # -- lookup -------------------------------------------------------------
    def known(self) -> set[str]:
        return {i.name for g in self.groups.values() for i in g.ideas.values()}

    def find(self, name: str) -> Idea | None:
        for group in self.groups.values():
            if name in group.ideas:
                return group.ideas[name]
        return None

    def all_ideas(self) -> Iterable[Idea]:
        for group in self.groups.values():
            yield from group.ideas.values()

    def group_of(self, name: str = "") -> Group | None:
        if not name:
            return next(iter(self.groups.values()), None)
        return self.groups.get(name)

    # -- read ---------------------------------------------------------------
    def snapshot(self) -> dict:
        known = self.known()
        return {
            "root": str(self.root),
            "groups": [
                {
                    **g.to_dict(),
                    "ideas": [
                        idea.to_dict(known)
                        for idea in sorted(g.ideas.values(), key=lambda x: -x.mtime)
                    ],
                }
                for g in self.groups.values()
            ],
            "broken": list(self.broken),
        }

    def list_lines(self, group: str = "", state: str = "", tag: str = "", limit: int = 40) -> list[str]:
        rows: list[str] = []
        for g in self.groups.values():
            if group and g.name != group:
                continue
            for idea in sorted(g.ideas.values(), key=lambda x: -x.mtime):
                if state and idea.state != state:
                    continue
                if tag and tag not in idea.tags:
                    continue
                extra = f" · [{', '.join(idea.tags)}]" if idea.tags else ""
                dead = idea.dead_links(self.known())
                dangling = f" · {len(dead)} dead link(s)" if dead else ""
                rows.append(f"{g.name}/{idea.name} · {idea.state} · p{idea.priority:g}{extra}{dangling} · {idea.title}")
                if len(rows) >= limit:
                    return rows
        return rows

    def read_doc(self, name: str) -> str:
        idea = self.find(name)
        if idea is None:
            raise KeyError(name)
        return idea.path.read_text(encoding="utf-8", errors="replace")

    # -- write --------------------------------------------------------------
    def _next_name(self, group: Group, base: str) -> str:
        name, n = base, 2
        while name in group.ideas or (group.path / f"{name}{SUFFIX}").exists():
            name = f"{base}-{n}"
            n += 1
        return name

    def spark(self, text: str, *, group: str = "", name: str = "", priority: float = 1.0) -> Idea:
        target = self.group_of(group)
        if target is None:
            raise KeyError(f"no group {group!r}")
        final = self._next_name(target, slugify(name or text))
        body = text.strip()
        idea = Idea(name=final, group=target.name, path=target.path / f"{final}{SUFFIX}",
                    priority=priority, body=body)
        idea.save()
        target.ideas[final] = idea
        return idea

    def note(self, name: str, text: str) -> Idea:
        idea = self._require(name)
        idea.body = (idea.body.rstrip() + "\n\n" + text.strip() + "\n") if idea.body.strip() else text.strip() + "\n"
        idea.save()
        return idea

    def set_state(self, name: str, state: str) -> Idea:
        if state not in STATES:
            raise ValueError(f"state must be one of {STATES}")
        idea = self._require(name)
        idea.state = state
        idea.save()
        return idea

    def set_priority(self, name: str, priority: float) -> Idea:
        idea = self._require(name)
        idea.priority = float(priority)
        idea.save()
        return idea

    def tag(self, name: str, word: str) -> Idea:
        idea = self._require(name)
        if word and word not in idea.tags:
            idea.tags.append(word)
            idea.save()
        return idea

    def untag(self, name: str, word: str) -> Idea:
        idea = self._require(name)
        if word in idea.tags:
            idea.tags.remove(word)
            idea.save()
        return idea

    def link(self, src: str, dst: str, kind: str = "related") -> Idea:
        idea = self._require(src)
        if self.find(dst) is None:
            raise KeyError(dst)
        if all(not (l.name == dst) for l in idea.links):
            idea.links.append(Link(name=dst, kind=kind if kind in KINDS else "related"))
            idea.save()
        return idea

    def unlink(self, src: str, dst: str) -> Idea:
        idea = self._require(src)
        kept = [l for l in idea.links if l.name != dst]
        if len(kept) != len(idea.links):
            idea.links = kept
            idea.save()
        return idea

    def rename(self, old: str, new: str) -> str:
        """Move the file *and* rewrite every reference — the name is the identity."""
        idea = self._require(old)
        target = slugify(new)
        group = self.groups[idea.group]
        if target in group.ideas:
            raise ValueError(f"{target!r} already exists in {group.name}")
        new_path = group.path / f"{target}{SUFFIX}"
        os.replace(idea.path, new_path)
        group.ideas.pop(old)
        idea.name, idea.path = target, new_path
        group.ideas[target] = idea
        touched = []
        for other in self.all_ideas():
            changed = False
            for l in other.links:
                if l.name == old:
                    l.name = target
                    changed = True
            if changed:
                other.save()
                touched.append(other.name)
        return f"{old} → {target}" + (f"; rewrote links in {', '.join(touched)}" if touched else "")

    # -- internals ----------------------------------------------------------
    def _require(self, name: str) -> Idea:
        idea = self.find(name)
        if idea is None:
            raise KeyError(name)
        return idea


# ---------------------------------------------------------------------- root discovery


EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"
"""What ships: copied into a root that does not exist yet, and never into one that does
— deleting the examples is a supported move, and re-seeding would undo it."""


def seed_if_missing(root: Path) -> bool:
    if root.exists():
        return False
    if not EXAMPLES_DIR.is_dir():
        root.mkdir(parents=True, exist_ok=True)
        return False
    shutil.copytree(EXAMPLES_DIR, root)
    return True


def default_root(node_dir: Path) -> Path:
    """Out of the box the data sits beside the node, gitignored, and is disposable."""
    return node_dir / "ideas"


def resolve_root(raw: str = "", *, node_dir: Path) -> Path:
    """`--root` > `MOSS_IDEAS_ROOT` > the node's own `ideas/`.

    The env var is the agreed handoff: whoever starts the process decides which
    directory this run governs, and every child inherits the answer.
    """
    candidate = raw or os.getenv("MOSS_IDEAS_ROOT", "")
    if not candidate:
        return default_root(node_dir).resolve()
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        # relative resolves against the caller's cwd — the same rule the shell uses
        path = Path.cwd() / path
    return path.resolve()
