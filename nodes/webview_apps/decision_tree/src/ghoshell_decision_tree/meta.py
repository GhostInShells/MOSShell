"""Axis-1 data for the decision-tree node — pure pydantic models, no IO.

The tree has three storage tiers (see FEATURE.md K4):

- ``DECISION_TREE_ROOT.md`` — the graph's meta: statuses table, edge vocabulary,
  prose plan. Hand-written, self-describing.
- ``tree.jsonl`` — the node structure + state, an append-only event log. The
  channel is its only writer; the current tree is a *fold* over the log, never a
  stored structure.
- ``graph-nodes/{name}/`` — a node's content, created lazily only when a node
  has material.

The seeds below are the ``text__`` JSON bodies of the channel commands; their
schema is rendered into the channel instruction (``instruction_schema``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ROOT_META_FILENAME = "DECISION_TREE_ROOT.md"
NODES_DIRNAME = "graph-nodes"
LOG_FILENAME = "tree.jsonl"

# The only vocabulary the renderer knows. A tree declares its statuses; each maps
# to one of these. `pruned` is always `muted` — orthogonal to status (K8).
Level = Literal["info", "success", "warn", "error", "muted"]
LEVELS: list[str] = ["info", "success", "warn", "error", "muted"]
PRUNED_LEVEL: str = "muted"

DEFAULT_STATUSES: dict[str, str] = {
    "open": "info",
    "discussing": "warn",
    "researching": "warn",
    "decided": "success",
}

DEFAULT_EDGES: dict[str, str] = {
    "decompose": "拆成一个子问题或子项",
}

DEFAULT_EDGE = "decompose"
DEFAULT_STATUS = "open"


class TreeMeta(BaseModel):
    """The frontmatter of ``DECISION_TREE_ROOT.md``."""

    name: str
    title: str
    description: str = ""
    kind: Literal["decision_tree"] = "decision_tree"
    created: str = ""
    updated: str = ""
    statuses: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_STATUSES))
    edges: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_EDGES))


class NodeState(BaseModel):
    """A node's current state — produced by folding the event log, not stored."""

    name: str
    title: str
    description: str = ""
    status: str = DEFAULT_STATUS
    status_note: str = ""
    pruned: bool = False
    created: str = ""
    updated: str = ""


class Event(BaseModel):
    """One line of ``tree.jsonl``. Each event is self-contained so a future
    compaction can fold them in order (K5)."""

    ev: Literal["create", "link", "update"]
    name: str = ""
    from_node: str = ""
    to: str = ""
    edge: str = DEFAULT_EDGE
    title: str = ""
    description: str = ""
    status: str = ""
    status_note: str = ""
    pruned: bool | None = None
    at: str = ""


class CreateTreeSeed(BaseModel):
    root: str = Field(description="graph root dir, relative to the project home")
    name: str = Field(description="graph id; also the root dir name (lowercase)")
    title: str = Field(description="human-readable title")
    description: str = Field(default="", description="one line, what this tree is for")
    statuses: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_STATUSES),
        description="status -> level map; levels are info/success/warn/error/muted",
    )
    edges: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_EDGES),
        description="edge name -> one-line meaning",
    )


class CreateNodeSeed(BaseModel):
    tree: str = Field(description="graph root dir")
    name: str = Field(description="node id; also the node dir name (lowercase, unique under this tree)")
    title: str = Field(description="human-readable title")
    description: str = Field(default="", description="one line, what this node is")
    from_node: str = Field(default="", description="parent node name; empty = attach to the graph root")
    edge: str = Field(default=DEFAULT_EDGE, description="edge name; must be declared in the tree meta")


class LinkNodeSeed(BaseModel):
    tree: str = Field(description="graph root dir")
    node: str = Field(description="source node name")
    linked_node: str = Field(description="target node name")
    edge: str = Field(default=DEFAULT_EDGE, description="edge name; must be declared in the tree meta")


class UpdateNodeSeed(BaseModel):
    tree: str = Field(description="graph root dir")
    name: str = Field(description="node name to update")
    status: str = Field(default="", description="new status; must be declared in the tree meta")
    status_note: str = Field(default="", description="one line, why it is in this state now")
    pruned: bool | None = Field(default=None, description="set true to prune (keep the node, mark it out of play)")
    title: str = Field(default="", description="new title (empty = unchanged)")
    description: str = Field(default="", description="new description (empty = unchanged)")


def _schema_lines(model: type[BaseModel]) -> list[str]:
    schema = model.model_json_schema()
    required = set(schema.get("required", []))
    lines: list[str] = []
    for name, spec in schema.get("properties", {}).items():
        if "enum" in spec:
            typ = "/".join(str(v) for v in spec["enum"])
        elif "anyOf" in spec:
            typ = " | ".join(str(v.get("type", "?")) for v in spec["anyOf"])
        else:
            typ = spec.get("type", "?")
        req = "" if name in required else " (optional)"
        desc = spec.get("description", "") or ""
        lines.append(f"    {name}: {typ}{req} — {desc}")
    return lines


def instruction_schema() -> str:
    """Render the seed schemas into the channel instruction — one line per field,
    derived from the models so they cannot drift from the code."""
    blocks: list[str] = []
    for title, model in [
        ("create_tree", CreateTreeSeed),
        ("create_node", CreateNodeSeed),
        ("link_node", LinkNodeSeed),
        ("update_node", UpdateNodeSeed),
    ]:
        blocks.append(title + " text__ (JSON):\n" + "\n".join(_schema_lines(model)))
    return "\n\n".join(blocks)
