"""The graph is the truth here: one `graph.json` per group, edges as entities.

These tests pin what the mechanism now rests on:

* a group is a directory holding one graph; discovery is a first-level scan;
* edges are entities with their own ids, so a label change cannot break one;
* weight decays with generation — depth is size, not taste;
* the op log is an account, not the truth: `diff` reads it, nothing folds it back;
* the markdown generation is imported once, and moved aside rather than rewritten.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from store import GRAPH, GraphStore, junctions, leaves, shortest_path, spine  # noqa: E402

LEGACY = """---
state: spark
priority: {priority}
label: {label}
links: [{links}]
---

{body}
"""


def group_dir(root: Path, name: str = "design") -> Path:
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def loaded(tmp_path: Path, name: str = "design"):
    store = GraphStore(tmp_path)
    store.reload()
    return store, store.require_group(name)


def write_legacy(group: Path, name: str, label: str, links: str = "", body: str = "", priority: float = 1.0):
    (group / f"{name}.idea.md").write_text(
        LEGACY.format(priority=priority, label=label, links=links, body=body or label), encoding="utf-8")


# -------------------------------------------------------------------- discovery


def test_a_group_is_a_directory_with_one_graph(tmp_path: Path):
    group_dir(tmp_path)
    store, group = loaded(tmp_path)
    assert group.graph_path == tmp_path / "design" / GRAPH
    assert group.graph_path.exists(), "an empty group still gets a graph on disk"
    assert group.rev == 1


def test_weight_decays_with_generation(tmp_path: Path):
    group_dir(tmp_path)
    store, group = loaded(tmp_path)
    root = store.add_node(group, "动机")
    child = store.add_node(group, "执行", parent=root.id)
    grandchild = store.add_node(group, "前端样式", parent=child.id)

    assert root.weight == 1.0
    assert child.weight == 0.8
    assert grandchild.weight == 0.64
    # the edge came with the child, pointing the right way
    assert [(e.src, e.dst) for e in group.edges] == [(root.id, child.id), (child.id, grandchild.id)]


# ------------------------------------------------------------------------ edges


def test_edges_are_entities_and_survive_a_rename(tmp_path: Path):
    group_dir(tmp_path)
    store, group = loaded(tmp_path)
    a = store.add_node(group, "存储策略")
    b = store.add_node(group, "md", parent=a.id)
    edge_id = group.edges[0].id

    store.set_node(group, a.id, label="存储")     # label is a display name, not an identity

    assert group.edges[0].id == edge_id
    assert group.edges[0].src == a.id
    assert group.find("存储") is a and group.find(a.id) is a
    assert group.children(a.id)[0].dst == b.id


def test_removing_a_node_takes_its_edges_with_it(tmp_path: Path):
    group_dir(tmp_path)
    store, group = loaded(tmp_path)
    a = store.add_node(group, "方案")
    b = store.add_node(group, "执行", parent=a.id)
    store.add_node(group, "验收", parent=a.id)

    gone = store.remove_node(group, b.id)

    assert gone["edges"] == 1
    assert b.id not in group.nodes
    assert all(e.dst != b.id and e.src != b.id for e in group.edges)


# --------------------------------------------------------------------- topology


def test_topology_answers_where_the_spine_is(tmp_path: Path):
    group_dir(tmp_path)
    store, group = loaded(tmp_path)
    root = store.add_node(group, "动机")
    mid = store.add_node(group, "执行", parent=root.id)
    store.add_node(group, "前端", parent=mid.id)
    store.add_node(group, "画布", parent=mid.id)
    store.add_node(group, "验收", parent=root.id)

    # 主路是最长的那条链，不是子树最大的那个——根永远子树最大，那样答的不是"主路"
    chain = spine(group)
    assert chain[:2] == [root.id, mid.id] and len(chain) == 3
    assert chain[-1] in {group.find("前端").id, group.find("画布").id}
    assert junctions(group)[0][0] == mid.id, "the branch point is where two children hang"
    assert set(leaves(group)) == {group.find("前端").id, group.find("画布").id, group.find("验收").id}
    assert shortest_path(group, root.id, group.find("画布").id) == [root.id, mid.id, group.find("画布").id]


# ------------------------------------------------------------------------ ops


def test_ops_are_an_account_and_diff_reads_them(tmp_path: Path):
    group_dir(tmp_path)
    store, group = loaded(tmp_path)
    start = group.rev
    store.add_node(group, "架构方案")
    store.add_node(group, "执行", parent="架构方案")
    store.set_node(group, "执行", flag=True)

    lines = store.diff_lines(group, since=start)
    joined = "\n".join(lines)
    assert "+ 架构方案" in joined
    assert "+ 执行" in joined and "flag=True" in joined

    # the log is disposable: dropping it leaves the graph intact
    group.ops_path.unlink()
    again = GraphStore(tmp_path)
    again.reload()
    g2 = again.require_group("design")
    assert len(g2.nodes) == 2
    assert "no ops recorded" in again.diff_lines(g2)[0]


def test_rev_moves_on_every_write(tmp_path: Path):
    group_dir(tmp_path)
    store, group = loaded(tmp_path)
    before = group.rev
    store.add_node(group, "a")
    store.set_node(group, "a", body="x")
    assert group.rev == before + 2
    on_disk = json.loads(group.graph_path.read_text(encoding="utf-8"))
    assert on_disk["rev"] == group.rev


# ---------------------------------------------------------------------- legacy


def test_markdown_ideas_are_imported_once(tmp_path: Path):
    group = group_dir(tmp_path)
    write_legacy(group, "motivation", "动机", links="execution")
    write_legacy(group, "execution", "执行", links="frontend, storage:depend")
    write_legacy(group, "frontend", "前端")
    write_legacy(group, "storage", "存储")

    store, g = loaded(tmp_path)

    assert len(g.nodes) == 4
    labels = {n.label for n in g.nodes.values()}
    assert labels == {"动机", "执行", "前端", "存储"}
    kinds = sorted(e.kind for e in g.edges)
    assert kinds == ["depend", "detail", "detail"]
    # the old files are moved aside, not deleted, and the import does not run twice
    assert not list(group.glob("*.idea.md"))
    assert len(list((group / ".trash").glob("*.idea.md"))) == 4

    store2 = GraphStore(tmp_path)
    store2.reload()
    assert len(store2.require_group("design").nodes) == 4


def test_every_group_carries_a_sign(tmp_path: Path):
    """IDEAS.md says what the group is and whether it is alive — it never lists nodes."""
    group_dir(tmp_path)
    store, group = loaded(tmp_path)

    assert group.ideas_path.exists()
    head = group.ideas_path.read_text(encoding="utf-8")
    assert "status: active" in head and "graph.json" in head
    assert "n1" not in head, "the sign must not copy the graph"

    store.set_status(group, "archived")
    assert "status: archived" in group.ideas_path.read_text(encoding="utf-8")
    again = GraphStore(tmp_path)
    again.reload()
    assert again.require_group("design").status == "archived"


# ------------------------------------------------------------------- discovery rules


def test_a_missing_root_starts_empty(tmp_path: Path):
    """A root that does not exist starts as nothing — not as a box of examples.

    Examples used to be copied in on first run; a booting ghost then read a demo
    graph as if it were the work, which is worse than reading nothing.
    """
    root = tmp_path / "ideas"
    store = GraphStore(root)
    store.reload()

    assert store.groups == {}
    assert store.focus_group() is None
    assert not root.exists(), "nothing is copied in behind the root's back"


def test_archived_is_never_the_focus(tmp_path: Path):
    """A bigger archived graph must not become the subject, and with nothing alive
    there is no subject at all — never a fallback into the drawer."""
    write_legacy(group_dir(tmp_path, "old"), "a", "旧")
    group_dir(tmp_path, "alive")

    store = GraphStore(tmp_path)
    store.reload()
    store.set_status(store.require_group("old"), "archived")
    store.reload()
    assert store.focus_group().name == "alive"

    store.set_status(store.require_group("alive"), "archived")
    store.reload()
    assert store.focus_group() is None


def test_a_group_can_be_made_from_the_page(tmp_path: Path):
    """Creating a group is a mkdir — the page's button and the shell do exactly this."""
    root = tmp_path / "ideas"
    store = GraphStore(root)
    store.reload()
    assert store.groups == {}

    group = store.create_group("stage3")

    assert (root / "stage3").is_dir()
    assert store.require_group("stage3").nodes == {}
    assert store.focus_group().name == "stage3"

    for bad in ("", ".hidden", "a/b", "  "):
        try:
            store.create_group(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} must be refused")
