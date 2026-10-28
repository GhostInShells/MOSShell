"""The graph is the truth here: one `graph.json` per group, edges as entities.

These tests pin what the mechanism now rests on:


* a group is a directory carrying `IDEAS.md`; discovery reads the marker, one level
  under the root — the position is the default radius, not the identity;* edges are entities with their own ids, so a label change cannot break one;
* weight decays with generation — depth is size, not taste;
* the op log is an account, not the truth: `diff` reads it, nothing folds it back;
* the markdown generation is imported once, and moved aside rather than rewritten.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from store import (  # noqa: E402
    GRAPH,
    LIVE_GRAPH,
    GraphStore,
    junctions,
    leaves,
    shortest_path,
    spine,
)

LEGACY = """---
state: spark
priority: {priority}
label: {label}
links: [{links}]
---

{body}
"""



SIGN = """---
status: {status}
graph: {graph}
---

# {name}

One line on what this group is.
"""


def group_dir(root: Path, name: str = "design", *, sign: bool = True) -> Path:
    """A group is a directory carrying the sign — so the helper writes the sign."""
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    if sign:
        (path / "IDEAS.md").write_text(
            SIGN.format(status="active", graph=GRAPH, name=name), encoding="utf-8")
    return path

def loaded(tmp_path: Path, name: str = "design"):
    store = GraphStore(tmp_path)
    store.reload()
    return store, store.require_group(name)


def write_legacy(group: Path, name: str, label: str, links: str = "", body: str = "", priority: float = 1.0):
    (group / f"{name}.idea.md").write_text(
        LEGACY.format(priority=priority, label=label, links=links, body=body or label), encoding="utf-8")


# -------------------------------------------------------------------- discovery


def test_a_group_keeps_a_live_graph_and_a_published_one(tmp_path: Path):
    """`graph.json` is the version product; `.graph.json` is the working state.

    Two names for two different promises: one changes on every gesture, the other
    only when a version is made — which is why only the second can enter git.
    """
    group_dir(tmp_path)
    store, group = loaded(tmp_path)

    assert group.graph_path == tmp_path / "design" / LIVE_GRAPH
    assert group.graph_path.exists(), "an empty group still gets a working graph on disk"
    assert group.published_path == tmp_path / "design" / GRAPH
    assert group.published_path.exists(), "the init version publishes graph.json"
    assert json.loads(group.published_path.read_text(encoding="utf-8"))["rev"] == group.rev
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


def test_every_group_carries_its_ignore_rule(tmp_path: Path):
    """A group ships its content (`IDEAS.md`, `graph.json`) and not its generations.

    Writing the sign and the ignore rule are the same act: a group directory made
    anywhere is safe to commit without a hand-written rule, so `git add -A` after a
    session of gestures cannot drag the live state, the account or the snapshots in.
    """
    store = GraphStore(tmp_path)
    store.reload()
    group = store.create_group("stage3")

    text = (group.path / ".gitignore").read_text(encoding="utf-8")
    rules = [ln.strip() for ln in text.splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    assert rules == [".graph.json", "ops.jsonl", ".lock", ".trash/", ".versions/"]
    assert "graph.json" not in rules and "IDEAS.md" not in rules, "the content is not ignored"

    # a group made before the default existed gets one on the next read
    old = group_dir(tmp_path)
    assert not (old / ".gitignore").exists()
    GraphStore(tmp_path).reload()
    assert (old / ".gitignore").exists()

    # and a hand-written rule is a decision — a default never overwrites one
    hand = group_dir(tmp_path, "hand")
    (hand / ".gitignore").write_text("mine\n", encoding="utf-8")
    GraphStore(tmp_path).reload()
    assert (hand / ".gitignore").read_text(encoding="utf-8") == "mine\n"


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
    """Creating a group writes the sign — the page, the CLI and the channel all do this."""
    root = tmp_path / "ideas"
    store = GraphStore(root)
    store.reload()
    assert store.groups == {}

    group = store.create_group("stage3")

    assert (root / "stage3").is_dir()
    assert (root / "stage3" / "IDEAS.md").is_file(), "the sign is the group"
    assert store.require_group("stage3").nodes == {}
    assert store.focus_group().name == "stage3"

    for bad in ("", "  "):
        try:
            store.create_group(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} must be refused")
    # A path is a path: a plain name lands under the root, an absolute one may live
    # anywhere, because the root is the default discovery radius rather than a wall.
    for nested in ("a/b", "a/b/c"):
        try:
            store.create_group(nested)
        except ValueError as exc:
            assert "one level deep" in str(exc), nested
            continue
        raise AssertionError(f"a nested group {nested!r} would be undiscoverable")

    outside = tmp_path / "elsewhere" / "journal"
    made = store.create_group(str(outside))
    assert (outside / "IDEAS.md").is_file()
    assert store.require_group(str(outside)).nodes == {}
    assert made.path == outside.resolve()


def test_a_directory_without_the_sign_is_not_a_group(tmp_path: Path):
    """The marker decides, not the position — and a stray graph is reported, not hidden."""
    group_dir(tmp_path, "design")
    (tmp_path / "just-a-folder").mkdir()
    stray = tmp_path / "stray"
    stray.mkdir()
    (stray / GRAPH).write_text("{}", encoding="utf-8")

    store = GraphStore(tmp_path)
    store.reload()

    assert sorted(store.groups) == ["design"]
    assert not (tmp_path / "just-a-folder" / "IDEAS.md").exists()
    assert any("stray" in line and "IDEAS.md" in line for line in store.broken)

# -------------------------------------------------------------------- versions


def test_a_group_starts_from_an_init_version(tmp_path: Path):
    """Every group has a base version, so history is never a stretch with no anchor."""
    group_dir(tmp_path)
    store, group = loaded(tmp_path)

    assert [v["name"] for v in store.versions(group)] == ["init"]
    assert group.saved_rev == group.rev


def test_save_writes_a_real_snapshot(tmp_path: Path):
    """A save is a copy of the graph, not a marker: going back is a file copy."""
    group_dir(tmp_path)
    store, group = loaded(tmp_path)
    root = store.add_node(group, "motive")
    store.add_node(group, "execution", parent=root.id)

    info = store.save(group, "two nodes")

    path = Path(info["file"])
    assert path.exists() and path.parent.name == ".versions"
    rev, name, data = store.read_version(group, "two nodes")
    assert (rev, name) == (info["rev"], "two nodes")
    assert len(data["nodes"]) == 2
    assert [v["rev"] for v in store.versions(group)] == sorted(
        [v["rev"] for v in store.versions(group)])


def test_restore_goes_back_and_appends_an_op(tmp_path: Path):
    """Going back does not rewrite the account — it says 'at r4 I went back to r2'."""
    group_dir(tmp_path)
    store, group = loaded(tmp_path)
    store.add_node(group, "motive")
    store.save(group, "one")
    version_rev = group.saved_rev
    store.add_node(group, "later")
    assert len(group.nodes) == 2

    info = store.restore(group, "one")

    assert len(group.nodes) == 1
    assert info["from"] == version_rev
    assert info["rev"] == group.rev == version_rev + 1
    assert any("readable" in row for row in store.segment_report(group))


def test_the_sign_carries_the_version_so_the_log_is_never_hot_read(tmp_path: Path):
    group_dir(tmp_path)
    store, group = loaded(tmp_path)
    store.add_node(group, "motive")
    store.save(group, "one")

    group.ops_path.unlink()          # the account is gone; the sign still knows
    again = GraphStore(tmp_path)
    again.reload()
    g = again.require_group("design")

    assert again.last_save(g) == (g.saved_rev, "one")
    assert again.ops_since_save(g) == 0


def test_a_broken_line_voids_only_its_segment(tmp_path: Path):
    group_dir(tmp_path)
    store, group = loaded(tmp_path)
    store.add_node(group, "motive")
    store.save(group, "one")
    store.add_node(group, "later")

    with group.ops_path.open("a", encoding="utf-8") as fh:
        fh.write("{not json\n")

    report = store.segment_report(group)
    assert any("one" in row and "readable" in row for row in report)
    assert any("ignored" in row for row in report)


def test_renaming_a_concept_keeps_its_place_and_moves_its_nodes(tmp_path: Path):
    """A rename changes the word, not the slot: order stays, and every node follows."""
    group_dir(tmp_path)
    store, group = loaded(tmp_path)
    order = list(group.concepts())
    colour = group.concepts()[order[0]]
    node = store.add_node(group, "motive", concept=order[0])

    store.rename_concept(group, order[0], "purpose")

    assert list(group.concepts())[0] == "purpose", "the slot keeps its position"
    assert group.concepts()["purpose"] == colour, "the colour is the slot's, not the word's"
    assert group.find(node.id).concept == "purpose"


def test_a_nodes_concept_can_be_changed(tmp_path: Path):
    group_dir(tmp_path)
    store, group = loaded(tmp_path)
    order = list(group.concepts())
    node = store.add_node(group, "motive")

    store.set_node(group, node.id, concept=order[2])

    assert group.find(node.id).concept == order[2]
    again = GraphStore(tmp_path)
    again.reload()
    assert again.require_group("design").find(node.id).concept == order[2]


def test_the_last_word_used_becomes_the_default(tmp_path: Path):
    """The default concept is expressed by using it, not by a setting nobody changes."""
    group_dir(tmp_path)
    store, group = loaded(tmp_path)
    order = list(group.concepts())

    store.add_node(group, "motive", concept=order[2])
    assert group.default_concept() == order[2]

    plain = store.add_node(group, "execution")
    assert plain.concept == order[2]

    again = GraphStore(tmp_path)
    again.reload()
    assert again.require_group("design").find(plain.id).concept == order[2]


def test_the_default_concept_survives_a_reload(tmp_path: Path):
    """It is group state, so it has to be in graph.json — not only in memory."""
    group_dir(tmp_path)
    store, group = loaded(tmp_path)
    order = list(group.concepts())
    store.add_node(group, "motive", concept=order[3])

    again = GraphStore(tmp_path)
    again.reload()
    g = again.require_group("design")

    assert g.default_concept() == order[3]
    assert again.add_node(g, "later").concept == order[3]


def test_undo_steps_back_one_write(tmp_path: Path):
    """Undo is an appended act, not a rewrite — and it only has to cover a few gestures."""
    group_dir(tmp_path)
    store, group = loaded(tmp_path)
    store.add_node(group, "motive")
    store.add_node(group, "execution")
    assert len(group.nodes) == 2

    info = store.undo(group)

    assert len(group.nodes) == 1
    assert info["nodes"] == 1
    assert '"op": "undo"' in group.ops_path.read_text(encoding="utf-8")
    # exactly one step back was available: once the first write is reached, it runs out
    try:
        store.undo(group)
    except KeyError:
        pass
    else:
        raise AssertionError("undo must run out after stepping back to the first write")
