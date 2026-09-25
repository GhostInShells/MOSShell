import pytest

from ghoshell_decision_tree.meta import (
    LOG_FILENAME,
    NODES_DIRNAME,
    PRUNED_LEVEL,
    ROOT_META_FILENAME,
    CreateNodeSeed,
    CreateTreeSeed,
    LinkNodeSeed,
    UpdateNodeSeed,
)
from ghoshell_decision_tree.store import DecisionTreeStore


@pytest.fixture
def store(tmp_path):
    return DecisionTreeStore(root=tmp_path)


def _tree(store, root="t"):
    store.create_tree(CreateTreeSeed(root=root, name=root, title="T"))
    return root


def _node(store, name, **kw):
    store.create_node("t", CreateNodeSeed(tree="t", name=name, title=name.upper(), **kw))


def test_create_tree_writes_files(store):
    _tree(store)
    root = store.tree_root("t")
    assert (root / ROOT_META_FILENAME).is_file()
    assert (root / LOG_FILENAME).is_file()
    assert (root / NODES_DIRNAME).is_dir()


def test_create_node_top_level(store):
    _tree(store)
    _node(store, "a")
    nodes, children = store.fold("t")
    assert "a" in nodes
    assert nodes["a"].status == "open"
    assert store.snapshot("t")["roots"] == ["a"]


def test_create_node_under_parent(store):
    _tree(store)
    _node(store, "a")
    _node(store, "b", from_node="a")
    _, children = store.fold("t")
    assert children["a"] == [{"name": "b", "edge": "decompose"}]
    assert store.snapshot("t")["roots"] == ["a"]


def test_update_status_and_level(store):
    _tree(store)
    _node(store, "a")
    store.update_node("t", UpdateNodeSeed(tree="t", name="a", status="decided", status_note="done"))
    nodes, _ = store.fold("t")
    assert nodes["a"].status == "decided"
    assert nodes["a"].status_note == "done"
    assert store.snapshot("t")["nodes"][0]["level"] == "success"


def test_pruned_level_is_muted(store):
    _tree(store)
    _node(store, "a")
    store.update_node("t", UpdateNodeSeed(tree="t", name="a", pruned=True))
    assert store.snapshot("t")["nodes"][0]["level"] == PRUNED_LEVEL


def test_prune_keeps_status(store):
    _tree(store)
    _node(store, "a")
    store.update_node("t", UpdateNodeSeed(tree="t", name="a", status="discussing"))
    store.update_node("t", UpdateNodeSeed(tree="t", name="a", pruned=True))
    nodes, _ = store.fold("t")
    assert nodes["a"].pruned is True
    assert nodes["a"].status == "discussing"


def test_invalid_status_rejected(store):
    _tree(store)
    _node(store, "a")
    with pytest.raises(ValueError):
        store.update_node("t", UpdateNodeSeed(tree="t", name="a", status="nope"))


def test_invalid_edge_rejected(store):
    _tree(store)
    with pytest.raises(ValueError):
        store.create_node("t", CreateNodeSeed(tree="t", name="a", title="A", edge="nope"))


def test_duplicate_name_rejected(store):
    _tree(store)
    _node(store, "a")
    with pytest.raises(ValueError):
        _node(store, "a")


def test_unknown_parent_rejected(store):
    _tree(store)
    with pytest.raises(ValueError):
        store.create_node("t", CreateNodeSeed(tree="t", name="a", title="A", from_node="ghost"))


def test_link_requires_both_nodes(store):
    _tree(store)
    _node(store, "a")
    with pytest.raises(ValueError):
        store.link_node("t", LinkNodeSeed(tree="t", node="a", linked_node="ghost"))


def test_create_tree_twice_rejected(store):
    _tree(store)
    with pytest.raises(ValueError):
        _tree(store)


def test_path_boundary_refuses_escape(store):
    with pytest.raises(ValueError):
        store.resolve("/etc")


def test_tree_root_rejects_absolute(store):
    with pytest.raises(ValueError):
        store.tree_root("/etc")


def test_tree_root_rejects_dot_and_parent(store):
    for rel in (".", "..", ""):
        with pytest.raises(ValueError):
            store.tree_root(rel)


def test_tree_root_accepts_relative_under_home(store):
    assert store.tree_root("a/b").is_relative_to(store._root)


def test_link_adds_cross_edge(store):
    _tree(store)
    _node(store, "a")
    _node(store, "b")
    store.link_node("t", LinkNodeSeed(tree="t", node="a", linked_node="b"))
    _, children = store.fold("t")
    assert {"name": "b", "edge": "decompose"} in children["a"]


def test_ls_lists_content_and_skips_dotfiles(store):
    _tree(store)
    _node(store, "a")
    d = store.node_dir("t", "a")
    d.mkdir(parents=True, exist_ok=True)
    (d / "note.md").write_text("x", encoding="utf-8")
    (d / ".hidden").write_text("x", encoding="utf-8")
    names = [e["name"] for e in store.ls("t", "a")]
    assert "note.md" in names
    assert ".hidden" not in names


def test_history_filters_by_name(store):
    _tree(store)
    _node(store, "a")
    store.update_node("t", UpdateNodeSeed(tree="t", name="a", status="decided"))
    assert [e["ev"] for e in store.history("t", "a")] == ["create", "update"]


def test_fold_is_append_only_replay(store):
    _tree(store)
    _node(store, "a", description="first")
    store.update_node("t", UpdateNodeSeed(tree="t", name="a", description="second"))
    nodes, _ = store.fold("t")
    assert nodes["a"].description == "second"
