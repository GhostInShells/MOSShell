import pytest

from ghoshell_decision_tree.meta import (
    CreateNodeSeed,
    CreateTreeSeed,
    TreeMeta,
    UpdateNodeSeed,
    instruction_schema,
)


def test_tree_meta_defaults():
    m = TreeMeta(name="t", title="T")
    assert m.statuses["open"] == "info"
    assert "decompose" in m.edges
    assert m.kind == "decision_tree"


def test_create_node_seed_defaults():
    s = CreateNodeSeed.model_validate({"tree": "t", "name": "a", "title": "A"})
    assert s.edge == "decompose"
    assert s.from_node == ""


def test_create_node_seed_requires_name():
    with pytest.raises(Exception):
        CreateNodeSeed.model_validate({"tree": "t", "title": "A"})


def test_update_seed_pruned_is_optional_tristate():
    s = UpdateNodeSeed.model_validate({"tree": "t", "name": "a"})
    assert s.pruned is None
    assert s.status == ""


def test_instruction_schema_is_self_describing():
    text = instruction_schema()
    for name in ("create_tree", "create_node", "link_node", "update_node"):
        assert name in text
    # the schema is derived from the models, so key fields must appear
    assert "tree" in text
    assert "name" in text
    assert "status" in text


def test_create_tree_seed_defaults():
    s = CreateTreeSeed.model_validate({"root": "t", "name": "t", "title": "T"})
    assert s.statuses["decided"] == "success"
