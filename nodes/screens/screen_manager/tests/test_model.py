import pytest

from ghoshell_screen_manager.model import ScreenModel, compute_layout, grid_template


def test_grid_layout_sizes():
    assert (compute_layout(1, "grid", "lr").cols, compute_layout(1, "grid", "lr").rows) == (1, 1)
    assert (compute_layout(2, "grid", "lr").cols, compute_layout(2, "grid", "lr").rows) == (2, 1)
    assert (compute_layout(2, "grid", "tb").cols, compute_layout(2, "grid", "tb").rows) == (1, 2)
    assert (compute_layout(3, "grid", "lr").cols, compute_layout(3, "grid", "lr").rows) == (3, 1)
    assert (compute_layout(4, "grid", "lr").cols, compute_layout(4, "grid", "lr").rows) == (2, 2)
    assert (compute_layout(6, "grid", "lr").cols, compute_layout(6, "grid", "lr").rows) == (3, 2)


def test_grid_cells_are_1x1_and_ordered():
    layout = compute_layout(6, "grid", "lr")
    assert [(c.r, c.c) for c in layout.cells] == [
        (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3),
    ]


def test_stack_master_spans_the_strip():
    lr = compute_layout(3, "stack", "lr")
    assert (lr.cols, lr.rows) == (2, 2)
    assert (lr.cells[0].r, lr.cells[0].c, lr.cells[0].rs, lr.cells[0].cs) == (1, 1, 2, 1)
    assert (lr.cells[1].r, lr.cells[1].c) == (1, 2)
    assert (lr.cells[2].r, lr.cells[2].c) == (2, 2)

    tb = compute_layout(3, "stack", "tb")
    assert (tb.cols, tb.rows) == (2, 2)
    assert (tb.cells[0].r, tb.cells[0].c, tb.cells[0].rs, tb.cells[0].cs) == (1, 1, 1, 2)


def test_grid_template_master_axis_is_2_to_1():
    assert grid_template(compute_layout(3, "stack", "lr")) == ("2fr 1fr", "repeat(2, 1fr)")
    assert grid_template(compute_layout(3, "stack", "tb")) == ("repeat(2, 1fr)", "2fr 1fr")
    assert grid_template(compute_layout(4, "grid", "lr")) == ("repeat(2, 1fr)", "repeat(2, 1fr)")


def test_stack_rejects_a_single_item():
    with pytest.raises(ValueError):
        compute_layout(1, "stack", "lr")


def test_bad_family_and_dir_rejected():
    with pytest.raises(ValueError):
        compute_layout(3, "circle", "lr")
    with pytest.raises(ValueError):
        compute_layout(3, "grid", "up")


def test_open_and_group_invariants():
    m = ScreenModel()
    m.open("a", "http://a", group="g")
    m.open("b", "http://b", group="g")
    assert m.groups() == ["g"]
    assert m.group_items("g") == ["a", "b"]
    assert m.group_of("a") == "g"


def test_close_empties_the_group():
    m = ScreenModel()
    m.open("a", "http://a", group="g")
    m.activate("g")
    m.close("a")
    assert m.groups() == []
    assert m.active() == ""


def test_close_clears_fullscreen():
    m = ScreenModel()
    m.open("a", "http://a", group="g")
    m.activate("g")
    m.set_fullscreen("a")
    m.close("a")
    assert m.fullscreen() is None


def test_arrange_requires_exact_permutation():
    m = ScreenModel()
    m.open("a", "http://a", group="g")
    m.open("b", "http://b", group="g")
    m.activate("g")
    m.arrange(["b", "a"], family="grid", dir="lr")
    assert m.active_items() == ["b", "a"]
    with pytest.raises(ValueError):
        m.arrange(["a"], family="grid", dir="lr")  # missing b


def test_open_duplicate_rejected():
    m = ScreenModel()
    m.open("a", "http://a", group="g")
    with pytest.raises(ValueError):
        m.open("a", "http://a", group="g")
