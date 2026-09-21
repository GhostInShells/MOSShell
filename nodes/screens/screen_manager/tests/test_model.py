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


# -- pool / desktop / grouping -----------------------------------------


def test_open_lands_on_the_desktop_by_default():
    m = ScreenModel()
    m.open("a", "http://a")
    assert m.desktop_items() == ["a"]
    assert m.group_of("a") == ""
    assert m.groups() == []


def test_open_into_a_group_creates_it():
    m = ScreenModel()
    m.open("a", "http://a", group="code")
    assert m.group_items("code") == ["a"]
    assert m.desktop_items() == []


def test_open_duplicate_rejected():
    m = ScreenModel()
    m.open("a", "http://a")
    with pytest.raises(ValueError):
        m.open("a", "http://a")


def test_arrange_pulls_items_off_the_desktop():
    m = ScreenModel()
    m.open("a", "http://a")
    m.open("b", "http://b")
    m.open("c", "http://c")
    m.arrange("code", ["a", "b"], family="grid", dir="lr")
    assert m.group_items("code") == ["a", "b"]
    assert m.desktop_items() == ["c"]
    assert m.arena() == "code"


def test_arrange_moves_an_item_out_of_another_group():
    m = ScreenModel()
    m.open("a", "http://a", group="code")
    m.open("b", "http://b", group="code")
    m.arrange("media", ["b"], family="grid", dir="lr")
    assert m.group_items("code") == ["a"]
    assert m.group_items("media") == ["b"]
    assert m.group_of("b") == "media"


def test_arranging_away_the_last_item_deletes_the_group():
    m = ScreenModel()
    m.open("a", "http://a", group="code")
    m.open("b", "http://b", group="media")
    m.arrange("media", ["a", "b"], family="grid", dir="lr")
    assert "code" not in m.groups()


def test_arrange_rejects_unknown_and_duplicate_ids():
    m = ScreenModel()
    m.open("a", "http://a")
    with pytest.raises(ValueError):
        m.arrange("code", ["a", "ghost"], family="grid", dir="lr")
    with pytest.raises(ValueError):
        m.arrange("code", ["a", "a"], family="grid", dir="lr")


def test_dismiss_sends_an_item_back_to_the_desktop():
    m = ScreenModel()
    m.open("a", "http://a", group="code")
    m.open("b", "http://b", group="code")
    m.dismiss("a")
    assert m.group_items("code") == ["b"]
    assert m.desktop_items() == ["a"]


def test_dismiss_last_item_deletes_the_group():
    m = ScreenModel()
    m.open("a", "http://a", group="code")
    m.activate("code")
    m.dismiss("a")
    assert m.groups() == []
    assert m.arena() == ""  # fell back to the desktop


def test_destroy_removes_an_item_entirely():
    m = ScreenModel()
    m.open("a", "http://a", group="code")
    m.destroy("a")
    assert m.items() == []
    assert m.groups() == []


def test_float_all_empties_the_active_group():
    m = ScreenModel()
    m.open("a", "http://a", group="code")
    m.open("b", "http://b", group="code")
    m.activate("code")
    freed = m.float_all()
    assert sorted(freed) == ["a", "b"]
    assert m.desktop_items() == ["a", "b"]
    assert m.groups() == []
    assert m.arena() == ""


# -- layout is per-group -------------------------------------------------


def test_layout_is_per_group():
    m = ScreenModel()
    m.open("a", "http://a", group="code")
    m.open("b", "http://b", group="code")
    m.open("c", "http://c", group="media")
    m.open("d", "http://d", group="media")

    m.activate("code")
    m.arrange("code", ["a", "b"], family="stack", dir="tb")
    assert (m.family(), m.dir()) == ("stack", "tb")

    m.activate("media")
    assert (m.family(), m.dir()) == ("grid", "lr")
    m.arrange("media", ["c", "d"], family="grid", dir="tb")
    assert (m.family(), m.dir()) == ("grid", "tb")

    m.activate("code")
    assert (m.family(), m.dir()) == ("stack", "tb")


def test_fullscreen_is_per_group():
    m = ScreenModel()
    m.open("a", "http://a", group="code")
    m.open("b", "http://b", group="media")
    m.activate("code")
    m.set_fullscreen("a")
    assert m.fullscreen() == "a"
    m.activate("media")
    assert m.fullscreen() is None
    m.activate("code")
    assert m.fullscreen() == "a"


def test_group_layout_is_forgotten_with_its_group():
    m = ScreenModel()
    m.open("a", "http://a", group="code")
    m.arrange("code", ["a"], family="grid", dir="tb")
    m.destroy("a")
    assert m.groups() == []
    m.open("b", "http://b", group="code")
    assert (m.family(), m.dir()) == ("grid", "lr")


def test_fullscreen_needs_a_group_on_stage():
    m = ScreenModel()
    m.open("a", "http://a")
    with pytest.raises(ValueError):
        m.set_fullscreen("a")


def test_fullscreen_cleared_when_its_item_leaves_the_group():
    m = ScreenModel()
    m.open("a", "http://a", group="code")
    m.open("b", "http://b", group="code")
    m.activate("code")
    m.set_fullscreen("a")
    m.dismiss("a")
    assert m.fullscreen() is None


# -- adoption / tombstones ----------------------------------------------


def test_adopt_lands_on_the_desktop_with_its_service():
    m = ScreenModel()
    item = m.adopt("cell/a/webview", "http://x", label="Terminal", item_id="term")
    assert item is not None
    assert item.service == "cell/a/webview"
    assert m.desktop_items() == ["term"]


def test_adopt_is_idempotent_for_a_live_service():
    m = ScreenModel()
    m.adopt("cell/a/webview", "http://x", label="A", item_id="a")
    assert m.adopt("cell/a/webview", "http://x", label="A", item_id="a") is None


def test_destroyed_service_is_tombstoned_until_it_releases():
    m = ScreenModel()
    m.adopt("cell/a/webview", "http://x", label="A", item_id="a")
    m.destroy("a")
    # The service is still live, but the tombstone holds — no re-adoption.
    assert m.adopt("cell/a/webview", "http://x", label="A", item_id="a") is None
    # The service goes away, then returns: adoption resumes.
    m.release("cell/a/webview")
    assert m.adopt("cell/a/webview", "http://x", label="A", item_id="a") is not None


def test_dismiss_is_soft_and_keeps_the_service():
    m = ScreenModel()
    m.adopt("cell/a/webview", "http://x", label="A", item_id="a")
    m.arrange("code", ["a"], family="grid", dir="lr")
    m.dismiss("a")
    # Dismiss only moves the item off its group; the service stays attached,
    # so the item still exists on the desktop (and adoption sees it).
    assert m.get("a") is not None
    assert m.desktop_items() == ["a"]
    assert m.service_of("a") == "cell/a/webview"


def test_release_removes_the_adopted_item():
    m = ScreenModel()
    m.adopt("cell/a/webview", "http://x", label="A", item_id="a")
    m.release("cell/a/webview")
    assert m.get("a") is None
