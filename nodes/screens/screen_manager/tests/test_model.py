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
    item = m.open("http://a")
    assert m.desktop_items() == [item.id]
    assert m.group_of(item.id) == ""
    assert m.groups() == []


def test_open_into_a_group_creates_it():
    m = ScreenModel()
    item = m.open("http://a", group="code")
    assert m.group_items("code") == [item.id]
    assert m.desktop_items() == []


def test_open_assigns_a_monotonic_handle():
    m = ScreenModel()
    a = m.open("http://a")
    b = m.open("http://b")
    assert a.id != b.id
    # A destroyed handle never recycles onto a later item.
    m.destroy(a.id)
    c = m.open("http://c")
    assert c.id != a.id
    assert c.id != b.id


def test_arrange_pulls_items_off_the_desktop():
    m = ScreenModel()
    a = m.open("http://a").id
    b = m.open("http://b").id
    c = m.open("http://c").id
    m.arrange("code", [a, b], family="grid", dir="lr")
    assert m.group_items("code") == [a, b]
    assert m.desktop_items() == [c]
    assert m.arena() == "code"


def test_arrange_moves_an_item_out_of_another_group():
    m = ScreenModel()
    a = m.open("http://a", group="code").id
    b = m.open("http://b", group="code").id
    m.arrange("media", [b], family="grid", dir="lr")
    assert m.group_items("code") == [a]
    assert m.group_items("media") == [b]
    assert m.group_of(b) == "media"


def test_arranging_away_the_last_item_deletes_the_group():
    m = ScreenModel()
    a = m.open("http://a", group="code").id
    b = m.open("http://b", group="media").id
    m.arrange("media", [a, b], family="grid", dir="lr")
    assert "code" not in m.groups()


def test_arrange_rejects_unknown_and_duplicate_ids():
    m = ScreenModel()
    a = m.open("http://a").id
    with pytest.raises(ValueError):
        m.arrange("code", [a, "ghost"], family="grid", dir="lr")
    with pytest.raises(ValueError):
        m.arrange("code", [a, a], family="grid", dir="lr")


def test_dismiss_sends_an_item_back_to_the_desktop():
    m = ScreenModel()
    a = m.open("http://a", group="code").id
    b = m.open("http://b", group="code").id
    m.dismiss(a)
    assert m.group_items("code") == [b]
    assert m.desktop_items() == [a]


def test_dismiss_last_item_deletes_the_group():
    m = ScreenModel()
    a = m.open("http://a", group="code").id
    m.activate("code")
    m.dismiss(a)
    assert m.groups() == []
    assert m.arena() == ""  # fell back to the desktop


def test_destroy_removes_an_item_entirely():
    m = ScreenModel()
    a = m.open("http://a", group="code").id
    m.destroy(a)
    assert m.items() == []
    assert m.groups() == []


def test_float_all_empties_the_active_group():
    m = ScreenModel()
    a = m.open("http://a", group="code").id
    b = m.open("http://b", group="code").id
    m.activate("code")
    freed = m.float_all()
    assert sorted(freed) == [a, b]
    assert m.desktop_items() == [a, b]
    assert m.groups() == []
    assert m.arena() == ""


# -- layout is per-group -------------------------------------------------


def test_layout_is_per_group():
    m = ScreenModel()
    a = m.open("http://a", group="code").id
    b = m.open("http://b", group="code").id
    c = m.open("http://c", group="media").id
    d = m.open("http://d", group="media").id

    m.activate("code")
    m.arrange("code", [a, b], family="stack", dir="tb")
    assert (m.family(), m.dir()) == ("stack", "tb")

    m.activate("media")
    assert (m.family(), m.dir()) == ("grid", "lr")
    m.arrange("media", [c, d], family="grid", dir="tb")
    assert (m.family(), m.dir()) == ("grid", "tb")

    m.activate("code")
    assert (m.family(), m.dir()) == ("stack", "tb")


def test_focus_is_a_view_and_picking_a_place_ends_it():
    """Focus is about *looking*, and choosing a place cancels it.

    The old per-group fullscreen failed both halves of this: a desktop item (no
    group) could not be shown at all, and clicking home while focused left you
    staring at the same item.
    """
    m = ScreenModel()
    a = m.open("http://a", group="code").id
    b = m.open("http://b").id            # 未入组：只在桌面上

    m.set_focus(b, by="human")
    assert m.focus() == b                # 桌面的 item 一样能整屏
    assert m.view_by() == "human"        # 来源属于状态，不靠可能丢的消息

    m.activate("code", by="human")
    assert m.focus() is None             # 选一个新地方 = 撤销覆盖
    assert a in m.group_items("code")    # 而分组没有被这件事碰到


def test_group_layout_is_forgotten_with_its_group():
    m = ScreenModel()
    a = m.open("http://a", group="code").id
    m.arrange("code", [a], family="grid", dir="tb")
    m.destroy(a)
    assert m.groups() == []
    b = m.open("http://b", group="code").id
    assert (m.family(), m.dir()) == ("grid", "lr")


def test_focus_clears_when_its_item_is_destroyed():
    """The item is gone — the view cannot keep pointing at nothing."""
    m = ScreenModel()
    a = m.open("http://a").id
    m.set_focus(a)
    assert m.focus() == a
    m.destroy(a)
    assert m.focus() is None


def test_focus_survives_a_membership_change_and_rejects_unknown_items():
    """Focus is about *looking*, so leaving a group must not clear it."""
    m = ScreenModel()
    a = m.open("http://a", group="code").id
    m.set_focus(a)
    m.dismiss(a)                       # 退回桌面
    assert m.focus() == a
    with pytest.raises(KeyError):
        m.set_focus("nope")


# -- adoption / tombstones ----------------------------------------------


def test_adopt_lands_on_the_desktop_with_its_service():
    m = ScreenModel()
    item = m.adopt("cell/a/webview", "http://x", label="Terminal")
    assert item is not None
    assert item.service == "cell/a/webview"
    assert m.desktop_items() == [item.id]


def test_adopt_is_idempotent_for_a_live_service():
    m = ScreenModel()
    m.adopt("cell/a/webview", "http://x", label="A")
    assert m.adopt("cell/a/webview", "http://x", label="A") is None


def test_destroyed_service_is_tombstoned_until_it_releases():
    m = ScreenModel()
    item = m.adopt("cell/a/webview", "http://x", label="A")
    m.destroy(item.id)
    # The service is still live, but the tombstone holds — no re-adoption.
    assert m.adopt("cell/a/webview", "http://x", label="A") is None
    # The service goes away, then returns: adoption resumes.
    m.release("cell/a/webview")
    assert m.adopt("cell/a/webview", "http://x", label="A") is not None


def test_dismiss_is_soft_and_keeps_the_service():
    m = ScreenModel()
    item = m.adopt("cell/a/webview", "http://x", label="A")
    m.arrange("code", [item.id], family="grid", dir="lr")
    m.dismiss(item.id)
    # Dismiss only moves the item off its group; the service stays attached,
    # so the item still exists on the desktop (and adoption sees it).
    assert m.get(item.id) is not None
    assert m.desktop_items() == [item.id]
    assert m.service_of(item.id) == "cell/a/webview"


def test_release_removes_the_adopted_item():
    m = ScreenModel()
    item = m.adopt("cell/a/webview", "http://x", label="A")
    m.release("cell/a/webview")
    assert m.get(item.id) is None


# -- navigate -----------------------------------------------------------


def test_navigate_repoints_a_hand_opened_item():
    m = ScreenModel()
    item = m.open("http://a")
    m.navigate(item.id, "http://b")
    assert m.get(item.id).url == "http://b"


def test_navigate_rejects_a_webview_item():
    m = ScreenModel()
    item = m.adopt("cell/a/webview", "http://x", label="A")
    with pytest.raises(ValueError):
        m.navigate(item.id, "http://y")


def test_navigate_rejects_an_unknown_item():
    m = ScreenModel()
    with pytest.raises(KeyError):
        m.navigate("ghost", "http://y")


# -- pool order ---------------------------------------------------------------


def test_a_new_item_lands_at_the_front_of_the_pool():
    m = ScreenModel()
    a = m.open("http://a").id
    b = m.open("http://b").id
    assert m.pool_order() == [b, a]


def test_raise_moves_items_to_the_front_and_the_rest_follow():
    m = ScreenModel()
    ids = [m.open(f"http://{k}").id for k in range(4)]
    assert m.pool_order() == [ids[3], ids[2], ids[1], ids[0]]
    assert m.raise_to_front([ids[0], ids[1]]) == [ids[0], ids[1], ids[3], ids[2]]


def test_raise_keeps_the_order_of_the_ids_it_names():
    m = ScreenModel()
    ids = [m.open(f"http://{k}").id for k in range(4)]
    assert m.raise_to_front([ids[1], ids[0]])[:2] == [ids[1], ids[0]]


def test_raise_is_idempotent():
    m = ScreenModel()
    ids = [m.open(f"http://{k}").id for k in range(3)]
    once = m.raise_to_front([ids[0]])
    assert m.raise_to_front([ids[0]]) == once


def test_raise_rejects_unknown_empty_and_duplicate_ids():
    m = ScreenModel()
    a = m.open("http://a").id
    with pytest.raises(ValueError):
        m.raise_to_front([])
    with pytest.raises(ValueError):
        m.raise_to_front([a, a])
    with pytest.raises(KeyError):
        m.raise_to_front(["ghost"])


def test_pool_order_drops_items_that_left_the_pool():
    m = ScreenModel()
    a = m.open("http://a").id
    b = m.open("http://b").id
    m.destroy(a)
    assert m.pool_order() == [b]


def test_raise_does_not_care_about_groups():
    """Order is about distance from the eye, not about where an item is arranged."""
    m = ScreenModel()
    a = m.open("http://a").id
    b = m.open("http://b").id
    m.arrange("code", [a], family="grid", dir="lr")
    assert m.raise_to_front([a]) == [a, b]
    assert m.group_items("code") == [a]


# ---- pip：主人铺满整块，角落卡浮在一角 -------------------------------------

def test_pip_master_is_full_stage():
    layout = compute_layout(2, "pip", "br")
    assert (layout.family, layout.dir) == ("pip", "br")
    master = layout.cells[0]
    assert (master.r, master.c, master.rs, master.cs) == (1, 1, layout.rows, layout.cols)


def test_pip_corner_moves_with_dir():
    br = compute_layout(2, "pip", "br").cells[1]
    bl = compute_layout(2, "pip", "bl").cells[1]
    tr = compute_layout(2, "pip", "tr").cells[1]
    tl = compute_layout(2, "pip", "tl").cells[1]
    # 角不对：br 贴右下，tl 贴左上，bl/tr 各占另外两角
    assert br.c > bl.c and br.r == bl.r
    assert tl.r < br.r and tl.c == bl.c
    assert tr.r == tl.r and tr.c == br.c
    # 角卡必须真的落在 20x20 的网格里，且是"小卡片"而不是半块屏
    for cell in (br, bl, tr, tl):
        assert cell.r + cell.rs - 1 <= 20 and cell.c + cell.cs - 1 <= 20
        assert cell.cs <= 4 and cell.rs <= 7


def test_pip_cells_overlap_on_purpose():
    layout = compute_layout(2, "pip", "br")
    assert [c.rs for c in layout.cells] == [20, 7] or layout.cells[0].rs > layout.cells[1].rs


def test_pip_rejects_wrong_count_and_dir():
    with pytest.raises(ValueError):
        compute_layout(1, "pip", "br")
    with pytest.raises(ValueError):
        compute_layout(3, "pip", "br")
    with pytest.raises(ValueError):
        compute_layout(2, "pip", "lr")


def test_pip_rail_keeps_the_stage_whole():
    layout = compute_layout(2, "pip", "rail")
    master = layout.cells[0]
    assert (master.r, master.c, master.rs, master.cs) == (1, 1, layout.rows, layout.cols)
    # 栏里的那块不占舞台的格子 —— 位置由页面 fixed 摆，格子只是占位
    assert layout.cells[1].rs < layout.rows and layout.cells[1].cs < layout.cols


# ---- overlay：站在屏上，而不是被屏展示 -------------------------------------

def test_overlay_does_not_change_what_the_screen_shows():
    m = ScreenModel()
    a = m.open("http://a", group="desk")
    b = m.open("http://b")
    m.activate("desk")
    m.set_overlay(b.id)
    assert m.overlay() == b.id
    # 舞台还是那个分组 —— 站在屏上的东西不该把台下换掉
    assert m.arena() == "desk"
    assert m.arena_items() == [a.id]


def test_overlay_clears_without_touching_focus():
    m = ScreenModel()
    a = m.open("http://a")
    m.set_focus(a.id)
    m.set_overlay(a.id)
    m.set_overlay(None)
    assert m.overlay() == ""
    assert m.focus() == a.id
