"""Identity + audit trail: labels are stable, perception flips are recorded."""

from __future__ import annotations

from ghoshell_ghost_in_web.model import PageModel


def test_label_is_stable_across_navigation():
    model = PageModel()
    model.update_content("s1", 1, "T1", "https://a.example")
    model.update_content("s1", 1, "T2", "https://b.example")
    page = model.page_by_label("p1")
    assert page is not None
    assert page.url == "https://b.example"
    assert page.title == "T2"
    assert len(model.all_pages()) == 1


def test_two_tabs_are_two_labels():
    model = PageModel()
    model.update_content("s1", 1, "A", "u")
    model.update_content("s1", 2, "B", "u")
    assert set(model.all_pages()) == {"p1", "p2"}


def test_perception_gate_and_audit():
    model = PageModel()
    model.update_content("s1", 1, "T", "u")
    assert model.perceived_pages() == {}
    model.set_perceived("s1", 1, True)
    assert set(model.perceived_pages()) == {"p1"}
    kinds = [(e.kind, e.detail, e.ok) for e in model.audit()]
    assert ("perception", "授权感知", True) in kinds


def test_close_tab_releases_label_and_logs():
    model = PageModel()
    model.update_content("s1", 1, "T", "u")
    model.close_tab("s1", 1)
    assert model.page_by_label("p1") is None
    assert any(e.detail == "页面关闭" for e in model.audit())


def test_audit_is_bounded():
    model = PageModel(audit_limit=5)
    for i in range(20):
        model.log("p1", "behavior", f"#{i}")
    assert len(model.audit(limit=100)) == 5
    assert model.audit()[-1].detail == "#19"
