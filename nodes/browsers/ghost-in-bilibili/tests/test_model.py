"""BridgeModel: label 持久身份 + bvid 易变属性 + 授权格."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ghoshell_ghost_in_bilibili.model import BridgeModel  # noqa: E402


def test_label_stable_across_autoplay():
    m = BridgeModel()
    m.update_content("s1", 7, "BV1", "标题1", "https://x/video/BV1")
    assert m.label_of("s1", 7) == "p1"
    # 自动播放切到下一个视频:label 不变,只有内容属性变
    m.update_content("s1", 7, "BV2", "标题2", "https://x/video/BV2")
    assert m.label_of("s1", 7) == "p1"
    assert m.page_by_label("p1").bvid == "BV2"


def test_two_tabs_two_labels():
    m = BridgeModel()
    m.update_content("s1", 1, "BV1", "", "")
    m.update_content("s1", 2, "BV1", "", "")  # 同 bvid 不同 tab
    assert m.label_of("s1", 1) == "p1"
    assert m.label_of("s1", 2) == "p2"


def test_label_never_reused_after_close():
    m = BridgeModel()
    m.update_content("s1", 1, "BV1", "", "")
    assert m.label_of("s1", 1) == "p1"
    m.close_tab("s1", 1)
    m.update_content("s1", 2, "BV2", "", "")  # 新 tab
    assert m.label_of("s1", 2) == "p2"  # 不复用 p1


def test_open_pages_only_presence():
    m = BridgeModel()
    m.update_content("s1", 1, "BV1", "", "")
    m.update_content("s1", 2, "BV2", "", "")
    assert m.open_pages() == {}  # 都没授权 presence
    m.set_presence("s1", 1, True)
    assert list(m.open_pages()) == ["p1"]


def test_grants():
    m = BridgeModel()
    m.update_content("s1", 1, "BV1", "", "")
    assert not m.granted("p1", "control")
    m.set_grant("s1", 1, "control", True)
    assert m.granted("p1", "control")
    # 未知组不生效
    m.set_grant("s1", 1, "nonsense", True)
    assert not m.granted("p1", "nonsense")


def test_reachability_on_hello_and_disconnect():
    m = BridgeModel()
    m.update_content("s1", 1, "BV1", "", "")
    m.update_content("s1", 2, "BV2", "", "")
    assert all(p.reachable for p in m.pages.values())
    m.on_disconnect("s1")
    assert all(not p.reachable for p in m.pages.values())
    m.on_hello("s1")
    assert all(p.reachable for p in m.pages.values())


def test_state_update():
    m = BridgeModel()
    m.update_state("s1", 1, t=42.5, paused=False, rate=2.0, duration=300.0)
    p = m.page_by_label("p1")
    assert (p.t, p.paused, p.rate, p.duration) == (42.5, False, 2.0, 300.0)
