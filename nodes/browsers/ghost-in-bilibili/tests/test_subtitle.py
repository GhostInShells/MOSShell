"""SubtitleStore: privacy whitelist + time-range query semantics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ghoshell_ghost_in_bilibili.subtitle import SubtitleStore  # noqa: E402

TRACK = [
    {"from": 0.0, "to": 2.5, "content": "你好"},
    {"from": 2.5, "to": 5.0, "content": "今天一起看视频"},
    {"from": 5.0, "to": 8.0, "content": "这段讲的是 AI"},
]


def _store(tmp_path: Path) -> SubtitleStore:
    return SubtitleStore(tmp_path / "subtitles")


def test_save_strips_unknown_fields(tmp_path):
    store = _store(tmp_path)
    store.save("BV1", [{"from": 0.0, "to": 1.0, "content": "x", "uid": 123, "nickname": "谁"}])
    raw = json.loads(store.path("BV1").read_text(encoding="utf-8"))
    assert raw[0] == {"from": 0.0, "to": 1.0, "content": "x"}


def test_save_sorts_by_from(tmp_path):
    store = _store(tmp_path)
    store.save("BV1", [TRACK[2], TRACK[0], TRACK[1]])
    assert [l["from"] for l in store.load("BV1")] == [0.0, 2.5, 5.0]


def test_available_reflects_file(tmp_path):
    store = _store(tmp_path)
    assert not store.available("BV1")
    store.save("BV1", TRACK)
    assert store.available("BV1")
    assert not store.available("BV2")


def test_query_overlap(tmp_path):
    store = _store(tmp_path)
    store.save("BV1", TRACK)
    # 跨两条
    assert [l["content"] for l in store.query("BV1", 2.0, 3.0)] == ["你好", "今天一起看视频"]
    # 精确边界:5.0 是两条线的公共边界,闭区间 query 返回两条(有意为之)
    assert [l["content"] for l in store.query("BV1", 5.0, 5.0)] == ["今天一起看视频", "这段讲的是 AI"]
    # 无重叠
    assert store.query("BV1", 20.0, 30.0) == []
    # start > end 自动交换
    assert [l["content"] for l in store.query("BV1", 3.0, 2.0)] == ["你好", "今天一起看视频"]


def test_at(tmp_path):
    store = _store(tmp_path)
    store.save("BV1", TRACK)
    assert store.at("BV1", 1.0)["content"] == "你好"
    assert store.at("BV1", 6.0)["content"] == "这段讲的是 AI"
    assert store.at("BV1", 100.0) is None
    # 半开 [from, to):边界落在后一条
    assert store.at("BV1", 5.0)["content"] == "这段讲的是 AI"
    assert store.at("BV1", 2.5)["content"] == "今天一起看视频"
    assert store.at("BV1", 8.0) is None


def test_window_recent(tmp_path):
    store = _store(tmp_path)
    store.save("BV1", TRACK)
    # t=6.0, before=15 → 三条全在窗口内
    assert [l["content"] for l in store.window("BV1", 6.0, 15.0)] == [
        "你好", "今天一起看视频", "这段讲的是 AI",
    ]
    # before 收紧 → 只有后两条
    assert [l["content"] for l in store.window("BV1", 6.0, 2.5)] == [
        "今天一起看视频", "这段讲的是 AI",
    ]


def test_missing_bvid_returns_empty(tmp_path):
    store = _store(tmp_path)
    assert store.query("nothing", 0, 10) == []
    assert store.window("nothing", 5) == []
