"""The file is the truth here: one `*.idea.md` per idea, one directory per group.

These tests pin the four properties the simplified mechanism rests on:

* discovery is a first-level directory scan, and the filename is the identity;
* frontmatter round-trips — editing a file by hand is a supported operation;
* a link whose target is gone is *reported*, not dropped;
* a broken file is reported too, because a file that silently disappears is
  indistinguishable from one that was never written.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from store import SUFFIX, Store, split_frontmatter  # noqa: E402

IDEA = """---
state: {state}
priority: {priority}
tags: [t1, t2]
links: [other, third:depend]
---

the body of {name}
"""


def write_group(root: Path, group: str, readme: str = "group context") -> Path:
    path = root / group
    path.mkdir(parents=True, exist_ok=True)
    (path / "README.md").write_text(readme, encoding="utf-8")
    return path


def write_idea(group: Path, name: str, state: str = "spark", priority: float = 1.0) -> Path:
    path = group / f"{name}{SUFFIX}"
    path.write_text(IDEA.format(state=state, priority=priority, name=name), encoding="utf-8")
    return path


def loaded(tmp_path: Path) -> Store:
    store = Store(tmp_path)
    store.reload()
    return store


# --------------------------------------------------------------------- discovery


def test_first_level_directories_are_groups(tmp_path: Path):
    write_group(tmp_path, "voice")
    write_group(tmp_path, "work")
    write_idea(tmp_path / "voice", "duplex-gap", state="active", priority=3)
    write_idea(tmp_path / "work", "bench-the-windows")

    store = loaded(tmp_path)

    assert sorted(store.groups) == ["voice", "work"]
    idea = store.find("duplex-gap")
    assert idea is not None and idea.group == "voice"
    assert idea.state == "active" and idea.priority == 3.0
    assert idea.tags == ["t1", "t2"]
    assert (tmp_path / "voice" / "README.md").read_text(encoding="utf-8") == "group context"


def test_name_is_the_identity(tmp_path: Path):
    write_group(tmp_path, "voice")
    write_idea(tmp_path / "voice", "duplex-gap")
    idea = loaded(tmp_path).find("duplex-gap")
    assert idea is not None and idea.name == "duplex-gap"
    assert idea.title == "the body of duplex-gap"      # first body line is the reading title


def test_frontmatter_round_trips(tmp_path: Path):
    write_group(tmp_path, "voice")
    path = write_idea(tmp_path / "voice", "duplex-gap")

    idea = loaded(tmp_path).find("duplex-gap")
    assert idea is not None
    idea.state = "waiting"
    idea.tags.append("later")
    idea.save()

    meta, body = split_frontmatter(path.read_text(encoding="utf-8"))
    assert meta["state"] == "waiting"
    assert meta["tags"] == ["t1", "t2", "later"]
    assert body.strip() == "the body of duplex-gap"


# ------------------------------------------------------------------------- links


def test_dead_link_is_reported_not_dropped(tmp_path: Path):
    write_group(tmp_path, "voice")
    write_idea(tmp_path / "voice", "other")
    write_idea(tmp_path / "voice", "duplex-gap")   # links: other, third:depend

    store = loaded(tmp_path)
    idea = store.find("duplex-gap")
    assert idea is not None
    known = store.known()
    assert [l.name for l in idea.live_links(known)] == ["other"]
    assert [l.name for l in idea.dead_links(known)] == ["third"]


def test_rename_rewrites_every_reference(tmp_path: Path):
    write_group(tmp_path, "voice")
    write_idea(tmp_path / "voice", "other")
    write_idea(tmp_path / "voice", "duplex-gap")

    store = loaded(tmp_path)
    store.rename("other", "other-name")

    assert store.find("other") is None
    assert store.find("other-name") is not None
    idea = store.find("duplex-gap")
    assert idea is not None
    assert [l.name for l in idea.links] == ["other-name", "third"]


# ------------------------------------------------------------------------ writes


def test_spark_slugifies_and_never_overwrites(tmp_path: Path):
    write_group(tmp_path, "voice")
    store = loaded(tmp_path)

    first = store.spark("双工开得太满: 两个人不在同一个时间点上说话", group="voice")
    second = store.spark("双工开得太满: 两个人不在同一个时间点上说话", group="voice")

    assert first.name == "双工开得太满-两个人不在同一个时间点上说话"
    assert second.name == first.name + "-2"
    assert first.path.exists() and second.path.exists()


def test_broken_file_is_reported_not_skipped(tmp_path: Path):
    write_group(tmp_path, "voice")
    write_idea(tmp_path / "voice", "fine")
    (tmp_path / "voice" / f"broken{SUFFIX}").write_text("---\nstate: [unclosed\n---\n", encoding="utf-8")

    store = loaded(tmp_path)

    assert store.find("fine") is not None
    assert len(store.broken) == 1 and "broken" in store.broken[0]


def test_snapshot_carries_dead_flag_for_the_page(tmp_path: Path):
    write_group(tmp_path, "voice")
    write_idea(tmp_path / "voice", "other")
    write_idea(tmp_path / "voice", "duplex-gap")

    snap = loaded(tmp_path).snapshot()
    idea = next(i for g in snap["groups"] for i in g["ideas"] if i["name"] == "duplex-gap")
    flags = {l["name"]: l["dead"] for l in idea["links"]}
    assert flags == {"other": False, "third": True}
