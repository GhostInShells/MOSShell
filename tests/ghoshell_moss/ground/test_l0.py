"""Tests for _l0.py — L0 file parsing 与 sediment 半程."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ghoshell_moss.ground.contract import GroundConvention, Pin
from ghoshell_moss.ground._l0 import (
    DEFAULT_L0_FILENAME,
    L0Contents,
    dump_l0_pins,
    load_l0,
)


# ---- load_l0 (缺文件, 空文件, 只有 frontmatter, 全套) --------------------


def test_load_l0_no_file(tmp_path: Path) -> None:
    c = load_l0(tmp_path)
    assert c.convention == GroundConvention()
    assert c.body == ""
    assert c.pins == []


def test_load_l0_frontmatter_only(tmp_path: Path) -> None:
    (tmp_path / DEFAULT_L0_FILENAME).write_text(
        "---\ntree_depth: 3\ncontext_budget: 8000\n---\n"
    )
    c = load_l0(tmp_path)
    assert c.convention.tree_depth == 3
    assert c.convention.context_budget == 8000
    # frontmatter 之外没有正文
    assert c.body.strip() == ""
    assert c.pins == []


def test_load_l0_body_no_frontmatter(tmp_path: Path) -> None:
    (tmp_path / DEFAULT_L0_FILENAME).write_text("# hello\n\nsome prose\n")
    c = load_l0(tmp_path)
    assert c.convention == GroundConvention()
    assert "some prose" in c.body
    assert c.pins == []


def test_load_l0_pin_section(tmp_path: Path) -> None:
    (tmp_path / DEFAULT_L0_FILENAME).write_text(
        "---\ntree_depth: 1\n---\n"
        "\n"
        "# governance\n\n"
        "law law law\n\n"
        "## ground:pins\n\n"
        "```yaml\n"
        "- addr: FEATURE.md\n"
        "  note: main\n"
        "  pinned_at: 100.0\n"
        "  seen_mtime: 90.0\n"
        "  seen_hash: abc\n"
        "```\n"
    )
    c = load_l0(tmp_path)
    assert c.convention.tree_depth == 1
    assert "law law law" in c.body
    assert "ground:pins" not in c.body  # pin 段被剥离
    assert len(c.pins) == 1
    p = c.pins[0]
    assert p.addr == "FEATURE.md"
    assert p.note == "main"
    assert p.pinned_at == 100.0
    assert p.seen_mtime == 90.0
    assert p.seen_hash == "abc"


def test_load_l0_pin_section_terminated_by_next_heading(tmp_path: Path) -> None:
    (tmp_path / DEFAULT_L0_FILENAME).write_text(
        "## ground:pins\n\n"
        "```yaml\n"
        "- addr: a.md\n"
        "  note: ''\n"
        "  pinned_at: 1.0\n"
        "```\n\n"
        "## Other\n\n"
        "unrelated body\n"
    )
    c = load_l0(tmp_path)
    assert len(c.pins) == 1
    assert c.pins[0].addr == "a.md"
    assert "unrelated body" in c.body
    assert "## Other" in c.body


def test_load_l0_pin_section_empty_yaml_list(tmp_path: Path) -> None:
    (tmp_path / DEFAULT_L0_FILENAME).write_text(
        "## ground:pins\n\n"
        "```yaml\n"
        "[]\n"
        "```\n"
    )
    c = load_l0(tmp_path)
    assert c.pins == []


def test_load_l0_body_preserved_before_pin_section(tmp_path: Path) -> None:
    (tmp_path / DEFAULT_L0_FILENAME).write_text(
        "# body\n\nprose\n\n## ground:pins\n\n```yaml\n[]\n```\n"
    )
    c = load_l0(tmp_path)
    assert "prose" in c.body
    assert "## ground:pins" not in c.body


# ---- dump_l0_pins ------------------------------------------------------


def _pin(addr: str, **kw) -> Pin:
    kw.setdefault("note", "")
    kw.setdefault("pinned_at", 0.0)
    return Pin(addr=addr, **kw)


def test_dump_creates_file_when_missing(tmp_path: Path) -> None:
    pins = [_pin("FEATURE.md", note="main")]
    dump_l0_pins(tmp_path, pins)
    path = tmp_path / DEFAULT_L0_FILENAME
    assert path.is_file()
    text = path.read_text()
    assert "## ground:pins" in text
    assert "FEATURE.md" in text


def test_dump_then_load_roundtrip(tmp_path: Path) -> None:
    original = [
        _pin("FEATURE.md", note="main", pinned_at=100.0),
        _pin("**/*.py", note="glob", pinned_at=200.0, seen_hash="xxx"),
        _pin("a.py:1-10", note="range", pinned_at=300.0, seen_mtime=99.0),
    ]
    dump_l0_pins(tmp_path, original)
    loaded = load_l0(tmp_path).pins
    assert len(loaded) == 3
    for a, b in zip(original, loaded):
        assert a.addr == b.addr
        assert a.note == b.note
        assert a.pinned_at == b.pinned_at
        assert a.seen_mtime == b.seen_mtime
        assert a.seen_hash == b.seen_hash


def test_dump_appends_pin_section_to_body_only_file(tmp_path: Path) -> None:
    (tmp_path / DEFAULT_L0_FILENAME).write_text("# preserved\n\nhello\n")
    dump_l0_pins(tmp_path, [_pin("a.md")])
    text = (tmp_path / DEFAULT_L0_FILENAME).read_text()
    assert "# preserved" in text
    assert "hello" in text
    assert "## ground:pins" in text
    # heading 顺序: body 先, pin 段后
    assert text.index("# preserved") < text.index("## ground:pins")


def test_dump_replaces_existing_pin_section(tmp_path: Path) -> None:
    (tmp_path / DEFAULT_L0_FILENAME).write_text(
        "# body\n\ntext\n\n"
        "## ground:pins\n\n"
        "```yaml\n- addr: old.md\n  note: ''\n  pinned_at: 1.0\n```\n\n"
        "## after\n\nafter-body\n"
    )
    dump_l0_pins(tmp_path, [_pin("new.md", note="fresh")])
    text = (tmp_path / DEFAULT_L0_FILENAME).read_text()
    assert "new.md" in text
    assert "old.md" not in text
    # body 与 pin 段之后的 heading 都保留
    assert "text" in text
    assert "## after" in text
    assert "after-body" in text


def test_dump_preserves_frontmatter(tmp_path: Path) -> None:
    (tmp_path / DEFAULT_L0_FILENAME).write_text(
        "---\ntree_depth: 5\ncontext_budget: 12345\n---\n\n"
        "# body\n"
    )
    dump_l0_pins(tmp_path, [_pin("a.md")])
    reloaded = load_l0(tmp_path)
    assert reloaded.convention.tree_depth == 5
    assert reloaded.convention.context_budget == 12345
    assert len(reloaded.pins) == 1


def test_dump_empty_pin_list_writes_empty_yaml_list(tmp_path: Path) -> None:
    dump_l0_pins(tmp_path, [])
    text = (tmp_path / DEFAULT_L0_FILENAME).read_text()
    # 里面应该有 `[]` 或空列表 yaml
    assert "```yaml" in text
    # 语法要 parse 得动
    section = text.split("## ground:pins", 1)[1]
    yaml_body = section.split("```yaml", 1)[1].split("```", 1)[0]
    assert yaml.safe_load(yaml_body) in (None, [])


def test_dump_idempotent(tmp_path: Path) -> None:
    pins = [_pin("FEATURE.md", pinned_at=1.0), _pin("**/*.py", pinned_at=2.0)]
    dump_l0_pins(tmp_path, pins)
    first = (tmp_path / DEFAULT_L0_FILENAME).read_text()
    dump_l0_pins(tmp_path, pins)
    second = (tmp_path / DEFAULT_L0_FILENAME).read_text()
    assert first == second


def test_dump_creates_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "sub" / "sub2"
    # 目录不存在
    dump_l0_pins(nested, [_pin("x.md")])
    assert (nested / DEFAULT_L0_FILENAME).is_file()


# ---- L0Contents.empty --------------------------------------------------


def test_l0_contents_empty() -> None:
    c = L0Contents.empty()
    assert c.convention == GroundConvention()
    assert c.body == ""
    assert c.pins == []


# ---- validation errors -------------------------------------------------


def test_load_l0_invalid_frontmatter_raises(tmp_path: Path) -> None:
    (tmp_path / DEFAULT_L0_FILENAME).write_text(
        "---\ntree_depth: not-a-number\n---\n"
    )
    with pytest.raises(Exception):  # pydantic ValidationError
        load_l0(tmp_path)
