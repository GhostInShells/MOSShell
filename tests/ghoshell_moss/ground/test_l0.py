"""Tests for _l0.py — GROUND.md serialization with discriminated union pins."""

from pathlib import Path

import pytest
import yaml

from ghoshell_moss.ground.contract import (
    FileArguments,
    FilePin,
    FrontmatterArguments,
    FrontmatterPin,
    GlobArguments,
    GlobPin,
    LsArguments,
    LsPin,
    Pin,
)
from ghoshell_moss.ground._l0 import (
    DEFAULT_L0_FILENAME,
    L0Contents,
    dump_l0_pins,
    load_l0,
)


# -- load ------------------------------------------------------------------


class TestLoadL0:
    def test_no_file_returns_empty(self, tmp_path):
        c = load_l0(tmp_path)
        assert c.pins == []
        assert c.body == ""

    def test_file_with_pins(self, tmp_path):
        (tmp_path / DEFAULT_L0_FILENAME).write_text(
            "---\n"
            '$id: "moss:test"\n'
            "---\n"
            "\n"
            "# My Ground\n\n"
            "some body text\n\n"
            "## ground:pins\n"
            "- verb: file\n"
            "  label: main\n"
            "  arguments: {path: src/main.py}\n"
            "  description: entry point\n"
            "- verb: glob\n"
            "  label: py\n"
            "  arguments: {pattern: '**/*.py'}\n"
            "- verb: ls\n"
            "  label: root\n"
            "  arguments: {path: ., depth: 1}\n"
            "- verb: frontmatter\n"
            "  label: status\n"
            "  arguments: {path: FEATURE.md}\n"
        )
        c = load_l0(tmp_path)
        assert len(c.pins) == 4

        file_pin = c.pins[0]
        assert isinstance(file_pin, FilePin)
        assert file_pin.label == "main"
        assert file_pin.arguments.path == "src/main.py"
        assert file_pin.description == "entry point"

        glob_pin = c.pins[1]
        assert isinstance(glob_pin, GlobPin)
        assert glob_pin.arguments.pattern == "**/*.py"

        ls_pin = c.pins[2]
        assert isinstance(ls_pin, LsPin)
        assert ls_pin.arguments.depth == 1

        fm_pin = c.pins[3]
        assert isinstance(fm_pin, FrontmatterPin)

        assert "some body text" in c.body
        assert "ground:pins" not in c.body

    def test_pin_section_terminated_by_next_heading(self, tmp_path):
        (tmp_path / DEFAULT_L0_FILENAME).write_text(
            "## ground:pins\n"
            "- verb: file\n"
            "  label: a\n"
            "  arguments: {path: a.md}\n\n"
            "## Other\n\n"
            "unrelated\n"
        )
        c = load_l0(tmp_path)
        assert len(c.pins) == 1
        assert "unrelated" in c.body

    def test_empty_pin_list(self, tmp_path):
        (tmp_path / DEFAULT_L0_FILENAME).write_text(
            "## ground:pins\n[]\n"
        )
        c = load_l0(tmp_path)
        assert c.pins == []

    def test_unknown_kind_skipped(self, tmp_path):
        (tmp_path / DEFAULT_L0_FILENAME).write_text(
            "## ground:pins\n"
            "- verb: file\n"
            "  label: good\n"
            "  arguments: {path: a.py}\n"
            "- verb: unknown_type\n"
            "  label: bad\n"
            "  arguments: {path: b.py}\n"
        )
        c = load_l0(tmp_path)
        assert len(c.pins) == 1
        assert c.pins[0].label == "good"

    def test_frontmatter_id_loaded(self, tmp_path):
        (tmp_path / DEFAULT_L0_FILENAME).write_text(
            "---\n"
            '$id: "moss:ghost"\n'
            "label: myground\n"
            "---\n"
            "\nbody\n"
        )
        c = load_l0(tmp_path)
        assert c.convention.id == "moss:ghost"
        assert c.convention.label == "myground"


# -- dump ------------------------------------------------------------------


class TestDumpL0:
    def test_creates_file_when_missing(self, tmp_path):
        dump_l0_pins(tmp_path, [FilePin(label="main", arguments=FileArguments(path="FEATURE.md"))])
        path = tmp_path / DEFAULT_L0_FILENAME
        assert path.is_file()
        assert "## ground:pins" in path.read_text()

    def test_roundtrip(self, tmp_path):
        original = [
            FilePin(label="main", arguments=FileArguments(path="FEATURE.md"), description="entry"),
            GlobPin(label="py", arguments=GlobArguments(pattern="**/*.py")),
            LsPin(label="root", arguments=LsArguments(path=".", depth=2)),
        ]
        dump_l0_pins(tmp_path, original)
        loaded = load_l0(tmp_path).pins
        assert len(loaded) == 3
        for a, b in zip(original, loaded):
            assert a.label == b.label
            assert a.verb == b.verb

    def test_preserves_frontmatter_and_body(self, tmp_path):
        (tmp_path / DEFAULT_L0_FILENAME).write_text(
            "---\n"
            '$id: "moss:test"\n'
            "---\n"
            "\n# Body\n\nlaw text\n"
        )
        dump_l0_pins(tmp_path, [FilePin(label="x", arguments=FileArguments(path="x.py"))])
        text = (tmp_path / DEFAULT_L0_FILENAME).read_text()
        assert '$id: "moss:test"' in text
        assert "law text" in text
        assert "x.py" in text

    def test_replaces_existing_pin_section(self, tmp_path):
        (tmp_path / DEFAULT_L0_FILENAME).write_text(
            "# body\n\ntext\n\n"
            "## ground:pins\n"
            "- verb: file\n  label: old\n  arguments: {path: old.md}\n\n"
            "## after\n\nafter\n"
        )
        dump_l0_pins(tmp_path, [FilePin(label="new", arguments=FileArguments(path="new.md"))])
        text = (tmp_path / DEFAULT_L0_FILENAME).read_text()
        assert "new.md" in text
        assert "old.md" not in text
        assert "after" in text

    def test_idempotent(self, tmp_path):
        pins = [FilePin(label="a", arguments=FileArguments(path="a.py")), GlobPin(label="b", arguments=GlobArguments(pattern="*.py"))]
        dump_l0_pins(tmp_path, pins)
        first = (tmp_path / DEFAULT_L0_FILENAME).read_text()
        dump_l0_pins(tmp_path, pins)
        second = (tmp_path / DEFAULT_L0_FILENAME).read_text()
        assert first == second

    def test_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "sub" / "deep"
        dump_l0_pins(nested, [FilePin(label="x", arguments=FileArguments(path="x.md"))])
        assert (nested / DEFAULT_L0_FILENAME).is_file()

    def test_empty_pins_writes_empty_list(self, tmp_path):
        dump_l0_pins(tmp_path, [])
        text = (tmp_path / DEFAULT_L0_FILENAME).read_text()
        # K55: bare YAML, check for empty list
        assert "[]" in text
        # YAML format verified above

    def test_pin_serialized_with_kind_field(self, tmp_path):
        dump_l0_pins(tmp_path, [FilePin(label="f", arguments=FileArguments(path="a.py", range="1-5"))])
        text = (tmp_path / DEFAULT_L0_FILENAME).read_text()
        assert "verb: file" in text
        assert "path: a.py" in text  # inside arguments
        assert "range: 1-5" in text  # inside arguments


class TestL0Contents:
    def test_empty(self):
        c = L0Contents.empty()
        assert c.pins == []
        assert c.body == ""
