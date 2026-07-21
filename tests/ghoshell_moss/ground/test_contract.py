"""Tests for contract.py — Pin models, GroundConvention, validation."""

import pytest

from ghoshell_moss.ground.contract import (
    FilePin,
    FrontmatterPin,
    GlobPin,
    GroundConvention,
    LsPin,
    Pin,
    PathOutsideRootError,
)


class TestPinModels:
    def test_file_pin_defaults(self):
        p = FilePin(label="main", path="src/main.py")
        assert p.kind == "file"
        assert p.range is None
        assert p.description == ""

    def test_file_pin_with_range(self):
        p = FilePin(label="hot", path="src/hot.py", range="80-140")
        assert p.range == "80-140"

    def test_file_pin_single_line_range(self):
        p = FilePin(label="l1", path="a.py", range="5")
        assert p.range == "5"

    def test_file_pin_invalid_range_rejected(self):
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            FilePin(label="x", path="a.py", range="not-a-range")

    def test_glob_pin(self):
        p = GlobPin(label="py", pattern="src/**/*.py")
        assert p.kind == "glob"

    def test_frontmatter_pin(self):
        p = FrontmatterPin(label="status", path="FEATURE.md")
        assert p.kind == "frontmatter"

    def test_ls_pin_defaults(self):
        p = LsPin(label="root", path=".")
        assert p.kind == "ls"
        assert p.depth == 2

    def test_ls_pin_custom_depth(self):
        p = LsPin(label="deep", path="src", depth=3)
        assert p.depth == 3

    def test_label_too_long_rejected(self):
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            FilePin(label="x" * 64, path="a.py")

    def test_label_empty_rejected(self):
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            FilePin(label="", path="a.py")

    def test_label_invalid_chars_rejected(self):
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            FilePin(label="my pin", path="a.py")

    def test_description_inherited(self):
        p = GlobPin(label="g", pattern="*", description="all files")
        assert p.description == "all files"

    def test_model_dump_includes_kind(self):
        d = FilePin(label="f", path="x.py").model_dump()
        assert d["kind"] == "file"
        assert d["label"] == "f"
        assert d["path"] == "x.py"

    def test_extra_fields_ignored(self):
        p = FilePin(label="f", path="x.py", extra_unknown="should_be_dropped")  # type: ignore[call-arg]
        assert not hasattr(p, "extra_unknown")


class TestGroundConvention:
    def test_defaults(self):
        c = GroundConvention()
        assert c.id is None
        assert c.label is None

    def test_with_id(self):
        c = GroundConvention(**{"$id": "moss:features"})
        assert c.id == "moss:features"

    def test_extra_keys_preserved(self):
        c = GroundConvention(foo="bar", baz=42)  # type: ignore[call-arg]
        assert c.foo == "bar"
        assert c.baz == 42


class TestPathOutsideRootError:
    def test_is_ground_error(self):
        from ghoshell_moss.ground.contract import GroundError

        err = PathOutsideRootError("escape")
        assert isinstance(err, GroundError)
