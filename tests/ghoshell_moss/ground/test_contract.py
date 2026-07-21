"""Tests for contract.py — Pin models, GroundConvention, validation."""

import pytest

from ghoshell_moss.ground.contract import (
    FileArguments,
    FilePin,
    FrontmatterArguments,
    FrontmatterPin,
    GlobArguments,
    GlobPin,
    GroundConvention,
    LsArguments,
    LsPin,
    Pin,
    PathOutsideRootError,
)


class TestPinModels:
    def test_file_pin_defaults(self):
        p = FilePin(label="main", arguments=FileArguments(path="src/main.py"))
        assert p.verb == "file"
        assert p.arguments.range is None
        assert p.description == ""

    def test_file_pin_with_range(self):
        p = FilePin(label="hot", arguments=FileArguments(path="src/hot.py", range="80-140"))
        assert p.arguments.range == "80-140"

    def test_file_pin_single_line_range(self):
        p = FilePin(label="l1", arguments=FileArguments(path="a.py", range="5"))
        assert p.arguments.range == "5"

    def test_file_pin_invalid_range_rejected(self):
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            FilePin(label="x", arguments=FileArguments(path="a.py", range="not-a-range"))

    def test_glob_pin(self):
        p = GlobPin(label="py", arguments=GlobArguments(pattern="src/**/*.py"))
        assert p.verb == "glob"

    def test_frontmatter_pin(self):
        p = FrontmatterPin(label="status", arguments=FrontmatterArguments(path="FEATURE.md"))
        assert p.verb == "frontmatter"

    def test_ls_pin_defaults(self):
        p = LsPin(label="root", arguments=LsArguments(path="."))
        assert p.verb == "ls"
        assert p.arguments.depth == 2

    def test_ls_pin_custom_depth(self):
        p = LsPin(label="deep", arguments=LsArguments(path="src", depth=3))
        assert p.arguments.depth == 3

    def test_label_too_long_rejected(self):
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            FilePin(label="x" * 64, arguments=FileArguments(path="a.py"))

    def test_label_empty_rejected(self):
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            FilePin(label="", arguments=FileArguments(path="a.py"))

    def test_label_invalid_chars_rejected(self):
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            FilePin(label="my pin", arguments=FileArguments(path="a.py"))

    def test_description_inherited(self):
        p = GlobPin(label="g", arguments=GlobArguments(pattern="*"), description="all files")
        assert p.description == "all files"

    def test_model_dump_includes_kind(self):
        d = FilePin(label="f", arguments=FileArguments(path="x.py")).model_dump()
        assert d["verb"] == "file"
        assert d["label"] == "f"
        assert d["arguments"]["path"] == "x.py"

    def test_extra_fields_ignored(self):
        p = FilePin(label="f", arguments=FileArguments(path="x.py"))
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
