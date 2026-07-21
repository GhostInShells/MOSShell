"""Tests for _chain.py — law chain collection from ancestor GROUND.md bodies."""

from pathlib import Path

from ghoshell_moss.ground._chain import collect_chain
from ghoshell_moss.ground._l0 import DEFAULT_L0_FILENAME, dump_l0_pins
from ghoshell_moss.ground.contract import FileArguments, FilePin


def _write_ground(root: Path, body: str) -> None:
    (root / DEFAULT_L0_FILENAME).write_text(
        f"---\n---\n\n{body}\n"
    )


class TestCollectChain:
    def test_empty_when_no_ground_files(self, tmp_path):
        result = collect_chain(tmp_path)
        assert result == ""

    def test_collects_body_from_ground_md(self, tmp_path):
        _write_ground(tmp_path, "law in root")
        result = collect_chain(tmp_path)
        assert "law in root" in result

    def test_walks_upward_root_first(self, tmp_path):
        outer = tmp_path
        inner = tmp_path / "sub" / "deep"
        inner.mkdir(parents=True)
        _write_ground(outer, "MARKER_OUTER")
        _write_ground(inner, "MARKER_INNER")

        result = collect_chain(inner, boundary=outer)
        outer_pos = result.index("MARKER_OUTER")
        inner_pos = result.index("MARKER_INNER")
        assert outer_pos < inner_pos  # root-first

    def test_stops_at_boundary(self, tmp_path):
        a = tmp_path
        b = tmp_path / "sub_b"
        c = tmp_path / "sub_b" / "sub_c"
        c.mkdir(parents=True)
        _write_ground(a, "MARKER_AAA")
        _write_ground(b, "MARKER_BBB")
        _write_ground(c, "MARKER_CCC")

        result = collect_chain(c, boundary=b)
        assert "MARKER_AAA" not in result
        assert "MARKER_BBB" in result
        assert "MARKER_CCC" in result

    def test_ground_outside_boundary_only_self(self, tmp_path):
        outside = tmp_path / "outside"
        inside = tmp_path / "inside"
        outside.mkdir()
        inside.mkdir()
        _write_ground(outside, "MARKER_OUT")

        result = collect_chain(outside, boundary=inside)
        assert "MARKER_OUT" in result

    def test_body_only_no_frontmatter_or_pins(self, tmp_path):
        # GROUND.md with frontmatter + body + pins — only body should appear
        root = tmp_path
        dump_l0_pins(root, [FilePin(label="f", arguments=FileArguments(path="a.py"))])
        # dump_l0_pins creates a fresh GROUND.md; add body manually
        text = (root / DEFAULT_L0_FILENAME).read_text()
        (root / DEFAULT_L0_FILENAME).write_text(
            "---\n$id: test\n---\n\n# My Body\n\nlaw here\n\n## ground:pins\n[]\n"
        )
        result = collect_chain(root)
        assert "# My Body" in result
        assert "law here" in result
        assert "ground:pins" not in result
        assert "$id" not in result  # frontmatter excluded

    def test_skips_dirs_without_ground_md(self, tmp_path):
        a = tmp_path
        b = tmp_path / "sub"
        b.mkdir()
        _write_ground(a, "MARKER_A")
        # b has no GROUND.md

        result = collect_chain(b, boundary=a)
        assert "MARKER_A" in result
