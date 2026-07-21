"""Integration tests — DefaultGroundSet + DefaultGround end-to-end.

Covers: open/close lifecycle, pin/unpin/update, sediment/load roundtrip,
label assignment, idempotent open, frame rendering, boundary checks.
"""

import asyncio
from pathlib import Path

import pytest

from ghoshell_moss.ground import DEFAULT_L0_FILENAME, DefaultGroundSet
from ghoshell_moss.ground._l0 import load_l0
from ghoshell_moss.ground.contract import FilePin, GlobPin, LsPin, PathOutsideRootError


def run(coro):
    return asyncio.run(coro)


# -- open / close / active -------------------------------------------------


class TestOpenClose:
    def test_open_returns_ground_with_correct_props(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(sub)
                assert g.root == sub.resolve()
                assert g.label == "sub"
                assert list(gs.active().keys()) == ["sub"]

        run(scenario())

    def test_open_absolute_path(self, tmp_path):
        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                assert g.root == tmp_path.resolve()

        run(scenario())

    def test_open_idempotent(self, tmp_path):
        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g1 = await gs.open(tmp_path)
                g2 = await gs.open(tmp_path, label="ignored")
                assert g1 is g2
                assert len(gs.active()) == 1

        run(scenario())

    def test_open_label_conflict_suffix(self, tmp_path):
        (tmp_path / "a" / "foo").mkdir(parents=True)
        (tmp_path / "b" / "foo").mkdir(parents=True)

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g1 = await gs.open(tmp_path / "a" / "foo")
                g2 = await gs.open(tmp_path / "b" / "foo")
                assert g1.label == "foo"
                assert g2.label == "foo-2"

        run(scenario())

    def test_open_explicit_label(self, tmp_path):
        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path, label="custom")
                assert g.label == "custom"

        run(scenario())

    def test_close_sediments_and_removes(self, tmp_path):
        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                g.pin(FilePin(label="x", path="some_file.md"))
                await gs.close(g.label)
                assert g.label not in gs.active()

            # 落盘后重开, pin 恢复
            async with DefaultGroundSet(workspace_root=tmp_path) as gs2:
                g2 = await gs2.open(tmp_path)
                assert "x" in {p.label for p in g2.pins()}

        run(scenario())

    def test_close_missing_raises_keyerror(self, tmp_path):
        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                with pytest.raises(KeyError):
                    await gs.close("no-such")

        run(scenario())

    def test_get_returns_none_for_missing(self, tmp_path):
        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                assert gs.get("no-such") is None
                g = await gs.open(tmp_path)
                assert gs.get(g.label) is g

        run(scenario())

    def test_aexit_sediments_all(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                ga = await gs.open(tmp_path / "a")
                gb = await gs.open(tmp_path / "b")
                ga.pin(FilePin(label="a_pin", path="a.md"))
                gb.pin(FilePin(label="b_pin", path="b.md"))

            a_pins = load_l0(tmp_path / "a").pins
            b_pins = load_l0(tmp_path / "b").pins
            assert {p.label for p in a_pins} == {"a_pin"}
            assert {p.label for p in b_pins} == {"b_pin"}

        run(scenario())

    def test_two_groundsets_independent(self, tmp_path):
        """Different GroundSet instances have independent label spaces."""
        async def scenario():
            gs1 = DefaultGroundSet(workspace_root=tmp_path)
            gs2 = DefaultGroundSet(workspace_root=tmp_path)
            async with gs1, gs2:
                g1 = await gs1.open(tmp_path, label="same-label")
                g2 = await gs2.open(tmp_path, label="same-label")
                assert g1.label == "same-label"
                assert g2.label == "same-label"
                assert g1 is not g2  # different instances

        run(scenario())


# -- pin / unpin / update --------------------------------------------------


class TestPins:
    def test_pin_stores_and_returns(self, tmp_path):
        (tmp_path / "target.py").write_text("hello\n")

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                p = g.pin(FilePin(label="t", path="target.py"))
                assert p.label == "t"
                assert p in g.pins()

        run(scenario())

    def test_pin_idempotent_overwrite(self, tmp_path):
        (tmp_path / "a.md").write_text("v1")

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                g.pin(FilePin(label="f", path="a.md", description="first"))
                g.pin(FilePin(label="f", path="a.md", description="second"))
                pins = g.pins()
                assert len(pins) == 1
                assert pins[0].description == "second"

        run(scenario())

    def test_pin_order_most_recent_first(self, tmp_path):
        (tmp_path / "a.md").write_text("a")
        (tmp_path / "b.md").write_text("b")
        (tmp_path / "c.md").write_text("c")

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                g.pin(FilePin(label="a", path="a.md"))
                g.pin(FilePin(label="b", path="b.md"))
                g.pin(FilePin(label="c", path="c.md"))
                assert [p.label for p in g.pins()] == ["c", "b", "a"]

        run(scenario())

    def test_unpin_removes(self, tmp_path):
        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                g.pin(FilePin(label="x", path="x.md"))
                g.unpin("x")
                assert "x" not in {p.label for p in g.pins()}

        run(scenario())

    def test_unpin_missing_raises_keyerror(self, tmp_path):
        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                with pytest.raises(KeyError):
                    g.unpin("no-such")

        run(scenario())

    def test_update_detects_change(self, tmp_path):
        target = tmp_path / "a.md"
        target.write_text("v1")

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                g.pin(FilePin(label="f", path="a.md"))
                target.write_text("v2")
                result = await g.update("f")
                assert result.changed is True
                assert result.old_hash != result.new_hash

        run(scenario())

    def test_update_no_change(self, tmp_path):
        target = tmp_path / "a.md"
        target.write_text("v1")

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                g.pin(FilePin(label="f", path="a.md"))
                result = await g.update("f")
                assert result.changed is False

        run(scenario())


# -- frame / context -------------------------------------------------------


class TestFrame:
    def test_frame_renders_pins(self, tmp_path):
        (tmp_path / "hello.md").write_text("line1\nline2\n")

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                g.pin(FilePin(label="greeting", path="hello.md", description="welcome"))
                frame = await g.context()
                assert "line1" in frame
                assert "<!-- ground:pin:greeting -->" in frame
                assert "<!-- /ground:pin:greeting -->" in frame

        run(scenario())

    def test_frame_marks_changed_on_disk(self, tmp_path):
        target = tmp_path / "shift.md"
        target.write_text("before")

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                g.pin(FilePin(label="s", path="shift.md"))
                target.write_text("after — different")
                frame = await g.context()
                assert "changed on disk" in frame

        run(scenario())

    def test_frame_marks_missing(self, tmp_path):
        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                g.pin(FilePin(label="nope", path="does-not-exist.md"))
                frame = await g.context()
                assert "[missing]" in frame

        run(scenario())

    def test_frame_glob_shows_hits(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.py").write_text("b")

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                g.pin(GlobPin(label="py", pattern="*.py"))
                frame = await g.context()
                assert "a.py" in frame
                assert "b.py" in frame

        run(scenario())

    def test_frame_declaration_block(self, tmp_path):
        (tmp_path / "a.py").write_text("x")

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                g.pin(FilePin(label="entry", path="a.py", range="1", description="start"))
                g.pin(LsPin(label="layout", path="."))
                frame = await g.context()
                # declaration block removed from frame; pin results use HTML comments
                assert "<!-- ground:pin:entry -->" in frame
                assert "<!-- /ground:pin:entry -->" in frame
                assert "<!-- ground:pin:layout -->" in frame
                assert "<!-- /ground:pin:layout -->" in frame

        run(scenario())


# -- sediment / load roundtrip ---------------------------------------------


class TestRoundtrip:
    def test_sediment_load_preserves_pins(self, tmp_path):
        (tmp_path / "a.md").write_text("aaa")
        (tmp_path / "b.py").write_text("bbb")

        async def phase1():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                g.pin(FilePin(label="first", path="a.md"))
                g.pin(GlobPin(label="second", pattern="*.py"))

        async def phase2():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                labels = {p.label for p in g.pins()}
                assert labels == {"first", "second"}

        run(phase1())
        assert (tmp_path / DEFAULT_L0_FILENAME).is_file()
        run(phase2())

    def test_sediment_preserves_body(self, tmp_path):
        (tmp_path / DEFAULT_L0_FILENAME).write_text(
            "---\n"
            '$id: "test"\n'
            "---\n"
            "\n# Governance\n\nrules here\n"
        )

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                g.pin(FilePin(label="x", path="x.md"))

            text = (tmp_path / DEFAULT_L0_FILENAME).read_text()
            assert "Governance" in text
            assert "rules here" in text
            assert "$id" in text

        run(scenario())


class TestBinaryFrame:
    def test_frame_shows_binary_marker(self, tmp_path):
        (tmp_path / "img.bin").write_bytes(b"\x00\x01\x02\x03" * 100)

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                g.pin(FilePin(label="img", path="img.bin"))
                frame = await g.context()
                assert "[binary file, not rendered]" in frame

        run(scenario())


# -- GroundSet forwarding --------------------------------------------------


class TestGroundSetForwarding:
    def test_pin_forwards(self, tmp_path):
        (tmp_path / "a.py").write_text("x")

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                p = gs.pin(g.label, FilePin(label="f", path="a.py"))
                assert p.label == "f"
                assert "f" in {pp.label for pp in g.pins()}

        run(scenario())

    def test_unpin_forwards(self, tmp_path):
        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                gs.pin(g.label, FilePin(label="f", path="x.md"))
                gs.unpin(g.label, "f")
                assert "f" not in {p.label for p in g.pins()}

        run(scenario())

    def test_frame_forwards(self, tmp_path):
        (tmp_path / "a.py").write_text("hello")

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                g.pin(FilePin(label="f", path="a.py"))
                frame = await gs.frame(g.label)
                assert "hello" in frame

        run(scenario())

    def test_update_forwards(self, tmp_path):
        target = tmp_path / "a.md"
        target.write_text("v1")

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                g.pin(FilePin(label="f", path="a.md"))
                target.write_text("v2")
                result = await gs.update(g.label, "f")
                assert result.changed

        run(scenario())
