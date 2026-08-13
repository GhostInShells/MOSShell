"""Integration tests — DefaultGroundSet + DefaultGround end-to-end.

Covers: open/close lifecycle, pin/unpin/update, sediment/load roundtrip,
label assignment, idempotent open, frame rendering, boundary checks.
"""

import asyncio
from pathlib import Path

import pytest

from ghoshell_moss.ground import DEFAULT_L0_FILENAME, DefaultGroundSet
from ghoshell_moss.ground._l0 import load_l0
from ghoshell_moss.ground.contract import (
    FileArguments,
    FilePin,
    GlobArguments,
    GlobPin,
    LsArguments,
    LsPin,
    PathOutsideRootError,
)


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
                g.pin(FilePin(label="x", arguments=FileArguments(path="some_file.md")))
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
                ga.pin(FilePin(label="a_pin", arguments=FileArguments(path="a.md")))
                gb.pin(FilePin(label="b_pin", arguments=FileArguments(path="b.md")))

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
                p = g.pin(FilePin(label="t", arguments=FileArguments(path="target.py")))
                assert p.label == "t"
                assert p in g.pins()

        run(scenario())

    def test_pin_idempotent_overwrite(self, tmp_path):
        (tmp_path / "a.md").write_text("v1")

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                g.pin(FilePin(label="f", arguments=FileArguments(path="a.md"), description="first"))
                g.pin(FilePin(label="f", arguments=FileArguments(path="a.md"), description="second"))
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
                g.pin(FilePin(label="a", arguments=FileArguments(path="a.md")))
                g.pin(FilePin(label="b", arguments=FileArguments(path="b.md")))
                g.pin(FilePin(label="c", arguments=FileArguments(path="c.md")))
                assert [p.label for p in g.pins()] == ["c", "b", "a"]

        run(scenario())

    def test_unpin_removes(self, tmp_path):
        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                g.pin(FilePin(label="x", arguments=FileArguments(path="x.md")))
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



# -- frame / context -------------------------------------------------------


class TestFrame:
    def test_frame_renders_pins(self, tmp_path):
        (tmp_path / "hello.md").write_text("line1\nline2\n")

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                g.pin(FilePin(label="greeting", arguments=FileArguments(path="hello.md"), description="welcome"))
                text = await g.context()
                assert "line1" in text
                assert "<!-- file-greeting: welcome -->" in text

        run(scenario())

    def test_frame_marks_missing(self, tmp_path):
        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                g.pin(FilePin(label="nope", arguments=FileArguments(path="does-not-exist.md")))
                frame = await g.context()
                assert "[missing]" in frame

        run(scenario())

    def test_frame_glob_shows_hits(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.py").write_text("b")

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                g.pin(GlobPin(label="py", arguments=GlobArguments(path="*.py")))
                frame = await g.context()
                assert "a.py" in frame
                assert "b.py" in frame

        run(scenario())

    def test_frame_declaration_block(self, tmp_path):
        (tmp_path / "a.py").write_text("x")

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                g.pin(FilePin(label="entry", arguments=FileArguments(path="a.py", range="1"), description="start"))
                g.pin(LsPin(label="layout", arguments=LsArguments(path=".")))
                text = await g.context()
                assert "<!-- file-entry: start -->" in text
                assert "<!-- ls-layout -->" in text
                assert "x" in text

        run(scenario())


# -- sediment / load roundtrip ---------------------------------------------


class TestRoundtrip:
    def test_sediment_load_preserves_pins(self, tmp_path):
        (tmp_path / "a.md").write_text("aaa")
        (tmp_path / "b.py").write_text("bbb")

        async def phase1():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                g.pin(FilePin(label="first", arguments=FileArguments(path="a.md")))
                g.pin(GlobPin(label="second", arguments=GlobArguments(path="*.py")))

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
                g.pin(FilePin(label="x", arguments=FileArguments(path="x.md")))

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
                g.pin(FilePin(label="img", arguments=FileArguments(path="img.bin")))
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
                p = gs.pin(g.label, FilePin(label="f", arguments=FileArguments(path="a.py")))
                assert p.label == "f"
                assert "f" in {pp.label for pp in g.pins()}

        run(scenario())

    def test_unpin_forwards(self, tmp_path):
        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                gs.pin(g.label, FilePin(label="f", arguments=FileArguments(path="x.md")))
                gs.unpin(g.label, "f")
                assert "f" not in {p.label for p in g.pins()}

        run(scenario())

    def test_frame_forwards(self, tmp_path):
        (tmp_path / "a.py").write_text("hello")

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(tmp_path)
                g.pin(FilePin(label="f", arguments=FileArguments(path="a.py")))
                frame = await gs.frame(g.label)
                assert "hello" in frame

        run(scenario())



# -- template open ----------------------------------------------------------


class TestTemplateOpen:
    def _make_template(self, tmp_path, name="mytmpl"):
        (tmp_path / ".grounds").mkdir(exist_ok=True)
        (tmp_path / ".grounds" / f"{name}.md").write_text(
            "---\n"
            "pins:\n"
            "- verb: ls\n"
            "  label: tree\n"
            "  arguments: {path: $CWD}\n"
            "  description: 目录树\n"
            "---\n"
            "# My Template\n\n"
            "body text\n"
        )

    def test_open_with_template_copies_body_and_pins(self, tmp_path):
        self._make_template(tmp_path)
        target = tmp_path / "proj"
        target.mkdir()

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(target, template="mytmpl")
                frame = await g.context()
                assert "body text" in frame
                assert any(p.label == "tree" for p in g.pins())

        run(scenario())

    def test_open_unknown_template_raises(self, tmp_path):
        target = tmp_path / "proj"
        target.mkdir()

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                with pytest.raises(KeyError):
                    await gs.open(target, template="nope")

        run(scenario())

    def test_template_init_sediments_on_close(self, tmp_path):
        self._make_template(tmp_path)
        target = tmp_path / "proj2"
        target.mkdir()

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(target, template="mytmpl")
                assert g.dirty  # 模板注入的 pins 未落盘 → close 触发 sediment
                await gs.close(g.label)
            assert (target / "GROUND.md").is_file()

        run(scenario())


class TestSnapshot:
    """Ground.snapshot() — 渲染 + 感知 digest + 对账 (auto-advance)."""

    @staticmethod
    def _make_ground(tmp_path) -> Path:
        root = tmp_path / "proj"
        root.mkdir()
        (root / "GROUND.md").write_text("---\nname: proj\npins: []\n---\n# body\n")
        return root

    def test_first_call_is_baseline(self, tmp_path):
        root = self._make_ground(tmp_path)

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(root)
                snap = await g.snapshot()
                assert snap.changed is False
                assert snap.hash
                assert snap.view.header.ground_path == str(root.resolve())

        run(scenario())

    def test_unchanged_stays_silent(self, tmp_path):
        root = self._make_ground(tmp_path)

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(root)
                s1 = await g.snapshot()
                s2 = await g.snapshot()
                assert s2.changed is False
                assert s2.hash == s1.hash

        run(scenario())

    def test_change_flags_once_then_acknowledges(self, tmp_path):
        root = self._make_ground(tmp_path)

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(root)
                s1 = await g.snapshot()
                (root / "GROUND.md").write_text(
                    "---\nname: proj\npins: []\n---\n# body changed\n"
                )
                await g.load()
                s2 = await g.snapshot()
                assert s2.changed is True
                assert s2.hash != s1.hash
                # 已承认 → 下一帧不变
                s3 = await g.snapshot()
                assert s3.changed is False
                assert s3.hash == s2.hash

        run(scenario())

    def test_explicit_ack_hash_is_baseline(self, tmp_path):
        root = self._make_ground(tmp_path)

        async def scenario():
            async with DefaultGroundSet(workspace_root=tmp_path) as gs:
                g = await gs.open(root)
                s1 = await g.snapshot()
                # ack 当前 hash → 无变化
                assert (await g.snapshot(ack_hash=s1.hash)).changed is False
                # ack 一个旧值 → 相对该基线变化
                assert (await g.snapshot(ack_hash="0" * 64)).changed is True

        run(scenario())
