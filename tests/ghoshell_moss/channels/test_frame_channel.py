"""Tests for the frame channel — question set as a session-scoped thinking frame."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from ghoshell_moss.channels.frame_channel import (
    FRAME_SUFFIX,
    Frame,
    SessionFrame,
    new_frame_channel,
    new_frame_channel_from_file,
)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


ORIENTATION = Frame(
    label="orientation",
    description="reconstruct the situation after context loss",
    questions=["What environment am I in?", "Who am I talking to?", "What must I not forget?"],
)


def _dump(frame: Frame) -> str:
    from ghoshell_common.helpers import generate_import_path, yaml_pretty_dump

    body = yaml_pretty_dump(frame.model_dump())
    return f"# dump from `{generate_import_path(Frame)}`\n{body}"


class TestModel:
    def test_frame_is_serialisable_round_trip(self):
        frame = ORIENTATION.model_copy(deep=True)
        frame.answers[0] = "a terminal"
        again = Frame.model_validate(frame.model_dump())
        assert again.answers == {0: "a terminal"}

    def test_answers_key_is_int(self):
        frame = Frame.model_validate({"label": "x", "questions": ["q"], "answers": {"0": "a"}})
        assert frame.answers == {0: "a"}

    def test_suffix_constant(self):
        assert FRAME_SUFFIX == ".frame.yml"


class TestSessionFrame:
    def test_all_questions_start_unresolved(self):
        sf = SessionFrame(Frame(label="x", questions=["a", "b"]))
        assert sf.unresolved == [0, 1]
        assert not sf.complete

    def test_resolve_lifts_from_unresolved(self):
        sf = SessionFrame(Frame(label="x", questions=["a", "b"]))
        sf.frame.answers[0] = "somewhere"
        assert sf.unresolved == [1]

    def test_notice_body_lists_unresolved_then_stamp(self):
        sf = SessionFrame(Frame(label="x", questions=["a", "b"]))
        sf.refreshed_at = "20:09"
        assert sf.render_notice() == "[0] a\n[1] b\nrefreshed at: 20:09"

    def test_complete_frame_renders_empty(self):
        sf = SessionFrame(Frame(label="x", questions=["a"]))
        sf.frame.answers[0] = "answered"
        assert sf.render_notice() == ""


class TestInitFrame:
    @pytest.mark.asyncio
    async def test_init_frame_lands_in_notice_and_instruction(self, tmp_path: Path):
        chan = new_frame_channel(root=tmp_path, init_frame=ORIENTATION)
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            meta = runtime.metas()[""]
        assert "What environment am I in?" in meta.named_notices["orientation"]
        assert "reconstruct the situation after context loss" in meta.instruction

    @pytest.mark.asyncio
    async def test_init_frame_is_deep_copied(self, tmp_path: Path):
        original = ORIENTATION.model_copy(deep=True)
        chan = new_frame_channel(root=tmp_path, init_frame=original)
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            await runtime.execute_command("resolve", kwargs={"question_index": 0, "answer": "x"})
        assert original.answers == {}

    @pytest.mark.asyncio
    async def test_from_file_drops_answers(self, tmp_path: Path):
        frame = ORIENTATION.model_copy(deep=True)
        frame.answers[0] = "stale"
        _write(tmp_path / "orientation.frame.yml", _dump(frame))
        chan = new_frame_channel_from_file(root=tmp_path, init_frame_file="orientation.frame.yml")
        async with chan.bootstrap() as runtime:
            out = await runtime.execute_command("resolved")
        assert out == "[orientation] nothing resolved"

    def test_from_file_rejects_escape(self, tmp_path: Path):
        outside = tmp_path / "outside"
        outside.mkdir()
        _write(outside / "f.frame.yml", _dump(ORIENTATION))
        root = tmp_path / "root"
        root.mkdir()
        with pytest.raises(ValueError):
            new_frame_channel_from_file(root=root, init_frame_file=outside / "f.frame.yml")


class TestLoad:
    @pytest.mark.asyncio
    async def test_load_without_answers_is_fresh(self, tmp_path: Path):
        frame = ORIENTATION.model_copy(deep=True)
        frame.answers[0] = "from disk"
        _write(tmp_path / "orientation.frame.yml", _dump(frame))
        chan = new_frame_channel(root=tmp_path)
        async with chan.bootstrap() as runtime:
            await runtime.execute_command("load", kwargs={"path": "orientation.frame.yml"})
            out = await runtime.execute_command("resolved")
        assert out == "[orientation] nothing resolved"

    @pytest.mark.asyncio
    async def test_load_with_answers_keeps_them(self, tmp_path: Path):
        frame = ORIENTATION.model_copy(deep=True)
        frame.answers[0] = "from disk"
        _write(tmp_path / "orientation.frame.yml", _dump(frame))
        chan = new_frame_channel(root=tmp_path)
        async with chan.bootstrap() as runtime:
            await runtime.execute_command(
                "load", kwargs={"path": "orientation.frame.yml", "with_answers": True}
            )
            out = await runtime.execute_command("resolved")
        assert "from disk" in out

    @pytest.mark.asyncio
    async def test_load_is_idempotent(self, tmp_path: Path):
        _write(tmp_path / "a.frame.yml", _dump(Frame(label="a", questions=["q"])))
        chan = new_frame_channel(root=tmp_path)
        async with chan.bootstrap() as runtime:
            first = await runtime.execute_command("load", kwargs={"path": "a.frame.yml"})
            second = await runtime.execute_command("load", kwargs={"path": "a.frame.yml"})
        assert "loaded [a] 0/1" in first
        assert "already loaded" in second

    @pytest.mark.asyncio
    async def test_load_rejects_path_outside_root(self, tmp_path: Path):
        outside = tmp_path / "outside.frame.yml"
        _write(outside, _dump(Frame(label="x", questions=["q"])))
        root = tmp_path / "root"
        root.mkdir()
        chan = new_frame_channel(root=root)
        async with chan.bootstrap() as runtime:
            with pytest.raises(Exception):
                await runtime.execute_command("load", kwargs={"path": "../outside.frame.yml"})

    @pytest.mark.asyncio
    async def test_load_missing_file_errors(self, tmp_path: Path):
        chan = new_frame_channel(root=tmp_path)
        async with chan.bootstrap() as runtime:
            with pytest.raises(Exception):
                await runtime.execute_command("load", kwargs={"path": "nope.frame.yml"})

    @pytest.mark.asyncio
    async def test_bad_yaml_errors_loudly(self, tmp_path: Path):
        _write(tmp_path / "bad.frame.yml", "label: [unclosed\n")
        chan = new_frame_channel(root=tmp_path)
        async with chan.bootstrap() as runtime:
            with pytest.raises(Exception):
                await runtime.execute_command("load", kwargs={"path": "bad.frame.yml"})


class TestDefine:
    @pytest.mark.asyncio
    async def test_define_from_json_body(self, tmp_path: Path):
        chan = new_frame_channel(root=tmp_path)
        body = json.dumps({"label": "debug", "questions": ["Is it a race?"]})
        async with chan.bootstrap() as runtime:
            out = await runtime.execute_command("define", kwargs={"text__": body})
            meta = runtime.metas()
            await runtime.refresh_metas()
            notice = runtime.metas()[""].named_notices["debug"]
        assert out == "defined [debug] 0/1"
        assert "Is it a race?" in notice

    @pytest.mark.asyncio
    async def test_define_doc_carries_json_schema(self, tmp_path: Path):
        chan = new_frame_channel(root=tmp_path)
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            meta = runtime.metas()[""]
        define_cmd = next(c for c in meta.commands if c.name == "define")
        rendered = str(define_cmd)
        assert "CDATA" in rendered
        assert "questions" in rendered

    @pytest.mark.asyncio
    async def test_define_rejects_invalid_json(self, tmp_path: Path):
        chan = new_frame_channel(root=tmp_path)
        async with chan.bootstrap() as runtime:
            with pytest.raises(Exception):
                await runtime.execute_command("define", kwargs={"text__": "{not json"})

    @pytest.mark.asyncio
    async def test_define_rejects_no_questions(self, tmp_path: Path):
        chan = new_frame_channel(root=tmp_path)
        body = json.dumps({"label": "x", "questions": []})
        async with chan.bootstrap() as runtime:
            with pytest.raises(Exception):
                await runtime.execute_command("define", kwargs={"text__": body})


class TestResolve:
    @pytest.mark.asyncio
    async def test_resolved_question_leaves_the_notice(self, tmp_path: Path):
        chan = new_frame_channel(root=tmp_path, init_frame=ORIENTATION)
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            await runtime.execute_command("resolve", kwargs={"question_index": 0, "answer": "a terminal"})
            await runtime.refresh_metas()
            notice = runtime.metas()[""].named_notices["orientation"]
        assert "What environment am I in?" not in notice
        assert "Who am I talking to?" in notice

    @pytest.mark.asyncio
    async def test_unknown_is_first_class(self, tmp_path: Path):
        chan = new_frame_channel(root=tmp_path, init_frame=ORIENTATION)
        async with chan.bootstrap() as runtime:
            await runtime.execute_command("resolve", kwargs={"question_index": 0, "answer": "unknown"})
            out = await runtime.execute_command("resolved")
        assert "unknown" in out

    @pytest.mark.asyncio
    async def test_complete_notice_is_removed(self, tmp_path: Path):
        chan = new_frame_channel(root=tmp_path, init_frame=Frame(label="solo", questions=["only?"]))
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            await runtime.execute_command("resolve", kwargs={"question_index": 0, "answer": "yes"})
            await runtime.refresh_metas()
            assert runtime.metas()[""].named_notices["solo"] is None

    @pytest.mark.asyncio
    async def test_out_of_range_errors(self, tmp_path: Path):
        chan = new_frame_channel(root=tmp_path, init_frame=ORIENTATION)
        async with chan.bootstrap() as runtime:
            with pytest.raises(Exception):
                await runtime.execute_command("resolve", kwargs={"question_index": 9, "answer": "x"})

    @pytest.mark.asyncio
    async def test_no_label_targets_latest(self, tmp_path: Path):
        chan = new_frame_channel(root=tmp_path, init_frame=Frame(label="a", questions=["q"]))
        async with chan.bootstrap() as runtime:
            out = await runtime.execute_command("resolve", kwargs={"question_index": 0, "answer": "here"})
        assert out == "[a] complete"


class TestReset:
    @pytest.mark.asyncio
    async def test_reset_clears_answers_keeps_questions(self, tmp_path: Path):
        chan = new_frame_channel(root=tmp_path, init_frame=ORIENTATION)
        async with chan.bootstrap() as runtime:
            await runtime.execute_command("resolve", kwargs={"question_index": 0, "answer": "x"})
            out = await runtime.execute_command("reset")
            resolved = await runtime.execute_command("resolved")
        assert out == "[orientation] reset"
        assert resolved == "[orientation] nothing resolved"


class TestExport:
    @pytest.mark.asyncio
    async def test_export_round_trips(self, tmp_path: Path):
        chan = new_frame_channel(root=tmp_path, init_frame=ORIENTATION)
        async with chan.bootstrap() as runtime:
            await runtime.execute_command("resolve", kwargs={"question_index": 0, "answer": "here"})
            await runtime.execute_command("export")
        written = tmp_path / "orientation.frame.yml"
        assert written.is_file()
        text = written.read_text(encoding="utf-8")
        assert text.startswith("# dump from `")
        loaded = Frame.model_validate(__import__("yaml").safe_load(text))
        assert loaded.answers == {0: "here"}

    @pytest.mark.asyncio
    async def test_export_custom_filename(self, tmp_path: Path):
        chan = new_frame_channel(root=tmp_path, init_frame=ORIENTATION)
        async with chan.bootstrap() as runtime:
            await runtime.execute_command("export", kwargs={"filename": "saved/copy"})
        assert (tmp_path / "saved" / "copy.frame.yml").is_file()

    @pytest.mark.asyncio
    async def test_export_rejects_escape(self, tmp_path: Path):
        root = tmp_path / "root"
        root.mkdir()
        chan = new_frame_channel(root=root, init_frame=ORIENTATION)
        async with chan.bootstrap() as runtime:
            with pytest.raises(Exception):
                await runtime.execute_command("export", kwargs={"filename": "../escape"})


class TestRefreshClock:
    @pytest.mark.asyncio
    async def test_first_refresh_always_signs(self, tmp_path: Path):
        chan = new_frame_channel(root=tmp_path, init_frame=Frame(label="a", questions=["q"]))
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            body = runtime.metas()[""].named_notices["a"]
        assert body.startswith("[0] q")
        assert "refreshed at: " in body

    @pytest.mark.asyncio
    async def test_stamp_holds_within_threshold(self, tmp_path: Path):
        chan = new_frame_channel(
            root=tmp_path, init_frame=Frame(label="a", questions=["q"]), refresh_seconds=10_000
        )
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            first = runtime.metas()[""].named_notices["a"]
            await runtime.refresh_metas()
            second = runtime.metas()[""].named_notices["a"]
        assert first == second

    @pytest.mark.asyncio
    async def test_stamp_renews_after_threshold(self, tmp_path: Path):
        """Past the threshold the frame is re-signed so the fragment is re-emitted."""
        chan = new_frame_channel(
            root=tmp_path, init_frame=Frame(label="a", questions=["q"]), refresh_seconds=120.0
        )
        holder: dict[str, SessionFrame] = {}
        orig = SessionFrame.render_notice

        def spy(self: SessionFrame) -> str:
            holder["sf"] = self
            return orig(self)

        SessionFrame.render_notice = spy  # type: ignore[method-assign]
        try:
            async with chan.bootstrap() as runtime:
                await runtime.refresh_metas()
                sf = holder["sf"]
                assert sf.refreshed_monotonic > time.monotonic() - 5
                sf.refreshed_monotonic = time.monotonic() - 10_000
                await runtime.refresh_metas()
                assert sf.refreshed_monotonic > time.monotonic() - 5
        finally:
            SessionFrame.render_notice = orig  # type: ignore[method-assign]
