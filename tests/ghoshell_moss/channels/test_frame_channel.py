"""Tests for frame channel — question-set as thinking framework.

Covers:
- question shape: blank-line-separated paragraphs, order = index
- load: label = path relative to root (suffix stripped); re-load does not reset
- resolve: minimal ack (command is truth), no state echo
- completion: the resolving call carries the ``nexts`` hint, once
- ``unknown`` is a first-class resolution
- status: unresolved (blind spots) listed before resolved
- spec: returns the format convention
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ghoshell_moss.channels.frame_channel import new_frame_channel
from ghoshell_moss.core.concepts.errors import CommandError

_FRAME = """\
---
description: orientation base
nexts:
  - env.frame.md
---

What environment am I in?

Who is the user?

What am I doing?
"""


def _write_root(tmp: Path) -> Path:
    root = tmp / "frames"
    root.mkdir()
    (root / "orientation.frame.md").write_text(_FRAME, encoding="utf-8")
    (root / "env.frame.md").write_text(
        "---\ndescription: environment\n---\n\nWhich toolchain am I running on?\n\n"
        "What constraints does the environment impose?\n",
        encoding="utf-8",
    )
    return root


@pytest.mark.asyncio
async def test_load_parses_question_shape():
    with tempfile.TemporaryDirectory() as tmp:
        root = _write_root(Path(tmp))
        chan = new_frame_channel(root=root, entry="orientation.frame.md")
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            instruction = runtime.self_meta().instruction
            assert "0/3" in instruction
            assert "What environment am I in?" in instruction
            assert "What am I doing?" in instruction


@pytest.mark.asyncio
async def test_resolve_returns_ack_not_state():
    with tempfile.TemporaryDirectory() as tmp:
        root = _write_root(Path(tmp))
        chan = new_frame_channel(root=root, entry="orientation.frame.md")
        async with chan.bootstrap() as runtime:
            result = await runtime.execute_command(
                "resolve", args=("orientation", 0, "a terminal in the MOSS repo")
            )
            assert result == "resolved"


@pytest.mark.asyncio
async def test_completion_carries_nexts_hint_once():
    with tempfile.TemporaryDirectory() as tmp:
        root = _write_root(Path(tmp))
        chan = new_frame_channel(root=root, entry="orientation.frame.md")
        async with chan.bootstrap() as runtime:
            await runtime.execute_command("resolve", args=("orientation", 0, "a"))
            await runtime.execute_command("resolve", args=("orientation", 1, "b"))
            result = await runtime.execute_command("resolve", args=("orientation", 2, "c"))
            assert "complete" in result
            assert "env.frame.md" in result
            # re-resolving an already-complete frame is a plain ack again
            result2 = await runtime.execute_command("resolve", args=("orientation", 0, "a2"))
            assert result2 == "resolved"


@pytest.mark.asyncio
async def test_unknown_is_a_valid_resolution():
    with tempfile.TemporaryDirectory() as tmp:
        root = _write_root(Path(tmp))
        chan = new_frame_channel(root=root, entry="orientation.frame.md")
        async with chan.bootstrap() as runtime:
            await runtime.execute_command("resolve", args=("orientation", 0, "unknown"))
            await runtime.execute_command("resolve", args=("orientation", 1, "unknown"))
            result = await runtime.execute_command("resolve", args=("orientation", 2, "unknown"))
            assert "complete" in result


@pytest.mark.asyncio
async def test_status_lists_unresolved_first():
    with tempfile.TemporaryDirectory() as tmp:
        root = _write_root(Path(tmp))
        chan = new_frame_channel(root=root, entry="orientation.frame.md")
        async with chan.bootstrap() as runtime:
            await runtime.execute_command("resolve", args=("orientation", 0, "answer0"))
            result = await runtime.execute_command("status", args=("orientation",))
            assert result.index("blind spots") < result.index("→")
            assert "What am I doing?" in result
            assert "answer0" in result


@pytest.mark.asyncio
async def test_load_additional_frame_and_resolve_by_label():
    with tempfile.TemporaryDirectory() as tmp:
        root = _write_root(Path(tmp))
        chan = new_frame_channel(root=root, entry="orientation.frame.md")
        async with chan.bootstrap() as runtime:
            result = await runtime.execute_command("load", args=("env.frame.md",))
            assert "loaded [env]" in result
            result2 = await runtime.execute_command(
                "resolve", args=("env", 0, "the MOSS toolchain")
            )
            assert result2 == "resolved"


@pytest.mark.asyncio
async def test_complete_instruction_surfaces_nexts_hint():
    with tempfile.TemporaryDirectory() as tmp:
        root = _write_root(Path(tmp))
        chan = new_frame_channel(root=root, entry="orientation.frame.md")
        async with chan.bootstrap() as runtime:
            for i in range(3):
                await runtime.execute_command("resolve", args=("orientation", i, "x"))
            await runtime.refresh_metas()
            instruction = runtime.self_meta().instruction
            assert "nexts: env.frame.md" in instruction


@pytest.mark.asyncio
async def test_malformed_frontmatter_raises():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "frames"
        root.mkdir()
        (root / "bad.frame.md").write_text(
            "---\nkey: [unclosed\n---\n\nWho?\n", encoding="utf-8"
        )
        chan = new_frame_channel(root=root)
        async with chan.bootstrap() as runtime:
            with pytest.raises(CommandError, match="malformed frontmatter"):
                await runtime.execute_command("load", args=("bad.frame.md",))


@pytest.mark.asyncio
async def test_spec_returns_convention():
    chan = new_frame_channel(root="frames")
    async with chan.bootstrap() as runtime:
        result = await runtime.execute_command("spec")
        assert "frontmatter" in result
        assert ".frame.md" in result


@pytest.mark.asyncio
async def test_list_marks_loaded_vs_available():
    with tempfile.TemporaryDirectory() as tmp:
        root = _write_root(Path(tmp))
        chan = new_frame_channel(root=root, entry="orientation.frame.md")
        async with chan.bootstrap() as runtime:
            result = await runtime.execute_command("list")
            assert "2 frame(s)" in result
            assert "loaded" in result
            assert "orientation" in result
            assert "env" in result


@pytest.mark.asyncio
async def test_reload_rereads_edited_questions_and_resets():
    with tempfile.TemporaryDirectory() as tmp:
        root = _write_root(Path(tmp))
        chan = new_frame_channel(root=root, entry="orientation.frame.md")
        async with chan.bootstrap() as runtime:
            await runtime.execute_command("resolve", args=("orientation", 0, "a"))
            # edit the frame file: add a question, then re-read
            (root / "orientation.frame.md").write_text(
                _FRAME + "\nA brand new question?\n", encoding="utf-8"
            )
            result = await runtime.execute_command("reload", args=("orientation.frame.md",))
            assert "reloaded [orientation]" in result
            assert "0/4" in result
            status = await runtime.execute_command("status", args=("orientation",))
            assert "A brand new question?" in status


@pytest.mark.asyncio
async def test_template_returns_starter():
    chan = new_frame_channel(root="frames")
    async with chan.bootstrap() as runtime:
        result = await runtime.execute_command("template")
        assert "description:" in result
        assert "<question" in result
