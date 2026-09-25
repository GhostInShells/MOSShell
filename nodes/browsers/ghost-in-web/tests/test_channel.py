"""Channel command contract: perception gating + dispatch + soft rejection."""

from __future__ import annotations

import pytest

from ghoshell_ghost_in_web.channel import build_channel
from ghoshell_ghost_in_web.model import PageModel

from fakes import RecordingDispatcher


def _page(model: PageModel, *, perceived: bool = True) -> str:
    model.update_content("s1", 1, "标题", "https://x.example")
    model.set_perceived("s1", 1, perceived)
    return "p1"


@pytest.mark.asyncio
async def test_exposes_command_set():
    chan = build_channel(PageModel(), dispatch=RecordingDispatcher())
    async with chan.bootstrap() as runtime:
        for name in ("read", "find", "click", "type", "say"):
            assert runtime.get_command(name) is not None, name


@pytest.mark.asyncio
async def test_read_dispatches_and_formats():
    model = PageModel()
    label = _page(model)
    rec = RecordingDispatcher({"ok": True, "result": "正文内容"})
    chan = build_channel(model, dispatch=rec)
    async with chan.bootstrap() as runtime:
        result = await runtime.execute_command("read", args=(label,))
        assert rec.actions == [(label, "read", None)]
        assert "正文内容" in str(result)
        assert label in str(result)


@pytest.mark.asyncio
async def test_read_without_perception_raises_and_does_not_dispatch():
    model = PageModel()
    label = _page(model, perceived=False)
    rec = RecordingDispatcher()
    chan = build_channel(model, dispatch=rec)
    async with chan.bootstrap() as runtime:
        with pytest.raises(Exception):
            await runtime.execute_command("read", args=(label,))
        assert rec.actions == []


@pytest.mark.asyncio
async def test_read_on_absent_page_raises():
    chan = build_channel(PageModel(), dispatch=RecordingDispatcher())
    async with chan.bootstrap() as runtime:
        with pytest.raises(Exception):
            await runtime.execute_command("read", args=("p99",))


@pytest.mark.asyncio
async def test_find_formats_hits():
    model = PageModel()
    label = _page(model)
    rec = RecordingDispatcher({"ok": True, "result": ['r3 <a> "Foo"']})
    chan = build_channel(model, dispatch=rec)
    async with chan.bootstrap() as runtime:
        result = await runtime.execute_command("find", args=(label, "Foo"))
        text = str(result)
        assert 'r3 <a> "Foo"' in text


@pytest.mark.asyncio
async def test_click_rejection_is_soft_not_raised():
    model = PageModel()
    label = _page(model)
    rec = RecordingDispatcher({"ok": False, "accepted": False, "result": "human rejected"})
    chan = build_channel(model, dispatch=rec)
    async with chan.bootstrap() as runtime:
        result = await runtime.execute_command("click", args=(label, "r3"))
        assert rec.actions == [(label, "click", "r3")]
        assert "拒绝" in str(result)


@pytest.mark.asyncio
async def test_type_passes_value():
    model = PageModel()
    label = _page(model)
    rec = RecordingDispatcher()
    chan = build_channel(model, dispatch=rec)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("type", args=(label, "r3", "你好"))
        assert rec.actions == [(label, "type", {"ref": "r3", "text": "你好"})]


@pytest.mark.asyncio
async def test_say_dispatches_without_gating():
    model = PageModel()
    label = _page(model)
    rec = RecordingDispatcher()
    chan = build_channel(model, dispatch=rec)
    async with chan.bootstrap() as runtime:
        result = await runtime.execute_command("say", args=(label, "你好"))
        assert rec.said == [(label, "你好")]
        assert label in str(result)
