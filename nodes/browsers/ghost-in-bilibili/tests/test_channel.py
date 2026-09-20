"""Channel command contract: authorization gating + dispatch + subtitle query."""

from __future__ import annotations

import pytest

from ghoshell_ghost_in_bilibili.channel import build_channel
from ghoshell_ghost_in_bilibili.model import BridgeModel
from ghoshell_ghost_in_bilibili.subtitle import SubtitleStore

from fakes import Recorder

TRACK = [
    {"from": 0.0, "to": 2.0, "content": "第一句"},
    {"from": 2.0, "to": 4.0, "content": "第二句"},
]


def _page(model, *, presence=True, grants=()) -> str:
    model.update_content("s1", 1, "BV1", "标题", "https://x/video/BV1")
    model.set_presence("s1", 1, presence)
    for g in grants:
        model.set_grant("s1", 1, g, True)
    return "p1"


@pytest.mark.asyncio
async def test_exposes_command_set(tmp_path):
    chan, _ = build_channel(BridgeModel(), SubtitleStore(tmp_path), dispatch=Recorder()), None
    async with chan.bootstrap() as runtime:
        for name in ("play", "pause", "seek", "speed", "subtitle", "say"):
            assert runtime.get_command(name) is not None, name


@pytest.mark.asyncio
async def test_play_dispatches_when_granted(tmp_path):
    model = BridgeModel()
    label = _page(model, grants=("control",))
    rec = Recorder()
    chan = build_channel(model, SubtitleStore(tmp_path), dispatch=rec)
    async with chan.bootstrap() as runtime:
        result = await runtime.execute_command("play", args=(label,))
        assert rec.actions == [(label, "play", None)]
        assert "play" in str(result)


@pytest.mark.asyncio
async def test_play_without_grant_raises_and_does_not_dispatch(tmp_path):
    model = BridgeModel()
    label = _page(model, grants=())  # presence 有,但 control 没授
    rec = Recorder()
    chan = build_channel(model, SubtitleStore(tmp_path), dispatch=rec)
    async with chan.bootstrap() as runtime:
        with pytest.raises(Exception):
            await runtime.execute_command("play", args=(label,))
        assert rec.actions == []


@pytest.mark.asyncio
async def test_play_on_absent_page_raises(tmp_path):
    model = BridgeModel()
    rec = Recorder()
    chan = build_channel(model, SubtitleStore(tmp_path), dispatch=rec)
    async with chan.bootstrap() as runtime:
        with pytest.raises(Exception):
            await runtime.execute_command("play", args=("p99",))


@pytest.mark.asyncio
async def test_seek_passes_value(tmp_path):
    model = BridgeModel()
    label = _page(model, grants=("control",))
    rec = Recorder()
    chan = build_channel(model, SubtitleStore(tmp_path), dispatch=rec)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("seek", args=(label, 60.0))
        assert rec.actions == [(label, "seek", 60.0)]


@pytest.mark.asyncio
async def test_subtitle_query_formats_lines(tmp_path):
    model = BridgeModel()
    label = _page(model, grants=("subtitle",))
    store = SubtitleStore(tmp_path)
    store.save("BV1", TRACK)
    chan = build_channel(model, store, dispatch=Recorder())
    async with chan.bootstrap() as runtime:
        result = await runtime.execute_command("subtitle", args=(label, 0.0, 2.0))
        text = str(result)
        assert "第一句" in text
        assert "0.0-2.0" in text


@pytest.mark.asyncio
async def test_say_dispatches_without_group(tmp_path):
    model = BridgeModel()
    label = _page(model)  # presence 即可,say 不需要卫星
    rec = Recorder()
    chan = build_channel(model, SubtitleStore(tmp_path), dispatch=rec)
    async with chan.bootstrap() as runtime:
        result = await runtime.execute_command("say", args=(label, "你好"))
        assert rec.saids == [(label, "你好")]
        assert label in str(result)
