import pytest

from ghoshell_screen_manager.audio import MockAudioSource
from ghoshell_screen_manager.channel import build_screen_channel
from ghoshell_screen_manager.model import ScreenModel

from fakes import Recorder


def _channel(model=None, audio=None, rec=None):
    rec = rec or Recorder()
    chan = build_screen_channel(
        model or ScreenModel(), surface=rec, audio=audio or MockAudioSource()
    )
    return chan, rec


@pytest.mark.asyncio
async def test_channel_exposes_the_command_set():
    chan, _ = _channel()
    async with chan.bootstrap() as runtime:
        for name in (
            "open", "close", "activate", "arrange", "fullscreen", "pool",
            "mark", "arrow", "text", "mock_scene", "mock_audio",
        ):
            assert runtime.get_command(name) is not None, name


@pytest.mark.asyncio
async def test_open_into_active_emits_open_then_arrange():
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("term", "http://x"), kwargs={"group": "code"})
        await runtime.execute_command("activate", args=("code",))
        await runtime.execute_command("open", args=("edit", "http://y"), kwargs={"group": "code"})
        assert rec.of("open")[-1]["item"]["id"] == "edit"
        assert rec.of("arrange")[-1]["ids"] == ["term", "edit"]


@pytest.mark.asyncio
async def test_open_into_nonactive_emits_no_arrange():
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("a", "http://a"), kwargs={"group": "g"})
        await runtime.execute_command("open", args=("b", "http://b"), kwargs={"group": "h"})
        assert not rec.of("arrange")


@pytest.mark.asyncio
async def test_arrange_emits_cells_and_order():
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("a", "http://a"), kwargs={"group": "g"})
        await runtime.execute_command("open", args=("b", "http://b"), kwargs={"group": "g"})
        await runtime.execute_command("activate", args=("g",))
        await runtime.execute_command(
            "arrange", args=("b,a",), kwargs={"family": "stack", "dir": "lr"}
        )
        frame = rec.of("arrange")[-1]
        assert frame["ids"] == ["b", "a"]
        assert frame["layout"]["family"] == "stack"
        assert frame["layout"]["cols"] == 2


@pytest.mark.asyncio
async def test_close_emits_close_and_arrange_when_active():
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("a", "http://a"), kwargs={"group": "g"})
        await runtime.execute_command("activate", args=("g",))
        await runtime.execute_command("close", args=("a",))
        assert rec.of("close")[-1]["id"] == "a"
        assert rec.of("arrange")[-1]["ids"] == []


@pytest.mark.asyncio
async def test_mark_emits_a_veil_frame():
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("term", "http://x"), kwargs={"group": "code"})
        await runtime.execute_command(
            "mark", args=("term",), kwargs={"region": "right", "duration": 0.01}
        )
        frame = rec.of("veil")[-1]
        assert frame["gesture"] == "mark"
        assert frame["item"] == "term"
        assert frame["region"] == "right"


@pytest.mark.asyncio
async def test_notice_carries_active_group_and_layout():
    model = ScreenModel()
    chan, _ = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("term", "http://x"), kwargs={"group": "code"})
        await runtime.execute_command("activate", args=("code",))
        await runtime.refresh_metas()
        notices = runtime.self_meta().named_notices
        assert "term" in notices["screen"]
        assert "code" in notices["screen"]
        assert "code" in notices["groups"]


@pytest.mark.asyncio
async def test_mock_scene_populates_groups():
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("mock_scene")
        assert len(model.groups()) >= 3
        assert rec.types()[-1] == "snapshot"


@pytest.mark.asyncio
async def test_open_refuses_a_duplicate_id():
    model = ScreenModel()
    chan, _ = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("a", "http://a"), kwargs={"group": "g"})
        with pytest.raises(Exception):
            await runtime.execute_command("open", args=("a", "http://b"), kwargs={"group": "g"})
