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
            "open", "arrange", "activate", "dismiss", "destroy", "fullscreen",
            "float", "pool", "mark", "arrow", "text", "mock_scene", "mock_audio",
        ):
            assert runtime.get_command(name) is not None, name


@pytest.mark.asyncio
async def test_open_emits_open_then_state():
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("term", "http://x"))
        assert rec.of("open")[-1]["item"]["id"] == "term"
        state = rec.of("state")[-1]
        assert state["desktop"] == ["term"]
        assert state["arena"] == ""


@pytest.mark.asyncio
async def test_arrange_emits_state_with_order_and_layout():
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("a", "http://a"))
        await runtime.execute_command("open", args=("b", "http://b"))
        await runtime.execute_command(
            "arrange", args=("b,a",), kwargs={"group": "code", "family": "stack", "dir": "lr"}
        )
        state = rec.of("state")[-1]
        assert state["arena"] == "code"
        assert state["ids"] == ["b", "a"]
        assert state["layout"]["family"] == "stack"
        assert state["layout"]["cols"] == 2
        assert state["desktop"] == []


@pytest.mark.asyncio
async def test_dismiss_emits_state():
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("a", "http://a"))
        await runtime.execute_command("arrange", args=("a",), kwargs={"group": "code"})
        await runtime.execute_command("dismiss", args=("a",))
        state = rec.of("state")[-1]
        assert state["desktop"] == ["a"]
        assert state["arena"] == ""


@pytest.mark.asyncio
async def test_destroy_emits_close_then_state():
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("a", "http://a"))
        await runtime.execute_command("destroy", args=("a",))
        assert rec.of("close")[-1]["id"] == "a"
        assert rec.of("state")[-1]["desktop"] == []


@pytest.mark.asyncio
async def test_fullscreen_emits_a_fullscreen_frame():
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("a", "http://a"))
        await runtime.execute_command("arrange", args=("a",), kwargs={"group": "code"})
        await runtime.execute_command("fullscreen", args=("a",))
        assert rec.of("fullscreen")[-1]["id"] == "a"


@pytest.mark.asyncio
async def test_activate_frame_carries_the_desktop():
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("activate", args=("",))
        assert rec.of("activate")[-1]["arena"] == ""


@pytest.mark.asyncio
async def test_mark_emits_a_veil_frame():
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("term", "http://x"))
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
        await runtime.execute_command("open", args=("term", "http://x"))
        await runtime.execute_command("arrange", args=("term",), kwargs={"group": "code"})
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
        await runtime.execute_command("open", args=("a", "http://a"))
        with pytest.raises(Exception):
            await runtime.execute_command("open", args=("a", "http://b"))


@pytest.mark.asyncio
async def test_notice_carries_the_surface_url_when_provided():
    model = ScreenModel()
    chan = build_screen_channel(
        model,
        surface=Recorder(),
        audio=MockAudioSource(),
        surface_url=lambda: "http://127.0.0.1:54321",
    )
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert runtime.self_meta().named_notices["url"] == "http://127.0.0.1:54321"


@pytest.mark.asyncio
async def test_views_notice_fragment_when_supplied():
    model = ScreenModel()
    chan = build_screen_channel(
        model,
        surface=Recorder(),
        audio=MockAudioSource(),
        views_notice=lambda: "1 view(s) floating: term",
    )
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert runtime.self_meta().named_notices["views"] == "1 view(s) floating: term"
