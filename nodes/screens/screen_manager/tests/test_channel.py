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
            "open", "navigate", "arrange", "activate", "dismiss", "destroy",
            "fullscreen", "float", "pool", "mark", "arrow", "text", "mock_scene",
            "mock_audio",
        ):
            assert runtime.get_command(name) is not None, name


@pytest.mark.asyncio
async def test_open_emits_open_then_state():
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("http://x",))
        item_id = rec.of("open")[-1]["item"]["id"]
        state = rec.of("state")[-1]
        assert state["desktop"] == [item_id]
        assert state["arena"] == ""


@pytest.mark.asyncio
async def test_open_assigns_monotonic_handles():
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("http://a",))
        await runtime.execute_command("open", args=("http://b",))
        ids = [f["item"]["id"] for f in rec.of("open")]
        assert ids[0] != ids[1]


@pytest.mark.asyncio
async def test_arrange_emits_state_with_order_and_layout():
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("http://a",))
        await runtime.execute_command("open", args=("http://b",))
        a, b = [f["item"]["id"] for f in rec.of("open")]
        await runtime.execute_command(
            "arrange", args=(f"{b},{a}",), kwargs={"group": "code", "family": "stack", "dir": "lr"}
        )
        state = rec.of("state")[-1]
        assert state["arena"] == "code"
        assert state["ids"] == [b, a]
        assert state["layout"]["family"] == "stack"
        assert state["layout"]["cols"] == 2
        assert state["desktop"] == []


@pytest.mark.asyncio
async def test_dismiss_emits_state():
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("http://a",))
        a = rec.of("open")[-1]["item"]["id"]
        await runtime.execute_command("arrange", args=(a,), kwargs={"group": "code"})
        await runtime.execute_command("dismiss", args=(a,))
        state = rec.of("state")[-1]
        assert state["desktop"] == [a]
        assert state["arena"] == ""


@pytest.mark.asyncio
async def test_destroy_emits_close_then_state():
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("http://a",))
        a = rec.of("open")[-1]["item"]["id"]
        await runtime.execute_command("destroy", args=(a,))
        assert rec.of("close")[-1]["id"] == a
        assert rec.of("state")[-1]["desktop"] == []


@pytest.mark.asyncio
async def test_fullscreen_emits_a_fullscreen_frame():
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("http://a",))
        a = rec.of("open")[-1]["item"]["id"]
        await runtime.execute_command("arrange", args=(a,), kwargs={"group": "code"})
        await runtime.execute_command("fullscreen", args=(a,))
        assert rec.of("fullscreen")[-1]["id"] == a


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
        await runtime.execute_command("open", args=("http://x",))
        item_id = rec.of("open")[-1]["item"]["id"]
        await runtime.execute_command(
            "mark", args=(item_id,), kwargs={"region": "right", "duration": 0.01}
        )
        frame = rec.of("veil")[-1]
        assert frame["gesture"] == "mark"
        assert frame["item"] == item_id
        assert frame["region"] == "right"


@pytest.mark.asyncio
async def test_navigate_repoints_a_hand_opened_item():
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("http://a",))
        item_id = rec.of("open")[-1]["item"]["id"]
        await runtime.execute_command("navigate", args=(item_id, "http://b"))
        assert model.get(item_id).url == "http://b"
        assert rec.of("open")[-1]["item"]["url"] == "http://b"


@pytest.mark.asyncio
async def test_navigate_rejects_a_webview_item():
    model = ScreenModel()
    item = model.adopt("cell/a/webview", "http://x", label="A")
    chan, _ = _channel(model)
    async with chan.bootstrap() as runtime:
        with pytest.raises(Exception):
            await runtime.execute_command("navigate", args=(item.id, "http://y"))


@pytest.mark.asyncio
async def test_notice_carries_active_group_and_layout():
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("http://x",))
        item_id = rec.of("open")[-1]["item"]["id"]
        await runtime.execute_command("arrange", args=(item_id,), kwargs={"group": "code"})
        await runtime.refresh_metas()
        notices = runtime.self_meta().named_notices
        assert item_id in notices["screen"]
        assert "code" in notices["screen"]
        assert "code" in notices["groups"]


@pytest.mark.asyncio
async def test_notices_separate_views_and_items():
    model = ScreenModel()
    model.adopt("cell/a/webview", "http://x", label="Terminal")
    chan, _ = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("http://a", "hand"))
        await runtime.refresh_metas()
        notices = runtime.self_meta().named_notices
        assert "views" in notices
        assert "items" in notices
        assert "Terminal" in notices["views"]
        assert "hand" in notices["items"]


@pytest.mark.asyncio
async def test_mock_scene_populates_groups():
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("mock_scene")
        assert len(model.groups()) >= 3
        assert rec.types()[-1] == "snapshot"


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
