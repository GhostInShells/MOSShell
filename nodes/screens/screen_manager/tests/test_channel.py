import asyncio

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
            "focus", "float", "raise", "pool", "mark", "box", "circle", "arrow",
            "danmaku", "paint", "veil", "clear_veil",
            "mock_scene", "mock_audio",
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
async def test_id_lists_accept_pipe_comma_and_a_real_sequence():
    """``|`` is the spelling to write; the other two are tolerated, not advertised.

    A comma list arrives as a *tuple*: CTML runs ``ast.literal_eval`` on an attribute
    with no type suffix, so ``ids="0,1"`` never reaches the command as the string the
    caller typed. ``|`` is not a Python literal, so it survives as written. Both
    spellings must mean the same thing, and a caller that passes a real sequence must
    not blow up either.
    """
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("http://a",))
        await runtime.execute_command("open", args=("http://b",))
        a, b = [f["item"]["id"] for f in rec.of("open")]
        for raw in (f"{b}|{a}", f"{b},{a}", (b, a)):
            await runtime.execute_command("arrange", args=(raw,), kwargs={"group": "code"})
            assert rec.of("state")[-1]["ids"] == [b, a], raw


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
async def test_focus_emits_a_focus_frame_for_any_pooled_item():
    """Focus needs no group: a desktop item is enough (that is the whole point)."""
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("http://a",))
        a = rec.of("open")[-1]["item"]["id"]
        await runtime.execute_command("focus", args=(a,))
        assert rec.of("focus")[-1]["id"] == a


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
async def test_every_gesture_carries_an_id_and_a_place():
    """The drawing tools are one family: each frame names a block so it can be
    taken back, and each end is an address (item + region), never a coordinate."""
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("http://a",))
        await runtime.execute_command("open", args=("http://b",))
        a, b = [f["item"]["id"] for f in rec.of("open")]
        for name, args, kwargs in (
            ("mark", (a,), {"region": "left"}),
            ("box", (a,), {"region": "top"}),
            ("circle", (b,), {"region": "bottom"}),
            ("arrow", (a, b), {"src_region": "right", "dst_region": "left"}),
            ("danmaku", ("hello",), {"motion": "orbit"}),
            ("paint", ("ctx.fillRect(box.x, box.y, 4, 4)",), {}),
        ):
            await runtime.execute_command(
                name, args=args, kwargs={**kwargs, "duration": 0.01}
            )
        frames = rec.of("veil")
        assert [f["gesture"] for f in frames] == [
            "mark", "box", "circle", "arrow", "danmaku", "paint",
        ]
        ids = [f["id"] for f in frames]
        assert len(set(ids)) == len(ids), "each block needs its own handle"
        assert frames[5]["script"].startswith("ctx.fillRect")
        arrow = frames[3]
        assert (arrow["src"], arrow["src_region"]) == (a, "right")
        assert (arrow["dst"], arrow["dst_region"]) == (b, "left")


@pytest.mark.asyncio
async def test_content_says_it_on_the_screen_instead_of_in_the_voice():
    """Character data on this channel is a bullet, not a caption.

    The same door `say` uses on `__main__`: a sentence goes straight in and the
    defaults decide the rest. Everything a caller might want to steer — direction,
    size, lifetime — stays `danmaku`'s job, so this one keeps a single argument.
    """
    chan, rec = _channel()

    async with chan.bootstrap() as runtime:
        async def chunks():
            yield "一"
            yield "句话"

        task = asyncio.ensure_future(
            runtime.execute_command("__content__", args=(chunks(),))
        )
        for _ in range(100):
            await asyncio.sleep(0.01)
            if rec.of("veil"):
                break
        frames = rec.of("veil")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert len(frames) == 1, "one scope of text is one bullet, not one per chunk"
    assert frames[0]["gesture"] == "danmaku"
    assert frames[0]["content"] == "一句话"
    assert frames[0]["duration"] == 6.0


@pytest.mark.asyncio
async def test_content_of_nothing_draws_nothing():
    """Whitespace is not a bullet: an empty scope stays silent instead of sending
    a blank line flying across the stage."""
    chan, rec = _channel()

    async with chan.bootstrap() as runtime:
        async def blank():
            yield "  "
            yield "\n"

        assert await runtime.execute_command("__content__", args=(blank(),)) is None

    assert rec.of("veil") == []


@pytest.mark.asyncio
async def test_veil_reads_back_what_is_still_drawn():
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("http://a",))
        item_id = rec.of("open")[-1]["item"]["id"]
        assert "clear" in await runtime.execute_command("veil")

        # 手势并行放着，读也不该排在后面：一个在放，另一个照发，arrange 照落。
        arrow = asyncio.ensure_future(
            runtime.execute_command("arrow", args=(item_id, item_id), kwargs={"duration": 5})
        )
        box = asyncio.ensure_future(
            runtime.execute_command("box", args=(item_id,), kwargs={"duration": 5})
        )
        await asyncio.sleep(0.05)
        await runtime.execute_command("arrange", args=(item_id,), kwargs={"group": "g"})
        assert model.groups() == ["g"]
        line = await runtime.execute_command("veil")
        assert "arrow" in line and "box" in line

        # 收也是一个动作：给 clear 一个时长，到点它自己收；被取消时 finally 也收。
        assert "cleared (2)" in await runtime.execute_command("clear_veil")
        assert "clear" in await runtime.execute_command("veil")
        await runtime.execute_command("box", args=(item_id,), kwargs={"duration": 5})
        await runtime.execute_command("clear_veil", kwargs={"duration": 0.05})
        assert "clear" in await runtime.execute_command("veil")

        for t in (arrow, box):
            t.cancel()
            try:
                await t
            except BaseException:      # 收尾：这里只关心任务不再挂着
                pass


@pytest.mark.asyncio
async def test_a_region_can_be_a_measured_fraction_of_the_item():
    """五档粗区域之外，还能给"这块砖里的哪一条"——比例矩形。比例而不是像素：
    砖会重排、窗口会缩放，像素的正确性只在画下那一瞬成立。"""
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("http://a",))
        item_id = rec.of("open")[-1]["item"]["id"]
        await runtime.execute_command(
            "box", args=(item_id, "0,0,1,0.035"), kwargs={"duration": 0.05}
        )
        assert rec.of("veil")[-1]["region"] == "0,0,1,0.035"
        # 写字符串、还是被 CTML 解成 list，都收 —— 注解不是权威（T1）。
        await runtime.execute_command(
            "box", args=(item_id, [0, 0, 1, 0.035]), kwargs={"duration": 0.05}
        )
        assert rec.of("veil")[-1]["region"] == "0,0,1,0.035"
        for bad in ("0,0,1", "0,0,1,1.5", "0,0,0,0.1", "left,top"):
            with pytest.raises(Exception):
                await runtime.execute_command(
                    "box", args=(item_id, bad), kwargs={"duration": 0.05}
                )


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
async def test_raise_emits_state_with_the_new_order():
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("http://a",))
        await runtime.execute_command("open", args=("http://b",))
        a, b = [f["item"]["id"] for f in rec.of("open")]
        # 新开的落在最前 —— 池序是状态，就随 state 帧发出去。
        assert rec.of("state")[-1]["order"] == [b, a]
        receipt = await runtime.execute_command("raise", args=(a,))
        assert rec.of("state")[-1]["order"] == [a, b]
        assert a in str(receipt)


@pytest.mark.asyncio
async def test_raise_rejects_an_unknown_id():
    model = ScreenModel()
    chan, _ = _channel(model)
    async with chan.bootstrap() as runtime:
        with pytest.raises(Exception):
            await runtime.execute_command("raise", args=("nope",))


@pytest.mark.asyncio
async def test_notice_carries_the_pool_order():
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("http://a",))
        await runtime.execute_command("open", args=("http://b",))
        a, b = [f["item"]["id"] for f in rec.of("open")]
        await runtime.refresh_metas()
        assert runtime.self_meta().named_notices["order"] == f"{b} {a}"
        await runtime.execute_command("raise", args=(a,))
        await runtime.refresh_metas()
        assert runtime.self_meta().named_notices["order"] == f"{a} {b}"


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
async def test_notices_separate_adopted_and_opened_windows():
    model = ScreenModel()
    model.adopt("cell/a/webview", "http://x", label="Terminal")
    chan, _ = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("http://a", "hand"))
        await runtime.refresh_metas()
        notices = runtime.self_meta().named_notices
        assert "adopted" in notices
        assert "opened" in notices
        assert "Terminal" in notices["adopted"]
        assert "hand" in notices["opened"]


@pytest.mark.asyncio
async def test_mock_scene_populates_groups():
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("mock_scene")
        assert len(model.groups()) >= 3
        assert rec.types()[-1] == "snapshot"


@pytest.mark.asyncio
async def test_notice_carries_the_human_side_rail_state():
    """侧栏的展开/收起是排布的前提：舞台可用区 = 视口 − 栏宽。它由页面上报，
    所以它必须真的走到 notice 里；没上报时不许编一个默认值出来。"""
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert "rail" not in runtime.self_meta().named_notices

        rec.rail = "collapsed"
        await runtime.refresh_metas()
        assert runtime.self_meta().named_notices["rail"] == "collapsed"

        rec.rail = "expanded"
        await runtime.refresh_metas()
        assert runtime.self_meta().named_notices["rail"] == "expanded"


@pytest.mark.asyncio
async def test_notice_carries_the_view_and_who_moved_it():
    """人的视图动作是**状态**，不是一条可能丢的消息。

    所以它必须出现在 notice 上：``focus`` 说看的是哪一个，``view_by`` 说是谁动的。
    两者都是 named 片段，且没人动过时不许编默认值 —— 与 ``rail`` 同一条规矩。
    """
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("http://a",))
        a = rec.of("open")[-1]["item"]["id"]

        await runtime.refresh_metas()
        notices = runtime.self_meta().named_notices
        assert "focus" not in notices            # 还没人看
        assert "view_by" not in notices          # 也还没人动

        await runtime.execute_command("focus", args=(a,))
        await runtime.refresh_metas()
        notices = runtime.self_meta().named_notices
        assert notices["focus"] == f"#{a}"
        assert notices["view_by"] == "model"     # 命令是我下的


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
async def test_a_numeric_handle_reaches_the_store_as_a_string():
    """CTML parses attribute values as literals, so focus(id="2") arrives as 2.

    Handles are strings in the store, so the command normalizes at its edge
    instead of making every caller remember :str — the same family as
    ids="0,1" and item="2". Before this, a numeric handle passed validation and
    then missed the lookup, which read as "no item 2" while item 2 was there.
    """
    model = ScreenModel()
    chan, rec = _channel(model)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("http://a",))
        a = rec.of("open")[-1]["item"]["id"]
        await runtime.execute_command("focus", args=(int(a),))
        assert rec.of("focus")[-1]["id"] == a
        await runtime.execute_command("dismiss", args=(int(a),))
        assert rec.of("state"), "dismiss on a numeric handle must reach the store"
