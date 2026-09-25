"""MockSession 冒烟验证 — signal / output / stream."""
import asyncio
import pytest
from ghoshell_moss.core.session.mock_session import MockSession
from ghoshell_moss.message import Message


# ── signal ──────────────────────────────────────

def test_signal_roundtrip():
    """add_signal → on_signal callback 同步触发, 历史记录写入."""
    sess = MockSession()
    received = []

    sess.on_signal(lambda s: received.append(s))
    sess.add_input_signal("hello", description="test signal")

    assert len(received) == 1
    assert received[0].description == "test signal"
    assert len(sess.signals) == 1
    assert sess.signals[0] is received[0]


def test_signal_multiple_callbacks():
    """多个 on_signal 都收到同一个 signal."""
    sess = MockSession()
    hits = []

    sess.on_signal(lambda s: hits.append(1))
    sess.on_signal(lambda s: hits.append(2))
    sess.add_input_signal("event")

    assert hits == [1, 2]


# ── output ──────────────────────────────────────

def test_output_roundtrip():
    """output → on_output 同步触发, 历史记录写入."""
    sess = MockSession()
    received = []

    sess.on_output(lambda item: received.append(item))
    sess.output("logos", Message.new().with_content("hello"), log="greeting")

    assert len(received) == 1
    assert received[0].role == "logos"
    assert received[0].log == "greeting"
    assert len(sess.outputs) == 1


def test_output_multiple_listeners():
    """多个 on_output 都收到同一个 OutputItem."""
    sess = MockSession()
    hits = []

    sess.on_output(lambda i: hits.append(i))
    sess.on_output(lambda i: hits.append(i))
    sess.output("task", log="done")

    assert len(hits) == 2
    assert hits[0].log == hits[1].log == "done"


def test_output_buffer_bridge():
    """output_buffer() 自动桥接 on_output."""
    sess = MockSession()
    buf = sess.output_buffer(maxsize=20)

    sess.output("task", log="step1")
    sess.output("task", log="step2")

    items = list(buf.values())
    assert len(items) >= 1
    buf.close()
    assert buf.is_closed()


# ── stream: sub_stream ──────────────────────────

def test_sub_stream_and_pub():
    """pub_stream_delta → sub_stream callback 同步触发."""
    sess = MockSession()
    received = []

    stop = sess.sub_stream("test/ch", lambda s: received.append(s))
    sess.pub_stream_delta("test/ch", b"hello")

    assert len(received) == 1
    assert received[0].relative_key == "test/ch"
    assert received[0].payload == b"hello"

    stop()
    sess.pub_stream_delta("test/ch", b"ghost")
    assert len(received) == 1  # stop 后不再收到


def test_stream_pub_history():
    """stream_pubs 记录所有 pub 的历史."""
    sess = MockSession()
    sess.pub_stream_delta("a", b"1")
    sess.pub_stream_delta("a", b"2")
    sess.pub_stream_delta("b", b"3")

    assert sess.stream_pubs["a"] == [b"1", b"2"]
    assert sess.stream_pubs["b"] == [b"3"]


# ── stream: get_stream async ────────────────────

@pytest.mark.asyncio
async def test_get_stream_iteration():
    """get_stream → async for 迭代消费 pub 的数据."""
    sess = MockSession()
    stream = sess.get_stream("test/iter")

    async with stream:
        sess.pub_stream_delta("test/iter", b"msg1")
        sess.pub_stream_delta("test/iter", b"msg2")

        s1 = await asyncio.wait_for(stream.__anext__(), timeout=1.0)
        s2 = await asyncio.wait_for(stream.__anext__(), timeout=1.0)

        assert s1.payload == b"msg1"
        assert s2.payload == b"msg2"


@pytest.mark.asyncio
async def test_get_stream_lifecycle():
    """get_stream 的 lifecycle 和 key 属性."""
    sess = MockSession()
    stream = sess.get_stream("test/life")

    async with stream:
        assert stream.full_key() == "MOSS/mock_scope/streams/test/life"
        assert stream.relative_key() == "test/life"


@pytest.mark.asyncio
async def test_get_stream_anext_before_enter_raises():
    """未 enter 就 anext 应抛 RuntimeError."""
    sess = MockSession()
    stream = sess.get_stream("test/guard")
    with pytest.raises(RuntimeError):
        await stream.__anext__()


@pytest.mark.asyncio
async def test_get_stream_stops_after_aexit():
    """aexit 后 stream 停止迭代."""
    sess = MockSession()
    stream = sess.get_stream("test/exit")

    async with stream:
        sess.pub_stream_delta("test/exit", b"only")
        s = await asyncio.wait_for(stream.__anext__(), timeout=1.0)
        assert s.payload == b"only"

    # stream 已退出, 新数据不会通过同一个 subscriber 拿到
    sess.pub_stream_delta("test/exit", b"never")


# ── lifecycle ───────────────────────────────────

@pytest.mark.asyncio
async def test_session_async_context():
    sess = MockSession()
    async with sess:
        assert sess.is_running() is True


# ── self_explain / stream_key_expr ───────────────

def test_self_explain():
    sess = MockSession()
    text = sess.self_explain()
    assert "mock (in-process)" in text
    assert sess.session_scope in text


def test_stream_key_expr():
    sess = MockSession()
    assert sess.stream_key_expr("ch") == "MOSS/mock_scope/streams/ch"
    assert sess.stream_key_expr("/ch/") == "MOSS/mock_scope/streams/ch"


# ── storage ─────────────────────────────────────

def test_storage_uses_tmpdir():
    sess = MockSession()
    storage = sess.storage
    assert storage is not None
    # 第二次访问返回同一个实例
    assert sess.storage.abspath() == storage.abspath()
