"""MossSessionWithZenoh + SimpleOutputBuffer 基础单测.

锚定 session 核心契约:
- SimpleOutputBuffer: 同 role 合并, maxsize 淘汰, 快照隔离
- MossSessionWithZenoh: output/signal 序列化往返, 构造期校验, output_buffer 桥接
- stream 协议: pub/sub roundtrip, StreamSubscriber 生命周期, logos 便捷方法
"""

import asyncio
import tempfile
import threading
import time
from pathlib import Path

import pytest
from ghoshell_moss.depends import depend_zenoh

depend_zenoh()
import zenoh

from ghoshell_moss.host.session.zenoh_session import SimpleOutputBuffer, MossSessionWithZenoh
from ghoshell_moss.core.blueprint.session import OutputItem, Sample
from ghoshell_moss.host.topics.zenoh_topics import ZenohTopicService
from ghoshell_moss.message import Message
from ghoshell_moss.contracts.logger import get_moss_logger
from ghoshell_moss.contracts.workspace import LocalStorage


# ── SimpleOutputBuffer ──────────────────────────────────


class TestSimpleOutputBuffer:
    """纯单测, 零外部依赖."""

    def test_same_role_merge(self):
        buf = SimpleOutputBuffer(maxsize=10)
        buf.add_output(OutputItem.new("task", log="cmd_a"))
        buf.add_output(OutputItem.new("task", log="cmd_b"))

        items = list(buf.values())
        assert len(items) == 1
        assert items[0].log == "cmd_a"

    def test_cross_role_no_merge(self):
        buf = SimpleOutputBuffer(maxsize=10)
        buf.add_output(OutputItem.new("task", log="task_done"))
        buf.add_output(OutputItem.new("error", log="oops"))

        items = list(buf.values())
        assert len(items) == 2
        assert items[0].role == "task"
        assert items[1].role == "error"

    def test_messages_extend_on_same_role(self):
        buf = SimpleOutputBuffer(maxsize=10)
        m1 = Message.new().with_content("a")
        m2 = Message.new().with_content("b")
        buf.add_output(OutputItem.new("task", m1))
        buf.add_output(OutputItem.new("task", m2))

        items = list(buf.values())
        assert len(items) == 1
        assert len(items[0].messages) == 2

    def test_maxsize_eviction(self):
        buf = SimpleOutputBuffer(maxsize=3)
        for i in range(5):
            buf.add_output(OutputItem.new(str(i)))

        items = list(buf.values())
        assert len(items) == 3
        assert items[0].role == "2"
        assert items[-1].role == "4"

    def test_values_is_snapshot(self):
        buf = SimpleOutputBuffer(maxsize=10)
        buf.add_output(OutputItem.new("first"))

        snapshot = list(buf.values())
        buf.add_output(OutputItem.new("second"))

        assert len(snapshot) == 1

    def test_updated_at(self):
        buf = SimpleOutputBuffer(maxsize=10)
        assert buf.updated_at() == 0.0
        buf.add_output(OutputItem.new("x"))
        assert buf.updated_at() > 0.0

    def test_closed_flag(self):
        buf = SimpleOutputBuffer(maxsize=10)
        buf.close()
        assert buf.is_closed()

    def test_empty_buffer_values(self):
        buf = SimpleOutputBuffer(maxsize=10)
        assert list(buf.values()) == []


# ── MossSessionWithZenoh ────────────────────────────────


class TestSessionWithZenoh:
    """需要本地 zenoh router."""

    @staticmethod
    def _new_session(
        zenoh_sess: zenoh.Session,
        scope: str = "test_session_scope",
    ) -> MossSessionWithZenoh:
        tmp = tempfile.mkdtemp()
        storage = LocalStorage(Path(tmp))
        topics = ZenohTopicService(
            session_scope=scope,
            session=zenoh_sess,
            address="test",
        )
        return MossSessionWithZenoh(
            session_scope=scope,
            session_root_storage=storage,
            session_tmp_root_storage=storage.sub_storage('tmp'),
            logger=get_moss_logger(),
            zenoh_session=zenoh_sess,
            topic_service=topics,
        )

    def test_construct_rejects_closed_zenoh(self):
        with zenoh.open(zenoh.Config()) as z:
            pass  # z 在 with 退出后已 close

        with pytest.raises(RuntimeError, match="closed"):
            self._new_session(z)

    def test_output_roundtrip(self):
        """output() → zenoh → on_output listener 收到 OutputItem."""
        received = []
        done = threading.Event()

        with zenoh.open(zenoh.Config()) as z:
            sess = self._new_session(z)
            sess.on_output(lambda item: (received.append(item), done.set()))
            sess.output("logos", Message.new().with_content("hello"), log="greeting")

            assert done.wait(timeout=2.0), "output roundtrip timed out"

        assert len(received) == 1
        item = received[0]
        assert item.role == "logos"
        assert item.log == "greeting"
        assert len(item.messages) == 1
        assert item.messages[0].to_content_string() == "hello"

    def test_output_log_only(self):
        """output() 只带 log 不带 messages."""
        received = []
        done = threading.Event()

        with zenoh.open(zenoh.Config()) as z:
            sess = self._new_session(z)
            sess.on_output(lambda item: (received.append(item), done.set()))
            sess.output("task", log="say done")

            assert done.wait(timeout=2.0)

        assert len(received) == 1
        assert received[0].role == "task"
        assert received[0].log == "say done"
        assert received[0].messages == []

    def test_signal_roundtrip(self):
        """add_signal() → zenoh → on_signal callback 收到 Signal."""
        received = []
        done = threading.Event()

        with zenoh.open(zenoh.Config()) as z:
            sess = self._new_session(z)
            sess.on_signal(lambda sig: (received.append(sig), done.set()))
            sess.add_input_signal("percept", description="test signal")

            assert done.wait(timeout=2.0), "signal roundtrip timed out"

        assert len(received) == 1
        sig = received[0]
        assert sig.description == "test signal"
        assert sig.name == "input"
        assert len(sig.messages) == 1
        assert sig.messages[0].to_content_string() == "percept"

    def test_signal_no_duplicate_delivery(self):
        """add_signal() 只触发 on_signal 恰好一次 — 排除 zenoh 自回环重复投递."""
        received = []
        first = threading.Event()

        with zenoh.open(zenoh.Config()) as z:
            sess = self._new_session(z)
            sess.on_signal(lambda sig: (received.append(sig), first.set()))
            sess.add_input_signal("once", description="dup check")

            assert first.wait(timeout=2.0), "signal not received at all"
            # 等一个足够长的窗口, 让潜在的第二次投递有机会到达
            time.sleep(0.3)

        assert len(received) == 1, (
            f"expected exactly 1 signal delivery, got {len(received)} — "
            f"zenoh may be delivering put() back to same-session subscriber twice"
        )

    def test_output_buffer_bridge(self):
        """output_buffer() 自动桥接 on_output, values() 返回快照."""
        with zenoh.open(zenoh.Config()) as z:
            sess = self._new_session(z)
            buf = sess.output_buffer(maxsize=20)

            sess.output("task", log="step 1")
            sess.output("task", log="step 2")
            time.sleep(0.05)  # zenoh 异步分发

            items = list(buf.values())
            assert len(items) >= 1  # 同 role 合并到至少一个 item

            buf.close()
            assert buf.is_closed()

    def test_multiple_output_listeners(self):
        """多个 on_output listener 都收到同样的 OutputItem."""
        hits = []
        done = threading.Event()
        counter = [0]

        def make_listener():
            def fn(item):
                hits.append(item)
                counter[0] += 1
                if counter[0] >= 2:
                    done.set()

            return fn

        with zenoh.open(zenoh.Config()) as z:
            sess = self._new_session(z)
            sess.on_output(make_listener())
            sess.on_output(make_listener())
            sess.output("log", log="broadcast")

            assert done.wait(timeout=2.0)

        assert len(hits) == 2
        assert hits[0].log == hits[1].log == "broadcast"

    def test_multiple_signal_callbacks(self):
        """多个 on_signal callback 都收到同样的 Signal."""
        hits = []
        done = threading.Event()
        counter = [0]

        def make_cb():
            def fn(sig):
                hits.append(sig)
                counter[0] += 1
                if counter[0] >= 2:
                    done.set()

            return fn

        with zenoh.open(zenoh.Config()) as z:
            sess = self._new_session(z)
            sess.on_signal(make_cb())
            sess.on_signal(make_cb())
            sess.add_input_signal("event", description="multicast")

            assert done.wait(timeout=2.0)

        assert len(hits) == 2
        assert hits[0].description == hits[1].description == "multicast"

    def test_on_output_before_output(self):
        """先注册 on_output 再 output — listener 在 subscriber 之后注册仍能收到."""
        with zenoh.open(zenoh.Config()) as z:
            sess = self._new_session(z)

            received = []
            done = threading.Event()
            sess.on_output(lambda item: (received.append(item), done.set()))
            sess.output("system", log="late listener")

            assert done.wait(timeout=2.0)
            assert len(received) == 1

    # ── stream 协议 base ─────────────────────────

    def test_stream_key_expr(self):
        with zenoh.open(zenoh.Config()) as z:
            sess = self._new_session(z)
            assert sess.stream_key_expr("ch") == "MOSS/test_session_scope/streams/ch"
            assert sess.stream_key_expr("/ch/") == "MOSS/test_session_scope/streams/ch"

    def test_is_running(self):
        with zenoh.open(zenoh.Config()) as z:
            sess = self._new_session(z)
            assert sess.is_running() is True

        # zenoh 关闭后 is_running 为 False
        with zenoh.open(zenoh.Config()) as z:
            sess = self._new_session(z)
        assert sess.is_running() is False

    def test_self_explain(self):
        with zenoh.open(zenoh.Config()) as z:
            sess = self._new_session(z)
            text = sess.self_explain()
            assert "test_session_scope" in text
            assert "zenoh" in text
            assert "streams" in text

    def test_sub_stream_and_pub_stream_delta(self):
        received = []
        done = threading.Event()

        with zenoh.open(zenoh.Config()) as z:
            sess = self._new_session(z)

            def on_sample(s: Sample) -> None:
                received.append(s)
                done.set()

            stop = sess.sub_stream("test/ch", on_sample)
            sess.pub_stream_delta("test/ch", b"hello")

            assert done.wait(timeout=2.0), "stream pub/sub timed out"
            assert len(received) == 1
            assert received[0].relative_key == "test/ch"
            assert received[0].payload == b"hello"

            stop()

    def test_sub_stream_stop_handle(self):
        received = []

        with zenoh.open(zenoh.Config()) as z:
            sess = self._new_session(z)

            def on_sample(s: Sample) -> None:
                received.append(s)

            stop = sess.sub_stream("test/stop", on_sample)
            stop()
            sess.pub_stream_delta("test/stop", b"ghost")
            time.sleep(0.1)

            assert len(received) == 0

    def test_sub_stream_rejects_when_session_closed(self):
        with zenoh.open(zenoh.Config()) as z:
            sess = self._new_session(z)
        # zenoh 已关闭
        with pytest.raises(RuntimeError):
            sess.sub_stream("x", lambda s: None)


# ── stream 异步测试 ──────────────────────────────


class TestSessionStreamAsync:
    """async stream 测试, 需要本地 zenoh router.

    stream 是广播模型: 先 sub 再 pub 才能收到数据. 测试用 create_task 并行,
    consumer 先启动阻塞等待, publisher 后发送, wait_for 防死锁.
    """

    @staticmethod
    def _new_session(zenoh_sess, scope="test_async_scope"):
        tmp = tempfile.mkdtemp()
        storage = LocalStorage(Path(tmp))
        topics = ZenohTopicService(
            session_scope=scope,
            session=zenoh_sess,
            address="test",
        )
        return MossSessionWithZenoh(
            session_scope=scope,
            session_root_storage=storage,
            session_tmp_root_storage=storage.sub_storage('tmp'),
            logger=get_moss_logger(),
            zenoh_session=zenoh_sess,
            topic_service=topics,
        )

    # ── get_stream 生命周期 ────────────────────────

    @pytest.mark.asyncio
    async def test_get_stream_lifecycle(self):
        with zenoh.open(zenoh.Config()) as z:
            sess = self._new_session(z)
            stream = sess.get_stream("test/life")

            async with stream:
                assert stream.full_key() == "MOSS/test_async_scope/streams/test/life"
                assert stream.relative_key() == "test/life"

    @pytest.mark.asyncio
    async def test_get_stream_iteration(self):
        """先进入 stream, 再 pub, 用 wait_for 取数据."""
        with zenoh.open(zenoh.Config()) as z:
            sess = self._new_session(z)
            stream = sess.get_stream("test/iter")

            async with stream:
                # pub 在 subscribe 之后
                sess.pub_stream_delta("test/iter", b"msg1")
                sess.pub_stream_delta("test/iter", b"msg2")

                s1 = await asyncio.wait_for(stream.__anext__(), timeout=2.0)
                s2 = await asyncio.wait_for(stream.__anext__(), timeout=2.0)

                assert s1.payload == b"msg1"
                assert s2.payload == b"msg2"

    # ── pub_logos / get_logos ──────────────────────

    @pytest.mark.asyncio
    async def test_pub_logos_get_logos(self):
        """并行: consumer task 先订阅阻塞, 主 task 再 pub, wait_for 收结果."""
        with zenoh.open(zenoh.Config()) as z:
            sess = self._new_session(z)
            sid = sess.session_id

            async def consume() -> list[str]:
                results = []
                async for text in sess.get_logos(session_id=sid):
                    results.append(text)
                    if len(results) >= 3:
                        break
                return results

            consumer = asyncio.create_task(consume())
            await asyncio.sleep(0.05)  # 让 subscriber 注册生效

            sess.pub_logos("hello ", "logos", session_id=sid)
            sess.pub_logos(" world", session_id=sid)

            results = await asyncio.wait_for(consumer, timeout=5.0)
            assert "".join(results) == "hello logos world"

    @pytest.mark.asyncio
    async def test_pub_logos_default_session_id(self):
        """pub_logos 不传 session_id 时默认使用当前 session id."""
        with zenoh.open(zenoh.Config()) as z:
            sess = self._new_session(z)

            async def consume() -> list[str]:
                results = []
                async for text in sess.get_logos():
                    results.append(text)
                    if len(results) >= 2:
                        break
                return results

            consumer = asyncio.create_task(consume())
            await asyncio.sleep(0.05)

            sess.pub_logos("hello", " world")

            results = await asyncio.wait_for(consumer, timeout=5.0)
            assert "".join(results) == "hello world"

    # ── 生命周期/并发 ──────────────────────────────

    @pytest.mark.asyncio
    async def test_stream_subscriber_stops_after_aexit(self):
        """__aexit__ 后 stream 不再收数据, 新 stream 可以独立订阅."""
        with zenoh.open(zenoh.Config()) as z:
            sess = self._new_session(z)

            # stream1: 收一条, 退出
            stream1 = sess.get_stream("test/exit")
            async with stream1:
                sess.pub_stream_delta("test/exit", b"msg1")
                s = await asyncio.wait_for(stream1.__anext__(), timeout=2.0)
                assert s.payload == b"msg1"

            # stream2: 独立订阅, 能收到新 pub 的数据
            stream2 = sess.get_stream("test/exit")
            async with stream2:
                sess.pub_stream_delta("test/exit", b"msg2")
                s = await asyncio.wait_for(stream2.__anext__(), timeout=2.0)
                assert s.payload == b"msg2"

    @pytest.mark.asyncio
    async def test_stream_multiple_subscribers(self):
        """同一 key 多个 subscriber 都收到同一条 pub."""
        with zenoh.open(zenoh.Config()) as z:
            sess = self._new_session(z)

            async def collect(n: int) -> list[bytes]:
                results = []
                stream = sess.get_stream("test/multi")
                async with stream:
                    for _ in range(n):
                        s = await asyncio.wait_for(stream.__anext__(), timeout=2.0)
                        results.append(s.payload)
                return results

            t1 = asyncio.create_task(collect(2))
            t2 = asyncio.create_task(collect(2))
            await asyncio.sleep(0.05)

            sess.pub_stream_delta("test/multi", b"A")
            sess.pub_stream_delta("test/multi", b"B")

            r1 = await asyncio.wait_for(t1, timeout=5.0)
            r2 = await asyncio.wait_for(t2, timeout=5.0)

            assert r1 == [b"A", b"B"]
            assert r2 == [b"A", b"B"]

    @pytest.mark.asyncio
    async def test_session_async_context(self):
        with zenoh.open(zenoh.Config()) as z:
            sess = self._new_session(z)
            async with sess:
                assert sess.is_running() is True

    # ── guard / 错误路径 ───────────────────────────

    @pytest.mark.asyncio
    async def test_anext_before_enter_raises(self):
        with zenoh.open(zenoh.Config()) as z:
            sess = self._new_session(z)
            stream = sess.get_stream("test/guard")
            with pytest.raises(RuntimeError, match="enter"):
                await stream.__anext__()

    @pytest.mark.asyncio
    async def test_aenter_twice_raises(self):
        with zenoh.open(zenoh.Config()) as z:
            sess = self._new_session(z)
            stream = sess.get_stream("test/double")
            async with stream:
                with pytest.raises(RuntimeError, match="already"):
                    async with stream:
                        pass

    @pytest.mark.asyncio
    async def test_aenter_rejects_closed_zenoh(self):
        with zenoh.open(zenoh.Config()) as z:
            sess = self._new_session(z)
            stream = sess.get_stream("test/closed")
        # zenoh 已 close, __aenter__ 应拒绝
        with pytest.raises(RuntimeError, match="closed"):
            async with stream:
                pass
