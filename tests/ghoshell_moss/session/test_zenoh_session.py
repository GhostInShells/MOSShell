"""MossSessionWithZenoh + SimpleOutputBuffer 基础单测.

锚定 session 核心契约:
- SimpleOutputBuffer: 同 role 合并, maxsize 淘汰, 快照隔离
- MossSessionWithZenoh: output/signal 序列化往返, 构造期校验, output_buffer 桥接
"""

import tempfile
import threading
import time
from pathlib import Path

import pytest
from ghoshell_moss.depends import depend_zenoh

depend_zenoh()
import zenoh

from ghoshell_moss.core.session.zenoh_session import SimpleOutputBuffer, MossSessionWithZenoh
from ghoshell_moss.core.blueprint.session import OutputItem
from ghoshell_moss.core.topic.zenoh_topics import ZenohTopicService
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
            session_storage=storage,
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
