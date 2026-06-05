"""Protocol-level TopicWindow tests — parametrized over QueueBased + Zenoh."""
import asyncio
import threading
import pytest
from ghoshell_moss.core.topic.suite_for_test import (
    TopicServiceSuite, QueueTopicServiceSuite,
)
from ghoshell_moss.core.concepts.topic import TopicService, ErrorTopic

try:
    from ghoshell_moss.host.topics.zenoh_topics import ZenohTopicServiceSuite
    _zenoh_available = True
except ImportError:
    ZenohTopicServiceSuite = None
    _zenoh_available = False

_topic_suites: list[TopicServiceSuite] = [QueueTopicServiceSuite()]
if _zenoh_available:
    _topic_suites.append(ZenohTopicServiceSuite())


@pytest.fixture(params=_topic_suites, ids=lambda s: s.name())
def service(request):
    suite: TopicServiceSuite = request.param
    svc = suite.create_service(sender="test_sender")
    yield svc
    suite.cleanup()


# ── helpers ──────────────────────────────────────────────


async def _new_window(service: TopicService, max_size: int = 10):
    """Create a window and wait for its subscription to become active."""
    win = service.create_window_for(ErrorTopic, max_size=max_size)
    await win.wait_started()
    return win


def _wait_callback(win, *, count: int, timeout: float = 2.0):
    """
    Register an on_change callback that sets a threading.Event after
    *count* arrivals. Returns (event, remove_handle).

    Use in tests to deterministically wait for N items instead of sleep().
    """
    event = threading.Event()
    counter = [0]  # mutable container for closure

    def _cb(w):
        counter[0] = len(w)
        if counter[0] >= count:
            event.set()

    remove = win.on_change(_cb)

    return event, remove


async def _await_event(event: threading.Event, timeout: float = 2.0):
    """Await a threading.Event without blocking the event loop."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, event.wait, timeout)


# ── protocol tests ──────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.usefixtures("service")
class TestTopicWindowProtocol:

    async def test_window_values_basic(self, service: TopicService):
        """values() returns items in order: oldest first."""
        async with service:
            win = await _new_window(service)
            event, remove = _wait_callback(win, count=3)

            service.pub(ErrorTopic(errmsg="a"))
            service.pub(ErrorTopic(errmsg="b"))
            service.pub(ErrorTopic(errmsg="c"))
            await _await_event(event)

            remove()
            vs = win.values()
            assert len(vs) == 3
            assert [v.errmsg for v in vs] == ["a", "b", "c"]

    async def test_window_max_size_eviction(self, service: TopicService):
        """Oldest items are evicted when max_size is exceeded."""
        async with service:
            win = await _new_window(service, max_size=3)
            event, remove = _wait_callback(win, count=6)

            for c in "abcdef":
                service.pub(ErrorTopic(errmsg=c))
            await _await_event(event)

            remove()
            vs = win.values()
            assert len(vs) == 3
            assert [v.errmsg for v in vs] == ["d", "e", "f"]

    async def test_window_len(self, service: TopicService):
        """__len__ reflects current item count."""
        async with service:
            win = await _new_window(service)
            event, remove = _wait_callback(win, count=3)

            assert len(win) == 0
            for _ in range(3):
                service.pub(ErrorTopic(errmsg="x"))
            await _await_event(event)

            remove()
            assert len(win) == 3

    async def test_window_changed_at_updates(self, service: TopicService):
        """changed_at() updates on every topic arrival."""
        async with service:
            win = await _new_window(service)
            event, remove = _wait_callback(win, count=1)

            assert win.changed_at() == 0.0
            service.pub(ErrorTopic(errmsg="first"))
            await _await_event(event)

            t1 = win.changed_at()
            assert t1 > 0

            event2, remove2 = _wait_callback(win, count=2)
            service.pub(ErrorTopic(errmsg="second"))
            await _await_event(event2)

            t2 = win.changed_at()
            assert t2 > t1
            remove()
            remove2()

    async def test_window_on_change_immediate(self, service: TopicService):
        """on_change with default flags fires on each arrival."""
        async with service:
            win = await _new_window(service)
            event, remove = _wait_callback(win, count=2)

            service.pub(ErrorTopic(errmsg="a"))
            service.pub(ErrorTopic(errmsg="b"))
            await _await_event(event)

            remove()
            assert len(win) >= 2

    async def test_window_on_change_remove(self, service: TopicService):
        """Returned handle unregisters the callback."""
        async with service:
            win = await _new_window(service)
            event, remove = _wait_callback(win, count=1)

            service.pub(ErrorTopic(errmsg="a"))
            await _await_event(event)
            assert len(win) == 1

            remove()
            # Second publish — callback should NOT fire since we removed it
            event2 = threading.Event()

            def _cb2(w):
                event2.set()

            win.on_change(_cb2)
            service.pub(ErrorTopic(errmsg="b"))
            await _await_event(event2)
            assert len(win) == 2

    async def test_window_closes_with_service(self, service: TopicService):
        """Window stops receiving after service is closed."""
        win = None
        async with service:
            win = await _new_window(service)
            event, remove = _wait_callback(win, count=1)
            service.pub(ErrorTopic(errmsg="before_close"))
            await _await_event(event)
            remove()
            assert len(win) == 1

        # service is closed now — window still readable
        assert len(win) == 1

    async def test_values_returns_copy(self, service: TopicService):
        """values() returns a copy — mutating it does not affect the window."""
        async with service:
            win = await _new_window(service)
            event, remove = _wait_callback(win, count=1)

            service.pub(ErrorTopic(errmsg="original"))
            await _await_event(event)
            remove()

            snapshot = win.values()
            snapshot.clear()
            assert len(win) == 1

    async def test_window_max_size_one(self, service: TopicService):
        """max_size=1 always keeps only the latest item."""
        async with service:
            win = await _new_window(service, max_size=1)
            event, remove = _wait_callback(win, count=3)

            service.pub(ErrorTopic(errmsg="first"))
            service.pub(ErrorTopic(errmsg="second"))
            service.pub(ErrorTopic(errmsg="third"))
            await _await_event(event)

            remove()
            vs = win.values()
            assert len(vs) == 1
            assert vs[0].errmsg == "third"

    async def test_window_multiple_windows_same_topic(self, service: TopicService):
        """Two windows on the same topic each receive independently."""
        async with service:
            win_a = await _new_window(service, max_size=5)
            win_b = await _new_window(service, max_size=5)
            event, remove = _wait_callback(win_a, count=2)

            service.pub(ErrorTopic(errmsg="x"))
            service.pub(ErrorTopic(errmsg="y"))
            await _await_event(event)

            remove()
            assert len(win_a) == 2
            assert len(win_b) == 2
            assert [v.errmsg for v in win_a.values()] == ["x", "y"]

    async def test_changed_at_starts_at_zero(self, service: TopicService):
        """changed_at() is 0.0 before any topics arrive."""
        async with service:
            win = await _new_window(service)
            assert win.changed_at() == 0.0
