import asyncio
import threading
import time

import pytest

from ghoshell_moss.tools.future_router import (
    FutureRouter,
    wait_future,
)


def test_create_returns_id_and_future_and_lists_pending():
    router = FutureRouter()
    fid, fut = router.create("hello")
    assert isinstance(fid, str) and len(fid) > 0
    assert not fut.done()

    pending = router.list_pending()
    assert len(pending) == 1
    assert pending[0].id == fid
    assert pending[0].arguments == "hello"
    assert router.list_done() == []


def test_resolve_archives_and_delivers_result():
    router = FutureRouter()
    fid, fut = router.create("ping")
    assert router.resolve(fid, "pong") is True
    assert fut.result(timeout=0.1) == "pong"

    assert router.list_pending() == []
    done = router.list_done()
    assert len(done) == 1 and done[0].id == fid


def test_reject_raises_and_archives():
    router = FutureRouter()
    fid, fut = router.create("x")
    assert router.reject(fid, "boom") is True

    with pytest.raises(RuntimeError, match="boom"):
        fut.result(timeout=0.1)
    assert router.list_pending() == []
    assert len(router.list_done()) == 1


def test_cancel_marks_future_cancelled():
    router = FutureRouter()
    fid, fut = router.create("x")
    assert router.cancel(fid) is True
    assert fut.cancelled()
    assert router.list_pending() == []


def test_resolve_unknown_id_returns_false():
    router = FutureRouter()
    assert router.resolve("nope", "x") is False
    assert router.reject("nope", "x") is False
    assert router.cancel("nope") is False


def test_double_resolve_returns_false():
    router = FutureRouter()
    fid, _ = router.create("x")
    assert router.resolve(fid, "first") is True
    assert router.resolve(fid, "second") is False


def test_get_finds_pending_and_done_and_returns_none_for_unknown():
    router = FutureRouter()
    fid, fut = router.create("x")
    assert router.get(fid) is fut

    router.resolve(fid, "ok")
    assert router.get(fid) is fut  # still findable in done window

    assert router.get("unknown") is None


def test_max_done_evicts_oldest():
    router = FutureRouter(max_done=2)
    ids = []
    for i in range(4):
        fid, _ = router.create(f"a{i}")
        router.resolve(fid, "ok")
        ids.append(fid)

    done = router.list_done()
    assert [e.id for e in done] == ids[2:]  # only last 2 survive
    assert router.get(ids[0]) is None
    assert router.get(ids[3]) is not None


def test_on_create_fires_with_entry_outside_lock():
    router = FutureRouter()
    seen: list = []

    def cb(entry):
        # Re-entering router methods must not deadlock — proves lock is released.
        snapshot = router.list_pending()
        seen.append((entry.id, entry.arguments, len(snapshot)))

    router.on_create(cb)
    fid, _ = router.create("payload")

    assert seen == [(fid, "payload", 1)]


def test_on_create_bad_callback_is_isolated():
    router = FutureRouter()
    log = []

    def bad(_entry):
        raise ValueError("intentional")

    def good(entry):
        log.append(entry.id)

    router.on_create(bad)
    router.on_create(good)

    fid, _ = router.create("x")
    assert log == [fid]  # good callback still fires


def test_direct_set_result_still_archives_via_done_callback():
    router = FutureRouter()
    fid, fut = router.create("x")
    fut.set_result("direct")

    # add_done_callback runs synchronously in the setter's thread.
    assert router.list_pending() == []
    assert any(e.id == fid for e in router.list_done())


def test_cross_thread_resolve_delivers_to_async_caller():
    router = FutureRouter()

    async def run():
        fid, fut = router.create("req")

        def worker():
            time.sleep(0.02)
            router.resolve(fid, "from-thread")

        threading.Thread(target=worker, daemon=True).start()
        return await wait_future(fut, timeout=1.0)

    assert asyncio.run(run()) == "from-thread"


def test_call_sugar_end_to_end():
    router = FutureRouter()

    def executor(entry):
        # on_create fires synchronously in caller's thread; settle via a worker
        # so the awaiter actually has to wait.
        def worker():
            router.resolve(entry.id, f"echo:{entry.arguments}")

        threading.Thread(target=worker, daemon=True).start()

    router.on_create(executor)

    async def run():
        return await router.call("hi", timeout=1.0)

    assert asyncio.run(run()) == "echo:hi"


def test_wait_future_timeout_cancels_underlying_future():
    router = FutureRouter()

    async def run():
        fid, fut = router.create("slow")
        with pytest.raises(asyncio.TimeoutError):
            await wait_future(fut, timeout=0.02)
        return fid, fut

    fid, fut = asyncio.run(run())
    assert fut.cancelled()
    # executor side sees the abandonment via resolve returning False
    assert router.resolve(fid, "too-late") is False


def test_wait_future_external_cancel_propagates():
    router = FutureRouter()

    async def run():
        fid, fut = router.create("x")
        task = asyncio.create_task(wait_future(fut))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return fut

    fut = asyncio.run(run())
    assert fut.cancelled()
