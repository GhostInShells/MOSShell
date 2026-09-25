import asyncio

import pytest

from push_node.session import SessionState
from push_node.store import PushStore


def test_request_creates_a_pending_session():
    store = PushStore()
    session = store.request("screen", label="desk")
    assert session.state == SessionState.PENDING
    assert session.owner == "model"


@pytest.mark.asyncio
async def test_settle_accept_resolves_the_waiter():
    store = PushStore()
    session = store.request("screen")

    async def park():
        return await store.waiter(session.id)

    task = asyncio.create_task(park())
    await asyncio.sleep(0)
    assert not task.done(), "waiter parks until a verdict lands"

    assert store.settle(session.id, "accept") is True
    assert await task == "accept"


def test_settle_is_debounced_once_decided():
    store = PushStore()
    session = store.request("screen")
    store.settle(session.id, "deny")
    store.set_state(session.id, SessionState.DENIED)
    assert store.settle(session.id, "accept") is False, "a decided session cannot be re-verdict"


def test_owner_distinguishes_the_two_faces():
    store = PushStore()
    model = store.request("screen", owner="model")
    human = store.request("camera", owner="human")
    assert model.owner != human.owner


def test_a_fresh_store_holds_no_sessions():
    # The store is in-memory only: a restart is a fresh store, so a live stream
    # can never be resurrected across a node restart.
    assert PushStore().sessions() == []
