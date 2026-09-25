"""Protocol tests for the webview service kind.

Each test pins one promise the kind makes to a consumer:

- discovery carries the identity, and the address is the identity
- a condition published before anyone listened is still observable (snapshot)
- later changes arrive without being asked for (stream)
- activity orders a view; there is no rank to claim
- focusing a view clears it for everyone, not just for the caller
- a view that goes offline leaves no trace

Run without a Matrix: two operators on one zenoh session, as the operator
canary does.
"""

import asyncio
import logging
import time

import pytest
import pytest_asyncio

import zenoh

from ghoshell_moss.matrix.operator import ZenohOperator
from ghoshell_moss.services.webview import (
    WebViewClient,
    WebViewDeclaration,
    WebViewServer,
)


def _operator(session: zenoh.Session, network_ns: str, address: str) -> ZenohOperator:
    return ZenohOperator(
        session=session,
        network_ns=network_ns,
        this_address=address,
        logger=logging.getLogger(address),
    )


def _declaration(title: str = 'Terminal', *, created: float | None = None, **kwargs) -> WebViewDeclaration:
    kwargs.setdefault('url', f'http://127.0.0.1:8768/{title.lower()}/')
    if created is not None:
        kwargs['created'] = created
    return WebViewDeclaration(title=title, **kwargs)


@pytest.fixture
def zenoh_session():
    session = zenoh.open(zenoh.Config())
    yield session
    session.close()


@pytest.fixture
def network_ns():
    return f'probe/webview/{time.time_ns()}'


@pytest_asyncio.fixture
async def server_operator(zenoh_session, network_ns):
    async with _operator(zenoh_session, network_ns, 'host/server') as operator:
        yield operator


@pytest_asyncio.fixture
async def client_operator(zenoh_session, network_ns):
    async with _operator(zenoh_session, network_ns, 'host/client') as operator:
        yield operator


# -- discovery ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_discovery_carries_the_identity(server_operator, client_operator):
    server = WebViewServer.from_operator(
        server_operator,
        _declaration(title='Terminal', description='bash cards'),
    )
    async with server, WebViewClient.from_operator(client_operator) as client:
        await asyncio.sleep(0.6)

        items = client.items()
        assert len(items) == 1
        assert items[0].declaration.title == 'Terminal'
        assert items[0].declaration.description == 'bash cards'
        assert items[0].declaration.url.endswith('/terminal/')
        # the address is the view's identity; it is not a declared field
        assert items[0].address == 'host/server'
        assert items[0].state.unread_count == 0


@pytest.mark.asyncio
async def test_a_view_that_goes_offline_leaves_no_trace(server_operator, client_operator):
    server = WebViewServer.from_operator(server_operator, _declaration())
    await server.__aenter__()
    async with WebViewClient.from_operator(client_operator) as client:
        await asyncio.sleep(0.6)
        assert len(client.items()) == 1

        await server.__aexit__(None, None, None)
        await asyncio.sleep(0.6)
        assert client.items() == []


# -- condition ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_condition_published_before_anyone_listened_is_still_observable(
        server_operator, client_operator,
):
    """``pub`` is not retained, so discovery alone is not enough: a consumer that
    arrives after the fact must still see what happened."""
    server = WebViewServer.from_operator(server_operator, _declaration())
    async with server:
        await asyncio.sleep(0.4)
        server.notify('build finished')  # nobody is subscribed yet
        await asyncio.sleep(0.3)

        async with WebViewClient.from_operator(client_operator) as client:
            await asyncio.sleep(0.5)

            items = client.items()
            assert len(items) == 1
            assert items[0].state.unread_count == 1
            assert items[0].state.last_message == 'build finished'


@pytest.mark.asyncio
async def test_later_changes_arrive_without_being_asked_for(server_operator, client_operator):
    server = WebViewServer.from_operator(server_operator, _declaration())
    async with server, WebViewClient.from_operator(client_operator) as client:
        await asyncio.sleep(0.6)
        assert client.items()[0].state.unread_count == 0

        server.notify('one')
        await asyncio.sleep(0.4)
        assert client.items()[0].state.unread_count == 1

        server.notify('two')
        await asyncio.sleep(0.4)
        assert client.items()[0].state.unread_count == 2
        assert client.items()[0].state.last_message == 'two'

        before = client.items()[0].state.last_activity_at
        server.touch()
        await asyncio.sleep(0.4)
        state = client.items()[0].state
        assert state.last_activity_at > before
        # claiming attention is not claiming there is something to read
        assert state.unread_count == 2


@pytest.mark.asyncio
async def test_a_consumer_is_told_about_every_change(server_operator, client_operator):
    server = WebViewServer.from_operator(server_operator, _declaration())
    async with server, WebViewClient.from_operator(client_operator) as client:
        await asyncio.sleep(0.6)

        seen = []
        client.on_change(lambda: seen.append(len(client.items())))
        await asyncio.sleep(0.1)

        server.notify('ping')
        await asyncio.sleep(0.4)
        assert seen and seen[-1] == 1

        await server.__aexit__(None, None, None)
        await asyncio.sleep(0.6)
        assert seen[-1] == 0


# -- ordering ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_ordering_follows_activity_not_a_claimed_rank(
        zenoh_session, network_ns, client_operator,
):
    """A view with no rank moves forward by being active — that is the whole
    substitute for a self-claimed priority."""
    now = time.time()
    first = _operator(zenoh_session, network_ns, 'host/one')
    second = _operator(zenoh_session, network_ns, 'host/two')
    async with first, second:
        # 'Two' is older, so it starts behind
        older = WebViewServer.from_operator(second, _declaration('Two', created=now - 10))
        newer = WebViewServer.from_operator(first, _declaration('One', created=now))
        async with older, newer, WebViewClient.from_operator(client_operator) as client:
            await asyncio.sleep(0.7)
            assert [i.declaration.title for i in client.items()] == ['One', 'Two']

            older.notify('two is alive')
            await asyncio.sleep(0.4)
            assert [i.declaration.title for i in client.items()] == ['Two', 'One']


# -- focus -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_focusing_a_view_clears_it_for_everyone(
        zenoh_session, network_ns, server_operator, client_operator,
):
    """Focus travels the mesh, so screens converge — the reason this is not a
    local decision."""
    other_operator = _operator(zenoh_session, network_ns, 'host/other')
    server = WebViewServer.from_operator(server_operator, _declaration())
    async with other_operator, server:
        async with (
            WebViewClient.from_operator(client_operator) as one,
            WebViewClient.from_operator(other_operator) as two,
        ):
            await asyncio.sleep(0.7)
            server.notify('ping')
            await asyncio.sleep(0.4)
            assert one.items()[0].state.unread_count == 1
            assert two.items()[0].state.unread_count == 1

            address = one.items()[0].address
            settled = await one.activate(address)
            assert settled is not None
            assert settled.unread_count == 0
            assert settled.focused_at is not None
            # having been read does not erase what it was
            assert settled.last_message == 'ping'

            await asyncio.sleep(0.4)
            converged = two.items()[0].state
            assert converged.unread_count == 0
            assert converged.focused_at is not None
