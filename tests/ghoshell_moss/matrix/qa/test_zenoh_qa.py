"""Integration tests for zenoh QA — requires a local zenoh router."""

import asyncio

import pytest
import zenoh

from ghoshell_moss.matrix.qa.zenoh_qa import (
    ZenohQA,
    ZenohAsker,
    ZenohWatcher,
    ZenohQAManager,
)


@pytest.fixture
def zenoh_session():
    with zenoh.open(zenoh.Config()) as session:
        yield session


@pytest.mark.asyncio
async def test_round_trip_approval(zenoh_session):
    """Asker issues → watcher sees → replies → owner gets verdict."""
    async with ZenohQAManager(
        issuer='ghost-1', prefix='test/qa',
        session=zenoh_session,
    ) as mgr:
        asker = mgr.asker('safemode')
        watcher = mgr.watch('safemode')

        seen: list[ZenohQA] = []
        watcher.on_question(lambda qa: seen.append(qa))

        qa = asker.ask_approval('delete file?')
        assert qa.owned()
        assert not qa.done()

        # wait for zenoh publish → subscriber delivery
        await asyncio.sleep(0.2)
        assert len(seen) == 1
        r_qa = seen[0]
        assert not r_qa.owned()
        assert r_qa.question.content == 'delete file?'

        r_qa.reply(r_qa.question.approve('ok'))

        await qa.wait()
        assert qa.done()
        assert qa.answer is not None
        assert not qa.answer.rejected

        # verdict broadcast reaches watcher copy
        await asyncio.sleep(0.2)
        assert r_qa.done()
        assert r_qa.answer is not None


@pytest.mark.asyncio
async def test_cancel(zenoh_session):
    """Owner cancels → done, broadcast to watcher."""
    async with ZenohQAManager(
        issuer='ghost-1', prefix='test/qa',
        session=zenoh_session,
    ) as mgr:
        asker = mgr.asker('ns')
        watcher = mgr.watch('ns')

        seen: list[ZenohQA] = []
        watcher.on_question(lambda qa: seen.append(qa))

        qa = asker.ask_approval('irrelevant')
        await asyncio.sleep(0.2)
        r_qa = seen[0]

        qa.cancel('aborted')
        await qa.wait()
        assert qa.done()
        assert qa.canceled()

        await asyncio.sleep(0.2)
        assert r_qa.done()
        assert r_qa.canceled()


@pytest.mark.asyncio
async def test_double_reply_raises(zenoh_session):
    """Responder cannot reply twice."""
    async with ZenohQAManager(
        issuer='ghost-1', prefix='test/qa',
        session=zenoh_session,
    ) as mgr:
        asker = mgr.asker('ns')
        watcher = mgr.watch('ns')

        seen: list[ZenohQA] = []
        watcher.on_question(lambda qa: seen.append(qa))

        asker.ask_approval('x')
        await asyncio.sleep(0.2)
        r_qa = seen[0]

        r_qa.reply(r_qa.question.approve('yes'))
        with pytest.raises(ValueError, match='already replied'):
            r_qa.reply(r_qa.question.reject('no'))


@pytest.mark.asyncio
async def test_cannot_cancel_unowned(zenoh_session):
    """Responder cannot cancel."""
    async with ZenohQAManager(
        issuer='ghost-1', prefix='test/qa',
        session=zenoh_session,
    ) as mgr:
        asker = mgr.asker('ns')
        watcher = mgr.watch('ns')

        seen: list[ZenohQA] = []
        watcher.on_question(lambda qa: seen.append(qa))

        asker.ask_approval('x')
        await asyncio.sleep(0.2)
        r_qa = seen[0]
        assert not r_qa.owned()

        with pytest.raises(ValueError, match='only owner'):
            r_qa.cancel('nope')


@pytest.mark.asyncio
async def test_asker_undone(zenoh_session):
    """Asker.undone() returns pending owner QA copies."""
    async with ZenohQAManager(
        issuer='ghost-1', prefix='test/qa',
        session=zenoh_session,
    ) as mgr:
        asker = mgr.asker('ns')

        assert asker.undone() == []
        qa1 = asker.ask_approval('q1')
        qa2 = asker.ask_approval('q2')
        assert len(asker.undone()) == 2

        qa1.cancel('done')
        await qa1.wait()
        assert len(asker.undone()) == 1
        assert asker.undone()[0] is qa2


@pytest.mark.asyncio
async def test_ask_confirm_round_trip(zenoh_session):
    """ask_confirm → watcher confirms → owner gets choice."""
    async with ZenohQAManager(
        issuer='ghost-1', prefix='test/qa',
        session=zenoh_session,
    ) as mgr:
        asker = mgr.asker('ns')
        watcher = mgr.watch('ns')

        seen: list[ZenohQA] = []
        watcher.on_question(lambda qa: seen.append(qa))

        qa = asker.ask_confirm('proceed?', yes='go ahead', no='stop')
        await asyncio.sleep(0.2)
        r_qa = seen[0]
        r_qa.reply(r_qa.question.confirm(result=True, content='go'))

        await qa.wait()
        assert qa.answer.choices == ['yes']
