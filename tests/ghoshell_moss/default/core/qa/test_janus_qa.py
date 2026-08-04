"""Integration tests for janus QA — asker/watcher round-trip, verdict, cancel."""

import asyncio

import pytest

from ghoshell_moss.core.qa.janus_qa import (
    JanusQA,
    JanusAsker,
    JanusWatcher,
    JanusQAManager,
)


@pytest.mark.asyncio
async def test_round_trip_approval():
    """Asker issues → watcher sees → replies → owner gets verdict."""
    async with JanusQAManager(issuer='ghost-1') as mgr:
        asker = mgr.asker('safemode')
        watcher = mgr.watch('safemode')

        seen: list[JanusQA] = []
        watcher.on_question(lambda qa: seen.append(qa))

        qa = asker.ask_approval('delete file?')
        assert qa.owned()
        assert not qa.done()

        await asyncio.sleep(0.05)
        assert len(seen) == 1
        r_qa = seen[0]
        assert not r_qa.owned()
        assert r_qa.question.content == 'delete file?'

        r_qa.reply(r_qa.question.approve('ok'))

        await qa.wait()
        assert qa.done()
        assert qa.answer is not None
        assert not qa.answer.rejected

        await asyncio.sleep(0.05)
        assert r_qa.done()
        assert r_qa.answer is not None


@pytest.mark.asyncio
async def test_reply_reject():
    """Responder rejects → owner receives rejected answer."""
    async with JanusQAManager(issuer='ghost-1') as mgr:
        asker = mgr.asker('ns')
        watcher = mgr.watch('ns')

        seen: list[JanusQA] = []
        watcher.on_question(lambda qa: seen.append(qa))

        qa = asker.ask_approval('delete?')
        await asyncio.sleep(0.05)

        seen[0].reply(seen[0].question.reject('busy'))

        await qa.wait()
        assert qa.answer is not None
        assert qa.answer.rejected
        assert qa.answer.content == 'busy'


@pytest.mark.asyncio
async def test_cancel():
    """Owner cancels → done, canceled flag set, broadcast to watcher."""
    async with JanusQAManager(issuer='ghost-1') as mgr:
        asker = mgr.asker('ns')
        watcher = mgr.watch('ns')

        seen: list[JanusQA] = []
        watcher.on_question(lambda qa: seen.append(qa))

        qa = asker.ask_approval('irrelevant')
        await asyncio.sleep(0.05)
        r_qa = seen[0]

        qa.cancel('aborted')
        await qa.wait()
        assert qa.done()
        assert qa.canceled()
        assert qa.question.canceled == 'aborted'

        await asyncio.sleep(0.05)
        assert r_qa.done()
        assert r_qa.canceled()
        assert r_qa.question.canceled == 'aborted'


@pytest.mark.asyncio
async def test_double_reply_raises():
    """Responder cannot reply twice to the same question."""
    async with JanusQAManager(issuer='ghost-1') as mgr:
        asker = mgr.asker('ns')
        watcher = mgr.watch('ns')

        seen: list[JanusQA] = []
        watcher.on_question(lambda qa: seen.append(qa))

        asker.ask_approval('x')
        await asyncio.sleep(0.05)
        r_qa = seen[0]

        r_qa.reply(r_qa.question.approve('yes'))

        with pytest.raises(ValueError, match='already replied'):
            r_qa.reply(r_qa.question.reject('no'))


@pytest.mark.asyncio
async def test_reply_after_done_raises():
    """Cannot reply after the question is resolved."""
    async with JanusQAManager(issuer='ghost-1') as mgr:
        asker = mgr.asker('ns')
        watcher = mgr.watch('ns')

        seen: list[JanusQA] = []
        watcher.on_question(lambda qa: seen.append(qa))

        qa = asker.ask_approval('x')
        await asyncio.sleep(0.05)
        r_qa = seen[0]

        r_qa.reply(r_qa.question.approve('yes'))
        await qa.wait()
        await asyncio.sleep(0.05)

        assert r_qa.done()
        with pytest.raises(ValueError, match='already replied'):
            r_qa.reply(r_qa.question.reject('no'))


@pytest.mark.asyncio
async def test_cannot_cancel_unowned():
    """Responder (unowned QA) cannot cancel."""
    async with JanusQAManager(issuer='ghost-1') as mgr:
        asker = mgr.asker('ns')
        watcher = mgr.watch('ns')

        seen: list[JanusQA] = []
        watcher.on_question(lambda qa: seen.append(qa))

        asker.ask_approval('x')
        await asyncio.sleep(0.05)
        r_qa = seen[0]
        assert not r_qa.owned()

        with pytest.raises(ValueError, match='only owner'):
            r_qa.cancel('nope')


@pytest.mark.asyncio
async def test_first_reply_wins():
    """With multiple watchers, first reply is the accepted answer."""
    async with JanusQAManager(issuer='ghost-1') as mgr:
        asker = mgr.asker('ns')
        w1 = mgr.watch('ns')
        w2 = mgr.watch('ns')

        seen1: list[JanusQA] = []
        seen2: list[JanusQA] = []
        w1.on_question(lambda qa: seen1.append(qa))
        w2.on_question(lambda qa: seen2.append(qa))

        qa = asker.ask_approval('who wins?')
        await asyncio.sleep(0.05)

        assert len(seen1) == 1
        assert len(seen2) == 1

        seen1[0].reply(seen1[0].question.approve('first'))
        seen2[0].reply(seen2[0].question.approve('second'))

        await qa.wait()
        assert qa.answer.content == 'first'


@pytest.mark.asyncio
async def test_asker_undone():
    """Asker.undone() returns pending owner QA copies."""
    async with JanusQAManager(issuer='ghost-1') as mgr:
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
async def test_watcher_questions_answered_filter():
    """Watcher.questions(answered=True) returns done QAs."""
    async with JanusQAManager(issuer='ghost-1') as mgr:
        asker = mgr.asker('ns')
        watcher = mgr.watch('ns')

        seen: list[JanusQA] = []
        watcher.on_question(lambda qa: seen.append(qa))

        asker.ask_approval('q1')
        await asyncio.sleep(0.05)

        assert len(watcher.questions()) == 1
        assert watcher.questions(answered=True) == []

        seen[0].reply(seen[0].question.approve('ok'))
        await asyncio.sleep(0.05)

        answered = watcher.questions(answered=True)
        assert len(answered) == 1
        assert answered[0].done()


@pytest.mark.asyncio
async def test_ask_confirm_round_trip():
    """ask_confirm → watcher confirms → owner gets choice."""
    async with JanusQAManager(issuer='ghost-1') as mgr:
        asker = mgr.asker('ns')
        watcher = mgr.watch('ns')

        seen: list[JanusQA] = []
        watcher.on_question(lambda qa: seen.append(qa))

        qa = asker.ask_confirm('proceed?', yes='go ahead', no='stop')
        await asyncio.sleep(0.05)

        r_qa = seen[0]
        r_qa.reply(r_qa.question.confirm(result=True, content='lets go'))

        await qa.wait()
        assert qa.answer.choices == ['yes']


@pytest.mark.asyncio
async def test_auto_cancel_on_context_exit():
    """QA async context manager auto-cancels on exit without resolution."""
    async with JanusQAManager(issuer='ghost-1') as mgr:
        asker = mgr.asker('ns')

        async with asker.ask_approval('will cancel') as qa:
            pass

        assert qa.done()
        assert qa.canceled()


@pytest.mark.asyncio
async def test_auto_cancel_on_exception():
    """QA async context manager cancels with exception message on error."""
    async with JanusQAManager(issuer='ghost-1') as mgr:
        asker = mgr.asker('ns')

        try:
            async with asker.ask_approval('will fail') as qa:
                raise RuntimeError('crash')
        except RuntimeError:
            pass

        assert qa.done()
        assert qa.canceled()
        assert 'crash' in qa.question.canceled


@pytest.mark.asyncio
async def test_qameta_is_set_on_answer():
    """reply() stamps answer.meta with the respondent's identity."""
    async with JanusQAManager(issuer='ghost-1') as mgr:
        asker = mgr.asker('ns')
        watcher = mgr.watch('ns')

        seen: list[JanusQA] = []
        watcher.on_question(lambda qa: seen.append(qa))

        qa = asker.ask_approval('x')
        await asyncio.sleep(0.05)

        r_qa = seen[0]
        r_qa.reply(r_qa.question.approve('ok'))

        await qa.wait()
        final = qa.answer
        assert final is not None
        assert final.meta is not None
        assert final.meta.refer_to == qa.question.meta.id
        assert final.meta.issuer == 'ghost-1'
