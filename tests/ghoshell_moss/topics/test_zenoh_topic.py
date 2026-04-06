import asyncio
import ghoshell_moss.core.concepts.topic as topic_concepts
from ghoshell_moss.core.concepts.topic import Topic, TopicMeta, ErrorTopic, TopicClosedError
from ghoshell_moss.topic.zenoh_topics import ZenohTopicService
import pytest
import zenoh


@pytest.mark.asyncio
async def test_topic_baseline():
    session = zenoh.open(zenoh.Config())
    with session:
        service = ZenohTopicService(
            sender="test",
            session_id="test",
            session=session,
        )
        listening_started = asyncio.Event()

        async def produce():
            publisher = service.model_publisher("publisher", ErrorTopic)
            async with publisher:
                assert publisher.is_running()
                await listening_started.wait()
                publisher.pub(ErrorTopic(errmsg="hello world"))
                await asyncio.sleep(0.0)
                publisher.pub(ErrorTopic(errmsg="hello world"))
                await asyncio.sleep(0.0)
                publisher.pub(ErrorTopic(errmsg="hello world"))
                await asyncio.sleep(0.0)
                publisher.pub(ErrorTopic(errmsg="hello world"))
                await asyncio.sleep(0.0)

        received = []

        async def consumer():
            async with service.subscribe_model(ErrorTopic) as subscriber:
                listening_started.set()
                assert len(service.listening()) == 1
                assert subscriber is not None
                assert subscriber.listening() == ErrorTopic.default_topic_name()
                assert subscriber.is_running()
                while subscriber.is_running():
                    item = await subscriber.poll_model()
                    received.append(item)
            assert not subscriber.is_running()

        async with service:
            producer_task = asyncio.create_task(produce())
            consumer_task = asyncio.create_task(asyncio.wait_for(consumer(), 0.01))
            await producer_task
            # 在 consumer 结束前退出.
            assert service.is_running()
            with pytest.raises(asyncio.TimeoutError):
                await consumer_task
        assert len(received) > 0


@pytest.mark.asyncio
async def test_topic_service_publish():
    session = zenoh.open(zenoh.Config())
    received = []
    started = asyncio.Event()
    with session:
        service = ZenohTopicService(sender="test", session_id="test", session=session)
        async with service:
            async def _consume():
                async with service.subscribe_model(ErrorTopic) as subscriber:
                    started.set()
                    item = await subscriber.poll_model()
                    received.append(item)

            task = asyncio.create_task(_consume())
            await started.wait()
            service.pub(ErrorTopic(errmsg="hello world"))
            await task
    assert len(received) == 1
    assert received[0].errmsg == "hello world"
