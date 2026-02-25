import asyncio

from ghoshell_moss.core.topic import QueueBasedTopicService, ErrorTopic, Subscriber
import pytest


@pytest.mark.asyncio
async def test_topic_baseline():
    service = QueueBasedTopicService(
        sender="test",
    )

    async def produce():
        publisher = service.publisher("publisher")
        assert publisher.is_running()
        await publisher.pub(ErrorTopic(errmsg="hello world"))
        await publisher.pub(ErrorTopic(errmsg="hello world"))
        await publisher.pub(ErrorTopic(errmsg="hello world"))
        await publisher.pub(ErrorTopic(errmsg="hello world"))

    received = []

    async def consumer():
        async with service.subscribe_model(ErrorTopic) as subscriber:
            assert len(service.listening()) == 1
            assert subscriber.listening() == ErrorTopic.default_topic_name()
            assert subscriber.is_running()
            while subscriber.is_running():
                item = await subscriber.poll_model()
                received.append(item)
        assert not service.is_running()

    async with service:
        producer_task = asyncio.create_task(produce())
        consumer_task = asyncio.create_task(consumer())
        await producer_task
        # 在 consumer 结束前退出.
        assert service.is_running()
        await service.wait_sent()

    await consumer_task
    assert len(received) == 4


@pytest.mark.asyncio
async def test_topic_publishers_and_consumers():
    service = QueueBasedTopicService(
        sender="test",
    )

    async def produce(o: int):
        publisher = service.publisher("publisher")
        assert publisher.is_running()
        for idx in range(5):
            await publisher.pub(ErrorTopic(errmsg="hello world %d:%d" % (o, idx)))

    received = []

    async def consumer(_subscriber: Subscriber):
        async with _subscriber:
            assert len(service.listening()) == 1
            assert _subscriber.listening() == ErrorTopic.default_topic_name()
            assert _subscriber.is_running()
            while _subscriber.is_running():
                item = await _subscriber.poll_model()
                received.append(item)
        assert not service.is_running()

    producers = []
    async with service:
        consumers = []
        for i in range(5):
            producer_task = asyncio.create_task(produce(i))
            producers.append(producer_task)
        for i in range(7):
            subscriber = service.subscribe_model(ErrorTopic)
            consumer_task = asyncio.create_task(consumer(subscriber))
            consumers.append(consumer_task)

        await asyncio.gather(*producers)
        # 在 consumer 结束前退出.
        assert service.is_running()
        await service.wait_sent()

    await asyncio.gather(*consumers)
    assert len(received) == 5 * 5 * 7
