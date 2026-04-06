import asyncio

import ghoshell_moss.core.concepts.topic as topic_concepts
from ghoshell_moss.core.concepts.topic import Topic, TopicMeta
from ghoshell_moss.topic import QueueBasedTopicService, ErrorTopic, Subscriber
import pytest


@pytest.mark.asyncio
async def test_topic_baseline():
    service = QueueBasedTopicService(
        sender="test",
    )

    async def produce():
        publisher = service.model_publisher("publisher", ErrorTopic)
        assert publisher.is_running()
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
        publisher = service.model_publisher("publisher", ErrorTopic)
        assert publisher.is_running()
        for idx in range(5):
            publisher.pub(ErrorTopic(errmsg="hello world %d:%d" % (o, idx)))
            await asyncio.sleep(0.0)

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


@pytest.mark.asyncio
async def test_topic_keep_latest():
    service = QueueBasedTopicService(
        sender="test",
    )

    consumer_started = asyncio.Event()
    producer_done = asyncio.Event()
    consumer_done = asyncio.Event()

    async def produce():
        await consumer_started.wait()
        publisher = service.model_publisher("publisher", ErrorTopic)
        async with publisher:
            for idx in range(5):
                publisher.pub(ErrorTopic(errmsg=str(idx)))
                await asyncio.sleep(0.0)
        producer_done.set()

    received = []

    async def consumer(_subscriber: Subscriber):
        async with _subscriber:
            consumer_started.set()
            await producer_done.wait()
            while _subscriber.is_running():
                item = await _subscriber.poll_model()
                received.append(item)
        consumer_done.set()

    async with service:
        producer_task = asyncio.create_task(produce())
        subscriber = service.subscribe_model(ErrorTopic, maxsize=1, keep="latest")
        consumer_task = asyncio.create_task(consumer(subscriber))
        await producer_task
    await consumer_task
    assert len(received) == 1
    assert received[0].errmsg == "4"


def test_topic_model():
    error = ErrorTopic(errmsg="hello world")
    topic = error.to_topic()
    new_error = ErrorTopic.from_topic(topic)
    assert new_error == error


@pytest.mark.asyncio
async def test_topic_keep_oldest():
    service = QueueBasedTopicService(
        sender="test",
    )

    consumer_started = asyncio.Event()

    async def produce():
        await consumer_started.wait()
        publisher = service.model_publisher("publisher", ErrorTopic)
        async with publisher:
            for idx in range(5):
                publisher.pub(ErrorTopic(errmsg=str(idx)))
                # 必须要让出, 否则 maxsize = 1 就无法测试了.
                await asyncio.sleep(0.0)

    received = []

    async def consumer(_subscriber: Subscriber):
        async with _subscriber:
            consumer_started.set()
            while _subscriber.is_running():
                item = await _subscriber.poll_model()
                received.append(item)

    async with service:
        producer_task = asyncio.create_task(produce())
        subscriber = service.subscribe_model(ErrorTopic, maxsize=1, keep="oldest")
        consumer_task = asyncio.create_task(consumer(subscriber))
        await producer_task
    await consumer_task
    assert len(received) == 1
    assert received[0].errmsg == "0"


def test_topic_is_overdue_logic(monkeypatch):
    topic = Topic(
        meta=TopicMeta(
            created_at=100.0,
            overdue=10.0,
        ),
        data={},
    )

    monkeypatch.setattr(topic_concepts.time, "time", lambda: 105.0)
    assert topic.is_overdue() is False

    monkeypatch.setattr(topic_concepts.time, "time", lambda: 111.0)
    assert topic.is_overdue() is True
