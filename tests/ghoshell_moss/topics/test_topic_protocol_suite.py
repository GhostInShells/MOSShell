import asyncio
import pytest
from ghoshell_moss.core.concepts.topic import Subscriber, TopicService, ErrorTopic, TopicClosedError
from ghoshell_moss.topic.suite_for_test import TopicServiceSuite, QueueTopicServiceSuite
from ghoshell_moss.topic.zenoh_topics import ZenohTopicServiceSuite

# 配置项：未来可以在这里增加 ZenohTopicSuite() 等
topic_suite_configs = [
    QueueTopicServiceSuite(),
    ZenohTopicServiceSuite(),
]


@pytest.fixture(params=topic_suite_configs, ids=lambda s: s.name())
def service(request):
    """每个测试用例都会拿到一个全新的、无污染的 TopicService"""
    suite: TopicServiceSuite = request.param
    yield suite.create_service(sender="test_sender")
    suite.cleanup()


@pytest.mark.asyncio
@pytest.mark.usefixtures("service")
class TestTopicProtocol:
    """Topic 协议一致性测试套件"""

    async def test_topic_baseline(self, service: TopicService):
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

    async def test_topic_keep_oldest(self, service: TopicService):
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
                try:
                    while _subscriber.is_running():
                        item = await _subscriber.poll_model()
                        received.append(item)
                except TopicClosedError:
                    pass

        async with service:
            producer_task = asyncio.create_task(produce())
            subscriber = service.subscribe_model(ErrorTopic, maxsize=1, keep="oldest")
            consumer_task = asyncio.create_task(consumer(subscriber))
            await producer_task
        # expect closed
        await consumer_task
        assert len(received) == 1
        assert received[0].errmsg == "0"

    async def test_topic_keep_latest(self, service: TopicService):
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
                # 等待 producer 生成完, 然后再拉.
                await producer_done.wait()
                # 稍微等一下调度, 否则轮不到 session 运行.
                await asyncio.sleep(0.2)
                item = await _subscriber.poll_model()
                received.append(item)

        async with service:
            producer_task = asyncio.create_task(produce())
            subscriber = service.subscribe_model(ErrorTopic, maxsize=1, keep="latest")
            consumer_task = asyncio.create_task(consumer(subscriber))
            await producer_task
            await consumer_task
        assert len(received) == 1
        # 考虑到并发测试性能的问题, 毕竟是全异步. 反正不等于 1 就对了.
        assert received[0].errmsg in ("3", "4")
