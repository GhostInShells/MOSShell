from typing import Literal, Optional

from ghoshell_common.helpers import uuid
from ghoshell_common.contracts import LoggerItf

from ghoshell_moss.message import Addition
from typing_extensions import Self
from ghoshell_moss.core.concepts.topic import *
from ghoshell_container import Provider, IoCContainer
import asyncio
import logging
import anyio


class QueueBasedSubscriber(Subscriber[TOPIC_MODEL | None]):
    """
    基于队列实现 Subscriber
    """

    def __init__(
            self,
            service_stopped: asyncio.Event,
            *,
            model: type[TOPIC_MODEL] | None,
            topic_name: str = "",
            uid: str | None = None,
            maxsize: int = 0,
            keep: Literal['latest', 'oldest'] = "latest",
            logger: LoggerItf | None = None
    ):
        self._model = model
        self._listening = topic_name or model.default_topic_name()
        self._uid = uid or uuid()
        self._queue: asyncio.Queue[Topic | None] = asyncio.Queue(maxsize=maxsize)
        self._receive_lock = asyncio.Lock()
        self._service_stopped = service_stopped
        self._logger = logger or logging.getLogger('moss')
        self._keep_policy = keep
        self._started = False
        self._closed = False
        self._service_wait_task: Optional[asyncio.Task] = None
        self._log_prefix = f"[QueueBasedSubscriber %s id=%s]" % (self._listening, self._uid)

    async def receive(self, topic: Topic, keep_policy: str = "") -> None:
        """
        接受上游发送的消息.
        """
        if topic.meta.name != self._listening:
            return
        if self._service_stopped.is_set():
            return
        await self._receive_lock.acquire()
        keep_policy = keep_policy or self._keep_policy
        try:
            if self._queue.full():
                if keep_policy == "oldest":
                    self._logger.info("%s drop topic %s cause full", self._log_prefix, topic.id)
                    return
                elif keep_policy == "latest":
                    if not self._queue.empty():
                        oldest = self._queue.get_nowait()
                        self._logger.info("%s drop oldest topic %s cause full", self._log_prefix, oldest)
                    self._queue.put_nowait(topic)
                else:
                    return
            else:
                self._queue.put_nowait(topic)
        except asyncio.QueueFull:
            self._logger.error("%s drop topic %s cause full", self._log_prefix, topic.id)
        finally:
            self._receive_lock.release()

    async def _wait_service_stopped(self) -> None:
        await self._service_stopped.wait()
        while self._queue.full():
            self._queue.get_nowait()
        self._queue.put_nowait(None)

    async def __aenter__(self) -> Self:
        self._started = True
        self._service_wait_task = asyncio.create_task(self._wait_service_stopped())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not self._closed:
            self._closed = True
            self._queue.put_nowait(None)

            if self._service_wait_task and not self._service_wait_task.done():
                self._service_wait_task.cancel()
                try:
                    await self._service_wait_task
                except asyncio.CancelledError:
                    pass
            self._service_wait_task = None
        if exc_val:
            if isinstance(exc_val, ClosedError):
                self._logger.info("%s stopped cause service closed", self._log_prefix)
                return True
            else:
                self._logger.error("%s stopped cause error: %s", self._log_prefix, exc_val)

    def listening(self) -> str:
        return self._listening

    def id(self) -> str:
        return self._uid

    async def poll(self, timeout: float | None = None) -> Topic:
        if self._queue.empty():
            if self._closed or self._service_stopped.is_set():
                raise ClosedError()
        item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
        if item is None:
            await self.close()
            raise ClosedError()
        # 业务侧才复制.
        return item.model_copy()

    async def poll_model(self, timeout: float | None = None) -> TOPIC_MODEL | None:
        if self._model is None:
            return None
        topic = await self.poll(timeout)
        return self._model(**topic.data)

    def is_closed(self) -> bool:
        return self._closed or self._service_stopped.is_set()

    def is_running(self) -> bool:
        return self._started and not self.is_closed()


class QueueBasedPublisher(Publisher):

    def __init__(
            self,
            *,
            creator: str,
            publish_queue: asyncio.Queue,
            service_stopped_event: asyncio.Event,
            uid: str | None = None,
            logger: LoggerItf | None = None,
    ):
        self._publish_queue = publish_queue
        self._service_stopped_event = service_stopped_event
        self._creator = creator
        self._logger = logger or logging.getLogger('moss')
        self._additions = []
        self._uid = uid or uuid()
        self._log_prefix = f"[QueueBasedPublisher %s id=%s]" % (self._creator, self._uid)

    def with_additions(self, *additions: Addition) -> Self:
        self._additions.extend(additions)
        return self

    def is_running(self) -> bool:
        return not self._service_stopped_event.is_set()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            if isinstance(exc_val, ClosedError):
                return True
            else:
                self._logger.exception("%s stopped cause error: %s", self._log_prefix, exc_val)

    async def pub(self, topic: Topic | TopicModel, *, name: str = "") -> None:
        if not self.is_running():
            self._logger.info("%s drop topic %s cause not running", self._log_prefix, topic.id)
            return
        if isinstance(topic, TopicModel):
            topic = topic.to_topic()
        if name:
            topic.meta.name = name
        topic.meta.creator = self._creator
        await self._publish_queue.put(topic)
        await asyncio.sleep(0.0)


class QueueBasedTopicService(TopicService):
    """
    实现最基本的协程 topic service.
    """

    def __init__(
            self,
            sender: str = "",
            *,
            logger: LoggerItf | None = None
    ):
        self._sender = sender or uuid()
        self._creator = f"TopicService/{self._sender}"
        self._started = False
        self._closing_event = asyncio.Event()
        self._main_loop_stopped_event = asyncio.Event()
        self._subscribers: dict[TopicName, dict[str, QueueBasedSubscriber]] = {}
        self._subscriber_lock = asyncio.Lock()
        self._publish_queue: asyncio.Queue[Topic] = asyncio.Queue()
        self._publish_queue_empty = asyncio.Event()
        self._main_loop_task: Optional[asyncio.Task] = None
        self._logger = logger or logging.getLogger('moss')
        self._log_prefix = "[QueueBasedTopicService] "

    async def start(self):
        if self._started:
            return
        self._started = True
        self._publish_queue_empty.set()
        self._main_loop_stopped_event.clear()
        self._main_loop_task = asyncio.create_task(self._main_publish_loop())

    async def close(self):
        if self._closing_event.is_set():
            return
        self._closing_event.set()
        if self._main_loop_task and not self._main_loop_task.done():
            self._main_loop_task.cancel()
            try:
                await self._main_loop_task
            except asyncio.CancelledError:
                pass
        self._main_loop_task = None

    async def wait_sent(self):
        wait_done = asyncio.create_task(self._main_loop_stopped_event.wait())
        wait_empty = asyncio.create_task(self._publish_queue_empty.wait())
        d, p = await asyncio.wait([wait_done, wait_empty], return_when=asyncio.FIRST_COMPLETED)
        for task in p:
            task.cancel()

    async def _main_publish_loop(self) -> None:
        try:
            async with anyio.create_task_group() as tg:
                while not self._closing_event.is_set():
                    if self._publish_queue.empty():
                        self._publish_queue_empty.set()
                    try:
                        topic = await asyncio.wait_for(self._publish_queue.get(), 0.2)
                        self._publish_queue_empty.clear()
                    except asyncio.TimeoutError:
                        continue
                    if not isinstance(topic, Topic):
                        self._logger.error("%s drop invalid topic item %s", self._log_prefix, topic)
                        continue
                    if topic.is_overdue():
                        self._logger.info("%s drop overdue topic item %s", self._log_prefix, topic)
                        continue
                    if topic.meta.sender == self._sender:
                        self._logger.info("%s drop self sending topic item %s", self._log_prefix, topic)
                        continue
                    topic.meta.sender = self._sender

                    # 向上广播.
                    tg.start_soon(self.on_topic_published, topic)

                    if topic.meta.name not in self._subscribers:
                        # 没有本地的监听.
                        continue

                    topic_name = topic.meta.name
                    subscribers = self._subscribers.get(topic_name, None)
                    if subscribers is None or len(subscribers) == 0:
                        continue
                    new_subscribers = {}
                    for subscriber in subscribers.values():
                        if subscriber.is_closed():
                            continue
                        new_subscribers[subscriber.id()] = subscriber
                        if not subscriber.is_running():
                            continue
                        # 创建分发任务.
                        tg.start_soon(self._dispatch_topic, subscriber, topic)
                    self._subscribers[topic_name] = new_subscribers
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._logger.exception("%s main publish loop failed: %r", self._log_prefix, e)
        finally:
            self._logger.info("%s main publish loop stopped", self._log_prefix)
            self._main_loop_stopped_event.set()
            self._publish_queue_empty.set()

    async def on_topic_published(self, topic: Topic) -> None:
        """
        重写这个函数, 支持向上游发送事件.
        """
        try:
            await self._on_topic_published(topic)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._logger.exception("%s handle topic published failed: %r", self._log_prefix, e)

    async def _on_topic_published(self, topic: Topic) -> None:
        pass

    async def on_topic_subscribed(self, topic_name: str) -> None:
        try:
            await self._on_topic_subscribed(topic_name)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._logger.exception("%s handle topic subscribed failed: %r", self._log_prefix, e)

    async def _on_topic_subscribed(self, topic_name: str) -> None:
        """
        重写这个函数, 支持向上游发送事件.
        """
        pass

    async def _dispatch_topic(self, subscriber: QueueBasedSubscriber, topic: Topic) -> None:
        try:
            if subscriber.id() == topic.meta.sender:
                # 不做循环发布.
                return
            await subscriber.receive(topic)
        except ClosedError:
            pass
        except Exception as e:
            self._logger.exception(
                "%s send topic %s to subscribe %s failed: %r",
                self._log_prefix,
                topic.meta, subscriber.id,
                e,
            )

    def is_running(self) -> bool:
        return self._started and not self._main_loop_stopped_event.is_set()

    def listening(self) -> list[TopicName]:
        return list(self._subscribers.keys())

    def subscribe(
            self,
            topic_name: str,
            *,
            uid: str | None = None,
            maxsize: int = 0,
            keep: Literal['latest', 'oldest'] = "latest",
    ) -> Subscriber[None]:
        return self._create_subscriber(
            topic_name=topic_name, uid=uid, maxsize=maxsize, keep=keep, model=None,
        )

    def subscribe_model(
            self,
            model: type[TOPIC_MODEL],
            *,
            topic_name: str = "",
            uid: str | None = None,
            maxsize: int = 0,
            keep: Literal['latest', 'oldest'] = "latest",
    ) -> Subscriber[TOPIC_MODEL]:
        return self._create_subscriber(
            topic_name=topic_name, uid=uid, maxsize=maxsize, keep=keep, model=model,
        )

    def _create_subscriber(
            self,
            model: type[TopicModel] | None,
            *,
            topic_name: str = "",
            uid: str | None = None,
            maxsize: int = 0,
            keep: Literal['latest', 'oldest'] = "latest",
    ) -> Subscriber:
        """
        """
        # 没有 await, 预计不会让出控制权. 所以这一版不加锁了.
        subscriber = QueueBasedSubscriber(
            self._main_loop_stopped_event,
            model=model,
            topic_name=topic_name,
            maxsize=maxsize,
            keep=keep,
            logger=self._logger,
        )
        sub_id = subscriber.id()
        topic_name = subscriber.listening()
        if topic_name not in self._subscribers:
            self._subscribers[topic_name] = {}
        self._subscribers[topic_name][sub_id] = subscriber
        return subscriber

    def publisher(self, creator: str, uid: str | None = None) -> Publisher:
        publisher = QueueBasedPublisher(
            creator=creator,
            publish_queue=self._publish_queue,
            service_stopped_event=self._main_loop_stopped_event,
            uid=uid,
            logger=self._logger,
        )
        return publisher

    async def pub(self, topic: Topic | TopicModel, *, name: str = "", creator: str = "") -> None:
        if not self.is_running():
            self._logger.info("%s drop topic %s cause not running", self._log_prefix, topic.id)
            return
        if isinstance(topic, TopicModel):
            topic = topic.to_topic()
        if name:
            topic.meta.name = name
        topic.meta.creator = creator or self._creator
        await self._publish_queue.put(topic)


class QueueBasedTopicProvider(Provider[TopicService]):
    """
    实现一个 provider.
    """

    def singleton(self) -> bool:
        return False

    def factory(self, con: IoCContainer) -> TopicService:
        return QueueBasedTopicService(
            logger=con.get(LoggerItf),
        )
