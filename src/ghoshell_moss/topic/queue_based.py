from typing import Literal, Optional

from ghoshell_common.helpers import uuid
from ghoshell_common.contracts import LoggerItf

from ghoshell_moss.message import Addition
from typing_extensions import Self
from ghoshell_moss.core.concepts.topic import *
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_container import Provider, IoCContainer
import asyncio
import logging
import anyio
import time
import janus


class QueueBasedSubscriber(Subscriber[TOPIC_MODEL | None]):
    """
    基于队列实现 Subscriber
    """

    def __init__(
            self,
            service_stopped: ThreadSafeEvent,
            *,
            model: type[TOPIC_MODEL] | None,
            topic_name: str = "",
            uid: str | None = None,
            maxsize: int = 0,
            keep: Literal["latest", "oldest"] = "latest",
            logger: LoggerItf | None = None,
    ):
        self._model = model
        if model is not None:
            topic_name = topic_name or model.default_topic_name()
        self._listening = topic_name
        self._uid = uid or uuid()
        self._queue: janus.Queue[Topic | None] = janus.Queue(maxsize=maxsize)
        self._receive_lock = asyncio.Lock()
        self._service_stopped = service_stopped
        self._logger = logger or logging.getLogger("moss")
        self._keep_policy = keep
        self._started = False
        self._closed = False
        self._service_wait_task: Optional[asyncio.Task] = None
        self._log_prefix = f"[QueueBasedSubscriber %s id=%s]" % (self._listening, self._uid)

    def receive(self, topic: Topic, keep_policy: str = "") -> None:
        """
        接受上游发送的消息.
        """
        if topic.meta.name != self._listening:
            return
        if self._service_stopped.is_set():
            raise TopicClosedError()
        keep_policy = keep_policy or self._keep_policy
        try:
            _queue = self._queue.sync_q
            if _queue.full():
                if keep_policy == "oldest":
                    self._logger.info("%s drop topic %s cause full", self._log_prefix, topic.meta.id)
                    return
                elif keep_policy == "latest":
                    if not _queue.empty():
                        oldest = _queue.get_nowait()
                        self._logger.info("%s drop oldest topic %s cause full", self._log_prefix, oldest)
                    _queue.put_nowait(topic)
                else:
                    return
            else:
                _queue.put_nowait(topic)
        except janus.QueueShutDown:
            raise TopicClosedError()
        except asyncio.QueueFull:
            self._logger.error("%s drop topic %s cause full", self._log_prefix, topic.meta.id)

    async def _wait_service_stopped(self) -> None:
        await self._service_stopped.wait()
        await self._close()

    async def __aenter__(self) -> Self:
        self._started = True
        self._service_wait_task = asyncio.create_task(self._wait_service_stopped())
        return self

    async def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.shutdown()
        if self._service_wait_task and not self._service_wait_task.done():
            self._service_wait_task.cancel()
            try:
                await self._service_wait_task
            except asyncio.CancelledError:
                pass
        self._service_wait_task = None

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._close()
        if exc_val:
            if isinstance(exc_val, TopicClosedError):
                self._logger.info("%s stopped cause service closed", self._log_prefix)
                return True
            else:
                self._logger.error("%s stopped cause error: %s", self._log_prefix, exc_val)
        return None

    def listening(self) -> str:
        return self._listening

    def id(self) -> str:
        return self._uid

    async def poll(self, timeout: float | None = None) -> Topic:
        if self._closed:
            raise TopicClosedError()
        _queue = self._queue.async_q
        try:
            item = await asyncio.wait_for(_queue.get(), timeout=timeout)
            if item is None:
                await self.close()
                raise TopicClosedError()
            # 业务侧才复制.
            return item.model_copy()
        except janus.QueueShutDown:
            raise TopicClosedError()

    async def poll_model(self, timeout: float | None = None) -> TOPIC_MODEL | None:
        if self._model is None:
            return None
        topic = await self.poll(timeout)
        return self._model.from_topic(topic)

    def is_closed(self) -> bool:
        return self._closed or self._service_stopped.is_set()

    def is_running(self) -> bool:
        return self._started and not self.is_closed()


class QueueBasedPublisher(Publisher):
    def __init__(
            self,
            topic_name: str,
            *,
            creator: str,
            publish_queue: janus.Queue[Topic],
            service_stopped_event: ThreadSafeEvent,
            uid: str | None = None,
            logger: LoggerItf | None = None,
            frequent: float = 0.0,
            model: type[TopicModel] | None = None,
    ):
        if model is not None:
            topic_name = topic_name or model.topic_name
        self._topic_name = topic_name
        self._publish_queue = publish_queue
        self._service_stopped_event = service_stopped_event
        self._creator = creator
        self._logger = logger or logging.getLogger("moss")
        self._additions = []
        self._uid = uid or uuid()
        self._log_prefix = f"[QueueBasedPublisher %s id=%s]" % (self._creator, self._uid)
        self._frequent = frequent
        self._last_sent: float = 0.0
        self._model_type = model

    def with_additions(self, *additions: Addition) -> Self:
        self._additions.extend(additions)
        return self

    def is_running(self) -> bool:
        return not self._service_stopped_event.is_set()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            if isinstance(exc_val, TopicClosedError):
                return True
            else:
                self._logger.exception("%s stopped cause error: %s", self._log_prefix, exc_val)
        return None

    def pub(self, topic: Topic | TOPIC_MODEL, *, name: str = "") -> None:
        if not self.is_running():
            self._logger.info("%s drop topic %s cause not running", self._log_prefix, topic.meta.id)
            return
        if self._frequent > 0 and self._last_sent + self._frequent > time.time():
            self._logger.error("%s drop topic %s cause too frequent", self._log_prefix, topic.meta.id)
            return

        if isinstance(topic, TopicModel):
            if self._model_type is not None:
                if not isinstance(topic, self._model_type):
                    raise ValueError(f"topic type {type(topic)} != allow topic type {self._model_type}")
            topic = topic.to_topic()
        if name:
            topic.meta.name = name

        if topic.meta.name != self._topic_name:
            raise ValueError(f"topic name {topic.topic_name} != allow topic name {self._topic_name}")

        if len(self._additions) > 0:
            topic.with_additions(*self._additions)
        topic.meta.creator = self._creator
        self._publish_queue.sync_q.put_nowait(topic)
        # 使用 async 做 api 唯一的目的就是为了这次调度.
        # 否则撑死是小, 并行调度阻塞事大.


class QueueBasedTopicService(TopicService):
    """
    实现最基本的协程 topic service.
    """

    def __init__(self, sender: str = "", *, logger: LoggerItf | None = None):
        self._sender = sender or uuid()
        self._creator = f"TopicService/{self._sender}"
        self._started = False
        self._closing_event = ThreadSafeEvent()
        self._main_loop_stopped_event = ThreadSafeEvent()
        self._subscribers: dict[TopicName, dict[str, QueueBasedSubscriber]] = {}
        self._subscriber_lock = asyncio.Lock()

        self._publish_queue: janus.Queue[Topic] = janus.Queue()
        self._publish_queue_empty = asyncio.Event()
        self._main_loop_task: Optional[asyncio.Task] = None
        self._logger = logger or logging.getLogger("moss")
        self._dispatch_tasks: set[asyncio.Task] = set()
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
            loop = asyncio.get_running_loop()
            removing_subscribe = []
            while not self._closing_event.is_set():
                try:
                    _queue = self._publish_queue
                    topic = await asyncio.wait_for(_queue.async_q.get(), 0.2)
                    self._publish_queue_empty.clear()
                except asyncio.TimeoutError:
                    if self._publish_queue.sync_q.empty() and self._publish_queue.async_q.empty():
                        self._publish_queue_empty.set()
                    continue
                except janus.AsyncQueueShutDown:
                    # old queue is shutdown.
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
                self._add_task(loop.create_task(self.on_topic_published(topic)))

                if topic.meta.name not in self._subscribers:
                    # 没有本地的监听.
                    continue
                if len(removing_subscribe) > 0:
                    removing_subscribe.clear()

                topic_name = topic.meta.name
                subscribers = self._subscribers.get(topic_name, None)
                if subscribers is None or len(subscribers) == 0:
                    continue
                for subscriber in subscribers.values():
                    if subscriber.is_closed():
                        continue
                    if not subscriber.is_running():
                        continue
                    # 创建分发任务.
                    if not self._dispatch_topic(subscriber, topic):
                        removing_subscribe.append(subscriber.id())
                if len(removing_subscribe) > 0:
                    for _id in removing_subscribe:
                        if _id in subscribers:
                            del subscribers[_id]
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._logger.exception("%s main publish loop failed: %r", self._log_prefix, e)
        finally:
            self._logger.info("%s main publish loop stopped", self._log_prefix)
            self._main_loop_stopped_event.set()
            self._publish_queue_empty.set()

    def _add_task(self, task: asyncio.Task) -> None:
        self._dispatch_tasks.add(task)
        task.add_done_callback(self._remove_task)

    def _remove_task(self, task: asyncio.Task) -> None:
        if task in self._dispatch_tasks:
            self._dispatch_tasks.remove(task)

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

    def _dispatch_topic(self, subscriber: QueueBasedSubscriber, topic: Topic) -> bool:
        try:
            if subscriber.id() == topic.meta.sender:
                # 不做循环发布.
                return True
            subscriber.receive(topic)
            return True
        except TopicClosedError:
            return False
        except Exception as e:
            self._logger.exception(
                "%s send topic %s to subscribe %s failed: %r",
                self._log_prefix,
                topic.meta,
                subscriber.id,
                e,
            )
            return True

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
            keep: Literal["latest", "oldest"] = "latest",
            model: type[TopicModel] = None,
    ) -> Subscriber:
        return self._create_subscriber(
            topic_name=topic_name,
            uid=uid,
            maxsize=maxsize,
            keep=keep,
            model=model,
        )

    def _create_subscriber(
            self,
            model: type[TopicModel] | None,
            *,
            topic_name: str = "",
            uid: str | None = None,
            maxsize: int = 0,
            keep: Literal["latest", "oldest"] = "latest",
    ) -> Subscriber:
        """ """
        # 没有 await, 预计不会让出控制权. 所以这一版不加锁了.
        subscriber = QueueBasedSubscriber(
            self._main_loop_stopped_event,
            model=model,
            topic_name=topic_name,
            maxsize=maxsize,
            keep=keep,
            logger=self._logger,
            uid=uid,
        )
        sub_id = subscriber.id()
        topic_name = subscriber.listening()
        if topic_name not in self._subscribers:
            self._subscribers[topic_name] = {}
        self._subscribers[topic_name][sub_id] = subscriber
        return subscriber

    def publisher(
            self,
            creator: str,
            topic_name: str,
            *,
            uid: str | None = None,
            model: type[TopicModel] | None = None,
    ) -> Publisher:
        publisher = QueueBasedPublisher(
            topic_name=topic_name,
            creator=creator,
            publish_queue=self._publish_queue,
            service_stopped_event=self._main_loop_stopped_event,
            uid=uid,
            logger=self._logger,
            model=model,
        )
        return publisher

    def pub(self, topic: Topic | TopicModel, *, name: str = "", creator: str = "") -> None:
        if not self.is_running():
            self._logger.info("%s drop topic %s cause not running", self._log_prefix, topic.meta.id)
            return
        if isinstance(topic, TopicModel):
            topic = topic.to_topic()
        if name:
            topic.meta.name = name
        topic.meta.creator = creator or self._creator
        self._publish_queue.sync_q.put_nowait(topic)


class QueueBasedTopicProvider(Provider[TopicService]):
    """
    实现一个 provider.
    """

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> TopicService:
        return QueueBasedTopicService(
            logger=con.get(LoggerItf),
        )
