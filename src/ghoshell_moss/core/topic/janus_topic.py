import asyncio
import logging
import threading
from typing import Literal, Optional, TypeVar

import janus
from typing_extensions import Self

from ghoshell_moss.core.concepts.topic import (
    ClosedError,
    Publisher,
    Subscriber,
    Topic,
    TopicModel,
    TopicName,
    TopicService,
)
from ghoshell_common.contracts import LoggerItf
from ghoshell_common.helpers import uuid

TOPIC_MODEL = TypeVar("TOPIC_MODEL", bound=TopicModel | None)

# by gemini pro
# todo: 考虑先不测试或实装, 还有很多问题没想明白. 用同步逻辑的确去掉了调度成本. 但生命周期管理感觉有严重问题.

class JanusSubscriber(Subscriber[TOPIC_MODEL]):
    """
    基于 Janus 队列的本地订阅者。
    支持跨线程的无阻塞 Push，以及 Asyncio 的无阻塞 Poll。
    """

    def __init__(
            self,
            service_stopped: threading.Event,
            *,
            model: type[TOPIC_MODEL] | None,
            topic_name: str = "",
            uid: str | None = None,
            maxsize: int = 0,
            keep: Literal["latest", "oldest"] = "latest",
            logger: LoggerItf | None = None,
    ):
        self._model = model
        self._listening = topic_name or (model.default_topic_name() if model else "")
        self._uid = uid or uuid()

        # 核心：Janus 混合队列。必须在 asyncio 事件循环中初始化。
        self._queue: janus.Queue[Topic | None] = janus.Queue(maxsize=maxsize)

        self._service_stopped = service_stopped
        self._logger = logger or logging.getLogger("moss")
        self._keep_policy = keep
        self._started = False
        self._closed = False
        self._log_prefix = f"[JanusSubscriber {self._listening} id={self._uid}]"

        # 用于保护 latest 丢弃策略的微小锁，极速释放
        self._sync_lock = threading.Lock()

    def receive_sync(self, topic: Topic) -> None:
        """
        供 Service 直接调用的同步推送方法。任何线程调用绝对安全。
        时间复杂度 O(1)。
        """
        if self._closed or self._service_stopped.is_set():
            return

        with self._sync_lock:
            try:
                self._queue.sync_q.put_nowait(topic)
            except janus.SyncQueueFull:
                if self._keep_policy == "oldest":
                    # 丢弃新消息
                    return
                elif self._keep_policy == "latest":
                    # 弹出最老的消息，压入新消息
                    try:
                        self._queue.sync_q.get_nowait()
                        self._queue.sync_q.put_nowait(topic)
                    except janus.SyncQueueEmpty:
                        # 极端并发下的防御
                        self._queue.sync_q.put_nowait(topic)
            except Exception as e:
                self._logger.error(f"{self._log_prefix} receive failed: {e}")

    async def __aenter__(self) -> Self:
        self._started = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close_sync()

    def close_sync(self):
        if not self._closed:
            self._closed = True
            try:
                self._queue.sync_q.put_nowait(None)  # 毒丸 (Poison Pill) 通知消费者退出
            except janus.SyncQueueFull:
                # 如果满了，强行清理一个空间放毒丸
                try:
                    self._queue.sync_q.get_nowait()
                    self._queue.sync_q.put_nowait(None)
                except Exception:
                    pass

    def listening(self) -> str:
        return self._listening

    def id(self) -> str:
        return self._uid

    async def poll(self, timeout: float | None = None) -> Topic:
        if self.is_closed():
            raise ClosedError()

        try:
            if timeout:
                item = await asyncio.wait_for(self._queue.async_q.get(), timeout=timeout)
            else:
                item = await self._queue.async_q.get()
        except asyncio.TimeoutError:
            raise

        if item is None:
            self._closed = True
            raise ClosedError()

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


class LocalPublisher(Publisher):
    def __init__(self, service: 'LocalTopicService', creator: str, uid: str | None = None):
        self._service = service
        self._creator = creator
        self._uid = uid or uuid()
        self._additions = []

    def with_additions(self, *additions) -> Self:
        self._additions.extend(additions)
        return self

    def is_running(self) -> bool:
        return self._service.is_running()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def pub(self, topic: Topic | TopicModel, *, name: str = "") -> None:
        """异步接口，底层直接调用同步分发"""
        self.pub_sync(topic, name=name)

    def pub_sync(self, topic: Topic | TopicModel, *, name: str = "") -> None:
        """允许外部线程直接同步调用"""
        if not self.is_running():
            return

        if isinstance(topic, TopicModel):
            topic = topic.to_topic()
        if name:
            topic.meta.name = name
        topic.meta.creator = self._creator

        # 直接调用 Service 的路由引擎
        self._service.dispatch_sync(topic)


class LocalTopicService(TopicService):
    """
    纯内存、线程安全的 Topic 路由引擎。
    没有任何后台 while 循环，完全事件驱动。
    """

    def __init__(self, sender: str = "", *, logger: LoggerItf | None = None):
        self._sender = sender or uuid()
        self._started = False

        # 使用 threading.Event 保证跨线程可见性
        self._service_stopped = threading.Event()

        # 路由表：topic_name -> {uid -> JanusSubscriber}
        self._subscribers: dict[str, dict[str, JanusSubscriber]] = {}
        self._route_lock = threading.RLock()  # 保护路由表的增删

        self._logger = logger or logging.getLogger("moss")
        self._log_prefix = "[LocalTopicService]"

        # 桥接钩子：用于对接 Channel (如 Zenoh/Circus)
        # 当 topic.meta.local == False 时，除了发给本地，还会塞入这个桥接队列
        self._bridge_outbound_queue: Optional[janus.Queue[Topic]] = None

    def set_bridge_queue(self, queue: janus.Queue[Topic]):
        """供外部 Proxy/Channel 注入，收集需要'出海'的 Topic"""
        self._bridge_outbound_queue = queue

    async def start(self):
        self._started = True
        self._service_stopped.clear()

    async def close(self):
        self.close_sync()

    def close_sync(self):
        if self._service_stopped.is_set():
            return
        self._service_stopped.set()

        with self._route_lock:
            for subs in self._subscribers.values():
                for sub in subs.values():
                    sub.close_sync()
            self._subscribers.clear()

    async def wait_sent(self):
        # 由于我们是点对点直推，没有缓冲队列，调用此方法时其实已经全部分发完毕
        pass

    def dispatch_sync(self, topic: Topic) -> None:
        """
        核心路由逻辑：O(1) 提取列表，直接推送。没有任何协程上下文切换。
        """
        if topic.is_overdue():
            return

        topic.meta.sender = self._sender
        topic_name = topic.meta.name

        # 1. 本地分发
        with self._route_lock:
            subs = self._subscribers.get(topic_name, {})
            # 创建快照，避免在派发时被修改
            active_subs = list(subs.values())

        for sub in active_subs:
            sub.receive_sync(topic)

        # 2. 桥接分发 (如果不是纯本地 Topic，且配置了出海队列)
        if not topic.meta.local and self._bridge_outbound_queue:
            try:
                self._bridge_outbound_queue.sync_q.put_nowait(topic)
            except janus.SyncQueueFull:
                self._logger.warning(f"{self._log_prefix} Bridge outbound queue full, dropping topic {topic.meta.id}")

    def is_running(self) -> bool:
        return self._started and not self._service_stopped.is_set()

    def listening(self) -> list[TopicName]:
        with self._route_lock:
            return list(self._subscribers.keys())

    def subscribe(self, topic_name: str, *, uid: str | None = None, maxsize: int = 0,
                  keep: Literal["latest", "oldest"] = "latest") -> Subscriber[None]:
        return self._create_subscriber(model=None, topic_name=topic_name, uid=uid, maxsize=maxsize, keep=keep)

    def subscribe_model(self, model: type[TOPIC_MODEL], *, topic_name: str = "", uid: str | None = None,
                        maxsize: int = 0, keep: Literal["latest", "oldest"] = "latest") -> Subscriber[TOPIC_MODEL]:
        return self._create_subscriber(model=model, topic_name=topic_name, uid=uid, maxsize=maxsize, keep=keep)

    def _create_subscriber(self, model: type[TopicModel] | None, *, topic_name: str = "", uid: str | None = None,
                           maxsize: int = 0, keep: Literal["latest", "oldest"] = "latest") -> Subscriber:
        sub = JanusSubscriber(
            self._service_stopped,
            model=model,
            topic_name=topic_name,
            uid=uid,
            maxsize=maxsize,
            keep=keep,
            logger=self._logger,
        )

        name = sub.listening()
        with self._route_lock:
            if name not in self._subscribers:
                self._subscribers[name] = {}
            self._subscribers[name][sub.id()] = sub

        return sub

    def publisher(self, creator: str, uid: str | None = None) -> LocalPublisher:
        return LocalPublisher(self, creator, uid)

    async def pub(self, topic: Topic | TopicModel, *, name: str = "", creator: str = "") -> None:
        publisher = self.publisher(creator=creator)
        publisher.pub_sync(topic, name=name)
        # 防御性让权：告诉 asyncio "我发完了，你可以去调度别的协程了"
        await asyncio.sleep(0)