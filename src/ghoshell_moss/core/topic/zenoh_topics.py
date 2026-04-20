from typing import Literal, Optional
from typing_extensions import Self

from ghoshell_moss import Addition
from ghoshell_moss.core.concepts.topic import (
    Publisher, Topic, Subscriber, TopicService, TopicModel, TOPIC_MODEL, TopicName,
    TopicClosedError,
)
from ghoshell_moss.depends import depend_zenoh
from ghoshell_moss.contracts import get_moss_logger, LoggerItf
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_common.helpers import uuid
from pydantic import ValidationError
from .suite_for_test import TopicServiceSuite
from .key_expr import MOSSTopicExpr
import janus
import asyncio
import threading
import orjson as json
import time

depend_zenoh()
import zenoh

__all__ = ['ZenohTopicSubscriber', 'ZenohTopicPublisher', 'ZenohTopicService', 'ZenohTopicServiceSuite']


class ZenohTopicService(TopicService):

    def __init__(
            self,
            session_id: str,
            session: zenoh.Session,
            node_name: str,
            *,
            logger: LoggerItf | None = None,
    ):
        self._session_id = session_id
        self._session = session
        # 一定要有一个 sender. 通常是 node name
        self._sender = node_name or uuid()
        self._logger = logger or get_moss_logger()
        self._subscriber_lock = asyncio.Lock()
        self._topic_key_expr = MOSSTopicExpr(session_id=session_id, node_name=node_name)

        self._publish_queue: janus.Queue[Topic] = janus.Queue()
        self._publish_queue_empty = asyncio.Event()
        self._main_loop_task: Optional[asyncio.Task] = None
        self._dispatch_tasks: set[asyncio.Task] = set()
        self._subscribing: set[TopicName] = set()
        self._publishing: set[TopicName] = set()
        self._log_prefix = "<ZenohBasedTopicService session_id=%s sender=%s>"
        self._started = False
        self._closing_event = ThreadSafeEvent()
        self._event_loop: asyncio.AbstractEventLoop | None = None

    def make_topic_key_expr(self, topic_name: str) -> str:
        return self._topic_key_expr.topic_key_expr(topic_name)

    def publishing(self) -> list[TopicName]:
        return list(self._publishing)

    def __repr__(self):
        return self._log_prefix

    async def start(self):
        if self._started:
            return
        self._started = True
        self._event_loop = asyncio.get_running_loop()

    async def close(self):
        self._closing_event.set()

    def is_running(self) -> bool:
        return self._started and not self._closing_event.is_set() and not self._session.is_closed()

    def subscribing(self) -> list[TopicName]:
        return list(self._subscribing)

    def subscribe(self, topic_name: str, *, uid: str | None = None, maxsize: int = 0,
                  model: type[TopicModel] | None = None) -> Subscriber:
        self._check_running()
        if model is not None:
            topic_name = topic_name or model.default_topic_name()

        key_expr = self.make_topic_key_expr(topic_name)
        self._subscribing.add(topic_name)
        return ZenohTopicSubscriber(
            session=self._session,
            service_stopped=self._closing_event,
            topic_name=topic_name,
            zenoh_key_expr=key_expr,
            uid=uid,
            maxsize=maxsize,
            model=model,
            logger=self._logger,
        )

    def _check_running(self):
        if not self.is_running():
            raise TopicClosedError(f"{self} is not running")

    def pub(self, topic: Topic | TopicModel, *, name: str = "", creator: str = "") -> None:
        self._check_running()
        if isinstance(topic, TopicModel):
            topic = topic.to_topic()
        if not isinstance(topic, Topic):
            raise TypeError("topic must be Topic")
        if name:
            topic.meta.name = name
        if not topic.meta.name:
            raise ValueError("topic must have a name")
        if creator:
            topic.meta.creator = creator
        key_expr = self.make_topic_key_expr(topic_name=topic.meta.name)

        def _publish():
            nonlocal key_expr, topic
            self._session.put(key_expr, topic.to_json())

        self._event_loop.run_in_executor(None, _publish)

    def publisher(self, creator: str, topic_name: str, *, uid: str | None = None,
                  model: type[TopicModel] | None = None) -> Publisher:
        self._check_running()
        if model is not None:
            topic_name = topic_name or model.default_topic_name()
        if not topic_name:
            raise ValueError("No topic name provided")
        key_expr = self.make_topic_key_expr(topic_name)
        self._publishing.add(topic_name)
        return ZenohTopicPublisher(
            session=self._session,
            service_stopped=self._closing_event,
            key_expr=key_expr,
            topic_name=topic_name,
            creator=creator,
            logger=self._logger,
            uid=uid,
        )


class ZenohTopicPublisher(Publisher):
    def __init__(
            self,
            *,
            session: zenoh.Session,
            service_stopped: ThreadSafeEvent,
            key_expr: str,
            topic_name: str,
            creator: str,
            uid: str | None = None,
            logger: LoggerItf | None = None,
            frequent: float = 0.0,
    ):
        self._zenoh_session = session
        self._zenoh_publisher: zenoh.Publisher | None = None
        self._service_stopped = service_stopped
        self._zenoh_key_expr = key_expr
        self._topic_name = topic_name
        self._creator = creator
        self._logger = logger or get_moss_logger()
        self._additions = []
        self._uid = uid or uuid()
        self._log_prefix = "<ZenohTopicPublisher creator=%s id=%s key=%s>" % (
            self._creator,
            self._uid,
            self._zenoh_key_expr,
        )
        self._frequent = frequent
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._undeclared_event = threading.Event()
        self._last_sent: float = 0.0
        self._started = False
        self._stopped = False

    def __repr__(self):
        return self._log_prefix

    def with_additions(self, *additions: Addition) -> Self:
        self._additions.extend(additions)
        return self

    def is_running(self) -> bool:
        return self._started and not self._stopped and not self._service_stopped.is_set()

    async def __aenter__(self) -> Self:
        if self._started:
            raise RuntimeError("Topic Service Already started")
        self._started = True
        self._zenoh_publisher = self._zenoh_session.declare_publisher(self._zenoh_key_expr)
        self._event_loop = asyncio.get_running_loop()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._stopped:
            return None
        self._stopped = True
        if self._zenoh_publisher is not None and not self._service_stopped.is_set():
            # undeclare for sure
            try:
                self._zenoh_publisher.undeclare()
            except RuntimeError:
                pass
            finally:
                self._undeclared_event.set()
        self._zenoh_publisher = None
        self._event_loop = None
        if exc_val is not None:
            if isinstance(exc_val, TopicClosedError):
                return True
            else:
                self._logger.exception("%s stopped cause error: %s", self._log_prefix, exc_val)
        return None

    def pub(self, topic: Topic | TopicModel, *, name: str = "") -> None:
        if not self.is_running():
            self._logger.info("%s drop topic %s cause not running", self._log_prefix, topic.meta.id)
            return
        if self._frequent > 0 and self._last_sent + self._frequent > time.time():
            self._logger.error("%s drop topic %s cause too frequent", self._log_prefix, topic.meta.id)
            return

        if isinstance(topic, TopicModel):
            topic = topic.to_topic()
        if name:
            topic.meta.name = name
        if len(self._additions) > 0:
            topic.with_additions(*self._additions)
        if topic.meta.name != self._topic_name:
            raise ValueError(f"topic name {topic.meta.name} != allowed topic name {self._topic_name}")
        topic.meta.creator = self._creator
        # 卸载到线程池里运行.
        self._event_loop.run_in_executor(None, self._pub_to_zenoh, topic)

    def _pub_to_zenoh(self, topic: Topic) -> None:
        try:
            if self._zenoh_session.is_closed():
                self._logger.info("%s drop topic %s cause session closed.", self._log_prefix, topic.meta)
                return None
            if self._zenoh_publisher is None:
                self._logger.info("%s drop topic %s cause publisher closed.", self._log_prefix, topic.meta)
                return None
            marshaled = topic.to_json()
            if self._undeclared_event.is_set():
                return None
            self._zenoh_publisher.put(marshaled)

        except zenoh.ZError as e:
            self._logger.exception("%s pub failed cause error: %s", self._log_prefix, e)
        except Exception as e:
            self._logger.exception("%s stopped cause error: %s", self._log_prefix, e)


class ZenohTopicSubscriber(Subscriber[TOPIC_MODEL | None]):

    def __init__(
            self,
            *,
            session: zenoh.Session,
            zenoh_key_expr: str,
            service_stopped: ThreadSafeEvent,
            model: type[TOPIC_MODEL] | None,
            topic_name: str = "",
            uid: str | None = None,
            maxsize: int = 0,
            logger: LoggerItf | None = None,
    ):
        self._session = session
        self._zenoh_key_expr = zenoh_key_expr
        self._declared_subscriber: zenoh.Subscriber | None = None
        self._zenoh_subscribing_thread: Optional[threading.Thread] = None
        self._service_stopped = service_stopped
        self._model: type[TopicModel] = model
        self._listening = topic_name or model.default_topic_name()
        self._uid = uid or uuid()
        self._queue: janus.Queue[Topic] = janus.Queue(maxsize=maxsize)
        self._receive_lock = asyncio.Lock()
        self._logger = logger or get_moss_logger()
        self._started = False
        self._closed = False
        self._service_wait_task: Optional[asyncio.Task] = None
        self._main_listening_loop_done_event = ThreadSafeEvent()
        self._log_prefix = f"<ZenohBasedSubscriber listening=%s id=%s>" % (self._listening, self._uid)

    def __repr__(self):
        return self._log_prefix

    async def __aenter__(self) -> Self:
        if self._started:
            return self
        self._started = True
        if self._session.is_closed():
            raise TopicClosedError(f"Zenoh session is closed")
        if self._service_stopped.is_set():
            raise TopicClosedError(f"Zenoh Topic Service is stopped")
        self._declared_subscriber = self._session.declare_subscriber(self._zenoh_key_expr)
        self._zenoh_subscribing_thread = threading.Thread(target=self._listening_loop, daemon=True)
        self._zenoh_subscribing_thread.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not self._closed:
            self._closed = True
            if self._declared_subscriber is not None and not self._service_stopped.is_set():
                try:
                    self._declared_subscriber.undeclare()
                except RuntimeError:
                    pass
            self._declared_subscriber = None
            self._zenoh_subscribing_thread = None
            # shutdown.
            self._queue.shutdown(immediate=False)
        if exc_val is not None:
            if isinstance(exc_val, TopicClosedError):
                return True
        return None

    def is_closed(self) -> bool:
        return (self._closed or self._main_listening_loop_done_event.is_set() or self._service_stopped.is_set()
                or self._session.is_closed())

    def is_running(self) -> bool:
        return self._started and not self._closed and not self._main_listening_loop_done_event.is_set()

    def _listening_loop(self):
        if self._declared_subscriber is None:
            return
        try:
            subscriber: zenoh.Subscriber = self._declared_subscriber
            for response in subscriber:
                if self._closed:
                    break
                sample = response
                self._logger.debug("%r receive sample from zenoh %s", self, sample.key_expr)
                self._receive_sample(sample)
        except janus.SyncQueueShutDown:
            # the service is done, will make the janus shutdown.
            pass
        except TopicClosedError:
            self._logger.info("%r zenoh subscribe listening loop stop on closed error", self)
        except zenoh.ZError as e:
            # 通常是 session 中断了.
            if self._session.is_closed():
                self._logger.info("%r zenoh subscribe listening loop stop on exception: %s", self, e)
            else:
                self._logger.exception("%r subscriber main loop failed: %s", self, e)
        finally:
            self._logger.info("%r listening loop stop on finally", self)
            self._main_listening_loop_done_event.set()
            self._queue.shutdown(immediate=False)

    def _receive_sample(self, sample: zenoh.Sample) -> None:
        """
        消化 Sample, 但是不要抛出特别异常.
        """
        try:
            # unserialize as json
            data = json.loads(sample.payload.to_bytes())
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            self._logger.exception("%r receive sample from zenoh failed: %s", self, e)
            return None

        try:
            topic = Topic(**data)
            self._receive(topic)
        except ValidationError as e:
            self._logger.warning(
                "%r receive sample from zenoh %s not valid topic: %s, value is %s",
                self, sample.key_expr, e, sample.payload.to_string()
            )
        except TopicClosedError:
            # 向上抛出.
            raise
        except Exception as e:
            self._logger.warning(
                "%r receive sample from zenoh key=%s failed: %s",
                self, sample.key_expr, e
            )

    def _receive(self, topic: Topic) -> None:
        """
        接受上游发送的消息.
        """
        if topic.meta.name != self._listening:
            return None
        elif topic.is_overdue():
            self._logger.info("%r drop overdue topic %s", self, topic.meta)
            return None
        elif self._service_stopped.is_set():
            self._logger.info("%r service stopped, drop topic %s", self, topic.meta)
            return None

        try:
            _queue = self._queue.sync_q
            if _queue.full():

                if not _queue.empty():
                    oldest = _queue.get_nowait()
                    self._logger.info("%r drop oldest topic %s cause full", self, oldest)
                _queue.put_nowait(topic)
            else:
                _queue.put_nowait(topic)
        except janus.SyncQueueShutDown:
            # shutdown
            raise TopicClosedError()
        except asyncio.QueueFull:
            self._logger.error("%s drop topic %s cause full", self._log_prefix, topic.meta.id)

    def listening(self) -> str:
        return self._listening

    def id(self) -> str:
        return self._uid

    async def poll(self, timeout: float | None = None) -> Topic:
        close_task = asyncio.create_task(self._service_stopped.wait())
        poll_task = asyncio.create_task(self._poll(timeout))
        done, pending = await asyncio.wait([close_task, poll_task], return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        if close_task in done:
            raise TopicClosedError()
        return await poll_task

    async def _poll(self, timeout: float | None = None) -> Topic:
        try:
            if timeout is not None and timeout > 0:
                return await asyncio.wait_for(self._queue.async_q.get(), timeout=timeout)
            else:
                return await self._queue.async_q.get()
        except asyncio.TimeoutError:
            raise
        except asyncio.CancelledError:
            raise
        except janus.AsyncQueueShutDown:
            raise TopicClosedError()

    async def poll_model(self, timeout: float | None = None) -> TOPIC_MODEL | None:
        topic = await self.poll(timeout)
        await asyncio.sleep(0)
        if self._model is not None:
            return self._model.from_topic(topic)
        return None


class ZenohTopicServiceSuite(TopicServiceSuite):

    def __init__(self):
        self._session: Optional[zenoh.Session] = None

    def name(self) -> str:
        return "zenoh"

    def create_service(self, sender: str) -> TopicService:
        self._session = zenoh.open(zenoh.Config())
        self._session.__enter__()
        return ZenohTopicService(
            session_id="session_id",
            session=self._session,
            node_name=sender,
        )

    def close(self) -> None:
        self._session.close()
