import threading

from ghoshell_container import IoCContainer, Container

from ghoshell_moss.depends import depend_zenoh

depend_zenoh()
import zenoh

from ghoshell_moss.core.duplex import (
    DuplexChannelProvider, Connection, ChannelEvent,
    ConnectionNotAvailable, ConnectionClosedError,
)
from ghoshell_moss.core.duplex.protocol import HeartbeatEvent
from ghoshell_moss.contracts import LoggerItf, get_moss_logger
from ._utils import NodeChannelBridgeExpr
from pydantic import ValidationError
import janus
import asyncio
import orjson
import time

__all__ = ['ZenohProviderConnection', 'ZenohChannelProvider']


class ZenohProviderConnection(Connection):
    """
    提供给 Zenoh Provider 的 connection.
    它应该:
    - 广播 Provider liveness
    - 监听 Proxy liveness
    - 推送给 Proxy receiver
    - 从 Provider receiver 拉取.

    Channel Provider 需要有唯一性. 不过考虑不通过 Connection 实现, 而是通过 Node 去管理.
    """

    def __init__(
            self,
            session: zenoh.Session,
            *,
            node_name: str,
            session_id: str,
            logger: LoggerItf | None = None,
    ) -> None:
        self._logger = logger or get_moss_logger()
        self._session_id = session_id
        self._session = session
        self._node = node_name
        self._bridge_expr = NodeChannelBridgeExpr(session_id=self._session_id, node_name=self._node)
        # 默认为 disconnected.
        self._disconnected_event = threading.Event()
        # 从 proxy 读取的队列.
        self._receive_from_proxy_queue: janus.Queue[ChannelEvent] = janus.Queue()
        self._logger_prefix = f"<ZenohProviderConnection node={node_name} session_id={self._session_id}>"
        # 标记最后通信联通时间.
        self._last_liveness_heartbeat: float = 0.0
        self._subscriber: zenoh.Subscriber | None = None
        self._publisher: zenoh.Publisher | None = None
        self._proxy_liveness_subscriber: zenoh.Subscriber | None = None
        self._liveness_token: zenoh.LivelinessToken | None = None
        self._started = False
        self._closed = False

    def __repr__(self):
        return self._logger_prefix

    def is_running(self) -> bool:
        return self._started and not self.is_closed()

    def _receive_proxy_event(self, sample: zenoh.Sample) -> None:
        try:
            data = orjson.loads(sample.payload.to_bytes())
            event = ChannelEvent(**data)
            self._last_liveness_heartbeat = time.time()
            if _ := HeartbeatEvent.from_channel_event(event):
                return None
            _queue = self._receive_from_proxy_queue
            _queue.sync_q.put(event)
        except (orjson.JSONDecodeError, TypeError, ValidationError) as e:
            self._logger.error(
                "%s receive invalid event %s, failed: %s",
                self._logger_prefix, sample.payload.to_string(), e,
            )
        except janus.SyncQueueShutDown:
            self._logger.info(
                "%s drop received event: %s",
                self._logger_prefix, sample.payload.to_string(),
            )

    def clear(self) -> None:
        if not self.is_running():
            return None
        # 清空所有数据发送, 不要浪费时间.
        if not self._receive_from_proxy_queue.sync_q.empty():
            old_receive_queue = self._receive_from_proxy_queue
            self._receive_from_proxy_queue = janus.Queue()
            old_receive_queue.shutdown(immediate=True)
        return None

    async def recv(self, timeout: float | None = None) -> ChannelEvent:
        if not self.is_running():
            raise ConnectionClosedError(f"{self._logger_prefix} connection closed")
        if self._disconnected_event.is_set():
            raise ConnectionNotAvailable(f"{self._logger_prefix} connection not available")
        try:
            if timeout is not None and timeout > 0:
                item = await asyncio.wait_for(self._receive_from_proxy_queue.async_q.get(), timeout=timeout)
            else:
                item = await self._receive_from_proxy_queue.async_q.get()
            return item
        except janus.AsyncQueueShutDown:
            raise ConnectionNotAvailable(f"{self._logger_prefix} connection not available")

    async def send(self, event: ChannelEvent) -> None:
        if not self.is_running():
            raise ConnectionClosedError(f"{self._logger_prefix} connection closed")
        if self._disconnected_event.is_set() or self._publisher is None:
            raise ConnectionNotAvailable(f"{self._logger_prefix} connection not available")
        try:
            self._send_event_to_proxy(event)
        except janus.AsyncQueueShutDown:
            raise ConnectionNotAvailable(f"{self._logger_prefix} connection not available")

    def _send_event_to_proxy(self, event: ChannelEvent) -> None:
        try:
            if self._publisher is None:
                return
            payload = orjson.dumps(event)
            # 卸载到线程池但是阻塞?
            self._publisher.put(payload)
        except TypeError as e:
            self._logger.error("%s send event to proxy failed: %s", self._logger_prefix, e)
        except zenoh.ZError as e:
            self._logger.info("%s send event to proxy failed: %s", self._logger_prefix, e)

    def is_closed(self) -> bool:
        return self._closed or self._session.is_closed()

    def is_connected(self) -> bool:
        return not self.is_closed() and not self._disconnected_event.is_set()

    async def start(self) -> None:
        if self._started:
            return
        if self._session.is_closed():
            raise RuntimeError(f"{self._logger_prefix} zenoh session closed")
        self._started = True
        # 创建 publisher
        publisher_key = self._bridge_expr.proxy_receiver_key
        self._publisher = self._session.declare_publisher(publisher_key)
        # 宣告 liveness
        provider_liveness_key = self._bridge_expr.provider_liveness_key
        self._liveness_token = self._session.liveliness().declare_token(provider_liveness_key)
        # 接受 Proxy 消息.
        subscriber_key = self._bridge_expr.provider_receiver_key
        self._subscriber = self._session.declare_subscriber(subscriber_key, self._receive_proxy_event)
        # 监听 proxy liveness.
        proxy_liveness_key = self._bridge_expr.proxy_liveness_key
        self._proxy_liveness_subscriber = self._session.liveliness().declare_subscriber(
            proxy_liveness_key,
            self._on_proxy_liveness_sample,
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._session.is_closed():
            if self._publisher is not None:
                try:
                    self._publisher.undeclare()
                except RuntimeError:
                    pass
            if self._subscriber is not None:
                try:
                    self._subscriber.undeclare()
                except RuntimeError:
                    pass
            if self._proxy_liveness_subscriber is not None:
                try:
                    self._proxy_liveness_subscriber.undeclare()
                except RuntimeError:
                    pass
            if self._liveness_token is not None:
                try:
                    self._liveness_token.undeclare()
                except RuntimeError:
                    pass
        self._publisher = None
        self._subscriber = None
        self._proxy_liveness_subscriber = None
        self._liveness_token = None
        self.clear()

    def _on_proxy_liveness_sample(self, sample: zenoh.Sample) -> None:
        if sample.kind == zenoh.SampleKind.PUT:
            self._disconnected_event.clear()
        elif sample.kind == zenoh.SampleKind.DELETE:
            self._disconnected_event.set()
            self.clear()


class ZenohChannelProvider(DuplexChannelProvider):
    """
    基于 Zenoh 提供的 Channel Provider.
    """

    def __init__(
            self,
            *,
            node_name: str,
            session_id: str,
            container: IoCContainer | None = None,
            session: zenoh.Session | None = None,
            liveness_check_interval: float = 3.0,
    ):
        self._node_name = node_name
        self._session_id = session_id
        if session is None:
            if container is None:
                raise ValueError("container or session must be provided")
            else:
                session = container.get(zenoh.Session)
        if session is None:
            raise ValueError("session must be provided as argument or from container")
        self._session = session
        if container is None:
            container = Container()
            container.set(zenoh.Session, session)
        self._session = session
        self._liveness_check_interval = liveness_check_interval
        connection = ZenohProviderConnection(
            session=session,
            session_id=session_id,
            node_name=node_name,
            logger=container.get(LoggerItf),
        )
        super().__init__(
            provider_connection=connection,
            container=container,
            reconnect_interval_seconds=self._liveness_check_interval
        )
