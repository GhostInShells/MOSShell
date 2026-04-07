from ghoshell_container import IoCContainer

from ghoshell_moss.depends import depend_zenoh

depend_zenoh()

import zenoh
from ghoshell_moss.core.duplex import (
    Connection, ChannelEvent,
    ConnectionNotAvailable, ConnectionClosedError,
    DuplexChannelProxy,
)
from ghoshell_moss.core.duplex.protocol import HeartbeatEvent
from ghoshell_moss.contracts import LoggerItf, get_moss_logger
from ._utils import NodeChannelBridgeExpr
from pydantic import ValidationError
import janus
import asyncio
import orjson
import threading

__all__ = ["ZenohProxyConnection", 'ZenohProxyChannel']


class ZenohProxyConnection(Connection):
    """
    提供给 Proxy 端的 connection。
    逻辑与 Provider 完全对称，但 Key 表达式的方向相反。
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

        # 状态控制
        self._disconnected_event = threading.Event()
        self._receive_from_provider_queue: janus.Queue[ChannelEvent] = janus.Queue()
        self._logger_prefix = f"<ZenohProxyConnection node={node_name} session_id={self._session_id}>"

        # Zenoh 句柄
        self._subscriber: zenoh.Subscriber | None = None
        self._publisher: zenoh.Publisher | None = None
        self._provider_liveness_subscriber: zenoh.Subscriber | None = None
        self._liveness_token: zenoh.LivelinessToken | None = None

        self._started = False
        self._closed = False

    def __repr__(self):
        return self._logger_prefix

    def _receive_provider_event(self, sample: zenoh.Sample) -> None:
        """从 Provider 接收消息的回调"""
        try:
            data = orjson.loads(sample.payload.to_bytes())
            event = ChannelEvent(**data)

            # 过滤业务心跳
            if _ := HeartbeatEvent.from_channel_event(event):
                return None

            _queue = self._receive_from_provider_queue
            _queue.sync_q.put(event)
        except (orjson.JSONDecodeError, TypeError, ValidationError) as e:
            self._logger.error(
                "%s receive invalid event %s, failed: %s",
                self._logger_prefix, sample.payload.to_string(), e,
            )
        except janus.SyncQueueShutDown:
            pass

    def clear(self) -> None:
        if not self.is_running():
            return None
        if not self._receive_from_provider_queue.sync_q.empty():
            old_queue = self._receive_from_provider_queue
            self._receive_from_provider_queue = janus.Queue()
            old_queue.shutdown(immediate=True)
        return None

    async def recv(self, timeout: float | None = None) -> ChannelEvent:
        if not self.is_running():
            raise ConnectionClosedError(f"{self._logger_prefix} connection closed")
        if self._disconnected_event.is_set():
            raise ConnectionNotAvailable(f"{self._logger_prefix} connection not available")
        try:
            if timeout is not None and timeout > 0:
                item = await asyncio.wait_for(self._receive_from_provider_queue.async_q.get(), timeout=timeout)
            else:
                item = await self._receive_from_provider_queue.async_q.get()
            return item
        except (janus.AsyncQueueShutDown, asyncio.TimeoutError):
            raise ConnectionNotAvailable(f"{self._logger_prefix} connection not available")

    async def send(self, event: ChannelEvent) -> None:
        if not self.is_running():
            raise ConnectionClosedError(f"{self._logger_prefix} connection closed")
        if self._disconnected_event.is_set() or self._publisher is None:
            raise ConnectionNotAvailable(f"{self._logger_prefix} connection not available")

        # 同样采用同步 put，避免过度的协程切换
        self._send_event_to_provider(event)

    def _send_event_to_provider(self, event: ChannelEvent) -> None:
        try:
            if self._publisher is None:
                return
            payload = orjson.dumps(event)
            self._publisher.put(payload)
        except Exception as e:
            self._logger.error("%s send event to provider failed: %s", self._logger_prefix, e)

    def is_closed(self) -> bool:
        return self._closed or self._session.is_closed()

    def is_connected(self) -> bool:
        return not self.is_closed() and not self._disconnected_event.is_set()

    def is_running(self) -> bool:
        return self._started and not self.is_closed()

    async def start(self) -> None:
        if self._started:
            return
        self._started = True

        if self._session.is_closed():
            raise RuntimeError(f"{self._logger_prefix} zenoh session closed")

        # 1. 创建 Publisher: Proxy 发送给 Provider 的 Receiver
        publisher_key = self._bridge_expr.provider_receiver_key
        self._publisher = self._session.declare_publisher(publisher_key)

        # 2. 宣告自身的 Liveness: Proxy 告诉 Provider 我在
        proxy_liveness_key = self._bridge_expr.proxy_liveness_key
        self._liveness_token = self._session.liveliness().declare_token(proxy_liveness_key)

        # 3. 接收消息: 订阅 Provider 的 Publisher (即 Proxy 的 Receiver)
        subscriber_key = self._bridge_expr.proxy_receiver_key
        self._subscriber = self._session.declare_subscriber(subscriber_key, self._receive_provider_event)

        # 4. 监听 Provider Liveness: Provider 掉线则 Proxy 断开
        provider_liveness_key = self._bridge_expr.provider_liveness_key
        self._provider_liveness_subscriber = self._session.liveliness().declare_subscriber(
            provider_liveness_key,
            self._on_provider_liveness_sample,
        )

    def _on_provider_liveness_sample(self, sample: zenoh.Sample) -> None:
        if sample.kind == zenoh.SampleKind.PUT:
            self._disconnected_event.clear()
        elif sample.kind == zenoh.SampleKind.DELETE:
            self._disconnected_event.set()
            self.clear()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if not self._session.is_closed():
            # 这里的 undeclare 逻辑保持一致
            for resource in [self._publisher, self._subscriber,
                             self._provider_liveness_subscriber, self._liveness_token]:
                if resource is not None:
                    try:
                        resource.undeclare()
                    except RuntimeError:
                        pass

        self._publisher = None
        self._subscriber = None
        self._provider_liveness_subscriber = None
        self._liveness_token = None
        self.clear()


class ZenohProxyChannel(DuplexChannelProxy):

    def __init__(
            self,
            *,
            node_name: str,
            session_id: str,
            name: str,
            description: str = "",
            session: zenoh.Session | None = None,
    ):
        self._node_name = node_name
        self._session_id = session_id
        self._zenoh_session = session
        super().__init__(
            name=name,
            description=description,
            to_provider_connection=None,
        )

    def _create_connection(self, container: IoCContainer) -> Connection:
        session = self._zenoh_session
        if session is None:
            # must find from container
            session = container.force_fetch(zenoh.Session)
        return ZenohProxyConnection(
            session,
            node_name=self._node_name,
            session_id=self._session_id,
            logger=container.get(LoggerItf),
        )
