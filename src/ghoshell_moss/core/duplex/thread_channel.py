import asyncio
from queue import Empty, Queue

from ghoshell_common.helpers import Timeleft
from ghoshell_container import Container, IoCContainer

from ghoshell_moss.core.duplex import (
    ChannelEvent,
    Connection,
    ConnectionClosedError,
    DuplexChannelProvider,
    DuplexChannelProxy,
)
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent

"""
# --- 测试专用 Channel --- 

测试专用的, 用多线程队列模拟一个 duplex channel.
用来测试 duplex 的各种功能设计. 
"""


class Provider2ProxyConnection(Connection):
    def __init__(
        self,
        *,
        provider_2_proxy_queue: Queue[ChannelEvent | None],
        proxy_2_provider_queue: Queue[ChannelEvent],
    ):
        self._closed = ThreadSafeEvent()
        self._send_queue = provider_2_proxy_queue
        self._recv_queue = proxy_2_provider_queue
        self._is_available = True

    def is_connected(self) -> bool:
        return not self._closed.is_set() and self._is_available

    async def recv(self, timeout: float | None = None) -> ChannelEvent:
        if self._closed.is_set():
            raise ConnectionClosedError("Connection closed")
        left = Timeleft(timeout or 0.0)

        def _recv_from_proxy() -> ChannelEvent:
            while not self._closed.is_set():
                try:
                    _timeout = left.left()
                    _timeout = _timeout if _timeout > 0.0 else 0.1
                    return self._recv_queue.get(block=True, timeout=_timeout)
                except Empty:
                    continue

        receiving = asyncio.create_task(asyncio.to_thread(_recv_from_proxy))
        closed = asyncio.create_task(self._closed.wait())
        done, pending = await asyncio.wait([receiving, closed], return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        if closed in done:
            raise ConnectionClosedError("Connection closed")
        return await receiving

    async def send(self, event: ChannelEvent) -> None:
        self._send_queue.put_nowait(event)

    def is_closed(self) -> bool:
        return self._closed.is_set()

    async def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._send_queue.put_nowait(None)

    async def start(self) -> None:
        pass


class Proxy2ProviderConnection(Connection):
    def __init__(
        self,
        *,
        provider_2_proxy_queue: Queue[ChannelEvent | None],
        proxy_2_provider_queue: Queue[ChannelEvent],
    ):
        self._closed = ThreadSafeEvent()
        self._send_queue = proxy_2_provider_queue
        self._recv_queue = provider_2_proxy_queue

    def is_connected(self) -> bool:
        return not self._closed.is_set()

    async def recv(self, timeout: float | None = None) -> ChannelEvent:
        if self._closed.is_set():
            raise ConnectionClosedError("Connection closed")

        _left = Timeleft(timeout or 0.0)

        def _recv_from_provider() -> ChannelEvent | None:
            while not self._closed.is_set():
                try:
                    _timeout = _left.left()
                    _timeout = _timeout if _timeout > 0.0 else 0.1
                    return self._recv_queue.get(block=True, timeout=_timeout)
                except Empty:
                    continue

        receiving = asyncio.create_task(asyncio.to_thread(_recv_from_provider))
        closed = asyncio.create_task(self._closed.wait())
        done, pending = await asyncio.wait([receiving, closed], return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        if closed in done:
            raise ConnectionClosedError("Connection closed")
        result = await receiving
        if result is None:
            self._closed.set()
            raise ConnectionClosedError("Connection closed")
        return result

    async def send(self, event: ChannelEvent) -> None:
        self._send_queue.put_nowait(event)

    def is_closed(self) -> bool:
        return self._closed.is_set()

    async def close(self) -> None:
        self._closed.set()

    async def start(self) -> None:
        pass


class ThreadChannelProvider(DuplexChannelProvider):
    def __init__(
        self,
        *,
        provider_connection: Provider2ProxyConnection,
        container: IoCContainer | None = None,
    ):
        super().__init__(
            provider_connection=provider_connection, container=Container(parent=container, name="ThreadChannelProvider")
        )


class ThreadChannelProxy(DuplexChannelProxy):
    def __init__(
        self,
        *,
        name: str,
        to_provider_connection: Proxy2ProviderConnection,
        description: str = "",
    ):
        super().__init__(
            name=name,
            description=description,
            to_provider_connection=to_provider_connection,
        )


def create_thread_channel(
    name: str,
    container: IoCContainer | None = None,
) -> tuple[ThreadChannelProvider, ThreadChannelProxy]:
    proxy_2_provider_queue = Queue()
    provider_2_proxy_queue = Queue()
    provider_side_connection = Provider2ProxyConnection(
        provider_2_proxy_queue=provider_2_proxy_queue,
        proxy_2_provider_queue=proxy_2_provider_queue,
    )
    proxy_side_connection = Proxy2ProviderConnection(
        provider_2_proxy_queue=provider_2_proxy_queue,
        proxy_2_provider_queue=proxy_2_provider_queue,
    )
    _provider = ThreadChannelProvider(
        provider_connection=provider_side_connection,
        container=container,
    )
    _proxy = ThreadChannelProxy(
        to_provider_connection=proxy_side_connection,
        name=name,
    )
    return _provider, _proxy
