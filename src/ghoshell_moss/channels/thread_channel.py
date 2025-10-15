from typing import Tuple
from ghoshell_moss.channels.duplex import *
import asyncio
from queue import Queue, Empty
from ghoshell_container import Container, IoCContainer
from ghoshell_moss.helpers.asyncio_utils import ThreadSafeEvent
from ghoshell_common.helpers import Timeleft


# --- 测试专用 Channel --- #
# 测试专用的, 用多线程队列模拟一个 duplex channel.

class Server2ClientConnection(Connection):

    def __init__(
            self,
            *,
            server_2_client_queue: Queue[ChannelEvent | None],
            client_2_server_queue: Queue[ChannelEvent],
    ):
        self._closed = ThreadSafeEvent()
        self._send_queue = server_2_client_queue
        self._recv_queue = client_2_server_queue

    async def recv(self, timeout: float | None = None) -> ChannelEvent:
        if self._closed.is_set():
            raise ConnectionClosedError(f"Connection closed")
        left = Timeleft(timeout or 0.0)

        def _recv_from_client() -> ChannelEvent:
            while not self._closed.is_set():
                try:
                    _timeout = left.left()
                    _timeout = _timeout if _timeout > 0.0 else 0.1
                    return self._recv_queue.get(block=True, timeout=_timeout)
                except Empty:
                    continue

        receiving = asyncio.create_task(asyncio.to_thread(_recv_from_client))
        closed = asyncio.create_task(self._closed.wait())
        done, pending = await asyncio.wait([receiving, closed], return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        if closed in done:
            raise ConnectionClosedError(f"Connection closed")
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


class Client2ServerConnection(Connection):

    def __init__(
            self,
            *,
            server_2_client_queue: Queue[ChannelEvent | None],
            client_2_server_queue: Queue[ChannelEvent],
    ):
        self._closed = ThreadSafeEvent()
        self._send_queue = client_2_server_queue
        self._recv_queue = server_2_client_queue

    async def recv(self, timeout: float | None = None) -> ChannelEvent:
        if self._closed.is_set():
            raise ConnectionClosedError(f"Connection closed")

        _left = Timeleft(timeout or 0.0)

        def _recv_from_server() -> ChannelEvent | None:
            while not self._closed.is_set():
                try:
                    _timeout = _left.left()
                    _timeout = _timeout if _timeout > 0.0 else 0.1
                    return self._recv_queue.get(block=True, timeout=_timeout)
                except Empty:
                    continue

        receiving = asyncio.create_task(asyncio.to_thread(_recv_from_server))
        closed = asyncio.create_task(self._closed.wait())
        done, pending = await asyncio.wait([receiving, closed], return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()
        if closed in done:
            raise ConnectionClosedError(f"Connection closed")
        result = await receiving
        if result is None:
            self._closed.set()
            raise ConnectionClosedError(f"Connection closed")
        return result

    async def send(self, event: ChannelEvent) -> None:
        self._send_queue.put_nowait(event)

    def is_closed(self) -> bool:
        return self._closed.is_set()

    async def close(self) -> None:
        self._closed.set()

    async def start(self) -> None:
        pass


class ThreadChannelServer(DuplexChannelServer):

    def __init__(
            self,
            *,
            to_client_connection: Server2ClientConnection,
            container: IoCContainer | None = None,
    ):
        super().__init__(
            to_client_connection=to_client_connection,
            container=Container(parent=container, name="ThreadChannelServer")
        )


class ThreadChannelProxy(DuplexChannelProxy):

    def __init__(
            self,
            *,
            name: str,
            block: bool,
            description: str = "",
            to_server_connection: Client2ServerConnection,
    ):
        super().__init__(
            name=name,
            block=block,
            description=description,
            to_server_connection=to_server_connection,
        )


def create_thread_channel(
        name: str,
        description: str = "",
        block: bool = True,
        container: IoCContainer | None = None,
) -> Tuple[ThreadChannelServer, ThreadChannelProxy]:
    client_2_server_queue = Queue()
    server_2_client_queue = Queue()
    server_side_connection = Server2ClientConnection(
        server_2_client_queue=server_2_client_queue,
        client_2_server_queue=client_2_server_queue,
    )
    client_side_connection = Client2ServerConnection(
        server_2_client_queue=server_2_client_queue,
        client_2_server_queue=client_2_server_queue,
    )
    _server = ThreadChannelServer(
        to_client_connection=server_side_connection,
        container=container,
    )
    _proxy = ThreadChannelProxy(
        to_server_connection=client_side_connection,
        block=block,
        name=name,
        description=description,
    )
    return _server, _proxy
