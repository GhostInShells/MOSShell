import zmq
import zmq.asyncio
from ghoshell_moss.channels.duplex.connection import Connection, ConnectionClosedError
from ghoshell_moss.channels.duplex.protocol import ChannelEvent, HeartbeatEvent
from ghoshell_moss.channels.duplex.server import DuplexChannelServer
from ghoshell_moss.channels.duplex.client import DuplexChannelProxy
from ghoshell_container import Container, IoCContainer
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import time
import asyncio
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging

__all__ = [
    'ZMQChannelProxy', 'ZMQChannelServer',
    'ZMQConnectionConfig', 'ZMQServerConnection', 'ZMQClientConnection',
    'ZMQSocketType',
    'create_zmq_channel',
    'ConnectionClosedError',
]

logger = logging.getLogger(__name__)


class ZMQSocketType(Enum):
    PAIR = "PAIR"
    REQ = "REQ"
    REP = "REP"
    PUB = "PUB"
    SUB = "SUB"
    PUSH = "PUSH"
    PULL = "PULL"
    DEALER = "DEALER"
    ROUTER = "ROUTER"


@dataclass
class ZMQConnectionConfig:
    """ZMQ 连接配置"""
    address: str = "tcp://127.0.0.1:5555"
    socket_type: ZMQSocketType = ZMQSocketType.PAIR
    bind: bool = True  # True 表示绑定，False 表示连接
    recv_timeout: Optional[float] = None  # 接收超时（秒）
    send_timeout: Optional[float] = None  # 发送超时（秒）
    linger: int = 0  # socket 关闭时的等待时间（毫秒）
    identity: Optional[bytes] = None  # socket 身份标识
    subscribe: Optional[str] = None  # 订阅主题（仅用于 SUB socket）
    context: Optional[zmq.asyncio.Context] = None  # 共享的 ZMQ 上下文
    heartbeat_interval: float = 1.0  # 心跳间隔（秒）
    heartbeat_timeout: float = 3.0  # 心跳超时时间（秒）


# 修改 BaseZMQConnection 类，使其成为抽象基类
class BaseZMQConnection(Connection, ABC):
    """ZMQ 连接基类，包含心跳机制"""

    def __init__(self, config: ZMQConnectionConfig):
        self._config = config
        self._ctx = config.context or zmq.asyncio.Context.instance()
        self._socket: Optional[zmq.asyncio.Socket] = None
        self._closed_event = asyncio.Event()
        self._last_activity = time.time()
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._recv_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()

    def is_closed(self) -> bool:
        return self._closed_event.is_set()

    async def close(self) -> None:
        if self._closed_event.is_set():
            return
        self._closed_event.set()
        if self._heartbeat_task:
            await self._heartbeat_task
            self._heartbeat_task = None

    async def _handle_heartbeat(self, event: ChannelEvent) -> None:
        """服务器处理心跳：响应心跳请求"""
        try:
            heartbeat = HeartbeatEvent.from_channel_event(event)
            if not heartbeat:
                return
            if heartbeat.direction == "request":
                # 响应心跳
                response = HeartbeatEvent(direction="response").to_channel_event()
                await self.send(response)
        except Exception as e:
            logger.warning("Failed to handle heartbeat: %s", e)

    async def start(self) -> None:
        """启动连接，创建并配置 socket"""
        if self._socket is not None:
            return

        # 创建 socket
        socket_type = getattr(zmq, self._config.socket_type.value)
        self._socket = self._ctx.socket(socket_type)

        # 配置 socket
        if self._config.recv_timeout is not None:
            self._socket.RCVTIMEO = int(self._config.recv_timeout * 1000)
        if self._config.send_timeout is not None:
            self._socket.SNDTIMEO = int(self._config.send_timeout * 1000)
        if self._config.linger is not None:
            self._socket.linger = self._config.linger
        if self._config.identity is not None:
            self._socket.identity = self._config.identity

        # 绑定或连接
        if self._config.bind:
            self._socket.bind(self._config.address)
        else:
            self._socket.connect(self._config.address)

        # 订阅主题（如果是 SUB socket）
        if (self._config.socket_type == ZMQSocketType.SUB and
                self._config.subscribe is not None):
            self._socket.subscribe(self._config.subscribe)

        # 启动心跳任务（只有客户端需要）
        if not self._config.bind:  # 客户端需要心跳
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    @abstractmethod
    async def _heartbeat_loop(self) -> None:
        """心跳循环任务"""
        pass

    async def recv(self, timeout: Optional[float] = None) -> ChannelEvent:
        """接收消息，处理心跳"""
        if self._closed_event.is_set():
            raise ConnectionClosedError("Connection closed")
        if self._socket is None:
            raise RuntimeError("Connection not started")

        start_time = time.time()

        async with self._recv_lock:
            while not self._closed_event.is_set():
                try:
                    # 计算剩余超时时间
                    remaining_timeout = None
                    if timeout is not None:
                        elapsed = time.time() - start_time
                        if elapsed >= timeout:
                            raise asyncio.TimeoutError("Receive timeout")
                        remaining_timeout = timeout - elapsed

                    # 使用 asyncio.wait 同时等待接收操作和心跳检查
                    receive_task = asyncio.ensure_future(self._socket.recv_json())
                    tasks = [receive_task]
                    check_remaining_task = None
                    if remaining_timeout is not None:
                        check_remaining_task = asyncio.create_task(asyncio.sleep(remaining_timeout))
                        tasks.append(check_remaining_task)
                    check_closed_task = asyncio.create_task(self._closed_event.wait())
                    tasks.append(check_closed_task)

                    # 等待第一个完成的任务
                    done, pending = await asyncio.wait(
                        tasks,
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    # 取消未完成的任务
                    for task in pending:
                        task.cancel()

                    # 处理心跳超时
                    if check_remaining_task in done:
                        raise asyncio.TimeoutError("Receive timeout")
                    elif check_closed_task in done:
                        self._closed_event.set()
                        raise ConnectionClosedError("Connection closed")
                    # 处理接收到的消息
                    else:
                        message = await receive_task
                        # 更新最后活动时间
                        self._last_activity = time.time()

                        # 处理心跳消息
                        if message.get("event_type") == HeartbeatEvent.event_type:
                            await self._handle_heartbeat(message)
                            continue  # 继续等待有效消息

                        return message

                except zmq.ZMQError as e:
                    if e.errno == zmq.ETERM:
                        raise ConnectionClosedError("Connection closed")
                    elif e.errno == zmq.EAGAIN:
                        # 超时，检查是否因活动超时
                        if not self._config.bind and time.time() - self._last_activity > self._config.heartbeat_timeout:
                            raise ConnectionClosedError("Heartbeat timeout")
                        raise asyncio.TimeoutError("Receive timeout")
                    else:
                        raise
                except asyncio.TimeoutError:
                    # 检查是否因活动超时
                    if not self._config.bind and time.time() - self._last_activity > self._config.heartbeat_timeout:
                        raise ConnectionClosedError("Heartbeat timeout")
                    raise
                except Exception as e:
                    # 检查是否因活动超时
                    if not self._config.bind and time.time() - self._last_activity > self._config.heartbeat_timeout:
                        raise ConnectionClosedError("Heartbeat timeout") from e
                    raise

    async def send(self, event: ChannelEvent) -> None:
        """发送消息"""
        if self._closed_event.is_set():
            raise ConnectionClosedError("Connection closed")
        if self._socket is None:
            raise RuntimeError("Connection not started")

        async with self._send_lock:
            try:
                await self._socket.send_json(event)
                # 更新最后活动时间（发送成功也算活动）
                self._last_activity = time.time()
            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM:
                    raise ConnectionClosedError("Connection closed")
                elif e.errno == zmq.EAGAIN:
                    raise asyncio.TimeoutError("Send timeout")
                else:
                    raise


class ZMQServerConnection(BaseZMQConnection):
    """服务端 ZMQ 连接"""

    async def _heartbeat_loop(self) -> None:
        pass


class ZMQClientConnection(BaseZMQConnection):
    """客户端 ZMQ 连接"""

    async def _heartbeat_loop(self) -> None:
        """心跳循环任务（只有客户端需要）"""
        try:
            while not self._closed_event.is_set():
                # 检查活动超时
                if time.time() - self._last_activity > self._config.heartbeat_timeout:
                    logger.warning("Heartbeat timeout, closing connection")
                    await self.close()
                    break

                # 发送心跳请求
                if time.time() - self._last_activity > self._config.heartbeat_interval:
                    try:
                        heartbeat_event = HeartbeatEvent(direction="request").to_channel_event()
                        await self.send(heartbeat_event)
                    except Exception as e:
                        logger.warning("Failed to send heartbeat: %s", e)

                await asyncio.sleep(self._config.heartbeat_interval / 2)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Heartbeat loop error: %s", e)


class ZMQChannelServer(DuplexChannelServer):
    def __init__(
            self,
            *,
            address: str = "tcp://127.0.0.1:5555",
            socket_type: ZMQSocketType = ZMQSocketType.PAIR,
            recv_timeout: Optional[float] = None,
            send_timeout: Optional[float] = None,
            linger: int = 0,
            heartbeat_interval: float = 1.0,
            heartbeat_timeout: float = 3.0,
            context: Optional[zmq.asyncio.Context] = None,
            container: IoCContainer | None = None,
    ):
        # 创建 server 连接配置
        config = ZMQConnectionConfig(
            address=address,
            socket_type=socket_type,
            bind=True,  # server 端绑定地址
            recv_timeout=recv_timeout,
            send_timeout=send_timeout,
            linger=linger,
            heartbeat_interval=heartbeat_interval,
            heartbeat_timeout=heartbeat_timeout,
            context=context,
        )

        connection = ZMQServerConnection(config)
        super().__init__(
            to_client_connection=connection,
            container=Container(parent=container, name="ZMQChannelServer")
        )


class ZMQChannelProxy(DuplexChannelProxy):
    def __init__(
            self,
            *,
            name: str,
            block: bool,
            address: str = "tcp://127.0.0.1:5555",
            socket_type: ZMQSocketType = ZMQSocketType.PAIR,
            recv_timeout: Optional[float] = None,
            send_timeout: Optional[float] = None,
            linger: int = 0,
            identity: Optional[bytes] = None,
            heartbeat_interval: float = 1.0,
            heartbeat_timeout: float = 3.0,
            context: Optional[zmq.asyncio.Context] = None,
            description: str = "",
    ):
        # 创建 client 连接配置
        config = ZMQConnectionConfig(
            address=address,
            socket_type=socket_type,
            bind=False,  # client 端连接地址
            recv_timeout=recv_timeout,
            send_timeout=send_timeout,
            linger=linger,
            identity=identity,
            heartbeat_interval=heartbeat_interval,
            heartbeat_timeout=heartbeat_timeout,
            context=context,
        )

        connection = ZMQClientConnection(config)
        super().__init__(
            name=name,
            block=block,
            description=description,
            to_server_connection=connection,
        )


def create_zmq_channel(
        name: str,
        address: str = "tcp://127.0.0.1:5555",
        socket_type: ZMQSocketType = ZMQSocketType.PAIR,
        recv_timeout: Optional[float] = None,
        send_timeout: Optional[float] = None,
        linger: int = 0,
        identity: Optional[bytes] = None,
        heartbeat_interval: float = 1.0,
        heartbeat_timeout: float = 3.0,
        description: str = "",
        block: bool = True,
        container: IoCContainer | None = None,
) -> Tuple[ZMQChannelServer, ZMQChannelProxy]:
    """创建配对的 ZMQ server 和 proxy"""
    # 使用共享的上下文以确保正确通信
    ctx = zmq.asyncio.Context.instance()

    # 创建 server
    server = ZMQChannelServer(
        address=address,
        socket_type=socket_type,
        recv_timeout=recv_timeout,
        send_timeout=send_timeout,
        linger=linger,
        heartbeat_interval=heartbeat_interval,
        heartbeat_timeout=heartbeat_timeout,
        context=ctx,
        container=container,
    )

    # 创建 proxy
    proxy = ZMQChannelProxy(
        name=name,
        block=block,
        address=address,
        socket_type=socket_type,
        recv_timeout=recv_timeout,
        send_timeout=send_timeout,
        linger=linger,
        identity=identity,
        heartbeat_interval=heartbeat_interval,
        heartbeat_timeout=heartbeat_timeout,
        context=ctx,
        description=description,
    )

    return server, proxy
