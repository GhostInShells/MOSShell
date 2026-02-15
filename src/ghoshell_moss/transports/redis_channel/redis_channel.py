import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Optional

try:
    from redis.asyncio import Redis
    from redis.exceptions import ConnectionError, ResponseError
except ImportError:
    raise ImportError('redis is not installed, please install it with "pip install ghoshell-moss[redis]"')

from ghoshell_container import Container, IoCContainer

from ghoshell_moss.core.duplex.connection import Connection, ConnectionClosedError
from ghoshell_moss.core.duplex.protocol import ChannelEvent
from ghoshell_moss.core.duplex.provider import DuplexChannelProvider
from ghoshell_moss.core.duplex.proxy import DuplexChannelProxy

logger = logging.getLogger(__name__)


class RedisStreamConnection(Connection):
    """基于Redis Stream的双工通信连接"""

    def __init__(
        self,
        redis: Redis,
        write_stream: str,
        read_stream: str,
        consumer_group: Optional[str] = None,
        consumer_id: Optional[str] = None,
    ):
        """
        初始化Redis流连接

        :param redis: Redis实例
        :param write_stream: 写入消息的流
        :param read_stream: 读取消息的流
        :param consumer_group: Redis消费者组名称（用于读取）
        :param consumer_id: 消费者ID（用于读取）
        """
        self._redis = redis
        self._write_stream = write_stream
        self._read_stream = read_stream
        self._consumer_group = consumer_group
        self._consumer_id = consumer_id or f"consumer-{uuid.uuid4().hex[:8]}"
        self._closed_event = asyncio.Event()
        self._recv_lock = asyncio.Lock()
        self._send_lock = asyncio.Lock()
        self._last_id = "0-0"
        self._group_created = False

    async def _ensure_group(self):
        """确保消费者组已创建（如果需要）"""
        if self._consumer_group and not self._group_created:
            try:
                await self._redis.xgroup_create(self._read_stream, self._consumer_group, id="0", mkstream=True)
                self._group_created = True
            except ResponseError as e:
                if "BUSYGROUP" in str(e):
                    # 消费者组已存在
                    self._group_created = True
                else:
                    logger.exception("Failed to create consumer group")
                    raise

    async def recv(self, timeout: Optional[float] = None) -> ChannelEvent:
        """从Redis Stream接收消息"""
        if self._closed_event.is_set():
            raise ConnectionClosedError("Connection closed")

        async with self._recv_lock:
            # 设置阻塞超时（毫秒）
            block = int(timeout * 1000) if timeout else 0

            while not self._closed_event.is_set():
                try:
                    # 如果需要消费者组
                    if self._consumer_group:
                        await self._ensure_group()
                        result = await self._redis.xreadgroup(
                            groupname=self._consumer_group,
                            consumername=self._consumer_id,
                            streams={self._read_stream: self._last_id},
                            count=1,
                            block=block,
                        )
                    else:
                        # 不使用消费者组，直接读取
                        result = await self._redis.xread(
                            streams={self._read_stream: self._last_id}, count=1, block=block
                        )
                        logger.debug("Raw Redis read result: %s", result)

                    if not result:
                        if block == 0:
                            # 非阻塞模式下没有消息直接返回
                            continue
                        # 超时处理
                        raise asyncio.TimeoutError("Receive timed out")

                    stream, messages = result[0]
                    if not messages:
                        continue

                    msg_id, message = messages[0]
                    self._last_id = msg_id

                    # 解析消息内容
                    # redis 默认是 bytes 类型，可以通过设置 decode_responses=True 来解码为字符串，举例：
                    # redis_client = redis.Redis(
                    #     host='localhost',
                    #     port=6379,
                    #     decode_responses=True,  # Import!!!
                    #     encoding='utf-8'
                    # )
                    payload = message.get(b"payload") or message.get("payload")
                    if not payload:
                        logger.warning("Received empty payload message: %s", message)
                        continue

                    event = json.loads(payload)
                    logger.info("RedisStreamConnection Received event: %s", event)
                    return event
                except ConnectionError as e:
                    logger.exception("Redis connection error")
                    raise ConnectionClosedError(f"Redis connection closed: {e}")

    async def send(self, event: ChannelEvent) -> None:
        """发送消息到Redis Stream"""
        if self._closed_event.is_set():
            raise ConnectionClosedError("Connection closed")

        async with self._send_lock:
            try:
                # 序列化事件并发送到相应流
                payload = json.dumps(event)
                await self._redis.xadd(self._write_stream, {"payload": payload})
                logger.info(
                    "RedisStreamConnection sending event to Redis stream %s: %s",
                    self._write_stream,
                    event,
                )
            except ConnectionError as e:
                logger.exception("Redis connection error")
                raise ConnectionClosedError(f"Redis connection failed: {e}")
            except Exception:
                logger.exception("Error sending message to Redis")
                raise

    def is_closed(self) -> bool:
        return self._closed_event.is_set()

    def is_connected(self) -> bool:
        return not self.is_closed() and self._redis is not None

    async def close(self) -> None:
        """关闭连接并清理资源"""
        if self._closed_event.is_set():
            return

        self._closed_event.set()
        # 注意：不关闭Redis连接，由外部管理

    async def start(self) -> None:
        """启动连接"""
        if self._consumer_group:
            await self._ensure_group()


@dataclass
class RedisConnectionConfig:
    """Redis Channel配置"""

    redis: Redis
    write_stream: str
    read_stream: str
    consumer_group: Optional[str] = None
    consumer_id: Optional[str] = None

    async def clear_stream(self):
        """清空Redis Stream"""
        await self.redis.delete(self.write_stream, self.read_stream)


class RedisChannelProxy(DuplexChannelProxy):
    """基于Redis的Channel代理（客户端）"""

    def __init__(
        self,
        config: RedisConnectionConfig,
        *,
        name: str,
        description: str = "",
    ):
        connection = RedisStreamConnection(
            redis=config.redis,
            write_stream=config.write_stream,
            read_stream=config.read_stream,
            consumer_group=config.consumer_group,
            consumer_id=config.consumer_id,
        )
        super().__init__(
            name=name,
            to_provider_connection=connection,
            description=description,
        )


class RedisChannelProvider(DuplexChannelProvider):
    """基于Redis的Channel提供者（服务端）"""

    def __init__(
        self,
        config: RedisConnectionConfig,
        *,
        container: Optional[IoCContainer] = None,
    ):
        connection = RedisStreamConnection(
            redis=config.redis,
            write_stream=config.write_stream,
            read_stream=config.read_stream,
            consumer_group=config.consumer_group,
            consumer_id=config.consumer_id,
        )
        super().__init__(
            provider_connection=connection, container=Container(parent=container, name="RedisChannelProvider")
        )
