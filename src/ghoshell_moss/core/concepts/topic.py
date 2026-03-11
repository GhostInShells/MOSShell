from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Literal

from pydantic import BaseModel, Field
from ghoshell_common.helpers import uuid
from ghoshell_moss.message import WithAdditional, Additional, Addition
from typing_extensions import Self
import time

__all__ = [
    "Topic",
    "TOPIC_MODEL",
    "TopicModel",
    "TopicService",
    "Subscriber",
    "Publisher",
    "ClosedError",
    "TopicName",
    "SubscribeKeep",
    "LogTopic",
    "ErrorTopic",
]

TopicName = str
SubscribeKeep = Literal["latest", "oldest"]
_TopicType = str


class TopicMeta(BaseModel):
    """
    定义 topic 可被复用的元信息.
    在传输和解析过程中它的数据结构不变, 也不占用 meta 之外的 keyword.
    """

    id: str = Field(default_factory=uuid, description="Unique identifier for the topic.")
    name: str = Field(default="", description="Name of the topic.")
    type: str = Field(default="", description="Type of the topic.")
    local: bool = Field(default=False, description="如果是 local 类型的 topic, 不会跨网络传输. ")
    creator: str = Field(
        default="",
        description="The unique identifier of the topic creator, in RESTFul format.",
    )
    sender: str = Field(
        default="",
        description="the address of whom sent this topic.",
    )
    created_at: float = Field(
        default_factory=lambda: round(time.time(), 4),
        description="Time when the topic was created. in seconds",
    )
    overdue: float = Field(
        default=0.0,
        description="Overdue after created, in seconds ",
    )


class Topic(BaseModel, WithAdditional):
    """
    MOSS 架构中的 Topic 信息, 也是基于 Pub/Sub 在全链路中广播.
    解决 Channel 与 Shell 主动通讯, Channel 之间通讯的基本问题.
    技术原理类似 Ros2 的 topics, 但是通信频率预期非长低, 应该是秒级的大脑事件才需要通过 topic 通讯.

    抽象设计之外, 底层逻辑完全可以自行实现. 比如在链路中独立一个 mqtt 用来做事件总线.

    可以慢慢迭代.
    """

    meta: TopicMeta = Field(description="meta information")

    data: dict = Field(
        description="the data of the topic",
    )

    def is_overdue(self) -> bool:
        if self.meta.overdue == 0.0:
            # 永不过期.
            return False
        return self.meta.created_at + self.meta.overdue <= time.time()


class TopicModel(BaseModel, ABC):
    meta: TopicMeta = Field(default_factory=TopicMeta, description="meta information")

    @classmethod
    @abstractmethod
    def topic_type(cls) -> str:
        """
        定义 topic 的类型. 对于使用 Topic 而非 TopicModel 的场景, 需要依赖 topic type 还原指定的 TopicModel.
        """
        pass

    @property
    def topic_name(self) -> str:
        return self.meta.name

    @classmethod
    @abstractmethod
    def default_topic_name(cls) -> str:
        """
        定义 topic name, 理论上一种 topic type 可以对应不同的 topic name 实现定向的分流.
        参考了 ros2 的模式.
        不过实际上, 可能绝大多数的 topic name 都使用默认的.
        """
        pass

    @classmethod
    def topic_schema(cls) -> dict:
        """
        通过这种方式, 一个服务可以展示它所有发送的 topic 和监听的 topic, 得到一个自解释的 schema 列表.
        """
        return cls.model_json_schema()

    def to_topic(
        self,
        *,
        name: str = "",
        overdue: float = 0.0,
        creator: str = "",
        sender: str = "",
    ) -> Topic:
        data = self.model_dump(exclude={"meta"})
        meta = self.meta
        meta.name = name or self.default_topic_name()
        meta.overdue = overdue
        meta.creator = creator
        meta.sender = sender
        return Topic(
            meta=meta,
            data=data,
            additional=None,
        )


class LogTopic(TopicModel):
    """
    实验性的范式, 考虑让 provider channel 实现的 logger 本质上是通过 topics 发送日志 topic
    然后 proxy 侧写入 topic.
    """

    level: Literal["debug", "info", "warning", "error"] = "info"
    message: str = Field(description="日志的正文讯息")

    @classmethod
    def topic_type(cls) -> str:
        return "system/log"

    @classmethod
    def default_topic_name(cls) -> str:
        return "system/log"


class ErrorTopic(TopicModel):
    """
    测试用的 topic.
    """

    errmsg: str = Field(
        description="the error message",
    )

    @classmethod
    def topic_type(cls) -> str:
        return "system/error"

    @classmethod
    def default_topic_name(cls) -> str:
        return "system/error"


TOPIC_MODEL = TypeVar("TOPIC_MODEL", bound=TopicModel | None)


class ClosedError(Exception):
    pass


class Subscriber(Generic[TOPIC_MODEL], ABC):
    """
    一个指定类型 topic 的监听者.
    """

    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    async def close(self) -> None:
        await self.__aexit__(None, None, None)

    @abstractmethod
    def listening(self) -> str:
        """
        监听的 topic name.
        """
        pass

    @abstractmethod
    def id(self) -> str:
        pass

    @abstractmethod
    async def poll(self, timeout: float | None = None) -> Topic:
        """
        :raise ClosedError: 服务已经关闭.
        :raise asyncio.TimeoutError: 超时.
        """
        pass

    @abstractmethod
    async def poll_model(self, timeout: float | None = None) -> TOPIC_MODEL | None:
        """
        :raise ClosedError: 服务已经关闭.
        :raise asyncio.TimeoutError: 超时.
        """
        pass

    @abstractmethod
    def is_closed(self) -> bool:
        """
        标记已经关闭.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        是否还在运行中.
        """
        pass


class Publisher(ABC):
    @abstractmethod
    def with_additions(self, *additions: Addition) -> Self:
        """
        注册所有 topic 都携带的 Addition 信息.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        是否还在运行中.
        """
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    async def pub(
        self,
        topic: Topic | TopicModel,
        *,
        name: str = "",
    ) -> None:
        """
        发布一个事件. 会在全链路里广播.
        :raise TopicServiceClosed: topic 已经停止运行.
        """
        pass


class TopicService(ABC):
    """
    实现一个基本的 TopicService, 能够实现 pub / sub
    现阶段没有人力和精力实现 QoS, 先基于基础链路来做.
    """

    @abstractmethod
    async def start(self):
        """
        启动 topic service.
        """
        pass

    @abstractmethod
    async def close(self):
        """
        关闭 Topic Service.
        """
        pass

    @abstractmethod
    async def wait_sent(self):
        pass

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_val and isinstance(exc_val, ClosedError):
            return True
        await self.close()

    @abstractmethod
    def is_running(self) -> bool:
        """
        是否正在运行中.
        """
        pass

    @abstractmethod
    def listening(self) -> list[TopicName]:
        """
        所有 subscribe 监听的 topic 名称.
        """
        pass

    @abstractmethod
    def subscribe(
        self,
        topic_name: str,
        *,
        uid: str | None = None,
        maxsize: int = 0,
        keep: SubscribeKeep = "latest",
    ) -> Subscriber[None]:
        pass

    @abstractmethod
    def subscribe_model(
        self,
        model: type[TOPIC_MODEL],
        *,
        topic_name: str = "",
        uid: str | None = None,
        maxsize: int = 0,
        keep: SubscribeKeep = "latest",
    ) -> Subscriber[TOPIC_MODEL]:
        """
        创建一个 subscriber.
        :param model: 监听的 Topic 模型.
        :param topic_name: 如果不为空, 会去迭代 topic_model.default_topic_name()
        :param uid: 每个 subscriber 都需要有指定的 uid. 可以自动生成.
        :param maxsize: 队列的最大数量. 为 0 表示无限, 为 1 表示只接受一个.
        :param keep: 当队列满了后, 新的 topic 发送过来的处理逻辑. oldest 会丢弃最新的 topic, latest 会丢弃最老的 topic.
        >>> async def consumer(service: TopicService):
        >>>     subscriber = service.subscribe_model(...)
        >>>     async with subscriber:
        >>>         while subscriber.is_running():
        >>>              topic = await subscriber.poll_model()
        """
        pass

    @abstractmethod
    async def pub(
        self,
        topic: Topic | TopicModel,
        *,
        name: str = "",
        creator: str = "",
    ) -> None:
        """
        发布一个事件. 会在全链路里广播.
        :raise TopicServiceClosed: topic 已经停止运行.
        """
        pass

    @abstractmethod
    def publisher(self, creator: str, uid: str | None = None) -> Publisher:
        """
        创建一个 publisher.

        :param creator: 确认发送者的身份.
        :param uid: 为发送者建立唯一 id.

        >>> async def publish(service: TopicService):
        >>>     publisher = service.publisher(...)
        >>>     async with publisher:
        >>>         await publisher.pub(...)
        """
        pass
