from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Literal, TypedDict, Required, Any, Protocol, ClassVar

from pydantic import BaseModel, Field
from ghoshell_common.helpers import uuid
from ghoshell_moss.message import WithAdditional, Addition
from typing_extensions import Self
import time

__all__ = [
    "Topic",
    "TOPIC_MODEL",
    "TopicModel",
    "TopicMeta",
    "TopicService",
    "Subscriber",
    "Publisher",
    "TopicClosedError",
    "TopicName",
    "SubscribeKeep",
    "LogTopic",
    "ErrorTopic",
]

TopicName = str
SubscribeKeep = Literal["latest", "oldest"]
_TopicType = str


class TopicSchema(TypedDict):
    """
    self describing Topic Schema
    """
    topic_name: Required[TopicName]
    topic_type: Required[_TopicType]
    json_schema: Required[dict[str, Any]]


class TopicMeta(BaseModel):
    """
    定义 topic 可被复用的元信息.
    在传输和解析过程中它的数据结构不变, 也不占用 meta 之外的 keyword.
    """

    id: str = Field(default_factory=uuid, description="Unique identifier for the topic.")
    name: str = Field(default="", description="Name of the topic.")
    type: str = Field(default="", description="Type of the topic.")
    # local 实现的两种方式: 1. 不跨网络传输. 2. 监听者发现 sender 不相同, 直接丢弃.
    local: bool = Field(default=False, description="如果是 local 类型的 topic, 不会跨网络传输. ")
    creator: str = Field(
        default="",
        description="The unique identifier of the topic creator, in RESTFul format.",
    )
    sender: str = Field(
        default="",
        description="the address of whom (topic service) sent this topic.",
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
        """topic 是否过期. 过期的 Service 应该直接丢弃. """
        if self.meta.overdue == 0.0:
            # 永不过期.
            return False
        return self.meta.created_at + self.meta.overdue <= time.time()

    def to_json(self) -> str:
        return self.model_dump_json(indent=0, ensure_ascii=False, exclude_defaults=True, exclude_none=True)


class TopicModel(BaseModel, ABC):
    """
    自解释的 Topic 协议约定.
    """

    meta: TopicMeta = Field(default_factory=TopicMeta, description="meta information")

    @classmethod
    @abstractmethod
    def topic_type(cls) -> str:
        """
        定义 topic 的类型. 对于使用 Topic 而非 TopicModel 的场景, 需要依赖 topic type 还原指定的 TopicModel.
        """
        pass

    @classmethod
    def topic_schema(cls, topic_name: str | None = None) -> TopicSchema:
        """
        get topic schema from model.
        """
        if topic_name is None:
            topic_name = cls.default_topic_name()
        # todo: 考虑 json_schema 里大量冗余都是 meta 的部分.
        return TopicSchema(
            topic_name=topic_name,
            topic_type=cls.topic_type(),
            json_schema=cls.model_json_schema(),
        )

    @classmethod
    def from_topic(cls, topic: Topic) -> Self | None:
        if topic.meta.type != cls.topic_type():
            return None
        meta = topic.meta
        data = topic.data.copy()
        data['meta'] = meta
        return cls(**data)

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

    def to_topic(
            self,
            *,
            name: str = "",
            overdue: float = 0.0,
            creator: str = "",
            sender: str = "",
    ) -> Topic:
        data = self.model_dump(exclude={"meta"}, exclude_none=True, exclude_defaults=True)
        meta = self.meta
        meta.name = name or self.default_topic_name()
        meta.overdue = overdue
        meta.creator = creator
        meta.sender = sender
        meta.type = self.topic_type()
        return Topic(
            meta=meta,
            data=data,
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


TOPIC_MODEL = TypeVar("TOPIC_MODEL", bound=TopicModel)


class TopicClosedError(Exception):
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


class Publisher(Generic[TOPIC_MODEL], ABC):
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
    def pub(
            self,
            topic: Topic | TOPIC_MODEL,
            *,
            name: str = "",
    ) -> None:
        """
        发布一个事件. 会在全链路里广播.
        :raise ClosedError: topic 已经停止运行.
        """
        pass


class TopicService(ABC):
    """
    实现一个基本的 TopicService, 能够实现 pub / sub
    注意!! TopicService 是业务层的实现, 并不是物理层的实现. 物理层的实现要充分考虑 MOSS 架构的多链路双工通讯问题.
    目前物理层通讯的底座是 Duplex Channel Connection.
    可以在 Channel 跨进程通讯之间提供统一的 Connection 层.

    这么做的核心原因是, 一个 MOSS 运行时可以通过 ChannelProxy => ChannelProvider 搭建多种异构的通讯通道.
    而单一的 Topic 依赖一个共同发现的总线, 会导致通讯链路的物理实现锁定.
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


    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_val and isinstance(exc_val, TopicClosedError):
            return True
        await self.close()
        return None

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
            model: type[TopicModel] | None = None,
    ) -> Subscriber:
        """
        声明一个 Subscribe, 只有启动后声明才生效.
        :param model: 监听的 Topic 模型.
        :param topic_name: 如果不为空, 会去迭代 topic_model.default_topic_name()
        :param uid: 每个 subscriber 都需要有指定的 uid. 可以自动生成.
        :param maxsize: 队列的最大数量. 为 0 表示无限, 为 1 表示只接受一个.
        :param keep: 当队列满了后, 新的 topic 发送过来的处理逻辑. oldest 会丢弃最新的 topic, latest 会丢弃最老的 topic.

        >>> async def consumer(service: TopicService):
        >>>     subscriber = service.subscribe_model(...)
        >>>     async with subscriber:
        >>>          try:
        >>>              topic = await subscriber.poll_model()
        >>>          except TopicClosedError:
        >>>              pass
        """
        pass

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
        提供一个强类型校验.
        """
        topic_name = topic_name or model.default_topic_name()
        return self.subscribe(
            topic_name,
            uid=uid,
            maxsize=maxsize,
            keep=keep,
            model=model,
        )

    @abstractmethod
    def pub(
            self,
            topic: Topic | TopicModel,
            *,
            name: str = "",
            creator: str = "",
    ) -> None:
        """
        发布一个事件. 会在全链路里广播.
        这种方式没有声明 topic publisher, 不利于被发现.
        :raise TopicServiceClosed: topic 已经停止运行.
        """
        pass

    @abstractmethod
    def publisher(
            self,
            creator: str,
            topic_name: str,
            *,
            uid: str | None = None,
            model: type[TopicModel] | None = None,
    ) -> Publisher:
        """
        创建一个 publisher. 声明自己的存在啊.
        :param creator: 确认发送者的身份. 基于约定.
        :param topic_name: the topic name to publish.
        :param uid: 为发送者建立唯一 id.
        :param model: 可以加一个强类型校验机制.

        >>> async def publish(service: TopicService):
        >>>     publisher = service.publisher(...)
        >>>     async with publisher:
        >>>         publisher.pub(...)
        """
        pass

    def model_publisher(
            self,
            creator: str,
            model: type[TOPIC_MODEL],
            *,
            topic_name: str = "",
            uid: str | None = None,
    ) -> Publisher[TOPIC_MODEL]:
        """
        提供一个强类型提示.
        """
        topic_name = topic_name or model.default_topic_name()
        return self.publisher(
            creator=creator,
            topic_name=topic_name,
            uid=uid,
            model=model,
        )


# --- todo: creator 的声明约定

class TopicCreator(Protocol):
    """
    方便未来做显示约定.
    暂时不使用.
    """

    @classmethod
    @abstractmethod
    def from_creator(cls, creator: str) -> Self | None:
        pass

    def to_creator(self) -> str:
        pass

    def __str__(self):
        return self.to_creator()


class ChannelCreator(TopicCreator):

    def __init__(self, channel_path: str):
        self.channel_path = channel_path
        self.creator = f"channel/{channel_path}"

    @classmethod
    def from_creator(cls, creator: str) -> Self | None:
        if creator.startswith("channel/"):
            parts = creator.split("/", maxsplit=1)
            channel_path = ''
            if len(parts) == 2:
                channel_path = parts[1]
            return cls(channel_path)
        return None

    def to_creator(self) -> str:
        return self.creator
