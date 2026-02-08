from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine, Iterable
from typing import (
    Any,
    ClassVar,
    Generic,
    Optional,
    Protocol,
    TypedDict,
    TypeVar,
    Union,
)

from ghoshell_common.helpers import generate_import_path, uuid
from pydantic import BaseModel, Field
from typing_extensions import Self

__all__ = ["ReqTopicModel", "Topic", "TopicBaseModel", "TopicCallback", "TopicModel"]


class Topic(TypedDict, total=False):
    """
    在 channel 之间广播的数据结构.
    不关心 topic broker 的通讯协议.
    """

    id: str
    """每个 topic 有唯一 id. """

    name: str
    """topic 的类型 id"""

    issuer: str
    """发布者的类型"""
    issuer_id: str
    """发布者的唯一 id. 这是假设有多个发布者的情况下. """

    req_id: Optional[str]
    """如果这个 topic 是对另一个 topic 的回复, 会携带那个 topic 的 id"""

    data: dict[str, Any] | list | str | bool | float | int | bytes | None
    """ topic 的数据结构. 基本要求是传递标量. """

    context: Optional[dict[str, Any]]
    """链路通讯, 追踪相关的上下文讯息. """


def make_topic_prefix(name: str, issuer: str = "", issuer_id: str = "") -> str:
    return f"{name}|{issuer}|{issuer_id}"


class TopicMeta(TypedDict):
    name: str
    description: str
    schema: dict[str, Any]


class TopicModel(Protocol):
    issuer: str
    issuer_id: str
    req_id: Optional[str]
    id: str

    @classmethod
    @abstractmethod
    def get_topic_name(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def to_topic_meta(cls) -> TopicMeta:
        pass

    @classmethod
    def from_topic(cls, topic: Topic) -> Self | None:
        pass

    @abstractmethod
    def new_topic(self, req_id: Optional[str] = None) -> Topic:
        pass


class TopicBaseModel(BaseModel, ABC):
    """
    一种简单的方式快速定义出 topic.
    """

    topic_name: ClassVar[str] = ""
    topic_description: ClassVar[str] = ""

    # topic 保留的关键字.

    issuer: str = Field(default="", description="Issuer of the topic")
    issuer_id: str = Field(default="", description="Issuer of the topic")
    req_id: Optional[str] = Field(default=None, description="the topic is response to topic id")
    id: str = Field(default_factory=uuid, description="the topic id")

    @classmethod
    def get_topic_name(cls) -> str:
        return cls.topic_name or generate_import_path(cls)

    @classmethod
    def to_topic_meta(cls) -> TopicMeta:
        return TopicMeta(
            name=cls.get_topic_name(),
            description=cls.topic_description or cls.__doc__ or "",
            schema=cls.model_json_schema(),
        )

    @classmethod
    def from_topic(cls, topic: Topic) -> Self | None:
        if topic["name"] != cls.get_topic_name():
            return None
        data = topic["data"]
        data["issuer"] = topic["issuer"]
        data["issuer_id"] = topic["issuer_id"]
        data["req_id"] = topic.get("req_id", None)
        data["id"] = topic["id"]

        model = cls(**data)
        return model

    def new_topic(self, issuer: str = "", req_id: Optional[str] = None) -> Topic:
        data = self.model_dump(exclude_none=True, exclude={"issuer", "req_id", "tid"})
        tid = self.topic_id or uuid()
        self.issuer = issuer or self.issuer
        self.req_id = req_id or self.req_id
        return Topic(
            id=tid,
            name=self.get_topic_name(),
            issuer=issuer,
            issuer_id=self.issuer_id,
            data=data,
            req_id=req_id,
        )


RESP = TypeVar("RESP", bound=TopicModel)


class ReqTopicModel(TopicBaseModel, Generic[RESP], ABC):
    """
    请求性质的 Topic. 它通常必须对应一个返回结果.
    """

    def new_response(self, resp: RESP) -> RESP:
        resp.req_id = self.id
        return resp


TopicCallback = Union[Callable[[Topic], Coroutine[None, None, None]] | Callable[[Topic], None]]
TopicModelCallback = Union[Callable[[TopicModel], Coroutine[None, None, None]] | Callable[[TopicModel], None]]


class Topics(ABC):
    @abstractmethod
    def on(self, topic_name: str, callback: TopicCallback) -> None:
        """
        注册 callback 函数, 同时监听这个 topic.
        todo: 未来增加更多过滤规则, 最好是通讯协议支持的.
        """
        pass

    @abstractmethod
    def on_model(self, topic_model: type[TopicModel], callback: TopicModelCallback) -> None:
        pass

    @abstractmethod
    def register(self, listening: Iterable[TopicMeta], sending: Iterable[TopicModel]) -> None:
        """
        注册本地可能的 topic 类型.
        """
        pass

    @abstractmethod
    async def send(self, topic: Topic | TopicModel) -> None:
        """
        发送一个 topic.
        """
        pass

    @abstractmethod
    async def call(self, req: ReqTopicModel[RESP], timeout: float | None) -> RESP:
        """
        发送一个 Topic, 并且等待结果.
        """
        pass

    @abstractmethod
    async def recv(self, timeout: float | None = None) -> Topic:
        """
        获取一个被广播的 topic
        :raise TimeoutError: 如果设置了 timeout.
        """
        pass
