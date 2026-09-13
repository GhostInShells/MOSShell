"""
Strongly-typed data (Topic) broadcast system implemented within the Shell layer,
used to build complex implementations.
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Literal, Any, Protocol, Annotated, Callable, List
from pydantic import BaseModel, Field, ValidationError
from ghoshell_moss.message import unique_id
from ghoshell_moss.message import WithAdditional, Addition
from typing_extensions import Self
import datetime
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
    "LogTopic",
    "ErrorTopic",
    "TopicNamePattern",
    "TopicSchema",
    "TopicWindow",
]

TopicNamePattern = r"^(|[a-zA-Z0-9]+(?:[._/-][a-zA-Z0-9]+)*)$"
TopicName = Annotated[str, Field(pattern=TopicNamePattern)]
TopicType = str


class TopicSchema(BaseModel):
    """
    self describing Topic Schema
    """
    topic_name: TopicName = Field(
        description="topic name",
        pattern=TopicNamePattern,
    )
    topic_type: TopicType = Field(
        description="topic type",
    )
    description: str = Field(
        default="",
        description="topic description",
    )
    json_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="topic json schema",
    )


class TopicMeta(BaseModel):
    """
    Reusable meta information for a topic.

    Its data structure is stable across transport and parsing, and it does not
    occupy keywords outside of meta.
    """

    id: str = Field(default_factory=unique_id, description="Unique identifier for the topic.")
    name: str = Field(
        default="",
        description="Name of the topic.",
        pattern=TopicNamePattern,
    )
    type: str = Field(default="", description="Type of the topic.")
    # local 实现的两种方式: 1. 不跨网络传输. 2. 监听者发现 sender 不相同, 直接丢弃.
    local: bool = Field(default=False, description="A local topic is not transported across the network.")
    creator: str = Field(
        default="",
        description="The unique identifier of the topic creator, in RESTFul format. "
                    "Unlike the sender: within the same communication link, multiple roles may create topics."
    )
    sender: str = Field(
        default="",
        description="The address of whom (topic service) sent this topic. "
                    "Unlike the creator, the sender is the identity on the communication link.",
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
    Topic information in the MOSS architecture, broadcast over the whole link via Pub/Sub.

    It solves the basic problem of proactive communication between Channel (经络,
    "meridian") and Shell, and between Channels.

    The technical principle resembles ROS2 topics, but the expected event frequency
    is very low — only second-scale brain events need to communicate through topics.

    Beyond this abstract design, the underlying transport can be implemented
    independently — for example, a dedicated MQTT event bus within the link.
    """

    meta: TopicMeta = Field(
        default_factory=TopicMeta,
        description="meta information",
    )

    data: dict = Field(
        description="the data of the topic",
    )

    @classmethod
    def from_data(cls, data: dict) -> Self:
        return cls(data=data)

    def is_overdue(self) -> bool:
        """Whether the topic is overdue. Overdue topics should be dropped immediately."""
        if self.meta.overdue == 0.0:
            # 永不过期.
            return False
        return self.meta.created_at + self.meta.overdue <= time.time()

    def to_json(self) -> str:
        return self.model_dump_json(indent=0, ensure_ascii=False, exclude_defaults=True, exclude_none=True)


class TopicModel(BaseModel, ABC, WithAdditional):
    """
    Self-describing Topic protocol contract.

    ``additional`` is a generic keyworded extension bag: a topic type can attach
    extra protocol data without declaring it as a field. It never travels inside
    ``Topic.data`` — ``to_topic()`` hoists it onto the ``Topic`` envelope and
    ``from_topic()`` restores it, so the wire has exactly one addition slot,
    shared with publisher-level additions (``Publisher.with_additions``).
    """

    meta: TopicMeta = Field(default_factory=TopicMeta, description="meta information")

    @property
    def created_at(self) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(self.meta.created_at)

    @classmethod
    @abstractmethod
    def topic_type(cls) -> str:
        """
        Defines the topic type. When using Topic rather than TopicModel, the topic
        type is needed to restore the specific TopicModel.
        """
        pass

    @classmethod
    def topic_schema(cls, topic_name: str | None = None) -> TopicSchema:
        """
        get topic schema from model.
        """
        if topic_name is None:
            topic_name = cls.default_topic_name()
        json_schema = cls.model_json_schema()
        # topic service generate meta
        del json_schema['properties']['meta']
        if '$defs' in json_schema:
            del json_schema['$defs']
        return TopicSchema(
            topic_name=topic_name,
            topic_type=cls.topic_type(),
            json_schema=json_schema,
            description=cls.__doc__ or '',
        )

    @classmethod
    def from_json(cls, js: bytes) -> Self | None:
        try:
            topic = Topic.model_validate_json(js)
            return cls.from_topic(topic)
        except ValidationError:
            return None

    @classmethod
    def from_topic(cls, topic: Topic) -> Self | None:
        if topic.meta.type != cls.topic_type():
            return None
        meta = topic.meta
        data = topic.data.copy()
        data['meta'] = meta
        if topic.additional:
            data['additional'] = topic.additional
        return cls.model_validate(data)

    @property
    def topic_name(self) -> TopicName:
        return self.meta.name

    @classmethod
    @abstractmethod
    def default_topic_name(cls) -> TopicName:
        """
        Defines the topic name. In principle one topic type can map to different topic
        names to route traffic selectively. Modeled after ROS2.

        In practice, the vast majority of topic names likely use the default.
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
        data = self.model_dump(exclude={"meta", "additional"}, exclude_none=True, exclude_defaults=True)
        meta = self.meta
        meta.name = name or self.default_topic_name()
        meta.overdue = overdue
        meta.creator = creator
        meta.sender = sender
        meta.type = self.topic_type()
        # additional 不进 data, 单独提到信封上, 与 publisher 级 additions 共用一个槽.
        additional = dict(self.additional) if self.additional else None
        # 由于是确定性的类型转换, 所以直接赋值.
        return Topic.model_construct(
            meta=meta,
            data=data,
            additional=additional,
        )


class LogTopic(TopicModel):
    """
    Experimental pattern: the logger implemented by a provider channel essentially
    sends logs as topics, and the proxy side writes those topics.
    """

    level: Literal["debug", "info", "warning", "error"] = "info"
    message: str = Field(description="The body text of the log message.")

    @classmethod
    def topic_type(cls) -> str:
        return "system/log"

    @classmethod
    def default_topic_name(cls) -> str:
        return "system/log"


class ErrorTopic(TopicModel):
    """
    A topic used for testing.
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
    A subscriber for a topic of a specified type.
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
        The topic name being listened to.
        """
        pass

    @abstractmethod
    def id(self) -> str:
        pass

    @abstractmethod
    async def poll(self, timeout: float | None = None) -> Topic:
        """
        :raise TopicClosedError: the service is already closed.
        :raise asyncio.TimeoutError: timed out.
        """
        pass

    @abstractmethod
    async def poll_model(self, timeout: float | None = None) -> TOPIC_MODEL | None:
        """
        :raise TopicClosedError: the service is already closed.
        :raise asyncio.TimeoutError: timed out.
        """
        pass

    @abstractmethod
    def is_closed(self) -> bool:
        """
        Whether it is marked as closed.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        Whether it is still running.
        """
        pass


class Publisher(Generic[TOPIC_MODEL], ABC):
    @abstractmethod
    def with_additions(self, *additions: Addition) -> Self:
        """
        Register Addition info carried by every topic.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        Whether it is still running.
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
            name: TopicName = "",
    ) -> None:
        """
        Publish an event. It is broadcast over the whole link.
        :raise TopicClosedError: the topic has stopped running.
        """
        pass


class TopicWindow(Generic[TOPIC_MODEL], ABC):
    """
    Bounded sliding window over a topic stream, backed by the TopicService lifecycle.

    Holds the most recent ``max_size`` typed ``TopicModel`` items. Designed for
    consumers that need the "latest N" pattern — monitoring dashboards, waveform
    displays, dialogue context windows, log tails.

    Every TopicWindow is created from and bound to a ``TopicService``. When the
    service closes, the window closes automatically — no manual lifecycle management.

    All read methods (``values()``, ``changed_at()``, ``__len__``) are thread-safe.
    ``on_change()`` callbacks are invoked from a thread pool — treat the callback
    body as a concurrent context.

    Call ``await wait_started()`` after creation and before publishing to ensure
    the underlying subscription is active.
    """

    @abstractmethod
    async def wait_started(self) -> None:
        """Block until the window's subscription is active and ready to receive."""
        pass

    @property
    @abstractmethod
    def max_size(self) -> int:
        """Maximum number of items retained. Fixed at creation time."""
        pass

    @abstractmethod
    def values(self) -> List[TOPIC_MODEL]:
        """
        Non-destructive snapshot of the current window contents.

        Index 0 is the oldest item, index -1 the newest. Returns a copy — the
        caller owns it and may mutate freely. Thread-safe.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Current number of items in the window. Thread-safe."""
        pass

    @abstractmethod
    def changed_at(self) -> float:
        """
        ``time.monotonic()`` timestamp of the most recent topic arrival.

        Updates on every receive, independent of ``on_change`` callback timing.
        Enables polling consumers to detect new data without registering a callback.
        Thread-safe.
        """
        pass

    @abstractmethod
    def on_change(
            self,
            callback: Callable[['TopicWindow'], None],
            *,
            debounce: float = 0,
            throttle: float = 0,
    ) -> Callable[[], None]:
        """
        Register a callback invoked when new topics arrive.

        The callback receives this window instance — call ``values()`` inside to
        get the current contents. Callbacks fire from a thread pool; keep the body
        fast and avoid blocking the pool.

        :param callback: Called with this window on new data.
        :param debounce: Quiet period in seconds. After a topic arrives, wait
            this long for further topics before firing. Resets on each arrival.
            Use for patterns like "transcribe after the speaker pauses."
        :param throttle: Maximum interval in seconds. If data keeps arriving
            without a debounce-sized gap, fire at least this often. Guarantees
            the callback won't be starved by a continuous stream.
        :return: A handle — call it to unregister the callback.
        """
        pass


class TopicService(ABC):
    """
    A basic TopicService implementing pub/sub in an asyncio environment.

    NOTE: TopicService is a business-layer implementation, not a physical-layer one.
    A physical-layer implementation must fully account for the MOSS architecture's
    multi-link duplex communication problem. Today the physical transport base is
    Duplex Channel Connection, which can provide a unified Connection layer between
    cross-process Channels.

    The core reason for this split: a MOSS runtime can build many heterogeneous
    communication channels through ChannelProxy => ChannelProvider, whereas a single
    Topic relying on a commonly discovered bus would lock in the physical
    implementation of the communication link.
    """

    @abstractmethod
    async def start(self):
        """
        Start the topic service.
        """
        pass

    @abstractmethod
    async def close(self):
        """
        Close the Topic Service.
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
        Whether it is currently running.
        """
        pass

    @abstractmethod
    def subscribing(self) -> list[TopicName]:
        """
        The names of all topics listened to via subscribe.
        """
        pass

    @abstractmethod
    def publishing(self) -> list[TopicName]:
        pass

    @abstractmethod
    def subscribe(
            self,
            topic_name: str,
            *,
            uid: str | None = None,
            maxsize: int = 0,
            model: type[TopicModel] | None = None,
    ) -> Subscriber:
        """
        Declare a subscribe; declarations only take effect after start.
        :param model: the Topic model to listen to.
        :param topic_name: if non-empty, falls back to topic_model.default_topic_name()
        :param uid: every subscriber needs an assigned uid; it can be auto-generated.
        :param maxsize: max queue size. 0 means unbounded, 1 means accept only one.

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
            topic_name: TopicName = "",
            uid: str | None = None,
            maxsize: int = 0,
    ) -> Subscriber[TOPIC_MODEL]:
        """
        Provides strong typing validation.
        """
        topic_name = topic_name or model.default_topic_name()
        return self.subscribe(
            topic_name,
            uid=uid,
            maxsize=maxsize,
            model=model,
        )

    @abstractmethod
    def create_window(
            self,
            topic_name: str,
            *,
            max_size: int,
            model: type[TopicModel] | None = None,
    ) -> TopicWindow:
        """
        Create a bounded sliding window over this topic.

        The window is bound to this service's lifecycle — when the service
        closes, the window closes automatically.

        :param topic_name: Topic to subscribe to.
        :param max_size: Maximum number of items retained. Must be >= 1.
        :param model: Optional typed model for deserialization.
        """
        pass

    def create_window_for(
            self,
            model: type[TOPIC_MODEL],
            *,
            topic_name: TopicName = "",
            max_size: int,
    ) -> TopicWindow[TOPIC_MODEL]:
        """Typed convenience wrapper around ``create_window()``."""
        topic_name = topic_name or model.default_topic_name()
        return self.create_window(
            topic_name,
            max_size=max_size,
            model=model,
        )

    @abstractmethod
    def pub(
            self,
            topic: Topic | TopicModel,
            *,
            name: TopicName = "",
            creator: str = "",
    ) -> None:
        """
        Publish an event. It is broadcast over the whole link.

        This form declares no topic publisher, which makes it hard to discover.
        :raise TopicClosedError: the topic has stopped running.
        """
        pass

    @abstractmethod
    def publisher(
            self,
            creator: str,
            topic_name: TopicName,
            *,
            uid: str | None = None,
            model: type[TopicModel] | None = None,
    ) -> Publisher:
        """
        Create a publisher — a publisher declares its own existence.
        :param creator: confirms the sender's identity, by convention.
        :param topic_name: the topic name to publish.
        :param uid: establishes a unique id for the sender.
        :param model: optionally adds a strong typing validation mechanism.

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
            topic_name: TopicName = "",
            uid: str | None = None,
    ) -> Publisher[TOPIC_MODEL]:
        """
        Provides a strong typing hint.
        """
        topic_name = topic_name or model.default_topic_name()
        return self.publisher(
            creator=creator,
            topic_name=topic_name,
            uid=uid,
            model=model,
        )
