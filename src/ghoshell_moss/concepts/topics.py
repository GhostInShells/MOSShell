import anyio
from pydantic import BaseModel, Field
from typing import TypedDict, Dict, Any, ClassVar, Literal, Optional, Union, List, Callable, Type, Coroutine, TypeVar
from typing_extensions import Self
from abc import ABC, abstractmethod
from ghoshell_common.helpers import generate_import_path, uuid


class Topic(TypedDict):
    id: str
    name: str
    issuer: str
    batch_id: Optional[str]
    complete: bool
    req_id: Optional[str]
    data: Dict[str, Any] | List | str | bool | float | int | bytes


def make_topic_prefix(name: str, issuer: str = "") -> str:
    return f"{name}||{issuer}" if issuer else name


class TopicMeta(TypedDict):
    name: str
    description: str
    schema: Dict[str, Any]


class TopicModel(BaseModel, ABC):
    topic_name: ClassVar[str] = ""
    topic_description: ClassVar[str] = ""

    issuer: str = Field(default='', description='Issuer of the topic')
    req_id: Optional[str] = Field(default=None, description='the topic is response to topic id')
    topic_id: str = Field(default_factory=uuid, description='the topic id')

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
        data['issuer'] = topic["issuer"]
        data['req_id'] = topic.get('req_id', None)
        data['tid'] = topic['id']

        model = cls(**data)
        return model

    def new_topic(self, issuer: str = "", req_id: Optional[str] = None) -> Topic:
        data = self.model_dump(exclude_none=True, exclude={'issuer', 'req_id', 'tid'})
        tid = self.topic_id or uuid()
        self.issuer = issuer or self.issuer
        self.req_id = req_id or self.req_id
        return Topic(
            id=tid,
            name=self.get_topic_name(),
            issuer=issuer,
            data=data,
            req_id=req_id,
        )


TopicModelType = TypeVar('TopicModelType', bound=TopicModel)


class TopicDispatcher(ABC):
    issuer: str

    @abstractmethod
    async def send(self, topic: Topic) -> None:
        pass

    @abstractmethod
    async def send_model(self, topic_model: TopicModel) -> None:
        pass

    @abstractmethod
    async def request(
            self,
            req: TopicModel,
            resp_type: Type[TopicModelType],
            timeout: float | None = None,
    ) -> TopicModelType:
        pass


TopicCallback = Callable[[Topic], Coroutine[None]]
TopicModelCallback = Callable[[TopicModel], Coroutine[None]]


class TopicListener(ABC):

    @abstractmethod
    def on(self, topic_prefix: str, callback: TopicCallback) -> None:
        pass

    @abstractmethod
    def on_model(self, topic_model: Type[TopicModel], callback: TopicModelCallback, issuer: str = "") -> None:
        pass

    @abstractmethod
    def listening(self) -> List[str]:
        pass


class TopicTransport(ABC):
    @abstractmethod
    def listener(self) -> TopicListener:
        pass

    @abstractmethod
    def dispatcher(self) -> TopicDispatcher:
        pass

    @abstractmethod
    async def start(self) -> None:
        """
        start to listen
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

    async def __aenter__(self):
        await self.start()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class CommandToken(TopicModel):
    topic_name: ClassVar[str] = "MOSShell.command_token"
    topic_description: ClassVar[str] = ""

    name: str = Field(description="command name")
    cid: str = Field(description="command id")
    chan: str = Field(description="channel name of the command")
    type: Literal['start', 'delta', 'end'] = Field(description="command token type")
    content: str = Field(description="the origin tokens")
    args: Optional[Dict[str, Any]] = Field(None, description="the command arguments")


class CommandResult(TopicModel):
    topic_name: ClassVar[str] = "MOSShell.command_result"
    topic_description: ClassVar[str] = ""

    cid: str = Field(description="command id")
    errcode: int = Field(0, description="command error code")
    errmsg: str = Field("", description="command error message")
    result: Union[list, bool, str, dict, None] = Field(None, description="the result of the command")
