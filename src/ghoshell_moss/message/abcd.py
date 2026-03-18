import json
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from typing import Any, Callable, Literal, Optional, Protocol, Required, Iterable, TypeVar, Generic, NamedTuple

from ghoshell_common.helpers import timestamp_ms, uuid_md5, generate_module_and_attr_name
from PIL import Image
from pydantic import BaseModel, Field, ValidationError, AwareDatetime
from typing_extensions import Self, TypedDict, is_typeddict
from datetime import datetime, UTC, timezone

__all__ = [
    "Addition",
    "Additional",
    "Content",
    "ContentModel",
    "HasAdditional",
    "Message",
    "MessageMeta",
    "MessageTypeName",
    "Role",
    "WithAdditional",
    "MessageAdapter",
    "MessageTransformer",
    "RawT",
    "ToRawT",
    "MessageProtocol",
]

"""
实现一个通用的消息协议。

1. 提供可以兼容 openai、gemini、claude 等主流模型消息协议的容器。
2. 可以无限扩展，而不需要重新定义消息结构。
3. 支持存储, 通过 adapter 可以定义对模型的请求. 
"""


class Role(str, Enum):
    """
    消息体的角色, 兼容 OpenAI, 未来会有更多类型的消息.
    由于消息本身兼顾应用侧传输, 和 AI 侧的上下文, 所以会存在一些 AI 看不到, 由系统发送的消息类型.
    默认模型调用时会根据消息角色进行过滤, 只保留符合条件的类型.
    """

    UNKNOWN = ""
    USER = "user"  # 代表用户的消息
    ASSISTANT = "assistant"  # 代表 ai 自身
    SYSTEM = "system"  # 兼容 openai 的 system 类型, 现在已经切换为 developer 类型了.
    DEVELOPER = "developer"  # 兼容 openai 的 developer 类型消息.

    @classmethod
    def all(cls) -> set[str]:
        return {member.value for member in cls}

    def new_meta(self, name: Optional[str] = None, stage: str = "") -> "MessageMeta":
        return MessageMeta(role=self.value, name=name, stage=str(stage))

    def __str__(self):
        return self.value


class MessageTypeName(str, Enum):
    """
    系统定义的一些消息类型.

    关于 MessageType 和 ContentType 的定位区别：
    1. content type 是多模态消息的不同类型，比如文本、音频、图片等等。
    2. message type 是高阶类型，定义了整个 Ghost 实现中哪些模块需要理解这个消息。
        - 举个例子, 链路传输可能包含 debug 类型的消息, 它对图形界面展示很重要, 但对大模型则不需要理解.
    3. 在解析消息/渲染消息时, 对应的 Handler 应该先理解 message type.
    """

    DEFAULT = ""  # 默认多模态消息类型


Additional = Optional[dict[str, dict[str, Any]]]
"""
各种数据类型的一种扩展协议.
它存储 弱类型/可序列化 的数据结构, 用 dict 来表示.
但它实际对应一个强类型的数据结构, 用 pydantic.BaseModel 来定义.
这样可以从弱类型容器中, 拿到一个强类型的数据结构, 但又不需要提前定义它. 
"""


class HasAdditional(Protocol):
    """
    用来做类型约束的协议, 描述一个拥有 additional 能力的对象.

    举例:
    >>> def foo(obj: HasAdditional):
    >>>     return obj.additional
    """

    additional: Additional


class Addition(BaseModel, ABC):
    """
    用来定义一个强类型的数据结构, 但它可以转化为 Dict 放入弱类型的容器 (additional) 中.
    从而可以无限扩展一个消息协议.

    典型的例子:
    大模型的 message 协议有很多扩展字段:
    - 是哪个 agent 发送的
    - 来自哪个 session
    - token 的使用量如何

    如果要把这些字段都定义出来, 数据结构很容易耦合某种具体的协议, 而且整个消息协议会非常庞大.
    用 addition 的缺点是, 不能直接看到一个 Message 对象上绑定了多少种 Addition
    好处是可以遍历去获取.

    在这种机制下, 一个传输协议的 protocol 不是一次性定义的, 而是在项目的某个类库中攒出来的.
    """

    @classmethod
    @abstractmethod
    def keyword(cls) -> str:
        """
        每个 Addition 数据对象都要求有一个唯一的关键字
        建议用 a.b.c 风格来定义, 目前还没形成约束.
        """
        pass

    def get_or_create(self, target: HasAdditional) -> Self:
        """
        语法糖, 从一个 target 获取 addition, 或返回自己.
        """
        obj = self.read(target)
        if obj is not None:
            return obj
        self.set(target)
        return self

    @classmethod
    def read(cls, target: HasAdditional, throw: bool = False) -> Self | None:
        """
        从一个目标对象中读取 Addition 数据结构, 并加工为强类型.
        """
        if not hasattr(target, "additional") or target.additional is None:
            return None
        keyword = cls.keyword()
        data = target.additional.get(keyword, None)
        if data is None:
            return None
        try:
            wrapped = cls(**data)
            return wrapped
        except ValidationError as e:
            # 如果协议未对齐, 解析失败, 通常不抛出异常.
            if throw:
                raise e
            return None

    def set(self, target: HasAdditional) -> None:
        """
        将 Addition 数据结构加工到目标上.
        """
        if target.additional is None:
            target.additional = {}

        keyword = self.keyword()
        data = self.model_dump(exclude_none=True)
        target.additional[keyword] = data


class WithAdditional:
    """
    语法糖, 爱用不用.
    """

    additional: Additional = None

    def with_additions(self, *additions: Addition) -> Self:
        for add in additions:
            add.set(self)
        return self


class AdditionList:
    """
    一个简单的全局数据对象, 可以用于注册所有系统用到的 Addition
    然后把它们用 schema 的形式下发.

    这个实现不一定要使用. 它的好处是, 可以集中地拼出一个新的 Additions 协议自解释模块.
    """

    def __init__(self, *types: type[Addition]):
        self.types = {t.keyword(): t for t in types}

    def add(self, addition_type: type[Addition], override: bool = True) -> None:
        """
        注册新的 Addition 类型.
        """
        keyword = addition_type.keyword()
        if override and keyword in self.types:
            raise KeyError(f"Addition {keyword} is already added.")
        self.types[keyword] = addition_type

    def schemas(self) -> dict[str, dict]:
        """
        返回所有的 Addition 的 Schema.
        """
        result = {}
        for t in self.types.values():
            keyword = t.keyword()
            schema = t.model_json_schema()
            result[keyword] = schema
        return result


class MessageMeta(BaseModel):
    """
    消息的元信息, 用来标记消息的维度.
    独立出数据结构, 是为了方便将 meta 在不同的数据结构中使用, 而不用持有整个 message.

    这部分的数据也可能直接反应到模型看到的消息协议上 (content).
    举例, Anthropic 等消息协议, 并没有特别明确的强类型约束, 类似 role / name 等字段都需要基于约定来定义.

    实际上对于模型请求而言, 只需要两种协议罢了:
    1. input
    2. output
    """

    id: str = Field(
        default_factory=uuid_md5,
        description="消息的全局唯一 ID",
    )
    stage: str = Field(
        default='',
        description="生产消息所属的阶段, 可以用于在历史消息中过滤消息. ",
    )
    role: str = Field(
        default="",
        description="消息体的角色",
    )
    name: Optional[str] = Field(
        default=None,
        description="消息的发送者身份. 作为 ghost in shells 架构中的标准概念.",
    )
    issuer: Optional[str] = Field(
        default=None,
        description="发送者的身份讯息. 在 ghost in shells 架构里, 输入和输出都是多端的. "
    )
    issuer_id: Optional[str] = Field(
        default=None,
        description="用来对 issuer 进行寻址. "
    )
    created_at: AwareDatetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="消息的创建时间, 一个消息只有一个创建时间",
    )
    stop_reason: Optional[str] = Field(default=None, description="消息体中断的原因")

    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="额外的 attributes 属性. "
    )

    def to_xml(self) -> str:
        attributes = self.attributes.copy()
        update = self.model_dump(exclude_none=True, exclude={'attributes', 'id'})
        attributes.update(update)
        parts = []
        for attr, value in attributes.items():
            if value == '':
                continue
            parts.append(f"{attr}='{value}'")
        attr_str = ' '.join(parts)
        return f'<meta {attr_str} />'


class Content(TypedDict):
    """
    消息的通用内容体. 目标是以字符串的形式呈现.
    """

    type: Required[str]
    data: str


class ContentModel(BaseModel, ABC):
    """
    多模态消息单元的强类型定义.
    可以用来展示成指定的 <xml> 格式文本.
    """

    @classmethod
    @abstractmethod
    def content_type(cls) -> str:
        pass

    @classmethod
    def from_content(cls, content: Content) -> Self | None:
        """
        从 content 弱类型容器中还原出强类型的数据结构.
        """
        if content["type"] != cls.content_type():
            return None
        try:
            data = cls.unmarshal(content['data'])
            return cls(**data)
        except ValidationError:
            return None

    @abstractmethod
    def marshal(self) -> str:
        pass

    @classmethod
    @abstractmethod
    def unmarshal(cls, content: str) -> dict:
        pass

    def to_content(self) -> Content:
        """
        将强类型的数据结构, 转成弱类型的 content 对象.
        """
        return Content(
            type=self.content_type(),
            data=self.marshal(),
        )


ContextType = Content | ContentModel | str | Image.Image | BaseModel

MessageProtocol = str | Literal['', 'anthropic', 'pydantic_ai', 'openai', 'gemini']


class Message(BaseModel, WithAdditional):
    """
    模型传输过程中的消息体. 本质上是兼具 存储/传输/展示 功能的通用数据容器.

    目标是:
    1. 兼容几乎所有的模型, 及其多模态消息类型.
    2. 可以跨网络传输, 所有数据可以序列化.
    3. 可以用于本地存储.
    4. 本身也是一个兼容弱类型的容器, 除了消息本身必要的讯息外, 其它的讯息都是弱类型的. 避免传输时需要转化各种数据类型.
    """

    protocol: MessageProtocol = Field(
        default='',
        description="消息协议的类型, 用来将 raw 反解析成一个具体的协议",
    )
    type: str = Field(
        default="",
        description="目标消息协议里的子类型. 用来生成具体的消息对象. ",
    )
    version: str = Field(
        default='',
        description="与 protocol 一致的版本控制. 未来势必陷入转义地狱. "
    )
    meta: MessageMeta = Field(
        default_factory=MessageMeta,
        description="消息的维度信息, 单独拿出来, 方便被其它数据类型所持有. ",
    )
    raw: dict | str | None = Field(
        default=None,
        description="原始消息协议的可序列化数据. 用来反序列化. "
    )

    contents: list[Content] | None = Field(
        default=None,
        description="moss 自身要用到的消息体, 仅在 protocol 为 '' 时有意义. ",
    )

    __raw__: Any | None = None
    '''原始的消息数据, 序列化时不会使用. 但是在类型转换时应该优先检查. '''

    @classmethod
    def from_raw(
            cls,
            protocol: str,
            raw_data: dict,
            type: str,
            *,
            version: str = '',
            meta: MessageMeta | None = None,
            raw: Any | None = None,
            additions: list[Addition] | None = None,
    ):
        meta = meta or MessageMeta()
        r = cls(
            meta=meta,
            type=type,
            protocol=protocol,
            version=version,
            raw=raw_data,
        )
        r.__raw__ = raw
        if additions:
            r.with_additions(*additions)
        return r

    @staticmethod
    def content_to_xml(content: Content) -> str:
        """
        将 content 对象转化成 xml.
        """
        tag = content['type']
        if not tag:
            return content['data']
        return f'<{tag}>{content}</{tag}>'

    @classmethod
    def new(
            cls,
            *,
            role: str = "",
            name: Optional[str] = None,
            id: Optional[str] = None,
    ):
        """
        语法糖, 用来极简地一条消息.

        >>> msg = Message.new()
        """
        meta = MessageMeta(
            role=role,
            name=name,
            id=id or uuid_md5(),
        )
        return cls(meta=meta)

    @property
    def role(self) -> str:
        """
        从 meta 里拿到 role.
        """
        return self.meta.role

    @property
    def name(self) -> str | None:
        """
        从 meta 里拿到 name.
        """
        return self.meta.name

    @property
    def id(self) -> str:
        """
        从 meta 里拿到 id.
        """
        return self.meta.id

    def with_content(self, *contents: ContextType) -> Self:
        """
        用来添加 content.
        :deprecated: 希望未来用不同类型的 raw message. 不要自己迭代了.
        """
        from .contents import Base64Image, Text

        if self.contents is None:
            self.contents = []

        for item in contents:
            if item is None:
                continue
            elif is_typeddict(item):
                _content = item
            elif isinstance(item, ContentModel):
                _content = item.to_content()
            elif isinstance(item, str) and item:
                _content = Text(text=item).to_content()
            elif isinstance(item, Image.Image):
                _content = Base64Image.from_pil_image(item).to_content()
            elif isinstance(item, BaseModel):
                _content = Content(
                    type=generate_module_and_attr_name(item) or '',
                    data=item.model_dump_json(indent=0, ensure_ascii=False, exclude_none=False),
                )
            elif isinstance(item, dict):
                _content = item
            else:
                continue
            self.contents.append(_content)
        return self

    def is_empty(self) -> bool:
        return self.contents is None and self.raw is None

    def dump(self) -> dict[str, Any]:
        """
        生成一个 dict 数据对象, 用于传输.
        会返回默认值, 以防修改默认值后无法从序列化中还原.
        但不会包含 none, 节省序列化空间.
        """
        return self.model_dump(exclude_none=True)

    def to_json(self, indent: int = 0) -> str:
        """
        语法糖, 用来生成序列化.
        """
        return self.model_dump_json(indent=indent, ensure_ascii=False, exclude_none=True)

    def xml_tag(self) -> str:
        tag = 'message'
        if self.protocol:
            tag += f':{self.protocol}'
            if self.version:
                tag += f'-{self.version}'
        return tag

    def to_xml(self) -> str:
        """
        将消息体化作 xml 信息.
        """
        tag = self.xml_tag()
        meta_str = self.meta.to_xml()
        content_parts = []
        if self.contents:
            for c in self.contents:
                content_parts.append(self.content_to_xml(c))
        content_str = ' '.join(content_parts)
        return f'<{tag}>{meta_str}{content_str}</{tag}>'

    def as_contents(self) -> Iterable[ContentModel]:
        """
        通过这种方式, 将当前消息协议 Protocol 为 '' 的, 自动转化成 ContentModel
        如果不支持 message meta, 可以兼容这个协议.
        只需要转换 Text/Image 即可, meta 等信息都可以作为 Text 保存 (XML 语法)

        核心目标是为了兼容 Anthropic 的消息协议
        """
        from ghoshell_moss.message.contents import ContentModelsDict, Text
        tag = self.xml_tag()
        # 返回消息的开标记.
        yield Text.new(f'<{tag}>')
        # 返回 Meta 信息.
        yield Text.new(self.meta.to_xml())
        if self.contents:
            for c in self.contents:
                if c['type'] in ContentModelsDict:
                    model_type = ContentModelsDict[c['type']]
                    model = model_type.from_content(c)
                    if model is not None:
                        yield model
                    continue
                else:
                    yield Text.new(Message.content_to_xml(c))
        yield Text.new(f'</{tag}>')

    def __str__(self):
        return self.to_xml()


RawT = TypeVar('RawT')


class MessageAdapter(Generic[RawT], ABC):
    """
    消息协议转换器.
    """

    @classmethod
    @abstractmethod
    def protocol(cls) -> str:
        pass

    @abstractmethod
    def raw_to_message(self, raw: RawT) -> Message:
        """
        将一个原始类型, 变成可存储传输的 Message 类型.
        """
        pass

    @abstractmethod
    def message_to_raw(self, message: Message) -> RawT | None:
        """
        将一个 Message 类型, 存储为 Raw 类型.
        """
        pass


FromRawT = TypeVar('FromRawT')
ToRawT = TypeVar('ToRawT')


class MessageProtocolBridge(Generic[FromRawT, ToRawT], ABC):
    """
    消息协议转换器. 不关 Message 的事情了.
    """

    @classmethod
    def from_protocol(cls) -> str:
        pass

    @classmethod
    def to_protocol(cls) -> str:
        pass

    @abstractmethod
    def transform(self, item: FromRawT) -> ToRawT | None:
        pass


class Expect(Generic[RawT]):
    def __init__(self, protocol: str, expect_type: type[RawT], type_checker: Callable[[Any], bool] | None = None):
        self.protocol = protocol
        self.expect_type = expect_type
        self.type_checker = type_checker

    def check_type(self, item: Any) -> bool | None:
        if self.type_checker is None:
            return None
        return self.type_checker(item)


class MessageTransformer:
    """
    多重类型转换.
    """

    def __init__(self, adapters: list[MessageAdapter], bridges: list[MessageProtocolBridge]):
        self._adapters: dict[str, MessageAdapter] = {}
        self._bridges: dict[str, dict[str, MessageProtocolBridge]] = {}
        for adapter in adapters:
            self._adapters[adapter.protocol()] = adapter

        for bridge in bridges:
            from_protocol = bridge.from_protocol()
            if from_protocol not in self._bridges:
                self._bridges[from_protocol] = {}
            self._bridges[from_protocol][bridge.to_protocol()] = bridge

    @staticmethod
    def expect(
            protocol: str,
            raw_type: type[ToRawT],
            *,
            type_checker: Callable[[Any], None] | None = None,
    ) -> Expect[ToRawT]:
        """
        用来做类型提示. 可以节省引用一个类.
        """
        return Expect(protocol, raw_type, type_checker=type_checker)

    def raw_to_message(self, raw: RawT, protocol: str) -> Message | None:
        if raw is None:
            return None
        if isinstance(raw, Message):
            return raw

        if protocol not in self._adapters:
            return None
        # 只转换一层.
        return self._adapters[protocol].raw_to_message(raw)

    def message_to_raw(
            self,
            message: Message,
            expect: Expect[ToRawT] | None = None,
    ) -> ToRawT | None:
        """
        做消息类型的多重转换.

        >>> def parse(transformer: MessageTransformer, msg: Message) -> str:
        >>>     # 一个极端的例子
        >>>     return transformer.message_to_raw(msg, transformer.expect('str', str))
        """
        raw_protocol = message.protocol
        if raw_protocol == "":
            raw_message = message
        elif raw_protocol not in self._adapters:
            return None
        else:
            raw_message = self._adapters[raw_protocol].message_to_raw(message)

        if expect is None:
            # 直接返回 raw message, 无论是什么类型.
            return raw_message
        if expect.protocol == raw_protocol:
            # 返回符合目标的协议.
            return raw_message

        # 还是原始消息协议.
        if raw_message.protocol == "":
            # 走 adapter 逻辑.
            if expect.protocol not in self._adapters:
                return None
            adapter = self._adapters[expect.protocol]
            return adapter.message_to_raw(message)

        # 走桥逻辑.
        if raw_protocol in self._bridges:
            if expect.protocol in self._bridges[raw_protocol]:
                bridge = self._bridges[raw_protocol][expect.protocol]
                return bridge.transform(raw_message)
        return None

    def parse_messages_to_raw(
            self,
            messages: list[Message],
            expect: Expect[RawT] | None = None,
    ) -> Iterable[RawT]:
        for message in messages:
            raw = self.message_to_raw(message, expect)
            if raw is not None:
                if expect is None or expect.check_type(raw) is not False:
                    yield raw

    def parse_raw_to_messages(
            self,
            protocol: str,
            raw: list[RawT],
            *,
            additions: list[Addition] | None = None,
            issuer: str | None = None,
            issuer_id: str | None = None,
    ) -> Iterable[Message]:
        """
        将目标类型的消息, 转换成 moss 的消息容器.
        """
        for msg in raw:
            item = self.raw_to_message(msg, protocol)
            if item is not None:
                if additions:
                    item.with_additions(*additions)
                if issuer:
                    item.meta.issuer = issuer
                if issuer_id:
                    item.meta.issuer_id = issuer_id
                yield item
