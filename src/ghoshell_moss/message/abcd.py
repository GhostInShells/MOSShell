import json
import html
from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Protocol, Iterable, TypeAlias, is_typeddict
from ghoshell_common.helpers import uuid, generate_module_and_attr_name
from PIL import Image
from pydantic import BaseModel, Field, ValidationError, AwareDatetime
from typing_extensions import Self
from datetime import datetime, timezone
from pydantic_ai import UserContent, MultiModalContent, BinaryImage

__all__ = [
    "Addition",
    "Additional",
    "Content",
    "ContentModel",
    "HasAdditional",
    "Message",
    "MessageMeta",
    "WithAdditional",
]

"""
实现一个消息协议容器. 这个容器经过了几个阶段的改造: 
- 一阶段: ghostos 项目中定义了面向 openai 的消息协议, 用来解决自己的 multi-ghosts 等问题. 
- 二阶段: 为了实现 MOSS 架构在 channel meta 中依赖的消息定义, 重新定义了 message, 并且费劲做了协议兼容. 

目前是三阶段, 考虑完全导向 pydantic ai 或者 anthropic 协议. 维护消息协议太辛苦, 但它又是系统最底层.

从设计思想上, Message 放弃了流式传输层协议, 回到存储和同步协议: 
1. 提供可以兼容 openai、gemini、claude 等主流模型消息协议的容器。考虑直接使用 Pydantic AI
2. 彻底放弃 OpenAI 的强类型约定. 目前行业共同指向了消息体自解释, 也是殊途同归. 
3. 放弃下行 (模型生成), 专注于上行消息协议. 
"""

Additional = Optional[dict[str, dict[str, Any]]]
"""
各种数据类型的一种扩展协议.
它存储 弱类型/可序列化 的数据结构, 用 dict 来表示.
但它实际对应一个强类型的数据结构, 用 pydantic.BaseModel 来定义.
这样可以从弱类型容器中, 拿到一个强类型的数据结构, 但又不需要提前定义它. 
这个数据不对 AI 暴露, 属于 Ghost In Shells 架构自身定义的交互数据. 
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


_now_utc = lambda: datetime.now(timezone.utc)


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
        default_factory=uuid,
        description="消息的全局唯一 ID",
    )
    issuer_id: Optional[str] = Field(
        default=None,
        description="用来对 issuer 进行寻址. "
    )
    tag: str = Field(
        default='message',
        description="message tag that wrap the message information.",
    )

    role: str | None = Field(
        default=None,
        description="消息体的角色类型. 来自 感知器/用户/AI/功能 等等",
    )
    issuer: Optional[str] = Field(
        default=None,
        description="消息的发送",
    )
    name: Optional[str] = Field(
        default=None,
        description="消息的发送者身份. 作为 ghost in shells 架构中的标准概念.",
    )
    created_at: AwareDatetime | None = Field(
        default=None,
        description="消息的创建时间, 一个消息只有一个创建时间",
    )
    completed_at: AwareDatetime | None = Field(
        default=None,
        description="消息的结束时间",
    )
    incomplete: bool | None = Field(
        default=None,
        description="消息是否未结束",
    )
    attributes: dict[str, Any] = Field(
        default_factory=dict,
        description="额外的 attributes 属性. "
    )

    def as_incomplete(self) -> Self:
        """
        Ghost In Shells 特殊的协议标记.
        由于时间是第一公民, 所以消息协议的开头与结尾时间很重要.
        更重要的是, 按 ghost in shells 的设计, 模型可以看到未结束的信息.
        比如响应的瞬间, 用户的 asr 解析尚未完成.
        """
        self.incomplete = True
        self.completed_at = None
        return self

    def as_completed(self) -> Self:
        """
        标记消息为已经结束的消息.
        """
        self.incomplete = None
        self.completed_at = _now_utc()

    def gen_attributes(self) -> dict[str, Any]:
        attributes = self.attributes.copy()
        # 排除掉 ghost in shells 架构自身的关键维度信息.
        update = self.model_dump(exclude_none=True, exclude={'attributes', 'id', 'issuer_id', 'stage', 'tag'})
        if len(update) > 0:
            attributes.update(update)
        return attributes

    def gen_attributes_str(self) -> str:
        attributes = self.gen_attributes()
        if len(attributes) == 0:
            return ''
        parts = []
        for attr, value in attributes.items():
            if value == '':
                continue
            # in case value has invalid mark
            value = str(value)
            value = html.escape(value, quote=True)
            parts.append(f'{attr}="{value}"')
        attr_str = ' '.join(parts)
        return attr_str

    def to_xml(self) -> str:
        """
        生成 XML 讯息, 其中时序感是默认必要的.
        """
        attr_str = self.gen_attributes_str()
        tag = self.tag or 'meta'
        return f'<{tag} {attr_str}/>'


Content: TypeAlias = str | MultiModalContent
"""
完全导向 pydantic ai 的技术实现. 而且只用 UserContent, 做上行通讯. 放弃下行协议存储. 
"""


class ContentModel(BaseModel, ABC):
    """
    多模态消息单元的强类型定义.
    可以用来展示成指定的 <xml> 格式文本.
    """

    @abstractmethod
    def to_content(self) -> Content:
        """
        将强类型的数据结构, 转成弱类型的 content 对象.
        """
        pass


ContextType = ContentModel | str | Image.Image | BaseModel


class Message(BaseModel, WithAdditional):
    """
    MOSS 体系上行给模型的消息体. 目前完全倾向 Pydantic AI 数据结构.

    目标是:
    1. 兼容几乎所有的模型, 及其多模态消息类型. 依赖 Pydantic AI.
    2. 可以跨网络传输, 所有数据可以序列化.
    3. 可以用于本地存储.
    """

    protocol: Literal['pydantic_ai'] = Field(
        default='pydantic_ai',
        description="消息协议的类型. 未来可能要考虑扩充支持 raw 消息类型",
    )
    type: str = Field(
        default="",
        description="目标消息协议里的子类型. 用来生成具体的消息对象. ",
    )
    meta: MessageMeta = Field(
        default_factory=MessageMeta,
        description="消息的维度信息, 单独拿出来, 方便被其它数据类型所持有. ",
    )
    contents: list[Content] = Field(
        default_factory=list,
        description="消息里的原始 Content 对象.",
    )

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
            id=id or uuid(),
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

    @classmethod
    def to_content(cls, item: ContextType | Content) -> Content:
        if isinstance(item, str):
            _content = item
        elif isinstance(item, dict) and 'kind' in item:
            _content = item
        elif isinstance(item, ContentModel):
            _content = item.to_content()
        elif isinstance(item, Image.Image):
            _content = BinaryImage(item)
        elif isinstance(item, BaseModel):
            tag = generate_module_and_attr_name(item) or ''
            serialized = item.model_dump_json(indent=0, ensure_ascii=False, exclude_none=False)
            if tag:
                _content = f'<pydantic-model cls="{tag}">{serialized}</pydantic-model>'
            else:
                _content = serialized
        elif isinstance(item, dict) or isinstance(item, list):
            _content = json.dumps(item)
        else:
            _content = item
        return _content

    def with_content(self, *contents: ContextType | Content) -> Self:
        """
        用来添加 content. 简单做一个向前兼容的.
        """

        if self.contents is None:
            self.contents = []

        for item in contents:
            if item is None:
                continue
            _content = self.to_content(item)
            self.contents.append(_content)
        return self

    def is_empty(self) -> bool:
        return len(self.contents) == 0

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

    def as_contents(
            self,
            with_meta: bool = True,
            tag: str = 'message',
    ) -> Iterable[UserContent]:
        """
        将整个消息体返回成 Pydantic AI 的 User Content.
        """
        if self.is_empty():
            yield from []
            return
        if not with_meta:
            yield from self.contents
            return

        attrs = self.meta.gen_attributes_str()
        if with_meta and attrs:
            yield f'<{tag} {attrs}>'
        for content in self.contents:
            yield content
        if attrs:
            yield f'</{tag}>'
