import orjson
import html
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Literal, Optional, Protocol, Iterable, TypeAlias
from PIL import Image
from pydantic import BaseModel, Field, ValidationError, AwareDatetime, dataclasses
from typing_extensions import Self
from datetime import datetime
from dateutil import tz
from .contents import ContentModel, Content, Text, Base64Image
from ulid import ULID
import ghoshell_common


def _ulid_gen() -> str:
    return str(ULID())


# patch uuid to ulid
ghoshell_common.helpers.uuid = _ulid_gen
from ghoshell_common.helpers import uuid

__all__ = [
    "Addition",
    "Additional",
    "HasAdditional",
    "Message",
    "MessageMeta",
    "WithAdditional",
]

# 实现一个消息协议容器. 这个容器经过了几个阶段的改造:
# - 一阶段: ghostos 项目中定义了面向 openai 的消息协议, 用来解决自己的 multi-ghosts 等问题.
# - 二阶段: 为了实现 MOSS 架构在 channel meta 中依赖的消息定义, 重新定义了 message, 并且费劲做了协议兼容.
# - 三阶段: 考虑完全导向 pydantic ai. 期望 pydantic ai 的消息协议更通用.
#
# 目前是四阶段, pydantic ai 的类库太重, 而它反序列化很困难, 仍然只适用于
#
# 从设计思想上, Message 放弃了流式传输层协议/存储, 回到上行消息协议:
# 1. 提供可以兼容 openai、gemini、claude 等主流模型消息协议的容器。直接使用 Pydantic AI 生态.
# 2. 彻底放弃 OpenAI 的强类型约定. 目前行业共同指向了消息体自解释, 也是殊途同归.
# 3. 放弃下行 (模型生成), 专注于上行消息协议.

Additional = Optional[dict[str, dict[str, Any]]]
"""
使用弱类型容器保存强类型数据结构的思想. 
它实际对应一个强类型的数据结构, 用 pydantic.BaseModel 来定义.
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
        if not isinstance(target.additional, dict):
            return None
        keyword = cls.keyword()
        data = target.additional.get(keyword, None)
        if data is None:
            return None
        if not isinstance(data, dict):
            return None
        try:
            wrapped = cls.model_validate(data)
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


_now_utc: Callable[[], datetime] = lambda: datetime.now(tz.gettz())


class MessageMeta(BaseModel):
    """
    消息的元信息, 用来标记消息的关键维度.
    独立出数据结构, 是为了方便将 meta 在 ghost in shells 的交互逻辑使用. 同时 **可以** 不污染消息 content.
    Meta 原生目标, 是当一个 Message 容器用 with_meta 的方式返回 contents 时, 带上必要的附加讯息, 比如时间戳.
    贯彻时间是第一公民的目标.
    """

    tag: str = Field(
        default="",
        description="当 Message 使用 meta 生成 xml 结构时, 用于包括 content 的 xml 标记. 如果为空, 意味着不包裹."
    )
    id: str = Field(
        default_factory=uuid,
        description="消息的全局唯一 ID",
    )
    role: str | None = Field(
        default=None,
        description="消息体的角色类型. 来自 感知器/用户/AI/功能 等等",
    )
    name: Optional[str] = Field(
        default=None,
        description="消息的发送者身份. 作为 ghost in shells 架构中的标准概念.",
    )
    created: AwareDatetime = Field(
        default_factory=_now_utc,
        description="消息的创建时间, 一个消息只有一个创建时间",
    )
    completed: AwareDatetime | None = Field(
        default=None,
        description="消息结束的时间戳",
    )
    timestamp: bool = Field(
        default=True,
        description="是否在容器展示时显示时间戳",
    )
    attributes: dict[str, str] = Field(
        default_factory=dict,
        description="额外的 attributes 属性. "
    )

    def gen_attributes(self, timestamp: bool = True) -> dict[str, Any]:
        attributes = self.attributes.copy()
        # 排除掉 ghost in shells 架构自身的关键维度信息.
        exclude = {'attributes', 'id', 'tag', 'timestamp'}
        if not self.timestamp or not timestamp:
            exclude.add('created')
            exclude.add('completed')

        update = self.model_dump(
            exclude_none=True,
            exclude_defaults=True,
            exclude=exclude,
        )
        if len(update) > 0:
            for key, value in update.items():
                if key not in attributes:
                    # 不覆盖 attributes. attributes 最高优.
                    attributes[key] = value
        return attributes

    def gen_attributes_str(self, timestamp: bool = True) -> str:
        attributes = self.gen_attributes(timestamp=timestamp)
        if len(attributes) == 0:
            return ''
        parts = []
        for attr, value in attributes.items():
            # in case value has invalid mark
            if isinstance(value, datetime):
                value = datetime.fromtimestamp(value.timestamp(), tz.gettz()).isoformat(timespec='seconds')
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
        tag = 'message'
        return f'<{tag} {attr_str}/>'


ContextType = ContentModel | str | Image.Image | BaseModel | Content


class Message(BaseModel, WithAdditional):
    """
    MOSS 体系上行给模型的消息体. 本质上是 content block 的分组.
    核心目标:
    1. 基于 meta 提供 moss 架构所必要的关键元信息.
    2. 默认将 meta 信息用 xml 格式包裹包含的 contents.
    3. 支持消息协议的多层嵌套. 用 xml 包裹.
    4. 可以通过 id 来去重.
    5. 用自定义的 addition 对象来做扩展.
    """

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
            tag: str = '',
            *,
            name: Optional[str] = None,
            attributes: dict[str, Any] | None = None,
            # 是否需要在生成的 xml 包裹容器中展示 timestamp.
            timestamp: bool = True,
    ) -> Self:
        """
        语法糖, 用来极简地一条消息.

        >>> msg = Message.new()
        """
        data: dict[str, Any] = {'tag': tag or ''}
        if name is not None:
            data['name'] = name
        if attributes is not None:
            data['attributes'] = attributes
        data['timestamp'] = timestamp
        meta = MessageMeta.model_validate(data)
        return cls(meta=meta)

    def is_completed(self) -> bool:
        return self.meta.completed is not None

    def as_complete(self, copy: bool = False) -> Self:
        item = self if copy is False else self.model_copy(deep=True)
        item.meta.completed = _now_utc()
        return item

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
        """
        以字符串优先的方式提供基础类型的数据转换.
        """
        if isinstance(item, str):
            _content = Text.new(item).to_content()
        elif isinstance(item, dict) and 'type' in item:
            # 盲目兼容.
            _content = item
        elif hasattr(item, 'kind'):
            _content = item
        elif isinstance(item, ContentModel):
            _content = item.to_content()
        elif isinstance(item, Image.Image):
            _content = Base64Image.from_pil_image(item)
        elif isinstance(item, BaseModel):
            serialized = item.model_dump_json(indent=0, ensure_ascii=False, exclude_none=False)
            _content = Text.new(serialized).to_content()
        elif isinstance(item, dict) or isinstance(item, list):
            serialized = orjson.dumps(item).decode('utf8')
            _content = Text.new(serialized).to_content()
        else:
            value = str(item)
            _content = Text.new(value).to_content()
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
            if isinstance(item, str) and item == '':
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

    def as_completed(self) -> Self:
        self.meta.completed = _now_utc()
        return self

    def as_contents(
            self,
            *,
            with_meta: bool = True,
            timestamp: bool = True,
    ) -> Iterable[Content]:
        """
        将整个消息体返回成 Pydantic AI 的 User Content.
        """
        if self.is_empty():
            yield from []
            return

        tag = self.meta.tag
        # 没有 tag 的情况下, 认为不包裹消息.
        if not with_meta or not tag:
            yield from self.contents
            return

        attrs = self.meta.gen_attributes_str(timestamp=timestamp)
        attr_str = ''
        if attrs:
            attr_str = ' ' + attrs
        yield Text.new(f'<{tag}{attr_str}>\n').to_content()
        for content in self.contents:
            yield content
        yield Text.new(f'\n</{tag}>').to_content()

    def with_messages(
            self,
            *messages: Self,
            with_meta: bool = True,
            timestamp: bool = True,
    ) -> Self:
        """
        join other messages.
        """
        for msg in messages:
            for content in msg.as_contents(with_meta=with_meta, timestamp=timestamp):
                self.contents.append(content)
        return self

    def get_copy(self) -> Self:
        return self.model_copy(deep=True)

    def to_xml(self) -> str:
        """
        debug method
        """
        result = []
        for content in self.as_contents(with_meta=True):
            if text := Text.from_content(content):
                result.append(text.text)
            else:
                content_type = content['type']
                result.append(f'<content type="{content_type}"/>')
        result = '\n'.join(result)
        return result.strip()
