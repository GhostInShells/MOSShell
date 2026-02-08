import json
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
from typing import Any, ClassVar, Literal, Optional, Protocol

from ghoshell_common.helpers import timestamp_ms, uuid_md5
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import Self, TypedDict

try:
    from typing import is_typeddict
except ImportError:  # pragma: no cover
    from typing_extensions import is_typeddict

__all__ = [
    "Addition",
    "Additional",
    "Content",
    "ContentModel",
    "Delta",
    "DeltaModel",
    "HasAdditional",
    "Message",
    "MessageMeta",
    "MessageStage",
    "MessageTypeName",
    "Role",
    "WithAdditional",
]

"""
实现一个通用的消息协议。

1. 可以兼容 openai、gemini、claude 等主流模型消息协议。
2. 同时兼具流式传输 + 存储的功能。
    - 流式传输考虑首包、间包、尾包
    - 消息类型可以扩展
    - 不一定是模型的消息，也可能是不能被模型读取的消息
    - 不同模型构建上下文时，可以筛选或排除特定类型的消息。
3. 可以无限扩展，而不需要重新定义消息结构。
4. 支持多模态。
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


class MessageStage(str, Enum):
    """
    生产消息的阶段.
    一个可用可不用, 可扩展的约束条件, 核心目标是在 Agent 架构中用来过滤历史消息.

    举个例子, 一个模型的 React 模式中, 返回的消息体可能包含了 reasoning, observe, response 三个阶段.
    其中 reasoning 是推理, observe 是工具调用, response 才是正规的回复.
    基于 function call 的做法, 只有在没有工具调用的那一轮输出, 才是真正的 response.

    这样用 stage 标记三个阶段生产的消息体, 在下一轮对话中, 可以从历史记忆里删除掉 reasoning 或者 observe, 保持干净.
    """

    DEFAULT = ""
    REASONING = "reasoning"
    OBSERVE = "observe"
    RESPONSE = "response"

    def new_meta(self, role: str = Role.ASSISTANT.value, name: Optional[str] = None):
        return MessageMeta(role=role, name=name, stage=self.value)


class MessageMeta(BaseModel):
    """
    消息的元信息, 用来标记消息的维度.
    这里的信息是不变化的.

    独立出数据结构, 是为了方便将 meta 在不同的数据结构中使用, 而不用持有整个 message.
    """

    id: str = Field(
        default_factory=uuid_md5,
        description="消息的全局唯一 ID",
    )
    stage: str = Field(
        default=MessageStage.DEFAULT.value,
        description="生产消息所属的阶段, 可以用于在历史消息中过滤消息. 比如 reasoning 就可以认为是一种过程.",
    )
    role: str = Field(
        default="",
        description="消息体的角色",
    )
    name: Optional[str] = Field(
        default=None,
        description="消息的发送者身份, 兼容 openai 的协议.",
    )
    additional: Optional[dict[str, dict[str, Any]]] = Field(
        default=None,
        description="消息体强类型的附属结构",
    )
    created_at: float = Field(
        default_factory=timestamp_ms,
        description="消息的创建时间, 一个消息只有一个创建时间",
    )
    updated_at: Optional[float] = Field(
        default=None,
        description="消息体最后更新时间",
    )
    completed_at: Optional[float] = Field(
        default=None,
        description="消息体的生成结束时间",
    )
    finish_reason: Optional[str] = Field(default=None, description="消息体中断的原因")


class Delta(TypedDict):
    """
    传输中的间包统一数据容器.

    这又是一个弱类型的容器, 其中 data 的数据结构没有自解释, 需要结合 type 去还原.
    """

    type: str
    data: dict


class DeltaModel(BaseModel, ABC):
    """
    传输的间包强类型数据结构.

    它用来定义一个 间包 的强类型数据结构, 但传输时会转成 Delta (弱类型)

    必须是可序列化的数据结构定义.
    """

    DELTA_TYPE: ClassVar[str] = ""
    """通过类常量的方式来定义 type 类型"""

    @classmethod
    def from_delta(cls, delta: Delta) -> Self | None:
        """
        从 delta 包中还原自身的强类型结构.
        """
        if delta["type"] != cls.DELTA_TYPE:
            return None
        try:
            return cls(**delta["data"])
        except ValidationError:
            return None

    def to_delta(self) -> Delta:
        """
        转换成弱类型.
        """
        return Delta(
            type=self.DELTA_TYPE,
            data=self.model_dump(exclude_none=True),
        )


class Content(TypedDict):
    """
    消息的通用内容体. 兼容各种模型.
    原理与 delta 一模一样.
    """

    type: str
    data: dict


class ContentModel(BaseModel, ABC):
    """
    多模态消息单元的强类型定义.
    """

    CONTENT_TYPE: ClassVar[str] = ""
    """通过类常量的方式来定义 type 类型"""

    @classmethod
    def from_content(cls, content: Content) -> Self | None:
        """
        从 content 弱类型容器中还原出强类型的数据结构.
        """
        if content["type"] != cls.CONTENT_TYPE:
            return None
        try:
            return cls(**content["data"])
        except ValidationError:
            return None

    def to_content(self) -> Content:
        """
        将强类型的数据结构, 转成弱类型的 content 对象.
        """
        return Content(
            type=self.CONTENT_TYPE,
            data=self.model_dump(exclude_none=True),
        )


class Message(BaseModel, WithAdditional):
    """
    模型传输过程中的消息体. 本质上是兼具 存储/传输/展示 功能的通用数据容器.

    目标是:
    1. 兼容几乎所有的模型, 及其多模态消息类型.
    2. 可以跨网络传输, 所有数据可以序列化.
    3. 可以用于本地存储.
    4. 本身也是一个兼容弱类型的容器, 除了消息本身必要的讯息外, 其它的讯息都是弱类型的. 避免传输时需要转化各种数据类型.
    5. 完整的内容数据, 都定义在 contents 里
    """

    type: str = Field(
        default="",
        description="消息的类型, 对应 MessageTypeName, 用来定义不同的处理逻辑. ",
    )
    meta: MessageMeta = Field(
        default_factory=MessageMeta,
        description="消息的维度信息, 单独拿出来, 方便被其它数据类型所持有. ",
    )
    seq: Literal["head", "delta", "incomplete", "completed"] = Field(
        default="completed",
        description="消息的传输状态, 目前分为首包, 间包和尾包."
                    "- 首包: 用来提示一个消息流已经被生产. 通常用来通知前端界面, 提前渲染消息容器"
                    "- 间包: 用最少的讯息传递一个 delta 包, 用于流式传输"
                    "- 尾包: 包含所有 delta 包粘包后的完整结果, 用来存储或展示."
                    "尾包分为 completed 和 incomplete 两种. "
                    "- completed 表示一个消息体完全传输完毕."
                    "- incomplete 表示虽然没传输完毕, 但可能也要直接使用."
                    "我们举一个具体的例子, 在模型处理多端输入时, 一个视觉信号让模型要反馈, 但一个 asr 输入还未全部完成;"
                    "这个时候, 大模型仍然要看到未完成的语音输入, 也就是 incomplete 消息."
                    "但是下一轮对话, 当 asr 已经完成时, 历史消息里不需要展示 incomplete 包."
                    "所以 incomplete 主要是用来在大模型思考的关键帧中展示一个粘包中的中间结果.",
    )
    delta: Optional[Delta] = Field(
        default=None,
        description="传输的间包, 非 head/delta 类型不会持有 delta. ",
    )
    contents: None | list[Content] = Field(default=None, description="弱类型的数据, 通常在尾包里. ")

    @classmethod
    def new(
            cls,
            *,
            role: Literal["assistant", "system", "developer", "user", ""] = "",
            name: Optional[str] = None,
            id: Optional[str] = None,
    ):
        """
        语法糖, 用来创建一条消息.

        >>> msg = Message.new().as_completed()
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
        语法糖, 用来从 meta 里拿到 role.
        其实挺多余的. 太想偷懒了.
        """
        return self.meta.role

    @property
    def name(self) -> str | None:
        """
        语法糖, 用来从 meta 里拿到 name.
        其实挺多余的. 太想偷懒了.
        """
        return self.meta.name

    @property
    def id(self) -> str:
        """
        语法糖, 用来从 meta 里拿到 id.
        其实挺多余的. 太想偷懒了.
        """
        return self.meta.id

    def with_content(self, *contents: Content | ContentModel | str | Image.Image) -> Self:
        """
        语法糖, 用来添加 content.
        """
        from .contents import Base64Image, Text

        for content in contents:
            if is_typeddict(content):
                self.contents = self.contents or []
                self.contents.append(content)
            elif isinstance(content, ContentModel):
                self.contents = self.contents or []
                self.contents.append(content.to_content())
            elif isinstance(content, str):
                self.contents = self.contents or []
                self.contents.append(Text(text=content).to_content())
            elif isinstance(content, Image.Image):
                self.contents = self.contents or []
                self.contents.append(Base64Image.from_pil_image(content).to_content())
        return self

    def is_completed(self) -> bool:
        """常用语法糖"""
        return self.seq == "completed"

    def is_incomplete(self) -> bool:
        """常用语法糖"""
        return self.seq == "incomplete"

    def is_done(self) -> bool:
        """
        常用语法糖
        尾包(done 包) 包含两种类型.
        """
        return (self.is_completed() or self.is_incomplete()) and len(self.contents) > 0

    def is_empty(self) -> bool:
        """
        标记一个无数据的空包.
        语法糖. 大模型理解消息时, 通常不允许传入空消息.
        """
        return not self.contents and not self.delta

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

    @classmethod
    def from_json(cls, json_data: str) -> Self:
        """
        糖. 是不是整个 message 会太甜了?
        """
        return cls(**json.loads(json_data))

    def get_copy(self) -> Self:
        """
        强类型复制的语法糖.
        """
        delta = None
        if self.delta is not None:
            delta = self.delta.copy()
        contents = None
        if self.contents is not None:
            contents = deepcopy(self.contents)
        return Message(
            meta=self.meta.model_copy(),
            seq=self.seq,
            delta=delta,
            contents=contents,
        )

    def as_head(self, delta: Optional[Delta | DeltaModel] = None) -> Self:
        """
        基于当前数据, 生成一个 Head 包.
        常见用法:
        >>> msg = Message.new().as_head()
        """
        if delta is not None and isinstance(delta, DeltaModel):
            delta = delta.to_delta()
        self.seq = "head"
        self.delta = delta
        self.contents = None
        self.meta.created_at = timestamp_ms()
        self.meta.updated_at = None
        self.meta.completed_at = None
        return self

    def as_delta(self, delta: DeltaModel | Delta) -> Self:
        """
        基于当前数据, 生成一个 delta 包.
        常见用法:
        >>> msg = Message.new().as_delta(delta)
        """
        if isinstance(delta, DeltaModel):
            delta = delta.to_delta()
        self.seq = "delta"
        self.delta = delta
        self.contents = None
        self.meta.updated_at = timestamp_ms()
        self.meta.completed_at = None
        return self

    def as_completed(self, contents: list[Content] | None = None) -> Self:
        """
        基于当前数据, 生成一个 尾包.
        常见用法:
        >>> msg = Message.new().as_completed(contents)
        >>> # 复制一个新的尾包.
        >>> copy_msg = msg.get_copy().as_completed()
        """
        if self.seq == "completed":
            return self
        contents = contents if contents is not None else self.contents.copy()
        self.seq = "completed"
        self.delta = None
        self.contents = contents
        self.meta.updated_at = timestamp_ms()
        self.meta.completed_at = self.meta.updated_at
        return self

    def as_incomplete(self, contents: list[Content] | None = None) -> Self:
        """
        与 as complete 类似, 生成一个未完成的尾包.
        """
        if self.seq == "completed":
            return self
        contents = contents if contents is not None else self.contents.copy()
        self.seq = "incomplete"
        self.delta = None
        self.contents = contents
        self.meta.updated_at = timestamp_ms()
        self.meta.completed_at = None
        return self
