from typing import Iterable, Generic, TypeVar
from typing_extensions import Self

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, AwareDatetime

from ghoshell_moss.message import Message, Content, WithAdditional
from ghoshell_common.helpers import uuid
from datetime import datetime
from dateutil import tz
import asyncio

__all__ = [
    'Conversation', 'ConversationStore',
    'Reaction', 'Moment', 'ConversationMeta',
    'ModelContext',
]


class Reaction(BaseModel, WithAdditional):
    """
    上一轮与外部世界互动的结果.
    由于现在模型并没有能支持全双工的实现,
    所以仍然需要一种粘合机制拼出交互.
    """
    moment_id: str = Field(
        default_factory=uuid,
        description="上一轮 Moment id.",
    )
    logos: str = Field(
        default='',
        description="上一轮交互, AI 输出的 logos. "
                    "驱动躯体与工具运行. 这里的 logos 是 符号/逻辑/指令/路径/现实规律 的含义. 对应中文的 道-言说 ",
    )
    outcomes: list[Message] = Field(
        default_factory=list,
        description="logos 执行同时或之后得到的内部 (比如躯体) 反馈结果. 是思维洞穴里的回声. ",
    )
    stop_reason: str = Field(
        default='',
        description="如果这是一个未完成的 Moment, 它可以被记录状态",
    )

    def new_moment(self) -> "Moment":
        """
        基于 Outcome 产生下一轮的观察.
        """
        return Moment(
            previous=self,
        )


class Moment(BaseModel, WithAdditional):
    """
    智能体上下文感知的关键帧.
    """

    id: str = Field(
        default_factory=uuid,
        description="为 observation 创建唯一 id",
    )

    # --- 以下缝合上一轮交互的讯息 --- #
    previous: Reaction | None = Field(
        default=None,
    )

    # --- 以下是新一轮交互的输入 --- #

    perspectives: dict[str, list[Message]] = Field(
        default_factory=dict,
        description="当前 Moment 生成的瞬间, 将不同类型的 context 合并进来, 提供一个动态上下文快照",
    )
    percepts: list[Message] = Field(
        default_factory=list,
        description="本轮的外部输入: 已经过解析/结构化/多模态对齐, 但尚未经过高层解读."
    )
    prompt: str = Field(
        default='',
        description="与本轮思考决策相关的提示讯息. 只在当前轮次生效",
    )

    def to_json(self, *, exclude_perspectives: bool = True) -> str:
        """
        标准的序列化方式, 也方便存储.
        """
        exclude = None
        if exclude_perspectives:
            exclude = {'perspectives'}
        return self.model_dump_json(
            exclude=exclude,
            ensure_ascii=False,
            exclude_none=True,
            exclude_defaults=True,
        )

    def new_reaction(self) -> Reaction:
        """生成下轮的接收池"""
        return Reaction(
            moment_id=self.id,
        )

    def previous_logos(self) -> str:
        if self.previous is None:
            return ''
        return self.previous.logos

    def with_perspective_context(self, key: str, messages: list[Message]) -> Self:
        """组合不同类型的动态内观上下文."""
        self.perspectives[key] = messages
        return self

    def last_moment_id(self) -> str | None:
        if self.previous is None:
            return None
        return self.previous.moment_id

    def perspective_messages(self) -> Iterable[Message]:
        if len(self.perspectives) == 0:
            yield from []
            return
        for messages in self.perspectives.values():
            yield from messages

    def as_request_messages(
            self,
            *,
            with_perspectives: bool = True,
            with_prompt: bool = True,
    ) -> Iterable[Message]:
        """
        所有这些消息, 理论上都会合并为一轮输入消息的 contents.
        本处是一个使用约定 (code as prompt), 不是硬性约束.
        """
        if self.previous is not None:
            reaction = self.previous
            if len(reaction.outcomes) > 0:
                yield Message.new().with_content('<outcomes>')
                yield from reaction.outcomes
                yield Message.new().with_content('</outcomes>')
            if reaction.stop_reason:
                yield Message.new(tag='stop_reason').with_content(reaction.stop_reason)

        perspectives_messages = list(self.perspective_messages())
        if len(perspectives_messages) > 0:
            if with_perspectives:
                yield Message.new().with_content("<perspectives>\n")
                yield from perspectives_messages
                yield Message.new().with_content("\n</perspectives>")
            else:
                count = len(perspectives_messages)
                yield Message.new().with_content(
                    f"<perspectives compacted>{count} messages compacted </perspectives>"
                )
        yield from self.percepts
        if with_prompt and self.prompt:
            yield Message.new(tag='prompt').with_content(self.prompt)

    def as_request_contents(self, *, with_context: bool = True) -> Iterable[Content]:
        """
        用这种方式, 可以拿到和 Anthropic 基本兼容的 Contents.
        可以包裹到 UserMessageParams 或 ToolMessageParams 里.
        """
        for msg in self.as_request_messages(with_perspectives=with_context):
            yield from msg.as_contents(with_meta=True)


class ConversationMeta(BaseModel, WithAdditional):
    """
    Conversation 用来保存会话历史.
    不可避免地需要进行历史分割, 经过分割的会话历史可以视作一棵树.
    ConversationMeta 用来快速还原一个会话的关键信息, 类似树节点的描述.
    """
    id: str = Field(
        default_factory=uuid,
        description="conversation uuid",
    )
    namespace: str = Field(
        default='',
        description="namespace of the conversation, such as session_scope/ghost_name/yyy-mm-dd",
    )
    title: str = Field(
        default='',
        description="conversation title",
    )
    description: str = Field(
        default='',
        description="conversation description",
    )
    recap: str = Field(
        default='',
        description="recap when the conversation was created",
    )
    root_id: str | None = Field(
        default=None,
        description="conversation tree root_id",
    )
    fork_from: str | None = Field(
        default=None,
        description="the conversation id which current one fork from",
    )
    created: AwareDatetime = Field(
        default_factory=lambda: datetime.now(tz.gettz()),
        description="the time when the conversation was created",
    )


_Logos = str


class ModelContext(BaseModel, WithAdditional):
    """
    为给大模型使用设计的数据结构.
    这个数据结构考虑可以存储, 方便调试还原每一个 AI 思考的关键帧.
    """
    request_id: str = Field(
        default_factory=uuid,
        description="作为请求的唯一 id. "
    )
    system_prompt: str = Field(
        default='',
        description="系统提示词. 这里可以有缓存锚点.",
    )
    memories: list[Message] = Field(
        default_factory=list,
        description="思考关键帧中出现在交互历史之前的内容. 可能包含多模态信息. 这里可以有二级缓存锚点. "
    )
    history: list[Moment] = Field(
        default_factory=list,
        description="在这个上下文中发生过的多轮交互. ",
    )
    current: Moment = Field(
        description="当前的瞬间. "
    )

    @classmethod
    def new(
            cls,
            current: Moment,
            *,
            system_prompt: str = '',
            memories: list[Message] | None = None,
            history: list[Moment] | None = None,
            request_id: str | None = None,
            strict: bool = False,
    ):
        data = dict(current=current, system_prompt=system_prompt, memories=memories or [], history=history or [])
        if request_id:
            data['request_id'] = request_id
        if strict:
            return cls(**data)
        else:
            return cls.model_construct(**data)

    def to_messages(
            self,
            with_system_prompt: bool = False,
            with_history_perspective_turns: int = 0,
    ) -> Iterable[tuple[list[Message], _Logos | None]]:
        """
        将 ModelFrame 还原为历史交互消息, 兼容 Anthropic / PydanticAI 的回合制思维.
        展示如何处理成多轮消息.
        :param with_system_prompt: 通常不包含 system prompt, 因为在很多 agent 或 模型的 api 中, system prompt 都是独立字段.
        :param with_history_perspective_turns: 携带最近 N 轮 Moment 快照的 moment.perspective 信息.
        """
        # 准备第一轮模型的请求信息.
        buffered_request_messages = []
        if with_system_prompt and self.system_prompt:
            buffered_request_messages.append(Message.new().with_content(self.system_prompt))
        buffered_request_messages.extend(self.memories)

        # 判断哪些轮次需要携带历史 perspectives. 最好都不要带.
        with_perspective_history_moment_turn_idx = len(self.history)
        if with_history_perspective_turns > 0:
            with_perspective_history_moment_turn_idx -= with_history_perspective_turns

        if len(self.history) > 0:
            idx = 0
            for moment in self.history:
                # 一旦有产生过不为空的 logos, 就发送.
                # 模型生产出来的 logos 是唯一现实交互的锚点.
                if logos := moment.previous_logos():
                    yield buffered_request_messages, logos
                    buffered_request_messages = []

                with_perspective = idx >= with_perspective_history_moment_turn_idx

                # 否则只是堆叠需要发送的消息.
                buffered_request_messages.extend(
                    moment.as_request_messages(with_perspectives=with_perspective, with_prompt=False)
                )
                idx += 1
        if logos := self.current.previous_logos():
            yield buffered_request_messages, logos
        yield list(self.current.as_request_messages(with_perspectives=True, with_prompt=True)), None


class Conversation(ABC):
    """
    Conversation 数据结构的抽象封装.
    内部可能包含 Conversation Policy 用来管理加工/截断逻辑.
    """

    @abstractmethod
    def meta(self) -> ConversationMeta:
        """返回 Meta 信息. """
        pass

    @abstractmethod
    def append(self, moment: Moment) -> None:
        """
        增加新的 observation.
        立刻生效, 不阻塞.
        """
        pass

    @abstractmethod
    def history(self, reverse_order: bool = True) -> Iterable[Moment]:
        """
        list observations in reverse chronological order.
        """
        pass

    @abstractmethod
    def get_effective_messages(self) -> Iterable[Message]:
        """
        这个方法负责根据当前的 compact 状态，
        返回 [压缩后的历史描述] + [近期的 Moment 序列]。
        这是推理层直接调用的接口。
        """
        pass

    @abstractmethod
    def save(self, compact: bool | None = None) -> asyncio.Future[ConversationMeta]:
        """
        保存当前 conversation.
        可以不阻塞当前流程. 返回更新后的 meta 信息. 可能实际上变更了 id.
        更新逻辑实际上会排队. 此外, Conversation 之所以是一个抽象类, 就是考虑内部实际上实现了 conversation policy.
        更新完毕后, Conversation 抽象内容物可能会变化. 具体的 Policy 由 Conversation 实现决定.
        :param compact: 为 None 表示 auto compact. 为 True 表示必须 Compact.
        """
        pass


CONVO = TypeVar('CONVO', bound=Conversation)


class ConversationStore(Generic[CONVO], ABC):
    """
    conversation 存储中心.
    """

    @abstractmethod
    def get(self, namespace: str, conversation_id: str, or_create: bool = False) -> CONVO:
        """
        get conversation by conversation id.
        raise: FileNotFoundError
        """
        pass

    @abstractmethod
    def create(self, namespace: str, conversation_id: str | None = None) -> CONVO:
        """
        create a new conversation.
        """
        pass
