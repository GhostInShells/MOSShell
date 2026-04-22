from typing import Iterable, Generic, TypeVar

from typing_extensions import Self, Literal
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, AwareDatetime, ValidationError

from ghoshell_moss.message import Message, Content, WithAdditional, Addition
from ghoshell_moss.core.concepts.command import ObserveError
from ghoshell_common.helpers import uuid
from PIL.Image import Image
from datetime import datetime
from dateutil import tz
import time
import asyncio
import enum


class Outcome(BaseModel, WithAdditional):
    id: str = Field(
        default_factory=uuid,
        description="为 observation 创建唯一 id",
    )
    logos: str = Field(
        default='',
        description="在这个 observation 触发前, 生成的 logos. 放入一个消息容器中. ",
    )
    messages: list[Message] = Field(
        default_factory=list,
        description="这个 observation 持有的未阅读 outcome",
    )
    stop_reason: str = Field(
        default='',
        description="如果这是一个未完成的 Observation, 它可以被记录状态",
    )

    def new_observation(self) -> "Observation":
        return Observation(
            previews=self,
        )


class Observation(BaseModel, WithAdditional):
    """
    智能体上下文感知的关键帧.
    """
    # 它包含以下核心概念的聚合.
    # - last: 上一轮 Observation 之后的讯息.
    #     - logos: 上一轮的 logos.
    #     - messages: 上一轮运行输出的讯息.
    #     - stop_reason: 上一轮的结束信息.
    # - context: observation 生成瞬间的动态上下文, 每一轮都会重新刷新.
    # - inputs: 触发 observation 的外部世界输入.
    # - prompt: 本轮思考时的提示信息.
    #
    # Observation 的定义用来将离散的关键帧交互, 缝合成一个连续的认知流.
    # 理论上 logos/outcome/inputs 三者在时间上是交错的, 但由于现阶段没有全双工的模型能力,
    # 为了防止认知撕裂, 考虑将它们按这种方式, 逻辑上重新排序.

    id: str = Field(
        default_factory=uuid,
        description="为 observation 创建唯一 id",
    )

    # --- 以下缝合上一轮交互的讯息 --- #
    previews: Outcome | None = Field(
        default=None,
    )

    # --- 以下是新一轮交互的输入 --- #

    context: dict[str, list[Message]] = Field(
        default_factory=dict,
        description="当前 Observation 生成的瞬间, 将不同类型的 context 合并进来, 提供上下文快照",
    )
    inputs: list[Message] = Field(
        default_factory=list,
        description="与本轮输入相关的上下文. 在连续的 observation 中, 通常只有第一轮有输入. "
    )
    prompt: str = Field(
        default='',
        description="与本轮思考决策相关的提示讯息. 只在当前轮次生效",
    )
    on_start_logos: str = Field(
        default='',
        description="the predefined logos before the reaction",
    )

    def new_outcome(self) -> Outcome:
        """生成下轮的接收池"""
        return Outcome(
            id=self.id,
        )

    def context_messages(self) -> Iterable[Message]:
        if len(self.context) == 0:
            yield from []
            return
        for messages in self.context.values():
            yield from messages

    def as_request_messages(self, *, with_context: bool = True) -> Iterable[Message]:
        """
        所有这些消息, 理论上都会合并为一轮输入消息的 contents.
        本处是一个使用约定 (code as prompt), 不是硬性约束.
        """
        if self.previews is not None:
            outcome = self.previews
            if len(outcome.messages) > 0:
                yield Message.new().with_content('<outcomes>')
                yield from outcome.messages
                yield Message.new().with_content('</outcomes>')
            if outcome.stop_reason:
                yield Message.new(tag='stop_reason').with_content(outcome.stop_reason)

        context_messages = list(self.context_messages())
        if len(context_messages) > 0:
            if with_context:
                yield Message.new().with_content("<context>\n")
                yield from context_messages
                yield Message.new().with_content("\n</context>")
            else:
                count = len(context_messages)
                yield Message.new().with_content(f"<compacted>{count} history messages compacted </compacted>")
        yield from self.inputs
        if self.prompt:
            yield Message.new(tag='prompt').with_content(self.prompt)

    def as_request_contents(self, *, with_context: bool = True) -> Iterable[Content]:
        """
        用这种方式, 可以拿到和 Anthropic 基本兼容的 Contents.
        可以包裹到 UserMessageParams 或 ToolMessageParams 里.
        """
        for msg in self.as_request_messages(with_context=with_context):
            yield from msg.as_contents(with_meta=True)


class ConversationMeta(BaseModel, WithAdditional):
    """meta information of conversation."""
    id: str = Field(
        default_factory=uuid,
        description="conversation uuid",
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
        description="recap before the conversation",
    )
    summary: str = Field(
        default='',
        description='the summary of the conversation',
    )
    root_id: str = Field(
        default='',
        description="conversation tree root_id",
    )
    parent_id: str = Field(
        default='',
        description="the current conversation fork from which",
    )
    created: AwareDatetime = Field(
        default_factory=lambda: datetime.now(tz.gettz()),
        description="the time when the conversation was created",
    )
    updated: AwareDatetime = Field(
        default_factory=lambda: datetime.now(tz.gettz()),
        description="the time when the conversation was updated",
    )
    total_observations: int = Field(
        default=0,
        description="total number of observations in this conversation",
    )


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
    def append(self, observation: Observation) -> None:
        """
        增加新的 observation.
        立刻生效, 不阻塞.
        """
        pass

    @abstractmethod
    def observations(self, reverse_order: bool = True) -> Iterable[Observation]:
        """
        list observations in reverse chronological order.
        """
        pass

    @abstractmethod
    def get_effective_context(self) -> Iterable[Message]:
        """
        这个方法负责根据当前的 compact 状态，
        返回 [压缩后的历史描述] + [近期的 Observation 序列]。
        这是推理层直接调用的接口。
        """
        pass

    @abstractmethod
    def save(self) -> asyncio.Future[ConversationMeta]:
        """
        保存当前 conversation, 可以不阻塞当前流程. 返回更新后的 meta 信息. 可能实际上变更了 id.
        更新逻辑实际上会排队.
        更新完毕后, Conversation 抽象可能会变化.
        """
        pass


CONVO = TypeVar('CONVO', bound=Conversation)


class ConversationStore(Generic[CONVO], ABC):
    """
    conversation 存储中心.
    """

    @abstractmethod
    def get(self, conversation_id: str, or_create: bool = False) -> CONVO:
        """
        get conversation by conversation id.
        raise: FileNotFoundError
        """
        pass

    @abstractmethod
    def list(self, offset: int = 0, limit: int = 10) -> Iterable[ConversationMeta]:
        """
        list the conversation metas in reverse chronological order.
        """
        pass
