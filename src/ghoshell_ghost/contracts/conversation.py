from typing import Any, Optional, Iterable, TypeVar, Generic
from typing_extensions import Self
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from ghoshell_common.helpers import uuid, timestamp_ms, yaml_pretty_dump
from ghoshell_moss.message import Message, WithAdditional

"""
用来管理大模型上下文的一种手段. 
"""

__all__ = [
    'ConversationTurn', 'Conversation', 'ConversationStore',
    'Recap', 'RecapStrategy',
]

RecapStrategy = str


class Recap(BaseModel, WithAdditional):
    """
    对话历史中的前情提要.
    通过 Additions 添加生成前情提要的必要讯息. 比如是谁生成的.
    不列入协议本体.
    """
    strategy: RecapStrategy = Field(
        description="定义谁生成的这个 Recap. "
    )
    messages: list[Message] = Field(
        default_factory=list,
        description="前情提要的消息体. 通常仅仅是一条文本消息. ",
    )


class ConversationTurn(BaseModel, WithAdditional):
    """
    对话历史中的一个回合, 单元, 可以用于对话历史的分叉.
    不做线程安全. 如果有线程安全的必要, 请 copy 一个实例.
    """

    turn_id: str = Field(
        default_factory=uuid,
        description="回合的全局唯一 id. "
    )
    last_turn_id: Optional[str] = Field(
        default=None,
        description="关联上一个对话历史 Item. ",
    )
    trace_id: Optional[str] = Field(
        default=None,
        description="生成这个 Item 的 trace id"
    )

    index: int = Field(
        default=0,
        description="在会话历史中的排序位置. 但是这个字段很难保持很好, 需要 conversation 完成排序后设置比较合适. ",
    )

    recaps: dict[RecapStrategy, Recap] = Field(
        default_factory=dict,
        description="之前上下文的前情提要. 给不同的 ConversationStrategy 存储 "
    )

    context: list[Message] = Field(
        default_factory=list,
        description="在思维的每个回合中动态上下文的快照信息. 它发生在 inputs 的同时. context 里的讯息不在历史消息里使用."
                    "需要记录到历史消息里的信息, 应该放入 inputs 的前端. ",
    )

    inputs: list[Message] = Field(
        default_factory=list,
        description="在一个回合中所有的输入信息. "
    )
    instruction: list[Message] = Field(
        default_factory=list,
        description="input 之后的 instruction 片段. 不会在对话历史中使用.  "
    )

    generates: list[Message] = Field(
        default_factory=list,
        description="在一个回合中 AI 生成的所有讯息, 需要被添加到记忆中的. 这些信息并不一定是 output, 可能没有发送到客户端上. "
    )

    created_at: float = Field(
        default_factory=timestamp_ms,
        description="创建的时间",
    )
    completed_at: Optional[float] = Field(
        default=None,
        description="运行结束的时间",
    )

    def dumps(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent, exclude_none=True, ensure_ascii=False)

    def to_yaml(self) -> str:
        return yaml_pretty_dump(self.dumps())

    def new_turn(
            self,
            *,
            context: list[Message] | None = None,
            inputs: list[Message] | None = None,
            instructions: list[Message] | None = None,
            turn_id: str | None = None,
            trace_id: str | None = None,
    ) -> Self:
        """
        基于当前回合, 生成一个新的回合.
        """
        data = {}
        if turn_id:
            data['turn_id'] = turn_id
        if trace_id:
            data['trace_id'] = trace_id
        if context:
            data['context'] = context
        if inputs:
            data['inputs'] = inputs
        if instructions:
            data['instruction'] = instructions
        new_turn = ConversationTurn(**data)
        new_turn.last_turn_id = self.turn_id
        new_turn.index = self.index + 1
        return new_turn

    def with_context(self, *msgs: Message) -> Self:
        """链式语法糖"""
        self.context.extend(msgs)
        return self

    def with_input(self, *msgs: Message) -> Self:
        """链式语法糖"""
        self.inputs.extend(msgs)
        return self

    def with_instructions(self, *msgs: Message) -> Self:
        """链式语法糖"""
        self.instruction.extend(msgs)
        return self

    def append(self, *generates: Message) -> None:
        """
        增加新的消息内容.
        但只接受消息尾包.
        """
        for item in generates:
            if item.is_done():
                self.generates.append(item.model_copy())

    def messages(
            self,
            *,
            recap_strategy: str | None = None,
            inputs: bool = True,
            context: bool = True,
            instruction: bool = True,
            generates: bool = True,
    ) -> Iterable[Message]:
        """
        生成这个回合的消息.
        """
        if recap_strategy and recap_strategy in self.recaps:
            recap = self.recaps[recap_strategy]
            for msg in recap.messages:
                if msg.is_done():
                    yield msg
        if context:
            for msg in self.context:
                if msg.is_done():
                    yield msg
        if inputs:
            for msg in self.inputs:
                if msg.is_done():
                    yield msg
        if instruction:
            for msg in self.instruction:
                if msg.is_done():
                    yield msg
        if generates:
            for msg in self.generates:
                if msg.is_done():
                    yield msg

    def update_message(self, message: Message) -> bool:
        """
        更新某一条消息.
        """

        def find_and_update_message(_messages: list[Message]) -> Optional[list[Message]]:
            nonlocal message
            found = False
            result = []
            for exists in _messages:
                if exists.msg_id == message.msg_id:
                    if not found:
                        found = True
                        result.append(message.get_copy())
                else:
                    result.append(exists)
            if found:
                return result
            else:
                return None

        if msgs := find_and_update_message(self.context):
            self.context = msgs
            return True
        if msgs := find_and_update_message(self.inputs):
            self.inputs = msgs
            return True
        if msgs := find_and_update_message(self.instruction):
            self.instruction = msgs
            return True
        if msgs := find_and_update_message(self.generates):
            self.generates = msgs
            return True
        return False

    def is_completed(self) -> bool:
        return self.completed_at is not None

    def complete(self, at: float | None = None) -> None:
        self.completed_at = at or timestamp_ms()


class ConversationMeta(BaseModel, WithAdditional):
    """
    Conversation 的元信息.
    """

    id: str = Field(
        default_factory=uuid,
        description="任何一个会话的全局唯一 id. ",
    )
    root_id: Optional[str] = Field(
        default=None,
        description="如果当前 Conversation 是另一个会话历史的 fork, 这里是起点目标会话的 id.",
    )
    title: str = Field(
        default="",
        description="关于会话的标题. ",
    )
    description: str = Field(
        default="",
        description="关于会话的简单描述. 主要用来召回."
    )
    summary: str | None = Field(
        default=None,
        description="对会话的历史摘要. "
    )
    fork_from: Optional[str] = Field(
        default=None,
        description="如果当前会话是一个 fork, 这个 id 是它 fork 的来源会话. ",
    )
    created_at: float = Field(
        default_factory=timestamp_ms,
        description="创建的时间"
    )

    def fork(
            self,
            fork_id: str | None = None,
    ) -> Self:
        """
        将 Conversation 的元信息用来分叉.
        """
        update = {
            'fork_from': self.id,
            'root_id': self.root_id or self.id,
            'created_at': timestamp_ms(),
        }
        if fork_id:
            update['id'] = fork_id
        copied = self.model_copy(deep=True, update=update)
        return copied


class Conversation(BaseModel, WithAdditional):
    """
    对话历史.
    存储时应该使用别的数据结构.
    """

    meta: ConversationMeta = Field(
        description="conversation 的元信息, 运行时不变. "
    )

    recap: Recap | None = Field(
        default=None,
        description="这个会话创建时已经设置好的消息. 对话历史裁剪时永远保留.",
    )

    history: list[ConversationTurn] = Field(
        default_factory=list,
        description="属于对话历史的部分. "
    )

    saved_at: float | None = Field(
        default=None,
        description="最后保存时间. 单位是秒, 精确到毫秒"
    )

    def dumps(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)

    def to_json(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent, exclude_none=True, ensure_ascii=False)

    def to_yaml(self) -> str:
        return yaml_pretty_dump(self.dumps())

    @classmethod
    def new(
            cls,
            *,
            id: Optional[str] = None,
            title: str = "",
            description: str = "",
            recap: Recap | None = None,
    ) -> "Conversation":
        """
        初始化一个 Conversation.
        :param id: 指定的 conversation id.
        :param title: 会话的名称.
        :param description: 会话的描述.
        :param recap: 创建时的前情提要, 永远不删减.
        :return:
        """

        data = {}
        if id:
            data["id"] = id
        if title is not None:
            data["title"] = title
        if description is not None:
            data["description"] = description

        meta = ConversationMeta(**data)
        return cls(
            meta=meta,
            recap=recap,
        )

    def add_turn(self, turn: ConversationTurn):
        """
        添加一个 Turn.
        """
        if len(self.history) > 0:
            last_turn = self.history[-1]
            turn.last_turn_id = last_turn.turn_id
            turn.index = last_turn.index + 1
        self.history.append(turn)

    def prepare_save(self) -> None:
        """
        语法糖, 保存前的操作.
        """
        self.sort_history()
        self.saved_at = timestamp_ms()

    def get_truncated_copy(self) -> "Conversation":
        # todo:
        raise NotImplementedError

    def get_history_turns(self, *, recap_strategy: RecapStrategy | None = None) -> list[ConversationTurn]:
        """
        返回历史消息的轮次.
        可以根据 recap_strategy 来指定首轮.
        """
        turns = []
        for turn in self.history:
            # use summary as truncate point
            if recap_strategy and recap_strategy in turn.recaps:
                turns = [turn]
            else:
                turns.append(turn)
        return turns

    def get_history_messages(self, *, recap_strategy: RecapStrategy | None = None) -> Iterable[Message]:
        """
        返回所有的历史消息.
        """
        turns = self.get_history_turns(recap_strategy=recap_strategy)
        for turn in turns:
            yield from turn.messages(recap_strategy=recap_strategy, context=False, instruction=False)

    def get_messages(self, *, recap_strategy: RecapStrategy | None = None) -> Iterable[Message]:
        """
        获取所有的消息.
        """
        if self.recap is not None:
            yield from self.recap.messages

        yield from self.get_history_messages(recap_strategy=recap_strategy)

    def update_message(self, message: Message) -> bool:
        if not message.is_done():
            return False
        for turn in self.get_history_turns():
            if turn.update_message(message):
                return True
        return False

    def new_turn(
            self,
            *,
            turn_id: str | None = None,
            trace_id: str | None = None,
    ) -> ConversationTurn:
        """
        新建一个对话回合.
        """
        if len(self.history) == 0:
            data = {}
            if turn_id:
                data["turn_id"] = turn_id
            if trace_id:
                data["trace_id"] = trace_id
            return ConversationTurn(**data)
        last_turn = self.history[-1]
        return last_turn.new_turn(turn_id=turn_id, trace_id=trace_id)

    def sort_history(self):
        idx = 0
        for turn in self.history:
            turn.index = idx
            idx += 1

    def fork(
            self,
            fork_id: Optional[str] = None,
    ) -> Self:
        """
        在当前基础上 fork 一个版本, 可以继续推进.
        """
        fork_meta = self.meta.fork(fork_id=fork_id)
        conversation = self.model_copy(update=dict(meta=fork_meta), deep=True)
        return conversation

    def fork_with_recap(self, recap: Recap, *, fork_id: Optional[str] = None, remain_turns: int = 0) -> Self:
        """
        通过摘要保留指定轮次.
        """
        fork_meta = self.meta.fork(fork_id=fork_id)
        history = self.history
        length = len(self.history)
        cut_from = length - remain_turns - 1

        if cut_from < 0:
            remaining_history = history
        else:
            remaining_history = history[cut_from:]

        return Conversation(
            meta=fork_meta,
            recap=recap,
            # 关键, 清空 history 从头开始.
            history=[turn.model_copy(deep=True) for turn in remaining_history],
        )

    def delete_turn(self, turn_id: str) -> bool:
        history = []
        found = False
        for turn in self.history:
            if turn.turn_id == turn_id:
                found = True
                continue
            history.append(turn)
        self.history = history
        if found:
            return True
        return False


CONVERSATION_STRATEGY_CONF = TypeVar("CONVERSATION_STRATEGY_CONF", bound=BaseModel)


class ConversationStrategy(Generic[CONVERSATION_STRATEGY_CONF], ABC):
    """
    Conversation 的特殊处理机制.
    考虑到线程安全和并发逻辑, 使用协程没有意义.
    之所以要允许使用配置项, 是为了使它未来可以 Channel 化.
    """

    @abstractmethod
    def name(self) -> str:
        """
        Strategy 的唯一名称.
        """
        pass

    @abstractmethod
    def description(self) -> str:
        """
        strategy 的描述
        """
        pass

    @abstractmethod
    def read(self, conversation: Conversation, conf: CONVERSATION_STRATEGY_CONF | None = None) -> list[Message]:
        """
        从一个 Conversation 中, 按约定的规则读取消息.
        """
        pass

    @abstractmethod
    def optimize(self, conversation: Conversation, conf: CONVERSATION_STRATEGY_CONF | None = None) -> Conversation:
        """
        优化一个 Conversation, 通常包含 Title, Description, 特殊轮次的 Recap, 或者干脆 Fork.
        一般应该在 Save Conversation 之前执行.
        """
        pass


class ConversationStore(ABC):
    """
    存储 Conversation 的模块.
    这里的实现要求线程安全, 有序, 尽快返回.
    所以实际运行的时候, 可能是通过队列等方式来实现保存的.

    如果要用 Asyncio 来调用, 需要使用 asyncio.to_thread 卸载到线程.
    """

    @abstractmethod
    def find(self, conversation_id: str) -> Optional[Conversation]:
        """
        获取一个 Conversation 实例. 如果不存在的话, 返回 None.
        :param conversation_id: conversation_id
        """
        pass

    @abstractmethod
    def find_or_create(self, conversation_id: str) -> Conversation:
        """
        如果不存在, 就创建一个.
        """
        pass

    @abstractmethod
    def save(self, conversation: Conversation) -> None:
        """
        全量保存一个 Conversation.
        实际上可能要做复杂的数据库对齐.
        底层逻辑要求:
        1. 线程安全.
        2. 严格有序.
        """
        pass

    @abstractmethod
    def list(self, offset: int = 0, limit: int = -1) -> Iterable[Conversation]:
        """
        按最后更新的时间正序排列 conversations.
        """
        pass
