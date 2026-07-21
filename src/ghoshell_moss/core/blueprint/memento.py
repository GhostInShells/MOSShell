from typing import Iterable, Any, Tuple

from typing_extensions import Self

from pydantic import BaseModel, Field, AwareDatetime

from ghoshell_moss.message import Message, WithAdditional
from ghoshell_moss.message import unique_id
from datetime import datetime
from dateutil import tz

__all__ = [
    'Reaction',
    'Moment',
]

_Logos = str


class Reaction(BaseModel, WithAdditional):
    """
    上一轮与外部世界互动的结果.
    由于现在模型并没有能支持全双工的实现,
    所以仍然需要一种粘合机制拼出交互.
    """
    moment_id: str = Field(
        default_factory=unique_id,
        description="上一轮 Moment id.",
    )
    executed_logos: _Logos = Field(
        default='',
        description="上一轮交互, 系统实际执行的 logos. "
                    "驱动躯体与工具运行. 这里的 logos 是 符号/逻辑/指令/路径/现实规律 的含义. 对应中文的 道-言说."
                    "系统执行的 logos 和模型生成的 logos 并不全等. ",
    )
    messages: list[Message] = Field(
        default_factory=list,
        description="logos 执行同时或之后得到的内部 (比如躯体) 反馈结果. 是思维洞穴里的回声. ",
    )
    stop_reason: str = Field(
        default='',
        description="如果这是一个未完成的 Moment, 它可以被记录状态",
    )

    def new_moment(
            self,
            *,
            percepts: dict[str, list[Message]] | None = None,
            hint: str = '',
            command_logos: _Logos = '',
    ) -> "Moment":
        """
        基于上一轮 Reaction 产生下一轮 Moment.
        hint 仅本轮生效，不持久化。
        """
        return Moment(
            previous=self,
            percepts=percepts or {},
            hint=hint,
            command_logos=command_logos,
        )


class Moment(BaseModel, WithAdditional):
    """
    智能体上下文感知的关键帧.
    每个关键帧都在缝合上一轮的输出, 到这一轮输入位置.
    关键的切分点是每一轮智能体的输出 (logos)
    """

    id: str = Field(
        default_factory=unique_id,
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
    compacted_perspectives: list[Message] | None = Field(
        default=None,
        description='对 perspectives 的压缩结果. 如果有的话. '
    )
    percepts: dict[str, list[Message]] = Field(
        default_factory=dict,
        description="本轮的外部输入, source-keyed. 每个 source 覆盖写入, 天然去重."
                    "与 perspectives 对齐: 同一帧内同 source 只保留最新. "
                    "多轮对话中 Moment 实例独立, 不跨帧累积."
                    "JSON 反序列化后 dict 无序 — 需由 memento 存储层按固定插入序"
                    "重建 (当前无持久化数据, 预留).",
    )
    hint: str = Field(
        default='',
        description='仅在本轮生效的提示',
    )
    command_logos: _Logos = Field(
        default='',
        description='按系统约定, 要在大脑思考前执行的 logos. 可以考虑给如何给模型看. ',
    )
    logos: _Logos = Field(
        default='',
        description="关键帧里由决策模块生成的 logos, 通常来自大模型",
    )
    created: AwareDatetime = Field(
        default_factory=lambda: datetime.now(tz.gettz()),
        description="the time when the conversation was created",
    )

    def to_dict(self) -> dict[str, Any]:
        """提示如何用字典查看 moment 数据, 更多实现参考 BaseModel"""
        return self.model_dump(
            exclude_none=True,
            exclude_defaults=True,
            mode='json',
        )

    def for_saving(
            self,
            *,
            compacted_perspectives: list[Message] | None = None,
    ) -> 'Moment':
        return self.model_copy(
            update={'compacted_perspectives': compacted_perspectives or [], 'perspectives': {}, 'hint': ''},
        )

    def to_json(
            self,
            *,
            exclude_perspectives: bool = True,
            exclude_hint: bool = True,
            indent: int = 0,
    ) -> str:
        """
        标准的序列化方式, 也方便存储.
        """
        exclude: set[str] = set()
        if exclude_perspectives:
            exclude.add('perspectives')
        if exclude_hint:
            exclude.add('hint')
        return self.model_dump_json(
            exclude=exclude or None,
            indent=indent,
            ensure_ascii=False,
            exclude_none=True,
            exclude_defaults=True,
        )

    def new_reaction(self) -> Reaction:
        """生成下轮的接收池"""
        return Reaction(
            moment_id=self.id,
        )

    def previous_executed_logos(self) -> str:
        if self.previous is None:
            return ''
        return self.previous.executed_logos

    def with_perspective(self, key: str, messages: list[Message]) -> Self:
        """组合不同类型的动态内观上下文. key 没有内容, 只用来做多次更新的去重. """
        self.perspectives[key] = messages
        return self

    def with_percepts(self, source: str, messages: list[Message]) -> Self:
        """source-keyed percepts 写入. 同 source 覆盖, 天然去重. 对标 with_perspective."""
        self.percepts[source] = list(messages)
        return self

    def percepts_messages(self) -> Iterable[Message]:
        """展平所有 source 的 percepts, 保持插入序 (Python 3.7+ dict ordered)."""
        for messages in self.percepts.values():
            yield from messages

    def percepts_texts(self) -> list[str]:
        """展平所有 percepts 并提取纯文本, 用于测试断言与调试."""
        return [msg.to_content_string() for msg in self.percepts_messages()]

    def last_moment_id(self) -> str | None:
        if self.previous is None:
            return None
        return self.previous.moment_id

    # --- 基于 code as prompt 的思路介绍各种字段的组合意义 --- #

    def perspective_messages(self, *, compact_first: bool = False) -> Iterable[Message]:
        """
        这 "一瞬间" 提供给思考模块的, 关键帧讯息.
        是一个认知的滑动窗口, 类似于电脑屏幕之于人类 (永远只输入最新的一帧)
        """
        if len(self.perspectives) == 0:
            yield from []
            return
        if compact_first:
            # 优先用压缩后的记录.
            if self.compacted_perspectives is not None:
                yield from self.compacted_perspectives
                return
        # 返回全量的数据.
        for messages in self.perspectives.values():
            yield from messages

    def previous_reaction_messages(self) -> Iterable[Message]:
        if self.previous is None:
            yield from []
            return
        reaction = self.previous
        if len(reaction.messages) > 0:
            yield from reaction.messages
        if reaction.stop_reason:
            yield Message.new(tag='stop_reason').with_content(reaction.stop_reason)

    def is_empty(self) -> bool:
        return self.previous is None and all(len(v) == 0 for v in self.percepts.values())

    def is_empty_request(self) -> bool:
        return all(len(v) == 0 for v in self.percepts.values())

    def inputs_messages(
            self,
            with_hint: bool = True,
            with_command_executing: bool = True,
    ) -> Iterable[Message]:
        """通过别名, 方便理解 percepts 相当于 agent 的 inputs messages. """
        yield from self.percepts_messages()
        if with_command_executing and self.command_logos:
            # 提示模型正在执行的命令:
            yield Message.new(tag='executing').with_content(self.command_logos)
        if with_hint and self.hint:
            # 给模型的提示.
            yield Message.new(tag='hint').with_content(self.hint)

    def as_request_messages(
            self,
            *,
            with_perspectives: bool = True,
            with_hint: bool = True,
            with_command_executing: bool = True,
    ) -> Iterable[Message]:
        """
        所有这些消息, 理论上都会合并为一轮输入消息的 contents.
        本处是一个使用约定 (code as prompt), 不是硬性约束.
        """
        yield from self.previous_reaction_messages()
        if with_perspectives:
            yield from self.perspective_messages(compact_first=False)
        elif self.compacted_perspectives is not None:
            yield from self.compacted_perspectives
        yield from self.inputs_messages(with_command_executing=with_command_executing, with_hint=with_hint)

    def as_history_messages(self) -> Iterable[Message]:
        """当 Moment 作为历史消息时, perspectives 无论是否存储都应该遗忘. """
        yield from self.previous_reaction_messages()
        if self.compacted_perspectives:
            yield from self.compacted_perspectives
        yield from self.percepts_messages()

    @classmethod
    def to_history_turns(
            cls,
            moments: Iterable['Moment'],
    ) -> Iterable[Tuple[list[Message], _Logos | None]]:
        """
        将一个连贯的 moment 转化为历史上的回合制交互.
        以模型生产的 logos 为分割线. 如果某一轮模型没有生产 logos, 但仍然执行了
        就将 executed logos 插入历史.
        """
        buffered_messages = []
        last_moment_has_logos = False
        for moment in moments:
            # 如果上一轮没有模型生成的历史 logos, 但存在 executed logos. 将它缝合到历史消息中.
            # Ghost 可能会执行一个 command, 这时候是不允许思考的. 这样就不会有生产出来的 logos.
            # 而执行的 logos 如果不记录到上下文, 就是单纯的幻觉.
            if not last_moment_has_logos:
                if executed_logos := moment.previous_executed_logos():
                    buffered_messages.append(Message.new(tag='executed').with_content(executed_logos))
            # 加载从上一轮响应到这一轮输入的历史消息.
            buffered_messages.extend(moment.as_history_messages())
            # 判断这一轮里包含了 logos.
            this_moment_has_logos = len(moment.logos) > 0
            if this_moment_has_logos and not buffered_messages:
                # logos 存在却无可入史消息: 多半由 perspectives (历史中被遗忘) 触发, 或上游记录异常.
                # 补一条占位, 让回合结构自洽, 避免 logos 被下面的 buffered_messages 守卫静默吞掉.
                buffered_messages.append(
                    Message.new(tag='perspective').with_content('本轮基于实时感知快照响应, 快照未在历史中保留.')
                )
            if buffered_messages:
                # 这一轮有 logos 才会拆分一个回合返回. 否则只是 buffer 历史.
                if this_moment_has_logos:
                    yield buffered_messages, moment.logos
                    buffered_messages = []
            # 为下一轮做准备.
            last_moment_has_logos = this_moment_has_logos
        if buffered_messages:
            yield buffered_messages, None
