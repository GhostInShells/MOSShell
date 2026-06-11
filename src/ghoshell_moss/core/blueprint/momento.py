from typing import Iterable, Any, List
from typing_extensions import Self

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, AwareDatetime

from ghoshell_moss.message import Message, WithAdditional
from ghoshell_moss.message import unique_id
from datetime import datetime
from dateutil import tz

__all__ = [
    'Momento',
    'Reaction',
    'Moment',
    'MomentoIndex', 'MomentoMetadata',
]


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
    logos: str = Field(
        default='',
        description="上一轮交互, 模型输出的 logos. 不应该包含条件反射等非模型的输出. "
                    "驱动躯体与工具运行. 这里的 logos 是 符号/逻辑/指令/路径/现实规律 的含义. 对应中文的 道-言说 ",
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
            percepts: list[Message] | None = None,
    ) -> "Moment":
        """
        基于 Outcome 产生下一轮的观察.
        """
        return Moment(
            previous=self,
            percepts=percepts or [],
        )


class Moment(BaseModel, WithAdditional):
    """
    智能体上下文感知的关键帧.
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
    percepts: list[Message] = Field(
        default_factory=list,
        description="本轮的外部输入: 已经过解析/结构化/多模态对齐, 但尚未经过高层解读."
                    "在多轮对话中保持, 在触发思考前可以继续添加内容.  "
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
            update={'compacted_perspectives': compacted_perspectives or [], 'perspectives': {}},
        )

    def to_json(self, *, exclude_perspectives: bool = True, indent: int = 0) -> str:
        """
        标准的序列化方式, 也方便存储.
        """
        exclude = None
        if exclude_perspectives:
            exclude = {'perspectives'}
        return self.model_dump_json(
            exclude=exclude,
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
        return self.previous is None and len(self.percepts) == 0

    def is_empty_request(self) -> bool:
        return len(self.percepts) == 0

    def inputs_messages(self) -> Iterable[Message]:
        """通过别名, 方便理解 percepts 相当于 agent 的 inputs messages. """
        yield from self.percepts

    def as_request_messages(
            self,
            *,
            with_perspectives: bool = True,
    ) -> Iterable[Message]:
        """
        所有这些消息, 理论上都会合并为一轮输入消息的 contents.
        本处是一个使用约定 (code as prompt), 不是硬性约束.
        """
        yield from self.previous_reaction_messages()
        if with_perspectives:
            yield from self.perspective_messages(compact_first=False)
        yield from self.inputs_messages()


class MomentoIndex(BaseModel):
    """
    Momento 的索引, 用于索引历史讯息.
    """
    branch_id: str = Field(
        default_factory=unique_id,
        description="认知片段自身的唯一 id. ",
    )
    session_id: str = Field(
        default='',
        description="当前 momento 从属的  session id. ",
    )
    root_id: str | None = Field(
        default=None,
        description="momento tree root_id",
    )
    from_branch_id: str | None = Field(
        default=None,
        description="当前分支从哪个轨迹上分离出来. ",
    )
    from_moment_id: str | None = Field(
        default=None,
        description="the moment id that the current momento checkout from ",
    )
    created: AwareDatetime = Field(
        default_factory=lambda: datetime.now(tz.gettz()),
        description="the time when the conversation was created",
    )

    def fork(self, moment_id: str) -> 'MomentoIndex':
        return MomentoIndex(
            session_id=self.session_id,
            root_id=self.root_id,
            from_branch_id=self.branch_id,
            from_moment_id=moment_id,
        )


class MomentoMetadata(BaseModel, WithAdditional):
    title: str = Field(
        default='',
        description="方便查询时理解的标题信息",
    )
    description: str = Field(
        default='',
        description="内容描述.",
    )
    recap: str = Field(
        default='',
        description='前情提要',
    )
    summary: str = Field(
        default='',
        description="摘要",
    )
    updated: AwareDatetime = Field(
        default_factory=lambda: datetime.now(tz.gettz()),
        description="the time when the conversation was created",
    )

    def fork(self) -> 'MomentoMetadata':
        return MomentoMetadata(
            title='',
            description='',
            recap=self.summary,
            summary='',
        )


_Logos = str


class MomentBranch(ABC):
    """
    Moment 交织而成的记忆碎片. 用来构建认知体系.
    作为 MOSS 架构的连续认知轨迹, 可以被 Ghost 或其它模型使用, 但也可以不使用.

    这个认知轨迹以对外交互为主, 不包含一次 moment 响应过程中, 所以不输出到 Moment 里的中间信息, 比如思考过程中工具调用,

    为何都是同步接口?
    moment 的更新理论上影响历史对话, 而且时序相关, 所以要做内存更新.
    而 moment 的存储是 io 操作, 要写的话, 卸载到子线程或协程线性写.
    一旦开 async 就永远有被 create task 的并发问题, 而 async 做内存更新没有任何收益.
    """

    @property
    def id(self) -> str:
        return self.index.branch_id

    @property
    def session_id(self) -> str:
        return self.index.session_id

    @property
    @abstractmethod
    def meta(self) -> MomentoMetadata:
        """返回 Meta 信息. """
        pass

    @property
    @abstractmethod
    def index(self) -> MomentoIndex:
        """返回 index 信息"""
        pass

    @abstractmethod
    def update(self, moment: Moment) -> None:
        """
        增加新的 observation.
        内存生效, 不阻塞.
        如果 moment 已经存在, 则修改历史; 不存在, 则更新它.
        """
        pass

    @abstractmethod
    def update_meta(self, meta: MomentoMetadata) -> None:
        """更新 metadata. """
        pass

    @abstractmethod
    def moments(self, reverse_order: bool = True, limit: int = -1) -> List[Moment]:
        """
        list observations in reverse chronological order.
        """
        pass

    @abstractmethod
    def fork(
            self,
            *,
            moment_id: str | None = None,
            title: str = '',
            description: str = '',
            recap: str = '',
    ) -> 'MomentBranch':
        """
        fork 一个对象或快照.
        :param moment_id: 为 None 的话, 以当前最后一帧 moment id 为起点.
        :param title: 新分支的标题.
        :param description: 新分支的描述.
        :param recap: 前情提要, 如果为空使用当前的 summary.
        """
        pass


class Momento(ABC):
    """
    所有 moment branches 的存储.
    通常是和 session scope 对齐.
    当前分支通常就是 session id 对应的分支.
    """

    @abstractmethod
    def main(self) -> MomentBranch:
        """
        整个 Momento 的当前主分支.
        """
        pass

    @abstractmethod
    def history(
            self,
            *,
            reverse_order: bool = True,
            limit: int = -1,
    ) -> List[MomentoMetadata]:
        """
        获取历史分支.
        """
        pass

    @abstractmethod
    def get_branch(
            self,
            branch_id: str,
            *,
            read_only: bool = False,
            or_create: bool = False,
    ) -> MomentBranch | None:
        """
        获取一个分支.
        """
        pass

    @abstractmethod
    def switch(self, branch: MomentBranch) -> None:
        """
        切换主分支, 同时会保存当前 Momento 的主分支.
        """
        pass
