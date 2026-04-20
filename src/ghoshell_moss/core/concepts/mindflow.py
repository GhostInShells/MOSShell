from typing import Callable, Coroutine, Protocol, Iterable, AsyncIterator, Any

from typing_extensions import Self, Literal
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, AwareDatetime, ValidationError

from ghoshell_moss.message import Message, Content, WithAdditional
from ghoshell_moss.core.concepts.command import ObserveError
from ghoshell_common.helpers import uuid
from PIL.Image import Image
import datetime
import dateutil
import time
import asyncio
import enum

"""
Mindflow 架构设计. 解决 感知/执行/思考 三循环的全双工状态管理问题. 
"""

# 关于三循环:
# 1. 思考循环: 模型接受信息, 思考并输出.
# 2. 感知循环: 接受外部世界各种感知信号, 产生冲动.
# 3. 执行循环: 执行流式指令, 同时获取流式的反馈.
# 双工:
# 1. 感知 -> 思考: 思考输出的同时, 感知在输入, 都是流式的.
# 2. 思考 -> 执行: 思考产生 token 的同时, 流式解释器立刻执行, 并且同时产生指令结果.
# 3. 执行 -> 感知: 当执行行为在外部世界产生效果, 会反馈到感知链路.
#
# 在这种场景下, 涉及一个复杂的状态管理体系.
# 1. 数据组织: 来自三个循环的信息需要有序记录.
# 2. 时序: 三循环的执行逻辑要对齐. 避免思维奔逸 (拿到反馈前就继续行动) 和裂脑 (感知/思考/行为消费不同时间轴上的信息.)
# 3. 中断: 来自三方的信号可能触发中断, 如高优打断事件, 模型调度异常, 执行错误指令等.
# 4. 结束: 状态需要有序地结束.
#
# 在当前 Mindflow 的体系中, signal + impulse + nucleus 是对感知的隔离建模, 预期用可迭代的单元将它们分割出去.
# Attention + Articulate + Action 是运行状态的管理调度体系.
# Mindflow 是中心管理单元.
# 如果要用多线程做资源隔离, 通常是 Mindflow + Nucleus / Articulate  在独立线程.
# 不过不建议用多线程做隔离, 最好在实现底层用多进程模型隔离.

__all__ = [
    'Priority', 'SignalName', 'Signal', 'SignalMeta', 'InputSignal', 'Impulse',
    'Flag',
    'Logos', 'Observation', 'Outcome',
    'Action', 'Articulate',
    'Nucleus', 'Mindflow', 'Attention',
    # 几个关键的通讯信号, 用来快速终止一些循环.
    'AttentionAbortedError', 'ObserveError', 'ActionAbortedError', 'ArticulateAbortedError',
    'PreemptedElseSuppress', 'BufferImpulse',
]

SignalName = str


class Priority(enum.IntEnum):
    """
    为了避免优先级无限膨胀, 因此做策略约定.
    """
    DEBUG = -1  # 通常只是保留在 Mindflow 的 context 列表中, 不会产生 Attention.
    INFO = 0  # 特殊的默认约定, 当相同 source 的 Impulse 在 Attention 生命周期中, 接受到了 INFO 级别的 Impulse, 就会唤起新的 observe.
    NOTICE = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4
    FATAL = 5  # 约定的最高级别, 永远抢占成功.


class Signal(BaseModel):
    """
    端侧发送给智能体响应的信号. 可能有以下几个关键特征:
    1. 多源头, 比如视觉/听觉/触觉/故障/通讯/异步回调....
    2. Partial, 典型的例子是 ASR 的首包到尾包, 每个分句都是一个 Partial 包.
    3. 保鲜, 过期的信号会直接丢弃.
    4. 以 AI 可以理解的消息为优先.
    """

    __state__: Literal['created', 'pending', 'dispatched', 'ignored'] = 'created'
    """内部用于 debug 的参数"""

    name: SignalName = Field(
        description="the signal name, if not match any mind pulse, the signal will be ignore",
    )
    id: str = Field(
        default_factory=uuid,
        description="unique identifier of the signal",
    )
    trace_id: str = Field(
        default='',
        description="the trace id of the signal. 通常系统自动标记, 不需要传值. ",
    )
    complete: bool = Field(
        default=True,
        description="whether the signal complete or partial."
                    "如果是 partial 包, 应该后续传递 complete = True 的尾包."
                    "但 partial 包仍然有存在意义, 比如打断, 占据注意力等. 举个例子, "
                    "一个高优的 ASR 首包打断了 AI 行为, 同时占据了注意力."
                    "抽象设计上不做粘包逻辑. 如果有粘包的需要, 需要结合 Nucleus 定义内部协议.",
    )
    max_hop: int = Field(
        default=1,
        description="maximum hop number, 为 0 不传播. 系统内部调度时会处理. 不应该修改它. Mindflow 内部使用这个字段. ",
    )
    issuer: str = Field(
        default="",
        description="the issuer of the signal, 不需要显示传递, 实际链路发布时会添加.",
    )
    priority: Priority = Field(
        default=Priority.INFO,
        description="信号的优先级, 越大优先级越高. 用于做抢占式调度. 来自边缘系统的输入本身应包含第一轮优先级"
    )
    strength: int = Field(
        default=100,
        description="信号的强度. 输入信号在 0~300 之间做设计, 常态位是 100. 通常直接用默认值即可."
                    "因为信号的衰减逻辑在 Attention 中设计, 所以在不耦合 attention 的情况下, 对信号强度的理解就按百分比处理."
                    "比如 100 * 1.2 表示加权 20%. ",
        ge=0,
        le=300,
    )
    description: str = Field(
        default='',
        description="short description of the signal."
                    "这个字段是可省略的. 它的作用是在极简的 Nucleus 实现中, 直接用它提示状态. "
                    "类似 IM 里红点展示的用户消息, 会保留一个缩略的一句话提示. ",
    )
    messages: list[Message] = Field(
        default_factory=list,
        description="被处理过的消息体.",
    )
    prompt: str = Field(
        default='',
        description="the prompt to handle the signal."
                    "prompt 也是可选的实现. 默认为空即可. 它的作用是一种补丁. 当一个输入进来时, 模型很可能按预训练约定去理解."
                    "典型案例如 图片, 模型会默认认为这是在 IM 里提交的一张照片. 而不知道这是自己的 vision. "
                    "这时就可以用补丁; 为什么拆到 prompt 字段呢? "
                    "因为 prompt 对多轮对话而言是一定要丢弃的; 放入 messages 里, 会导致上下文里被 prompt 补丁淹没. ",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="meta data of the signal follow the protocol of the name."
                    "可扩展的强类型约定, 通过 SignalMeta 可以提供一个 JSON Schema 协议去定义细节. ",
    )
    stale_timeout: float = Field(
        default=0,
        description="the stale signal will be ignored. ",
    )
    created_at: AwareDatetime = Field(
        default_factory=lambda: datetime.datetime.now(dateutil.tz.gettz()),
    )

    @classmethod
    def new(
            cls,
            name: SignalName,
            *messages: Message,
            priority: Priority = Priority.INFO,
            description: str = '',
            metadata: dict[str, Any] | None = None,
            strength: int = 100,
            stale_timeout: float = 0,
            complete: bool = True,
    ) -> Self:
        return cls(
            name=name,
            messages=list(messages),
            priority=priority,
            description=description,
            metadata=metadata or {},
            strength=strength,
            stale_timeout=stale_timeout,
            complete=complete,
        )

    def priority_strength(self) -> int:
        return self.priority * 1000 + self.strength

    def is_stale(self) -> bool:
        if self.stale_timeout <= 0:
            return False
        delta = time.time() - self.created_at.timestamp()
        return delta > self.stale_timeout

    def to_json(self) -> str:
        return self.model_dump_json(indent=0, exclude_none=True, exclude_defaults=True, ensure_ascii=False)

    def __repr__(self):
        return f"<Signal id={self.id} trace={self.trace_id} name={self.name}>"


class SignalMeta(BaseModel, ABC):
    """
    定义一个 Signal 的补充协议 (围绕 metadata), 用于在环境中被发现, 从而可以做到自解释.
    所有字段应该都是支持序列化的, 否则会在传输时报错.
    同时 Pydantic BaseModel 定义的 Signal Meta 可以作为协议被发现, 提供 metadata 的 json schema 协议.
    """

    @classmethod
    @abstractmethod
    def signal_name(cls) -> SignalName:
        """定义唯一的 signal 名称. """
        pass

    @classmethod
    def priority(cls) -> Priority:
        return Priority.INFO

    @classmethod
    def match(cls, signal: Signal) -> bool:
        return signal.name == cls.signal_name()

    @classmethod
    def from_signal(cls, signal: Signal) -> Self | None:
        """
        快速做 signal metadata 的数据还原加工

        典型用法:
        >>> def match_signal(s: Signal):
        >>>     if input_signal := InputSignal.from_signal(s):
        >>>        ...
        """
        if cls.signal_name() != signal.name:
            return None
        try:
            metadata = signal.metadata
            return cls.model_validate(metadata)
        except ValidationError:
            return None

    def to_signal(
            self,
            *messages: Message | str | Image,
            description: str = '',
            stale_timeout: float = 0,
            priority: int | None = None,
    ) -> Signal:
        """快速用 meta 定义一个 signal. 提示两者的使用机制. """
        name = self.signal_name()
        wrapped_messages = []
        for msg in messages:
            if isinstance(msg, Image):
                wrapped_messages.append(Message.new().with_content(msg))
            elif isinstance(msg, str):
                wrapped_messages.append(Message.new().with_content(msg))
            elif isinstance(msg, Message):
                wrapped_messages.append(msg)
        priority = self.priority() if priority is None else priority
        return Signal(
            name=name,
            messages=wrapped_messages,
            metadata=self.model_dump(exclude_defaults=True, exclude_none=True),
            description=description,
            stale_timeout=stale_timeout,
            priority=priority,
        )


class InputSignal(SignalMeta):
    """
    系统最基础的 Input 讯号. 代表一个明确的输入.
    """

    @classmethod
    def signal_name(cls) -> SignalName:
        return 'input'

    @classmethod
    def priority(cls) -> Priority:
        return Priority.NOTICE


class Impulse(BaseModel):
    """
    the impulse that raise mindflow attention
    Impulse 可以是 Nucleus 加工后的产物, 也可以是 Signal 的原样复制 (极简情况下).
    它的核心目的是隔离原始信号, 将之转换成更明确的调度信号.
    """
    id: str = Field(
        default_factory=uuid,
        description="the impulse id",
    )
    source: str = Field(
        default='',
        description="the nucleus source name",
    )
    priority: Priority | int = Field(
        default=Priority.NOTICE,
        description="the impulse priority",
    )
    strength: int = Field(
        default=100,
        description="the impulse 初始强度, 在 attention 中设计强度计算曲线用来解决相同优先级打断机制.",
        ge=0,
        le=300,
    )
    on_logos_start: str = Field(
        default='',
        description="the start logos insert into the stream. 可以理解为条件反射, 在思考启动前就会执行. ",
    )
    complete: bool = Field(
        default=True,
        description="if the impulse is complete, or just occupy the attention until complete impulse from the same id",
    )
    description: str = Field(
        default='',
        description="the impulse short description. 这个描述可以理解为 IM 消息列表上的摘要. ",
    )
    messages: list[Message] = Field(
        default_factory=list,
        description="the messages of the impulse. if empty, no need to think",
    )
    prompt: str = Field(
        default='',
        description="the prompt to handle the impulse",
    )

    stale_timeout: float = Field(
        default=0,
        description="当一个 Impulse 无法占据到 Attention 时的过期时间. "
    )

    # -- 系统内部字段 -- #

    trace_id: str = Field(
        default='',
        description="the impulse trace id, 向上溯源.",
    )
    created_at: AwareDatetime = Field(
        default_factory=lambda: datetime.datetime.now(dateutil.tz.gettz()),
        description="the creation time of the impulse",
    )
    strength_decay_seconds: float = Field(
        default=20,
        description="Strength decay 约定时间. 如果不定义的话, 使用系统默认的约定. 作为最底层的约束存在. ",
    )

    @classmethod
    def from_signal(cls, signal: Signal, source: str, stale_timeout: float | None = None) -> Self:
        """
        一个简单的示例, 直接将 signal 转化成 impulse 不做任何处理.
        实际上 Impulse 并不见得来源于单一 Signal. 这种涉及只为了通讯使用.
        """
        stale_timeout = stale_timeout if stale_timeout is not None else signal.stale_timeout
        if stale_timeout > 0:
            stale_timeout = stale_timeout - (time.time() - signal.created_at.timestamp())
        return Impulse(
            source=source,
            trace_id=signal.trace_id or signal.id,
            priority=signal.priority,
            strength=signal.strength,
            messages=signal.messages.copy(),
            description=signal.description,
            prompt=signal.prompt,
            complete=signal.complete,
            stale_timeout=stale_timeout,
        )

    def priority_strength(self) -> int:
        return self.priority * 1000 + self.strength

    def is_stale(self) -> bool:
        if self.stale_timeout <= 0:
            return False
        delta = time.time() - self.created_at.timestamp()
        return delta > self.stale_timeout

    def __repr__(self):
        return f"<Impulse id={self.id} trace={self.trace_id} source={self.source}>"


class Nucleus(ABC):
    """
    并行 感知/思考/决策 单元的统一抽象. 它接受输入信号, 返回动机, 属于 “单生产者-单消费者”的有界缓冲区
    在输入场景中, 它是输入信号的治理层, 用于将高频的输入信号治理/加工/降频/加权后, 转化为 Mindflow 可以处理的 Impulse.
    可以拥有各种实现机制, 比如:
    1. lru buffer, 将所有的信号合并
    2. summary, 将信号合并摘要
    3. priory queue, 结合 maxsize 做单一信号量.
    4. arbiter, 加入仲裁者模型做快速校验.
    5. sidecar, 旁路思考, 向主路广播...

    同样, 它可以作为 MultiTasks/Planner/Timer/Ticker/MultiAgent 等各种机制, 通过 signal 和 impulse 两个大一统抽象管理特别复杂的
    异步通讯逻辑, 与主交互脑通讯. 理想情况下它不应该包含调度逻辑, 而只作为通讯调度层.
    """

    @abstractmethod
    def name(self) -> str:
        """
        用于区分不同的 Nucleus 单元.
        """
        pass

    @abstractmethod
    def description(self) -> str:
        """
        所有的 Nucleus 都应该是自解释的, 而且这个自解释要足够高效, 能一句话自我描述.
        """
        pass

    @abstractmethod
    def status(self) -> str:
        """
        当前 Nucleus 的状态提示, 参考 IM 的消息红点, 要简短而精准.
        如果为空, 会被忽略.
        """
        pass

    @abstractmethod
    def signals(self) -> list[SignalName]:
        """
        声明监听的信号类型.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        排空讯号, 应该强制清空所有状态.
        用于做极限故障下的还原, 作为最基础的恢复手段.
        """
        pass

    @abstractmethod
    def on_signal(self, signal: Signal) -> None:
        """
        接受一个信号量, 在内部开始执行校验逻辑, 生成 impulse.
        没有背压, 应当尽可能快地入队或丢弃，不执行任何耗时或异步操作。内部应有独立的任务循环消费队列。
        """
        pass

    @abstractmethod
    def with_bus(self, signal_broadcast: Callable[[Signal], None], impulse_notify: Callable[[Impulse], None]) -> None:
        """
        注册总线, 可以广播信号, 或者发送 impulse.
        1. Nucleus 可以广播 signal 给其它监听者.
        2. Nucleus 产生了 Impulse, 可以回调通知, 比如回调 Mindflow.
        注意, Impulse 回调时不能 pop, 如果回调的 Impulse 无法抢占 attention, 应该会收到一个 suppress 信号.

        关于通讯, 目前设计上 Nucleus 和 Mindflow 的接口层在相同循环内.
        但实际上总线的调用可能在不同线程. 所以总线函数底层必须是线程安全的 (比如用 janus.Queue).
        """
        pass

    @abstractmethod
    def suppress(self, suppress_by: Impulse) -> None:
        """
        如果产生的 impulse 不能被接纳, Nucleus 应该收到一个 suppress 信号
        可以在内部实现加权/降权 逻辑.
        :param suppress_by: 被别的信号压制, 得到别的信号. 未来可以通过决策单元判断是否要加权.
        """
        pass

    @abstractmethod
    def pop_impulse(self, impulse: Impulse) -> None:
        """
        通知 Nucleus 一个 Impulse 被 pop 了.
        """
        pass

    @abstractmethod
    def peek(self, no_stale: bool = True) -> Impulse | None:
        """
        查看一下最新的 Impulse.
        方便做 ranking.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        """
        启动 Nucleus 自身的生命周期, 包含异步逻辑, 或者启动子进程.
        """
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        退出生命周期.
        """
        pass


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

    def new_outcome(self) -> Outcome:
        """生成下轮的接收池"""
        return Outcome(
            id=self.id,
        )

    def as_messages(self, *, with_context: bool = True) -> Iterable[Message]:
        """
        所有这些消息, 理论上都会合并为一轮输入消息的 contents.
        本处是一个使用示范 (code as prompt), 不是硬性约束.
        """
        if self.previews is not None:
            outcome = self.previews
            if len(outcome.messages) > 0:
                yield Message.new().with_content('<outcomes>')
                yield from outcome.messages
                yield Message.new().with_content('</outcomes>')
            if outcome.stop_reason:
                yield Message.new(tag='stop_reason').with_content(outcome.stop_reason)

        if len(self.context) > 0:
            if with_context:
                yield Message.new().with_content("<context>\n")
                for context_messages in list(self.context.values()):
                    yield from context_messages
                yield Message.new().with_content("\n</context>")
            else:
                count = 0
                for compacted in self.context.values():
                    count += len(compacted)
                yield Message.new().with_content(f"<compacted>{count} history messages compacted </compacted>")
        yield from self.inputs
        if self.prompt:
            yield Message.new(tag='prompt').with_content(self.prompt)

    def as_contents(self, *, with_context: bool = True) -> Iterable[Content]:
        """
        用这种方式, 可以拿到和 Anthropic 基本兼容的 Contents.
        可以包裹到 UserMessageParams 或 ToolMessageParams 里.
        """
        for msg in self.as_messages(with_context=with_context):
            yield from msg.as_contents(with_meta=True)


Logos = AsyncIterator[str]
"""
智能体输出用来驱动躯体/工具/交互/思考 等一切能力的讯息. 对应中文的 "道". 目前在项目里主要是 CTML. 它包含四重含义:
1. 它本身是语言, 在 MOSS 架构里包含了运行时控制的魔力 (CTML). 
2. 它是逻辑的编织, 要符合现实世界的规律 (时间第一公民, 时序拓扑, 结构化并行)
3. 它驱动了躯体/工具/思维 的运行轨迹
4. 它包含了智能体与现实世界交互的底层原则, 一个智能体通过它输出的 logos 来展示它自身的 logos. 

经过和 Gemini/Deepseek 的多轮讨论, 没有更好的词能够精准涵盖它所包含的 哲学/技术拓扑, 又屏蔽掉底层实现 (比如 CTML). 

在 MOSS 架构中运行的智能体, 更像是 "魔法师". 它不是用精确到舵机电平的神经脉冲控制外部世界, 而是用符号流.
类似用魔法吟唱的方式驱动火球, 石头人 等. 
"""


class Flag(Protocol):
    """
    对齐 Event 对应的接口, 要实现线程安全 (参考 ghoshell_moss.core.helpers.ThreadSafeEvent) 同时支持信号回调.
    """

    @abstractmethod
    async def wait(self) -> None:
        pass

    @abstractmethod
    def set(self) -> None:
        pass

    @abstractmethod
    def is_set(self) -> bool:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


PreemptedElseSuppress = bool
BufferImpulse = None
UnreadOutcome = list[Message]
StopReason = str


class AttentionAbortedError(Exception):
    """
    方便 Attention 模块明确关闭整个 Attention.
    在各个子模块均生效.
    """
    pass


class ArticulateAbortedError(Exception):
    pass


class ActionAbortedError(Exception):
    pass


class Articulate(ABC):
    """
    推理决策单元, 将推理的结果发送给执行单元.
    需要实现线程安全.
    """

    @property
    @abstractmethod
    def observation(self) -> Observation:
        """
        推理时的关键帧片段.
        """
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        """
        启动推理单元.
        """
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        关闭本轮推理单元.
        """
        pass

    @abstractmethod
    def abort(self, error: str | AttentionAbortedError | Exception | None) -> None:
        """
        显式声明退出 Attention.
        当 abort 提交时, 它所注册的任务全部会执行结束.
        """
        pass

    def raise_observe(self, message: str) -> None:
        """
        抛出一个 ObserveError 方便快速退出调用栈.
        被 __aexit__ 捕获后, 会标记为需要下一轮观察.
        """
        raise ObserveError(message)

    @abstractmethod
    async def send_logos(self, logos: Logos) -> None:
        """
        发送 Logos 流
        """
        pass

    @abstractmethod
    def create_task(self, cor: Coroutine) -> asyncio.Future:
        """
        创建和 Attention 生命周期同步的 task.
        如果 task 抛出 CancelError 之外的 Error, 会中断整个 Attention 运行.
        """
        pass

    @abstractmethod
    def flag(self, name: str) -> Flag:
        """
        声明一个 flag, 用于生命周期通讯, 需要是一个线程安全的可阻塞对象.
        因为未来  躯体/思考/感知 可能运行在三个线程中.
        执行协议可以定义不同的生命周期节点, 方便一些运行逻辑做很复杂的交叉阻塞.
        目前只是预留的一个扩展, 暂时不做约定实现.
        """
        pass

    @abstractmethod
    async def send(self, delta: str) -> None:
        """
        发送单个 logos delta.
        logos 是无背压的, 因为 logos 的执行也是并行流式的, 无法感受到真实队列膨胀.
        所以最终应该靠积压量做快速失败.
        """
        pass


class Action(ABC):
    """
    控制 Logos 的执行循环.
    """

    @abstractmethod
    def logos(self) -> Logos:
        """
        返回本轮生成的执行文本.
        """
        pass

    @abstractmethod
    def outcome(self, *messages: Message | str, observe: bool = False) -> None:
        """
        提交 outcome, 标记是否要引发下一轮观察.
        如果在一个 Action 的生命周期中 Observe 被标记了, 或者发生了特殊的异常,
        Attention 会循环下一组调用.
        如果没有需要观察的 outcome, Attention 会自然结束.
        """
        pass

    @abstractmethod
    def flag(self, name: str) -> Flag:
        """
        声明一个 flag, 用于生命周期通讯, 需要是一个线程安全的可阻塞对象.
        因为未来  躯体/思考/感知 可能运行在三个线程中.
        执行协议可以定义不同的生命周期节点, 方便一些运行逻辑做很复杂的交叉阻塞.
        目前只是预留的一个扩展, 暂时不做约定实现.
        """
        pass

    @abstractmethod
    def abort(self, error: str | AttentionAbortedError | Exception | None) -> None:
        """
        显式声明退出 Attention.
        当 abort 提交时, 它所注册的任务全部会执行结束.
        """
        pass

    def raise_observe(self, message: str) -> None:
        """
        抛出一个 ObserveError 方便快速退出调用栈.
        被 __aexit__ 捕获后, 会标记为需要下一轮观察.
        """
        raise ObserveError(message)

    @abstractmethod
    def create_task(self, cor: Coroutine) -> asyncio.Future:
        """
        创建和 Attention 生命周期同步的 task.
        如果有一个任务抛出了 Cancel 之外的 Error, 会停止其它的任务.
        """
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        """
        启动本轮执行单元.
        """
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        关闭本轮执行单元.
        如果发生了异常, 根据其影响决定是否触发下一轮.
        还是直接关闭 Attention.
        """
        pass


class Attention(ABC):
    """
    一种三循环全双工运行时的资源和状态调度单元.
    它通常是 Impulse 创建出来的实例, 一直到 思考/执行 都结束后退出.
    它可以连续地输出 observation, 直到注意力自身被中断.
    因此思考流程可以不断从 attention 中获取连续的 Re-Act 讯号, Mindflow 负责打断.
    """

    @abstractmethod
    def peek(self) -> Impulse:
        """
        快速窥探已经持有的 impulse.
        """
        pass

    @property
    def id(self) -> str:
        return self.peek().id

    @abstractmethod
    def is_aborted(self) -> bool:
        """
        快速校验运行时状态.
        """
        pass

    @abstractmethod
    def is_started(self) -> bool:
        """
        如果一个 Attention 从未启动就被取消了.
        下一个继承它的 Attention 应该要拿到的, 是它尚未处理过的上一轮 outcome.
        """
        pass

    @abstractmethod
    def on_observation(self, callback: Callable[[Observation], None]) -> None:
        """
        注册 Observation 回调, 通常用来整理历史记录.
        当正常运行的过程中, 一个 observation 被创建时会使用它.
        """
        pass

    @abstractmethod
    def flag(self, name: str) -> Flag:
        """
        声明一个 flag, 用于生命周期通讯, 需要是一个线程安全的可阻塞对象.
        因为未来  躯体/思考/感知 可能运行在三个线程中.
        执行协议可以定义不同的生命周期节点, 方便一些运行逻辑做很复杂的交叉阻塞.
        目前只是预留的一个扩展, 暂时不做约定实现.
        """
        pass

    @abstractmethod
    def with_context_func(
            self,
            context_name: str,
            context_func: Callable[[], list[Message]],
    ) -> Self:
        """
        注册一个 context func, 在运行时 attention 可以随时用 context func 编织当前的 context, 更新上下文.
        这个函数是一个同步函数, 它的目标不是并行调度, 而是以最快速度拿到一个快照, 实际上应该从缓存里拿.
        计划中要拿到的快照包括:
        1. Mindflow 的快照, 可以看到所有 nucleus 的最新状态. 类似飞书/微信 这样 IM 的红点提示.
        2. Shell 的快照, 也就是 MOSS dynamic 动态上下文.
        3. Interpreter 的快照, 记录当前瞬间, 哪些命令正在执行, 有多少被取消, 多少执行完毕.
        """
        pass

    @abstractmethod
    async def wait_aborted(self) -> None:
        """
        阻塞到 Attention 停止运行.
        实际上 Attention 启动时就会内部创建生命周期检查, 即便其它 task 死锁也会强制退出.

        >>> async def run_attention(attention: Attention) -> None:
        >>>     async with attention:
        >>>         ...
        >>>         await attention.wait_aborted()
        >>>     await attention.wait_closed()
        """
        pass

    @abstractmethod
    async def wait_closed(self) -> None:
        """
        可用于阻塞到 Attention 生命周期运行结束. 也就是 __aexit__ 完成阻塞.
        wait_aborted 和 wait_closed 是两个不同的信号.
        """
        pass

    @abstractmethod
    def on_challenge(self, challenger: Impulse) -> PreemptedElseSuppress | BufferImpulse:
        """
        仲裁新的 impulse. 决定自身是否被中断. 调度发起者是 mindflow.
        最基础的仲裁逻辑:
        0. 启动保护期, 随时间衰减.
        1. 如果 id 和当前 Impulse 相同, complete 取代 incomplete 并解除 impulse 阻塞.
        2. 挑战的 impulse priory 低于当前 impulse 优先级, 返回 False, 目标 impulse 发起方接受 suppress 回调.
        3. 优先级相同, 应该基于同源提权, 异元降权的原理做强度比较.
        4. 如果挑战者优先级更高, 则挑战一定成功. 当前 Attention 应该 abort.
        5. 如果 priority 为 Fatal, 应该永远被打断.

        这是最简单的规则. Attention 更好的做法是有一个速度极快的仲裁者. 它要具备响应大量讯号挑战的极简算法.

        - Preempted(True):
            如果挑战成功, Mindflow 应该实例化新的 Attention 之后, abort 当前的 Attention.
        - Supress (False):
            挑战失败, Mindflow 应该 supress impulse 的源头.
        - BufferImpulse (None):
            这个 Impulse 被 Attention 吸收了, 当 Attention 没被中断时, 会将 Impulse 提供到下一轮 Observation.
            Buffer Impulse 提供连续观察思考的语义. 只有同源的 Impulse, 且级别为 Info 时会更新.

        attention 管理一个源响应的生命周期.
        在这个生命周期中, 如果想要抢占, 则应该走 Impulse 逻辑打断.
        想要观察, 则走 outcome.
        想要提供低优先级的补充信息, 走 INFO.

        OnChallenge 在系统内最核心要解决的问题, 是消除大多数情况下的仲裁风暴和无限抖动.
        这在早期工程复杂度简单的时候, 直接通过约定的设计范式解决.
        更复杂的情况下会引入高阶反身性仲裁, 那属于甜蜜的烦恼.
        """
        pass

    @abstractmethod
    def loop(self) -> AsyncIterator[tuple[Articulate, Action]]:
        """
        循环生成 Articulate 和 Action, 将它们发送到两个循环中 (可能是独立线程).
        当一组里的 Articulate / Action 都执行完毕时, 循环会进入下一轮检查.
        如果 Attention 没有任何需要 Observe 的讯息, 则会自然退出 Attention.
        Attention 将自身的 API 封装成线程安全给后两者.
        """
        pass

    @abstractmethod
    def is_closed(self) -> bool:
        """
        是否已经运行结束.
        """
        pass

    @abstractmethod
    def last_outcome(self) -> Outcome:
        """
        用来返回当前 Attention 的未处理状态.
        即便运行结束也会保留, 直到垃圾删除.
        用来保障 Mindflow 生成下一帧 Attention 时, 能够正确地携带上一轮的未处理结果.
        """
        pass

    @abstractmethod
    def abort(self, error: str | AttentionAbortedError | Exception | None) -> None:
        """
        显式声明退出 Attention.
        当 abort 提交时, 它所注册的任务全部会执行结束.
        """
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        """可重入的生命周期, 用来拦截未处理异常. """
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """整个生命周期结束"""
        pass


class Mindflow(ABC):
    """
    三循环全双工智能体的思维调度中枢.
    它解决的核心问题是, 如何 管理/描述/隔离 一个全双工三循环系统的运行逻辑.

    三循环: 1. 感知体系;  2. AI 思考单元. 3. 躯体运行时.  除此之外还有一个控制循环.
    双工: 1. 躯体输出; 2. 感知输入. 两者并行.
    有复杂的中断逻辑: 0. 强制命令, 比如熔断, 急停. 1. 思考异常; 2. 执行异常; 3. 执行结束; 4. 输入更强的信号, 中断.

    同时有很多个状态和讯号通讯, 而在一个时间片里只有一组行为拥有可运行资源.

    Mindflow 的作用就是统筹所有的实现模块:
    1. nucleus: 感知单元, 接受原始信号量, 通过加工后返回有优先级效果的 Impulse. 解决并行感知后聚合/行为仲裁的问题.
    2. attention: 单一执行状态管理, 能同时接受多方的讯号, 维持一个可被抢占的运行时状态. 交换数据, 管理所有生命周期.
    """

    @abstractmethod
    def faculties(self) -> Iterable[Nucleus]:
        """
        持有的并行感知, 思考, 裁决单元.
        这里的 nucleus 并不一定是个执行单元, 也可以仅仅是一个通讯单元或 Adapter.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        pass

    @abstractmethod
    async def wait_started(self) -> None:
        """等待启动完成."""
        pass

    @abstractmethod
    def wait_started_sync(self, timeout: float | None = None) -> bool:
        pass

    @abstractmethod
    def is_quiet(self) -> bool:
        """
        has no attention and impulse
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        排空讯号, 应该强制清空所有状态.
        用于做极限故障下的还原, 作为最基础的恢复手段.
        """
        pass

    @abstractmethod
    def context_messages(self) -> list[Message]:
        """
        通过一个 message func, mindflow 可以快速描述自身当前的状态.
        类似 IM 红点的机制, 描述所有有状态 Nuclei 最新的情况.
        """
        pass

    @abstractmethod
    async def add_nucleus(self, nucleus: Nucleus) -> Self:
        """
        动态注册新的感知单元. 理论上可以在运行时添加启动.
        """
        pass

    @abstractmethod
    def on_impulse(self, impulse: Impulse) -> None:
        """
        接受一个 impulse, 并进入和当前 attention 的 challenge 仲裁.
        注意, 这里的 on_signal / on_impulse 作为总线提供给 Nucleus 时, 要防止信号成环无限传播.
        似乎没有系统机制可以百分之百预防.
        """
        pass

    @abstractmethod
    def on_signal(self, signal: Signal) -> None:
        """
        接受 signal 回调. 由于 Signal 的回调很可能和 Mindflow 不是在同一个线程或循环,
        所以内测需要卸载到当前循环, 并且考虑做好讯号闸门.
        Signal 的限频最好不在 Mindflow 侧做, 而应该通过发送者/环境中间件解决限频问题.
        """
        pass

    @abstractmethod
    def attention(self) -> Attention | None:
        """
        返回当前的 Attention.
        """
        pass

    @abstractmethod
    def set_impulse(self, impulse: Impulse) -> None:
        """
        直接添加一个 Impulse 到池中.
        """
        pass

    @abstractmethod
    def pause(self, toggle: bool) -> None:
        """
        急停, 仍然接受 signal/impulse, 但不会分发, 而是直接丢弃. 只有 set_ 系统指令有意义.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        立刻关闭 Mindflow.
        """
        pass

    @abstractmethod
    def loop(self) -> AsyncIterator[Attention]:
        """
        在生命周期中返回最新的 Attention, 方便定义清晰的 loop.
        每一轮 aborted 的 attention 应该要把异常结果提交给下一轮作为开始.
        """
        pass

    @abstractmethod
    async def __aenter__(self):
        """启动"""
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出"""
        pass


if __name__ == "__example__":
    """
    整套实现思路的应用构想. 只是一个举例, 细节未打磨. 
    """
    import janus


    def model(observation: Observation) -> Logos:
        """
        reasoning actions from observation
        generate logos for action.
        """
        pass


    side_thinking = False
    never_observe_again = False
    endless_thinking = False
    articulate_queue = janus.Queue[Articulate]()
    action_queue = janus.Queue[Action]()


    async def articulate_loop() -> None:
        """
        在单整个生命周期中, 连续响应多次 observation.
        """

        # 定义一个函数, 方便做独立生命周期管理.
        async def articulate_func(_articulate: Articulate) -> None:
            await articulate.send_logos(model(articulate.observation))

        while True:
            articulate = await articulate_queue.async_q.get()
            async with articulate:
                # 将生命周期与 articulate 的生命周期绑定.
                # 使之可以被异常取消.
                await articulate.create_task(articulate_func(articulate))


    def interpret(logos: Logos) -> AsyncIterator[tuple[list[Message], bool]]:
        """解释执行器"""
        pass


    async def _run_action(action: Action) -> None:
        async for messages, observe in interpret(action.logos()):
            action.outcome(*messages, observe=observe)


    async def action_loop() -> None:
        """
        执行 action 的循环.
        """
        while True:
            action = await action_queue.async_q.get()
            async with action:
                await action.create_task(_run_action(action))


    async def mindflow_main_loop(mindflow: Mindflow) -> None:
        async with mindflow:
            async for attention in mindflow.loop():
                # 展开 attention 的异常拦截作用域. 不拦截 fatal
                async with attention:
                    # 阻塞到 attention 运行结束或者中断.
                    async for articulate, action in attention.loop():
                        articulate_queue.sync_q.put_nowait(articulate)
                        action_queue.sync_q.put_nowait(action)
