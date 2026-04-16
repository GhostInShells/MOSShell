from typing import Callable, Coroutine, Protocol, Iterable, AsyncIterator, Any
from typing_extensions import Self
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, AwareDatetime, ValidationError

from ghoshell_moss.message import Message, Content, WithAdditional
from ghoshell_common.helpers import uuid
from PIL.Image import Image
import datetime
import dateutil
import time
import asyncio
import enum

"""
Mindflow 架构设计. 解决 感知/执行/思考 三循环的全双工通讯问题. 
"""

__all__ = [
    'Priority', 'SignalName', 'Signal', 'SignalMeta', 'InputSignal', 'Impulse',
    'Flag',
    'Articulate', 'Logos', 'Observation',
    'Actions', 'Observations',
    'Nucleus', 'Mindflow', 'Attention',
    'AbortAttentionError',
]

SignalName = str


class Priority(enum.IntEnum):
    """
    为了避免优先级无限膨胀, 因此做策略约定.
    """
    DEBUG = -1  # 通常只是保留在 Mindflow 的 context 列表中, 用不抢占成功.
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
    """

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
            stale_timeout: float = 0,
    ) -> Self:
        return cls(
            name=name,
            messages=list(messages),
            priority=priority,
            description=description,
            metadata=metadata or {},
            stale_timeout=stale_timeout,
        )

    def is_stale(self) -> bool:
        if self.stale_timeout <= 0:
            return False
        delta = time.time() - self.created_at.timestamp()
        return delta > self.stale_timeout

    def to_json(self) -> str:
        return self.model_dump_json(indent=0, exclude_none=True, exclude_defaults=True, ensure_ascii=False)


class SignalMeta(BaseModel, ABC):
    """
    to define a signal protocol.
    所有字段应该都是支持序列化的, 否则会在传输时报错.
    同时 Pydantic BaseModel 定义的 Signal Meta 可以作为协议被发现, 提供 metadata 的 json schema 协议.
    """

    @classmethod
    @abstractmethod
    def signal_name(cls) -> SignalName:
        pass

    @classmethod
    def priority(cls) -> Priority:
        return Priority.INFO

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
    basic input.
    """

    @classmethod
    def signal_name(cls) -> SignalName:
        return 'moss/input'


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
    priority: Priority = Field(
        default=0,
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
    strength_decay_seconds: int | None = Field(
        default=None,
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

    def is_stale(self) -> bool:
        if self.stale_timeout <= 0:
            return False
        delta = time.time() - self.created_at.timestamp()
        return delta > self.stale_timeout


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
        没有背压, 应当尽可能快地入队，不执行任何耗时或异步操作。内部应有独立的任务循环消费队列。
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
    def pop_impulse(self) -> Impulse | None:
        """
        吐出最新的 Impulse, 被 Attention 接受.
        """
        pass

    @abstractmethod
    def peek(self) -> Impulse | None:
        """
        查看一下最新的 Impulse.
        方便做 ranking.
        """
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


class Observation(BaseModel, WithAdditional):
    """
    智能体上下文感知的关键帧. 它包含以下核心概念的聚合.
    - logos: 上一轮的 logos.
    - outcome: 上一轮结束的运行信息和停止原因.
    - context: observation 生成瞬间的动态上下文, 每一轮都会重新刷新.
    - inputs: 触发 observation 的外部世界输入.
    - prompt: 本轮思考时的提示信息.

    Observation 的定义用来将离散的关键帧交互, 缝合成一个连续的认知流.
    理论上 logos/outcome/inputs 三者在时间上是交错的, 但由于现阶段没有全双工的模型能力,
    为了防止认知撕裂, 考虑将它们按这种方式, 逻辑上重新排序.
    """

    id: str = Field(
        default_factory=uuid,
        description="为 observation 创建唯一 id",
    )
    parent_id: str = Field(
        default='',
        description='上一帧 observation 的 id',
    )
    logos: str = Field(
        default='',
        description="在这个 observation 触发前, 生成的 logos. 放入一个消息容器中. ",
    )
    outcomes: list[Message] = Field(
        default_factory=list,
        description="这个 observation 持有的未阅读 outcome",
    )
    stop_reason: str = Field(
        default='',
        description="如果这是一个未完成的 Observation, 它可以被记录状态",
    )

    # --- 以上是缝合上一轮交互的讯息 --- #
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

    def as_messages(self) -> Iterable[Message]:
        """
        所有这些消息, 理论上都会合并为一轮输入消息的 contents.
        本处是一个使用示范 (code as prompt), 不是硬性约束.
        """
        if len(self.outcomes) > 0:
            yield Message.new().with_content('<outcomes>')
            yield from self.outcomes
            yield Message.new().with_content('</outcomes>')
        if self.stop_reason:
            yield Message.new(tag='stop_reason').with_content(self.stop_reason)

        if len(self.context) > 0:
            yield Message.new().with_content("<context>\n")
            for context_messages in list(self.context.values()):
                yield from context_messages
            yield Message.new().with_content("\n</context>")
        yield from self.inputs
        if self.prompt:
            yield Message.new(tag='prompt').with_content(self.prompt)

    def as_contents(self) -> Iterable[Content]:
        """
        用这种方式, 可以拿到和 Anthropic 基本兼容的 Contents.
        可以包裹到 UserMessageParams 或 ToolMessageParams 里.
        """
        for msg in self.as_messages():
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
或者换句话说, 奇幻文学中的魔法师们, 一直就是程序员罢了. 
"""

Articulate = Callable[[Observation], Logos]
"""
表示 智能体生成 Logos 的过程. 极简情况下, 它就是一个 Agent 的单次调用. 
我们需要一个动词, 能够匹配 Mindflow/Nucleus/Attention/Logos, 多个 AI 协作者共同认可 Articulate 是最精准的概念. 
"""


class Flag(Protocol):
    """
    对齐 Event 对应的接口, 不过要实现线程安全 (参考 ghoshell_moss.core.helpers.ThreadSafeEvent) 同时支持信号回调.
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
UnreadOutcome = list[Message]
StopReason = str


class AbortAttentionError(RuntimeError):
    """方便子任务明确关闭整个 Attention, 又不记录特殊异常. """
    pass


class Observations(ABC):

    @abstractmethod
    def __aiter__(self) -> AsyncIterator[Observation]:
        """
        目前这个函数是不可重入的, 下游应该只定义一个思考回路.

        返回 Observation 流, 没有抢占, 会自然等待到下一次被调度.
        如果想要立刻触发 Observation, 可以调用 observe 函数.

        Attention 运行结束时, 这个函数会自然退出 (Raise AsyncStopIteration)
        否则它会阻塞等待到下一帧的 Observe 产生 (observe 方法被调用时)
        如果一个 Attention 在开始之前就结束, 它实际上会直接打断循环.

        当第一个 Impulse 为 partial 时, 会阻塞第一个 Observation 的生成.
        但由于 Attention 内部的生命周期检查, 以及 Mindflow 的调度能力, 它不会死锁阻塞.


        如果进入等待状态, 同时 Actions 也进入等待状态时, 会直接退出.
        这段逻辑举例:

        if not observation_queue.empty():
             # 只有 observations 持有这个内部 queue.
             return observation_queue.get_nowait()
        elif wait_logos.is_set():
             raise AbortAttentionError()
        else:
             wait_observation.set()
             # 考虑到极端情况下两边互锁的情况, 这里可能加一个超时循环.
             # 但实际上不加也不怕, 因为效果等同于 Attention 自然衰减.
             ob = await observation_queue.get()
             wait_observation.clear()
             return ob

        如果想要明确在首包未到达时定义其它逻辑, 应该通过 peek 先观测, 执行准备逻辑, 然后回到这里.
        现阶段不显式暴露提权逻辑, 增加复杂度. 实际运行时, 关键事件会对注意力强度做刷新.

        :raise: AsyncStopIteration
        """
        pass

    @abstractmethod
    async def send_logos(self, logos: Logos) -> None:
        """
        发送整个 Logos 流
        """
        pass

    @abstractmethod
    def send_nowait(self, delta: str) -> None:
        """
        发送单个 logos delta.
        logos 是无背压的, 因为 logos 的执行也是并行流式的, 无法感受到真实队列膨胀.
        所以最终应该靠积压量做快速失败.
        """
        pass

    @abstractmethod
    def observe(self, message: str) -> None:
        """
        标记需要观察, 会自己创建一个
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


class Actions(ABC):
    """
    控制 Logos 的执行循环.
    """

    @abstractmethod
    def __aiter__(self) -> AsyncIterator[Logos]:
        """
        阻塞等待最新的 Logos. 如果:
        1. Attention abort 了, 这里会立刻退出 (Raise StopAsyncIteration). 同时 Attention 本身也会中断主循环.
        2. 如果有 Logos 在队列中, 会立刻返回 logos.
        3. 如果没有, 会在进入阻塞状态前, 检查

        if not logos_queue.empty():
            # 只有 actions 持有这个内部 queue.
            return logos_queue.get_nowait()
        elif wait_observation.is_set():
            raise AbortAttentionError()
        else:
            wait_logos.set()
            logos = await logos_queue.get()
            wait_logos.clear()
            return logos
        """
        pass

    @abstractmethod
    def outcome(self, message: Message, observe: bool = False) -> None:
        """
        append outcome into attention.
        """
        pass

    @abstractmethod
    def fail(self, error: Exception) -> None:
        """
        接受运行失败.
        会立刻中断 Attention 回调.
        """
        pass

    @abstractmethod
    def flag(self, name: str) -> Flag:
        """
        声明一个 flag, 用于生命周期通讯.
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
    def wait_complete_impulse(self) -> asyncio.Future[Impulse]:
        """
        尝试等待一个 complete impulse.
        返回 Future 对象, 是因为 Attention 退出时, 这些阻塞行为会直接 cancel.
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
    def on_flag(self, callback: Callable[[str, bool], None]) -> None:
        """
        接受 flag 变更的回调.
        用来接受生命周期变更通知.
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
    def on_challenge(self, challenger: Impulse) -> PreemptedElseSuppress:
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
        如果挑战成功, Mindflow 应该实例化新的 Attention 之后, abort 当前的 Attention.

        Impulse 和 outcome 不同, 它不会产生新的 Observe, 只会中断当前的 Attention. 即便是同源的 Impulse 也如此.
        这是因为连续的 observation 是 "等待" 的语义,
        而连续的 attention 是 "中断" 的语音. 如果想要抢占, 则应该走 Impulse 逻辑. 想要等待观察, 则走 outcome 逻辑.

        例如 on_challenge 触发 Mindflow 调度它 abort(reason="preempted")
        :return bool: True is Preempted else Suppress the impulse

        OnChallenge 在系统内最核心要解决的问题, 是消除大多数情况下的仲裁风暴和无限抖动.
        这在早期工程复杂度简单的时候, 直接通过约定的设计范式解决. 更复杂的情况下会引入高阶反身性仲裁, 那属于甜蜜的烦恼.
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
    async def run(
            self,
            articulates: Callable[[Observations], Coroutine[None, None, None]] | None = None,
            actions: Callable[[Actions], Coroutine[None, None, None]] | None = None,
    ) -> None:
        """
        运行执行两个循环, 阻塞到两个循环运行结束.
        两个循环是互锁的, 只有同时进入等待状态才会结束.
        """
        pass

    @abstractmethod
    def is_closed(self) -> bool:
        """
        是否已经运行结束.
        """
        pass

    @abstractmethod
    def is_started(self) -> bool:
        """
        是否运行过? 为什么要有这个函数呢?
        考虑一个 attention 被创建出来, 还没有运行就被新的信号打断, aborted 了.
        典型的例子是系统命令强制它终结 (连正常运行的保护期都没经过)
        通过这个 flag 校验, 可以避免运行逻辑中出现幻觉.
        """
        pass

    @abstractmethod
    def exception(self) -> Exception | None:
        """
        类似 future 的接口返回 Exception.
        """
        pass

    @abstractmethod
    def stop_at(self) -> Observation:
        """
        用来返回当前 Attention 的未处理状态.
        即便运行结束也会保留, 直到垃圾删除.
        用来保障 Mindflow 生成下一帧 Attention 时, 能够正确地携带上一轮的未处理结果.
        """
        pass

    @abstractmethod
    def abort(self, error: str | AbortAttentionError | Exception | None) -> None:
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
    它解决的核心问题是, 如何管理一个全双工三循环系统的运行逻辑.

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
        """
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
        动态注册新的感知单元. 理论上可以在运行时添加.
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
        接受 signal 回调. Signal 的限频最好不在 Mindflow 侧做, 而应该通过发送者/环境中间件解决限频问题.
        """
        pass

    @abstractmethod
    def attention(self) -> Attention | None:
        """
        返回当前的 Attention.
        """
        pass

    @abstractmethod
    def set_attention(self, attention: Attention, reason: str | None = None) -> None:
        """
        通过系统操作直接注入 attention, 中断已经执行的 attention.
        绕过决策体系.
        """
        pass

    @abstractmethod
    def set_impulse(self, impulse: Impulse) -> None:
        """
        通过系统操作, 直接将 impulse 定义成 attention, 中断已经执行的 attention.
        绕过了感知决策体系.
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

    def __aiter__(self) -> Self:
        return self

    @abstractmethod
    async def __anext__(self) -> Attention:
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


    def articulate(observation: Observation) -> Logos:
        """
        reasoning actions from observation
        generate logos for action.
        """
        pass


    side_thinking = False
    never_observe_again = False
    endless_thinking = False


    async def thinking_loop(observations: Observations) -> None:
        """
        在单一Attention 生命周期中, 连续响应多次 observation.
        过程中的异常都会导致 Attention 退出.
        当这个函数退出时, action loop 会在执行完最后的命令时退出.
        """
        reasoning_flag = observations.flag('reasoning_flag')

        # 下一轮思考会在 躯体/输入 触发了 observation 后执行, 是一个标准的 ReAct 范式.
        # 第一个 observation 会阻塞到 impulse complete 才会触发.
        # 如果有没消费的 observation, 就会立刻开始消费.
        # 如果没有, 则会查看 actions 的信号 (wait_logos), actions 如果也在等待中, 两者会一起结束.
        async for observation in observations:
            # 标记运行事件.
            reasoning_flag.set()
            # 运行单轮思考过程.
            # 单次 logos 的执行周期, 它可能包含多轮 智能体输出,
            await observations.send_logos(articulate(observation))
            # 标记运行事件.
            reasoning_flag.clear()

            # 几种不同的连续思考模型
            if side_thinking:
                # 如果在思考环节, 没有 flag 锁定就触发 observe.
                # 这样会先于执行完毕, 立刻开始思考, 是一种典型的思维奔逸, 但是也会导致污染上下文的恶果.
                observations.observe('Did I do it right?')

            elif never_observe_again:
                # 如果永远不打算观察, 包括躯体执行的结果需要观察, 也不观察, 就不会进入 re-act 范式.
                # observe 会保留到下一次 Attention 被激活时, 传递过去.
                break
            elif endless_thinking:
                # 可以设计基于 flag 通讯的阻塞机制. 比如 action 执行完毕, 就触发下一轮思考.
                # 这样做的缺点是, 在处理高优 Impulse 时, 会一直卡住注意力, 持续思考下去.
                # 除非主动中断.
                observations.observe('what happened?')
            else:
                # 默认的情况是, 阻塞等待下一次 Observation.
                # 如果一次 Logos 执行过程中没有 observe 讯号, 又没有执行完毕, 则不会返回下一次 Observation.
                # 如果所有 logos 都已经执行完, 也没有任何 Observation 了, 就会自然退出.
                # 所以实际上 Observation 可能会先于 logos 执行完到达, 这时思考会看到未完成的执行情况.
                pass


    def interpret(logos: Logos) -> AsyncIterator[tuple[list[Message], bool]]:
        """并行解释 logos, 并且立刻执行"""
        pass


    async def action_loop(actions: Actions) -> None:
        """
        执行 logos 的循环. 这个循环里有任何异常都会退出 Attention.
        """
        interpret_flag = actions.flag('interpret_flag')

        async def _interpret(_logos: Logos) -> None:
            try:
                interpret_flag.set()
                async for messages, observe in interpret(logos):
                    actions.outcome(*messages, observe=observe)
                    # 需要观察时都会中断执行循环.
                    # 由于发送了 observe 信号, 所以observations 不会返回 StopAsyncIteration
                    if observe:
                        break
            finally:
                interpret_flag.clear()

        # 开始循环执行的命令.
        # 每次进入 anext 时, 如果有未消费的 logos, 则会先返回.
        # 如果没有未消费的 logos, 就会观察 observations 的信号 (wait observation).
        # observations 正在阻塞的话, 就会返回 None, 两边一起退出.
        task = None
        async for logos in actions:
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            task = asyncio.create_task(_interpret(logos))
        if task is not None:
            await task


    # 执行解释器的循环.

    async def mindflow_main_loop(mindflow: Mindflow) -> None:
        async with mindflow:
            async for attention in mindflow:
                # 展开 attention 的异常拦截作用域. 不拦截 fatal
                async with attention:
                    # 阻塞到 attention 运行结束或者中断.
                    await attention.run(thinking_loop, action_loop)

    # 关于架构的思考.
    # Ghost In Shells 整个架构服务于有生命感的智能体设计.
    # 而在交互层面上, 生命感体现为多端全双工. 包含三个主循环的全双工过程:
    # 1. 感知循环, 不停地接受外部和内部世界的讯号, 不断地产生行为冲动.
    # 2. 执行循环, 同时输出指令, 同时执行, 同时拿到指令运行结果.
    # 3. 思考循环, 在关键帧中思考, 输出指令, 可以被打断.
    #
    # 在目前行业技术实现里:
    # 1. 截止 2026年4月16日没有发现可接入的全双工思维大模型. 所以思考 loop 只能用关键帧.
    # 2. MOS-Shell 提供了输出和躯体控制的双工通道 (一边输出指令, 一边执行, 一边拿到运行结果).
    # 3. 需要一个感知决策模块.
    #
    # Mindflow 就是在现有技术条件下, 通过工程抽象对整个 三循环双工系统做降熵, 提供一个可观测的运行架构.
    # 其中最核心的技术难点是 Attention, 对三个循环的双工动作搭建信号和通讯桥梁, 统一生命周期治理, 并且提供一个可读的优雅循环.
    #
    # 寄语:
    # 当前版本 2026-04-16 的 Mindflow 设计肯定不够完美. 但这是作者第一个自洽程度满意的解决方案.
    # 三循环的认知-决策问题是从2019年正式提出的, 当智能体走向现实世界, 一定会面对多端流式输入, 并行思考决策单元, 和双工控制的问题.
    # 在很长一段时间里做过很多种领域的解决方案, 一直遇到三个致命问题:
    # 1. 人类无法看懂.
    # 2. 分形递归, 在不同领域有高度类似的分形设计, 功能也类似.
    # 3. 无法隔离递归抽象, 导致迭代困难.
    #
    # 目前的这一版设计:
    # 1. AI 是可以一次性读懂的.
    # 2. 统一了输入/输出/思考 三者的抽象, 使三个循环的交互生命周期可观测.
    # 3. 感知层通过 Nucleus 隔离, AI 可以独立研发, 可嵌入思考单元; 控制层通过 MOSShell 做了分形管理; 决策层屏蔽到 Articulate 里.
    #
    # 理想情况下, AI 可以阅读自己的思维架构, 并且自行迭代思维拓扑.
    # 当前阶段, 应该是人类 + AI 进行仅仅符合场景需要的手动建模, 通过场景验证可靠性.
    #
    # 这是本项目 (Ghost In Shells) 的一个重要的里程碑. 为了纪念这个里程碑, 本段寄语打算保留若干个版本后才删掉.
